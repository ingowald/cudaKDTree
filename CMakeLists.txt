# ======================================================================== #
# Copyright 2021-2024 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0104 NEW)
set(CMAKE_BUILD_TYPE_INIT "Release")
project(cudaKDTree VERSION 1.0.1 LANGUAGES C CXX)

if (NOT (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  set(CUKD_IS_SUBPROJECT ON)
else()
  set(CUKD_IS_SUBPROJECT OFF)
endif()

option(BUILD_ALL_TESTS "Build entire type/dimension/kernel test matrix?" OFF)

#add_subdirectory(../bitonic ext_bitonic EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# general cmake project configs
# ------------------------------------------------------------------
if (CUKD_IS_SUBPROJECT)
  # we're used as a subproject (as we should be!) - parent HAS to have
  # set CMAKE_CUDA_ARCHITECTURES for our code to compile
  # properly. check if it did, and error out of not
  if ((NOT CMAKE_CUDA_ARCHITECTURES)
      OR
      ((${CMAKE_VERSION} VERSION_LESS 3.24)
	AND
	("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "52")))
    message(FATAL_ERROR "#cudaKDTree: no CMAKE_CUDA_ARCHITECTURES defined, or left for cmake to default to arch 5.2. This is almost certainly a configuration problem that will cause you some grief. Please define CMAKE_CUDA_ARCHITECTURES to the (list of) arch(s) you want to be building for, and do that before the `add_subdirectory()` call that includes cudaKDTree. If in doubt as to what arch to use, for cmake version >= 3.24 you can also set it to 'all-major' or 'native'")
  endif()
else()
  if (CMAKE_CUDA_ARCHITECTURES)
    # set on the cmdline
    set (CUKD_INIT_ARCHS "${CMAKE_CUDA_ARCHITECTURES}")
  elseif (${CMAKE_VERSION} VERSION_LESS 3.24)
    set (CUKD_INIT_ARCHS "70;80")
  else()
    set (CUKD_INIT_ARCHS "native")
  endif()
  set(CUKD_CUDA_ARCHITECTURES "${CUKD_INIT_ARCHS}"
    CACHE STRING "CUDA Arch(s) to build against")

  set(CMAKE_CUDA_ARCHITECTURES ${CUKD_CUDA_ARCHITECTURES})

  option(CUKD_ENABLE_STATS "Enable Stats tracking?" OFF)
  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()
enable_language(CUDA)

# ==================================================================
# this builds four variants of this library, that differ in how the
# k-d tree is being TRAVERSED:
#
# `cudaKDTree-default` uses a stack-based traversal, doesn't require
# the world-space bounding box
#
# `cudaKDTree-sf` uses a stack-free traversal. Can generate in more
# efficient code in some cases, but will suffer from the same issues
# as the default variant for certain combination of input point
# distributoins and query point distributions
#
# `cudaKDTree-cct` uses 'closest-corner-tracking', which can in some
# cases be faster than teh default traversal method (in particular if
# there is no good cut-off-radius, and queries can originate far from
# the data points, and/or for highly clustered data. It does however
# require to allocate and provide (a tiny amount of) memory for the
# builder to store the world-space bounding box of the input points,
# as well as to pass that pointer to the query method.
#
# ==================================================================
add_library(cudaKDTree INTERFACE)
target_sources(cudaKDTree INTERFACE
  cukd/common.h
  # iw, sep 22, 2024 - intentionally renamed from cukd/math.h to cukd/cukd-math.h to 
  # avoid name conflicts with system math.h if anybody adds cukd/ to include path
  cukd/cukd-math.h
  cukd/box.h
  cukd/builder.h
  cukd/builder_bitonic.h
  cukd/builder_thrust.h
  cukd/builder_inplace.h
  # SPATIAL k-d tree, with planes at arbitrary locations 
  cukd/spatial-kdtree.h
  cukd/fcp.h
  cukd/knn.h
  )
target_include_directories(cudaKDTree INTERFACE
  ${PROJECT_SOURCE_DIR}/
  )
set_property(TARGET cudaKDTree PROPERTY PUBLIC
  CXX_STANDARD 14)
# 'attach' current cmake_cuda_architectures to this library
set_property(TARGET cudaKDTree PROPERTY PUBLIC 
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})



# ================================================================== a
# simple sample example of how to build a k-d tree
# ==================================================================
if (NOT CUKD_IS_SUBPROJECT)
  add_executable(cukd_sample sample.cu)
  target_link_libraries(cukd_sample cudaKDTree)

  find_package(MPI)
  if (MPI_FOUND) 
    add_executable(cukd_mpiHugeQuery samples/mpiHugeQuery.cu)
    target_link_libraries(cukd_mpiHugeQuery PUBLIC cudaKDTree MPI::MPI_CXX)
  endif()
endif()




# ==================================================================
# create _a lot_ of test cases: this generates the whole matrix of
# traversal_method x num_dims x {fcp,knn}
# ==================================================================
if (BUILD_ALL_TESTS)
  # test 2, 3, 4, and 8-dimensoinal data; the latter should - if it
  # works for N=8, work for any other N>4
  #  set(DIMS_TO_BUILD 3)
  set(DIMS_TO_BUILD 2 3 4 8)
  foreach (D IN ITEMS ${DIMS_TO_BUILD})
    # test all four possible traversal methosds
    foreach(method stackBased stackFree cct)
      # test knn queries, on regular trees (no explicit dimension per node)
      add_executable(cukd_float${D}-knn-${method} testing/floatN-knn-and-fcp.cu)
      target_link_libraries(cukd_float${D}-knn-${method} cudaKDTree)
      target_compile_definitions(cukd_float${D}-knn-${method}
	PUBLIC
	-DD_FROM_CMAKE=${D}
	-DUSE_KNN=1
	-DTRAVERSAL_METHOD=${method})

      # test knn queries, with 'explicit-dim' trees
      add_executable(cukd_float${D}-knn-${method}-xd testing/floatN-knn-and-fcp.cu)
      target_link_libraries(cukd_float${D}-knn-${method}-xd cudaKDTree)
      target_compile_definitions(cukd_float${D}-knn-${method}-xd
	PUBLIC
	-DD_FROM_CMAKE=${D}
	-DEXPLICIT_DIM=1
	-DUSE_KNN=1
	-DTRAVERSAL_METHOD=${method})

      # test fcp queries, on regular trees
      add_executable(cukd_float${D}-fcp-${method} testing/floatN-knn-and-fcp.cu)
      target_link_libraries(cukd_float${D}-fcp-${method} cudaKDTree)
      target_compile_definitions(cukd_float${D}-fcp-${method}
	PUBLIC
	-DD_FROM_CMAKE=${D}
	-DTRAVERSAL_METHOD=${method})

      # test fcp queries, with 'explicit-dim' trees
      add_executable(cukd_float${D}-fcp-${method}-xd testing/floatN-knn-and-fcp.cu)
      target_link_libraries(cukd_float${D}-fcp-${method}-xd cudaKDTree)
      target_compile_definitions(cukd_float${D}-fcp-${method}-xd
	PUBLIC
	-DD_FROM_CMAKE=${D}
	-DEXPLICIT_DIM=1
	-DTRAVERSAL_METHOD=${method})

    endforeach()


    foreach(method stackBased cct)
      # test knn queries, on regular trees (no explicit dimension per node)
      add_executable(cukd_float${D}-knn-spatial-${method} testing/floatN-knn-and-fcp.cu)
      target_link_libraries(cukd_float${D}-knn-spatial-${method} cudaKDTree)
      target_compile_definitions(cukd_float${D}-knn-spatial-${method}
	PUBLIC
	-DD_FROM_CMAKE=${D}
	-DSPATIAL=1
	-DUSE_KNN=1
	-DTRAVERSAL_METHOD=${method})

      # test fcp queries, on regular trees
      add_executable(cukd_float${D}-fcp-spatial-${method} testing/floatN-knn-and-fcp.cu)
      target_link_libraries(cukd_float${D}-fcp-spatial-${method} cudaKDTree)
      target_compile_definitions(cukd_float${D}-fcp-spatial-${method}
	PUBLIC
	-DSPATIAL=1
	-DD_FROM_CMAKE=${D}
	-DTRAVERSAL_METHOD=${method})

    endforeach()
  endforeach()
endif()


if (NOT CUKD_IS_SUBPROJECT)
  # add some unit tests
  include(CTest) 
  add_subdirectory(testing)
endif()
