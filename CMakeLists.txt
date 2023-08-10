# ======================================================================== #
# Copyright 2021-2023 Ingo Wald                                            #
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

cmake_policy(SET CMP0048 NEW)
project(cudaKDTree VERSION 0.0.1 LANGUAGES C CXX CUDA)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 NEW)
endif()

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
if (NOT CUKD_IS_SUBPROJECT)
  option(CUKD_ENABLE_STATS "Enable Stats tracking?" OFF)
  if(NOT SET_UP_CONFIGURATIONS_DONE)
    set(SET_UP_CONFIGURATIONS_DONE 1)

    # No reason to set CMAKE_CONFIGURATION_TYPES if it's not a multiconfig generator
    # Also no reason mess with CMAKE_BUILD_TYPE if it's a multiconfig generator.
    if(CMAKE_CONFIGURATION_TYPES) # multiconfig generator?
      set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE) 
    else()
      if(NOT CMAKE_BUILD_TYPE)
	#            message("Defaulting to release build.")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
      endif()
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
      # set the valid options for cmake-gui drop-down list
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")
    endif()
  endif()

  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

  set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "Specific CUDA Arch(s) to build against")
  #set(CMAKE_CUDA_ARCHITECTURES 70 75 86)
  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.17)
    # nothing to do, setting CMAKE_CUDA_ARCHITECTURES is the way to go
  else()
    foreach (arch IN ITEMS ${CMAKE_CUDA_ARCHITECTURES})
      #    message("status adding arch=${arch}")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
    endforeach()
  endif()

else()
  if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
  endif()
endif()


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
# as well as to pass aht pointer to the query method.
#
# ==================================================================
add_library(cudaKDTree INTERFACE)
target_sources(cudaKDTree INTERFACE
  cukd/common.h
  cukd/math.h
  cukd/box.h
  cukd/builder.h
  cukd/builder_bitonic.h
  cukd/builder_thrust.h
  cukd/builder_inplace.h
  # SPATIAL k-d tree, with planes at arbitrary locatoins 
  cukd/spatial-kdtree.h
  cukd/fcp.h
  cukd/knn.h
  )
target_include_directories(cudaKDTree INTERFACE
  ${PROJECT_SOURCE_DIR}/
  )


# ================================================================== a
# simple sample example of how to build a k-d tree
# ==================================================================
if (NOT CUKD_IS_SUBPROJECT)
  add_executable(cukd_sample sample.cu)
  target_link_libraries(cukd_sample cudaKDTree)
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
	-DUSE_KNN=1
	-DTRAVERSAL_METHOD=${method})

      # test fcp queries, on regular trees
      add_executable(cukd_float${D}-fcp-spatial-${method} testing/floatN-knn-and-fcp.cu)
      target_link_libraries(cukd_float${D}-fcp-spatial-${method} cudaKDTree)
      target_compile_definitions(cukd_float${D}-fcp-spatial-${method}
	PUBLIC
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
