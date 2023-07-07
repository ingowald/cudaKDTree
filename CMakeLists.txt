# ======================================================================== #
# Copyright 2021-2022 Ingo Wald                                            #
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
  set(CUKD_IS_SUBMODULE ON)
else()
  set(CUKD_IS_SUBPROJECT OFF)
endif()

# ------------------------------------------------------------------
# general cmake project configs
# ------------------------------------------------------------------
if (NOT CUKD_IS_SUBPROJECT)
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
endif()

set(CMAKE_CUDA_ARCHITECTURES 70)
#set(CMAKE_CUDA_ARCHITECTURES 70 75 86)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.17)
  # nothing to do, setting CMAKE_CUDA_ARCHITECTURES is hte way to go
else()
  foreach (arch IN ITEMS ${CMAKE_CUDA_ARCHITECTURES})
#    message("status adding arch=${arch}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
  endforeach()
endif()



#set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
#if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.17)
  # nothing to do, setting CMAKE_CUDA_ARCHITECTURES is hte way to go
#else()
#  foreach (arch IN ITEMS ${CMAKE_CUDA_ARCHITECTURES})
#    message("status adding arch=${arch}")
#    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
#  endforeach()
#endif()


foreach(stack_free ON OFF)
  foreach(improved ON OFF)
    message(STATUS "stack_free ${stack_free}")
    message(STATUS "improved ${improved}")
    if (${stack_free})
      set(suffix "-sf")
    else()
      set(suffix "-sb")
    endif()
    
    if (${improved})
      set(suffix "${suffix}-imp")
    else()
      set(suffix "${suffix}-reg")
    endif()

    add_library(cudaKDTree${suffix} INTERFACE)
    target_sources(cudaKDTree${suffix} INTERFACE
      cukd/common.h
      cukd/builder.h
      cukd/fcp.h
      cukd/knn.h
      )
    target_include_directories(cudaKDTree${suffix} INTERFACE
      #target_link_libraries(cudaKDTree INTERFACE
      ${PROJECT_SOURCE_DIR}/
      )
    if (${improved})
      target_compile_definitions(cudaKDTree${suffix} INTERFACE -DCUKD_IMPROVED_TRAVERSAL=1)
    endif()
    if (${stack_free})
      target_compile_definitions(cudaKDTree${suffix} INTERFACE -DCUKD_STACK_FREE=1)
    endif()

    # add_executable(cukd_float2-fcp${suffix} testing/float2-fcp.cu)
    # target_link_libraries(cukd_float2-fcp${suffix} cudaKDTree${suffix})
    
    # add_executable(cukd_float3-fcp${suffix} testing/float3-fcp.cu)
    # target_link_libraries(cukd_float3-fcp${suffix} cudaKDTree${suffix})
        
    # add_executable(cukd_float4-fcp${suffix} testing/float4-fcp.cu)
    # target_link_libraries(cukd_float4-fcp${suffix} cudaKDTree${suffix})

    # add_executable(cukd_float2-knn${suffix} testing/float2-knn.cu)
    # target_link_libraries(cukd_float2-knn${suffix} cudaKDTree${suffix})
    
    # add_executable(cukd_float3-knn${suffix} testing/float3-knn.cu)
    # target_link_libraries(cukd_float3-knn${suffix} cudaKDTree${suffix})

    
    set(DIMS_TO_BUILD 2 3 4 8)
    foreach (D IN ITEMS ${DIMS_TO_BUILD})
      add_executable(cukd_float${D}-knn${suffix} testing/floatN-knn.cu)
      target_link_libraries(cukd_float${D}-knn${suffix} cudaKDTree${suffix})
      target_compile_definitions(cukd_float${D}-knn${suffix} PUBLIC -DD_FROM_CMAKE=${D})

      add_executable(cukd_float${D}-fcp${suffix} testing/floatN-fcp.cu)
      target_link_libraries(cukd_float${D}-fcp${suffix} cudaKDTree${suffix})
      target_compile_definitions(cukd_float${D}-fcp${suffix} PUBLIC -DD_FROM_CMAKE=${D})
    endforeach()

  endforeach()
endforeach()
