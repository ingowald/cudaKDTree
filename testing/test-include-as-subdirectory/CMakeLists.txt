cmake_minimum_required(VERSION 3.28)

project(test-cukd)

set(CMAKE_CUDA_ARCHITECTURES native)
add_subdirectory(../../ BUILD_CUKD EXCLUDE_FROM_ALL)

add_executable(testBuild-sample-from-subdirectory
  ../../sample.cu
  )
target_link_libraries(testBuild-sample-from-subdirectory
  cudaKDTree
  )
