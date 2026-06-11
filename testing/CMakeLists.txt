# ======================================================================== #
# Copyright 2023-2024 Ingo Wald                                            #
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

# directory for both "real" test cases and unit testing

if (USE_HIP)
  project(cukdTests LANGUAGES HIP CXX)
else()
  project(cukdTests LANGUAGES CUDA CXX)
endif()

# test all builders on both simple float3 and Photon types, empty inputs
add_executable(cukdTestBitonicEmptyInput testBuilderEmptyInput.cu)
target_compile_definitions(cukdTestBitonicEmptyInput PUBLIC -DBUILDER_TO_TEST=buildTree_bitonic)
target_link_libraries(cukdTestBitonicEmptyInput PRIVATE cudaKDTree)
cukd_gpu_sources(cukdTestBitonicEmptyInput)
add_test(NAME cukdTestBitonicEmptyInput COMMAND cukdTestBitonicEmptyInput)

add_executable(cukdTestThrustEmptyInput testBuilderEmptyInput.cu)
target_compile_definitions(cukdTestThrustEmptyInput PUBLIC -DBUILDER_TO_TEST=buildTree_thrust)
target_link_libraries(cukdTestThrustEmptyInput PRIVATE cudaKDTree)
cukd_gpu_sources(cukdTestThrustEmptyInput)
add_test(NAME cukdTestThrustEmptyInput COMMAND cukdTestThrustEmptyInput)

add_executable(cukdTestInPlaceEmptyInput testBuilderEmptyInput.cu)
target_compile_definitions(cukdTestInPlaceEmptyInput PUBLIC -DBUILDER_TO_TEST=buildTree_inPlace)
target_link_libraries(cukdTestInPlaceEmptyInput PRIVATE cudaKDTree)
cukd_gpu_sources(cukdTestInPlaceEmptyInput)
add_test(NAME cukdTestInPlaceEmptyInput COMMAND cukdTestInPlaceEmptyInput)

# test all builders on both simple float3 and Photon types, simple 1000 random points
add_executable(cukdTestBitonicSimpleInput testBuilderSimpleInput.cu)
target_compile_definitions(cukdTestBitonicSimpleInput PUBLIC -DBUILDER_TO_TEST=buildTree_bitonic)
target_link_libraries(cukdTestBitonicSimpleInput PRIVATE cudaKDTree)
cukd_gpu_sources(cukdTestBitonicSimpleInput)
add_test(NAME cukdTestBitonicSimpleInput COMMAND cukdTestBitonicSimpleInput)

add_executable(cukdTestThrustSimpleInput testBuilderSimpleInput.cu)
target_compile_definitions(cukdTestThrustSimpleInput PUBLIC -DBUILDER_TO_TEST=buildTree_thrust)
target_link_libraries(cukdTestThrustSimpleInput PRIVATE cudaKDTree)
cukd_gpu_sources(cukdTestThrustSimpleInput)
add_test(NAME cukdTestThrustSimpleInput COMMAND cukdTestThrustSimpleInput)

add_executable(cukdTestInPlaceSimpleInput testBuilderSimpleInput.cu)
target_compile_definitions(cukdTestInPlaceSimpleInput PUBLIC -DBUILDER_TO_TEST=buildTree_inPlace)
target_link_libraries(cukdTestInPlaceSimpleInput PRIVATE cudaKDTree)
cukd_gpu_sources(cukdTestInPlaceSimpleInput)
add_test(NAME cukdTestInPlaceSimpleInput COMMAND cukdTestInPlaceSimpleInput)


add_executable(cukdTestPayloadSampleFromReadme testPayloadSampleFromReadme.cu)
target_compile_definitions(cukdTestPayloadSampleFromReadme PUBLIC -DBUILDER_TO_TEST=buildTree_thrust)
target_link_libraries(cukdTestPayloadSampleFromReadme PRIVATE cudaKDTree)
cukd_gpu_sources(cukdTestPayloadSampleFromReadme)
add_test(NAME cukdTestPayloadSampleFromReadme COMMAND cukdTestPayloadSampleFromReadme)



add_executable(cukdTestHostBuilderEmptyInput testHostBuilderEmptyInput.cu)
target_link_libraries(cukdTestHostBuilderEmptyInput PRIVATE cudaKDTree)
cukd_gpu_sources(cukdTestHostBuilderEmptyInput)
add_test(NAME cukdTestHostBuilderEmptyInput COMMAND cukdTestHostBuilderEmptyInput)

add_executable(cukdTestHostBuilderSimpleInput testHostBuilderSimpleInput.cu)
target_link_libraries(cukdTestHostBuilderSimpleInput PRIVATE cudaKDTree)
cukd_gpu_sources(cukdTestHostBuilderSimpleInput)
add_test(NAME cukdTestHostBuilderSimpleInput COMMAND cukdTestHostBuilderSimpleInput)


# tests, for a wide range of input data, whether host, thrust,
# bitonic, and inplace builders all produce the same tree.
add_executable(cukdTestBuildersSameResult testBuildersSameResult.cu)
target_link_libraries(cukdTestBuildersSameResult PRIVATE cudaKDTree)
cukd_gpu_sources(cukdTestBuildersSameResult)
add_test(NAME cukdTestBuildersSameResult COMMAND cukdTestBuildersSameResult)



# make sure all knn variants for a _spatial_ k-d tree will at least compile
add_executable(cukdTestCompileSpatialKNN compileSpatialKNN.cu)
target_link_libraries(cukdTestCompileSpatialKNN PRIVATE cudaKDTree)
cukd_gpu_sources(cukdTestCompileSpatialKNN)
add_test(NAME cukdTestCompileSpatialKNN COMMAND cukdTestCompileSpatialKNN)
# make sure all knn variants for a _spatial_ k-d tree will at least compile
add_executable(cukdTestCompileKNN compileKNN.cu)
target_link_libraries(cukdTestCompileKNN PRIVATE cudaKDTree)
cukd_gpu_sources(cukdTestCompileKNN)
add_test(NAME cukdTestCompileKNN COMMAND cukdTestCompileKNN)



# add a (compile-only) test to see if we can link two different object
# files (that both include the same builders) without getting any
# multiple definition errors.
add_executable(cukdTestMultipleDefinitions
  testMultipleDefinitions_a.cu  testMultipleDefinitions_b.cu)
target_link_libraries(cukdTestMultipleDefinitions PRIVATE cudaKDTree)
cukd_gpu_sources(cukdTestMultipleDefinitions)


# ==================================================================
# issue 5: reported wrong/inconsisten results for differnet builders,
# in a given set generated by given random seed
# ==================================================================
#
add_executable(cukdTestIssue5_thrust issue5.cu)
target_link_libraries(cukdTestIssue5_thrust PRIVATE cudaKDTree)
target_compile_definitions(cukdTestIssue5_thrust PUBLIC BUILDER_TO_TEST=buildTree_thrust)
cukd_gpu_sources(cukdTestIssue5_thrust)
add_test(NAME cukdTestIssue5_thrust COMMAND cukdTestIssue5_thrust)
#
add_executable(cukdTestIssue5_bitonic issue5.cu)
target_link_libraries(cukdTestIssue5_bitonic PRIVATE cudaKDTree)
target_compile_definitions(cukdTestIssue5_bitonic PUBLIC BUILDER_TO_TEST=buildTree_bitonic)
cukd_gpu_sources(cukdTestIssue5_bitonic)
add_test(NAME cukdTestIssue5_bitonic COMMAND cukdTestIssue5_bitonic)
#
add_executable(cukdTestIssue5_inPlace issue5.cu)
target_link_libraries(cukdTestIssue5_inPlace PRIVATE cudaKDTree)
target_compile_definitions(cukdTestIssue5_inPlace PUBLIC BUILDER_TO_TEST=buildTree_inPlace)
cukd_gpu_sources(cukdTestIssue5_inPlace)
add_test(NAME cukdTestIssue5_inPlace COMMAND cukdTestIssue5_inPlace)



enable_testing()
