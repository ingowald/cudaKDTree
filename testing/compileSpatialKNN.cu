// ======================================================================== //
// Copyright 2018-2023 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "cukd/knn.h"

using namespace cukd;

__global__
void invokeQueries(SpatialKDTree<float3> *d_tree, float *d_results, float3 *d_queries)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  
  HeapCandidateList<100> stackHeapResults(10.f);
  stackBased::knn(stackHeapResults,*d_tree,d_queries[tid]);

  FixedCandidateList<4> stackListResults(10.f);
  stackBased::knn(stackListResults,*d_tree,d_queries[tid]);
  
  HeapCandidateList<100> cctHeapResults(10.f);
  cct::knn(cctHeapResults,*d_tree,d_queries[tid]);
  
  FixedCandidateList<4> cctListResults(10.f);
  cct::knn(cctListResults,*d_tree,d_queries[tid]);

  d_results[tid]
    = stackHeapResults.maxRadius2()
    + stackListResults.maxRadius2()
    + cctHeapResults.maxRadius2()
    + cctListResults.maxRadius2();
}

int main(int, const char **)
{
  /* this only tests _compile_ capability */
  return 0;
}

