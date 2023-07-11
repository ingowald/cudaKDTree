#if 1
#define USE_KNN 1
#include "floatN-fcp.cu"
#else
// ======================================================================== //
// Copyright 2022-2023 Ingo Wald                                            //
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

#include "cukd/builder.h"
// knn = "k-nearest-neighbor" query
#include "cukd/knn.h"
#include <queue>
#include <limits>
#include <iomanip>

using namespace cukd;
using namespace cukd::common;


#if D_FROM_CMAKE == 2
using floatN = float2;
#elif D_FROM_CMAKE == 3
using floatN = float3;
#elif D_FROM_CMAKE == 4
using floatN = float4;
#elif D_FROM_CMAKE == 8
using floatN = cukd::vec_float<8>;
#else
#pragma error("error ... should get a value of 2, 3, or 4 from cmakefile...")
#endif

// ==================================================================
template<typename CandidateList>
__global__ void d_knn(unsigned long long *d_stats,
                      float *d_results,
                      floatN *d_queries,
                      int numQueries,
#if CUKD_IMPROVED_TRAVERSAL
                      const cukd::box_t<floatN> *d_bounds,
#endif
                      floatN *d_nodes,
                      int numNodes,
                      float maxRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;
  
  CandidateList result(maxRadius);
  float sqrDist
    = cukd::knn(d_stats,result,d_queries[tid],
#if CUKD_IMPROVED_TRAVERSAL
                *d_bounds,
#endif
                d_nodes,numNodes);
  d_results[tid] = sqrtf(sqrDist);
}

template<typename CandidateList>
void knn(float *d_results,
         floatN *d_queries,
         int numQueries,
#if CUKD_IMPROVED_TRAVERSAL
         const cukd::box_t<floatN> *d_bounds,
#endif
         floatN *d_nodes,
         int numNodes,
         float maxRadius)
{
  int bs = 128;
  int nb = divRoundUp(numQueries,bs);
  unsigned long long *d_stats = 0;
  static bool firstTime = true;
  if (firstTime) {
    cudaMallocManaged((char **)&d_stats,sizeof(*d_stats));
    *d_stats = 0;
  }
  d_knn<CandidateList><<<nb,bs>>>(d_stats,d_results,d_queries,numQueries,
#if CUKD_IMPROVED_TRAVERSAL
                                  d_bounds,
#endif
                                  d_nodes,numNodes,maxRadius);
  if (firstTime) {
    cudaDeviceSynchronize();
    std::cout << "KDTREE_STATS " << *d_stats << std::endl;
    std::cout << "NICE_STATS " << prettyNumber(*d_stats) << std::endl;
    cudaFree(d_stats);
    firstTime = false;
  }
}



// __global__ void d_knn4(unsigned long long *d_stats,
//                        float *d_results,
//                        floatN *d_queries,
//                        int numQueries,
// #if CUKD_IMPROVED_TRAVERSAL
//                        const cukd::box_t<floatN> *d_bounds,
// #endif
//                        floatN *d_nodes,
//                        int numNodes,
//                        float maxRadius)
// {
//   int tid = threadIdx.x+blockIdx.x*blockDim.x;
//   if (tid >= numQueries) return;

//   cukd::FixedCandidateList<4> result(maxRadius);
//   float sqrDist
//     = cukd::knn
//     <cukd::FixedCandidateList<4>>
//     (d_stats,result,d_queries[tid],
// #if CUKD_IMPROVED_TRAVERSAL
//      *d_bounds,
// #endif
//      d_nodes,numNodes);
//   d_results[tid] = sqrtf(sqrDist);
// }

// void knn4(float *d_results,
//           floatN *d_queries,
//           int numQueries,
// #if CUKD_IMPROVED_TRAVERSAL
//           const cukd::box_t<floatN> *d_bounds,
// #endif
//           floatN *d_nodes,
//           int numNodes,
//           float maxRadius)
// {
//   int bs = 128;
//   int nb = divRoundUp(numQueries,bs);
//   unsigned long long *d_stats = 0;
//   static bool firstTime = true;
//   if (firstTime) {
//     cudaMallocManaged((char **)&d_stats,sizeof(*d_stats));
//     *d_stats = 0;
//   }
//   d_knn4<<<nb,bs>>>(d_stats,d_results,d_queries,numQueries,
// #if CUKD_IMPROVED_TRAVERSAL
//                     d_bounds,
// #endif
//                     d_nodes,numNodes,maxRadius);
//   if (firstTime) {
//     cudaDeviceSynchronize();
//     std::cout << "KDTREE_STATS " << *d_stats << std::endl;
//     std::cout << "NICE_STATS " << prettyNumber(*d_stats) << std::endl;
//     cudaFree(d_stats);
//     firstTime = false;
//   }
// }


// // ==================================================================
// __global__ void d_knn8(unsigned long long *d_stats,
//                        float *d_results,
//                        floatN *d_queries,
//                        int numQueries,
// #if CUKD_IMPROVED_TRAVERSAL
//                        const cukd::box_t<floatN> *d_bounds,
// #endif
//                        floatN *d_nodes,
//                        int numNodes,
//                        float maxRadius)
// {
//   int tid = threadIdx.x+blockIdx.x*blockDim.x;
//   if (tid >= numQueries) return;

//   cukd::FixedCandidateList<8> result(maxRadius);
//   float sqrDist
//     = cukd::knn
//     <cukd::TrivialFloatPointTraits<floatN>>
//     (d_stats,result,d_queries[tid],
// #if CUKD_IMPROVED_TRAVERSAL
//      *d_bounds,
// #endif
//      d_nodes,numNodes);
//   d_results[tid] = sqrtf(sqrDist);
// }

// void knn8(float *d_results,
//           floatN *d_queries,
//           int numQueries,
// #if CUKD_IMPROVED_TRAVERSAL
//           const cukd::box_t<floatN> *d_bounds,
// #endif
//           floatN *d_nodes,
//           int numNodes,
//           float maxRadius)
// {
//   int bs = 128;
//   int nb = divRoundUp(numQueries,bs);
//   unsigned long long *d_stats = 0;
//   static bool firstTime = true;
//   if (firstTime) {
//     cudaMallocManaged((char **)&d_stats,sizeof(*d_stats));
//     *d_stats = 0;
//   }
//   d_knn8<<<nb,bs>>>(d_stats,d_results,d_queries,numQueries,
// #if CUKD_IMPROVED_TRAVERSAL
//                     d_bounds,
// #endif
//                     d_nodes,numNodes,maxRadius);
//   if (firstTime) {
//     cudaDeviceSynchronize();
//     std::cout << "KDTREE_STATS " << *d_stats << std::endl;
//     std::cout << "NICE_STATS " << prettyNumber(*d_stats) << std::endl;
//     cudaFree(d_stats);
//     firstTime = false;
//   }
// }


// // ==================================================================
// __global__ void d_knn64(unsigned long long *d_stats,
//                         float *d_results,
//                         floatN *d_queries,
//                         int numQueries,
// #if CUKD_IMPROVED_TRAVERSAL
//                         const cukd::box_t<floatN> *d_bounds,
// #endif
//                         floatN *d_nodes,
//                         int numNodes,
//                         float maxRadius)
// {
//   int tid = threadIdx.x+blockIdx.x*blockDim.x;
//   if (tid >= numQueries) return;

//   cukd::FixedCandidateList<64> result(maxRadius);
//   float sqrDist
//     = cukd::knn
//     <cukd::TrivialFloatPointTraits<floatN>>
//     (d_stats,result,d_queries[tid],
// #if CUKD_IMPROVED_TRAVERSAL
//      *d_bounds,
// #endif
//      d_nodes,numNodes);
//   d_results[tid] = sqrtf(sqrDist);
// }

// void knn64(float *d_results,
//            floatN *d_queries,
//            int numQueries,
// #if CUKD_IMPROVED_TRAVERSAL
//            const cukd::box_t<floatN> *d_bounds,
// #endif
//            floatN *d_nodes,
//            int numNodes,
//            float maxRadius)
// {
//   int bs = 128;
//   int nb = divRoundUp(numQueries,bs);
//   unsigned long long *d_stats = 0;
//   static bool firstTime = true;
//   if (firstTime) {
//     cudaMallocManaged((char **)&d_stats,sizeof(*d_stats));
//     *d_stats = 0;
//   }
//   d_knn64<<<nb,bs>>>(d_stats,d_results,d_queries,numQueries,
// #if CUKD_IMPROVED_TRAVERSAL
//                      d_bounds,
// #endif
//                      d_nodes,numNodes,maxRadius);
//   if (firstTime) {
//     cudaDeviceSynchronize();
//     std::cout << "KDTREE_STATS " << *d_stats << std::endl;
//     std::cout << "NICE_STATS " << prettyNumber(*d_stats) << std::endl;
//     cudaFree(d_stats);
//     firstTime = false;
//   }
// }


// // ==================================================================
// __global__ void d_knn20(unsigned long long *d_stats,
//                         float *d_results,
//                         floatN *d_queries,
//                         int numQueries,
// #if CUKD_IMPROVED_TRAVERSAL
//                         const cukd::box_t<floatN> *d_bounds,
// #endif
//                         floatN *d_nodes,
//                         int numNodes,
//                         float maxRadius)
// {
//   int tid = threadIdx.x+blockIdx.x*blockDim.x;
//   if (tid >= numQueries) return;

//   cukd::HeapCandidateList<20> result(maxRadius);
//   d_results[tid]
//     = sqrtf(cukd::knn
//             <cukd::TrivialFloatPointTraits<floatN>>
//             (d_stats,result,d_queries[tid],
// #if CUKD_IMPROVED_TRAVERSAL
//              *d_bounds,
// #endif
//              d_nodes,numNodes));
// }

// void knn20(float *d_results,
//            floatN *d_queries,
//            int numQueries,
// #if CUKD_IMPROVED_TRAVERSAL
//            const cukd::box_t<floatN> *d_bounds,
// #endif
//            floatN *d_nodes,
//            int numNodes,
//            float maxRadius)
// {
//   int bs = 128;
//   int nb = divRoundUp(numQueries,bs);
//   unsigned long long *d_stats = 0;
//   static bool firstTime = true;
//   if (firstTime) {
//     cudaMallocManaged((char **)&d_stats,sizeof(*d_stats));
//     *d_stats = 0;
//   }
//   d_knn20<<<nb,bs>>>(d_stats,d_results,d_queries,numQueries,
// #if CUKD_IMPROVED_TRAVERSAL
//                      d_bounds,
// #endif
//                      d_nodes,numNodes,maxRadius);
//   if (firstTime) {
//     cudaDeviceSynchronize();
//     std::cout << "KDTREE_STATS " << *d_stats << std::endl;
//     std::cout << "NICE_STATS " << prettyNumber(*d_stats) << std::endl;
//     cudaFree(d_stats);
//     firstTime = false;
//   }
// }


// // ==================================================================
// __global__ void d_knn50(unsigned long long *d_stats,
//                         float *d_results,
//                         floatN *d_queries,
//                         int numQueries,
// #if CUKD_IMPROVED_TRAVERSAL
//                         const cukd::box_t<floatN> *d_bounds,
// #endif
//                         floatN *d_nodes,
//                         int numNodes,
//                         float maxRadius)
// {
//   int tid = threadIdx.x+blockIdx.x*blockDim.x;
//   if (tid >= numQueries) return;

//   cukd::HeapCandidateList<50> result(maxRadius);
//   d_results[tid] = sqrtf(cukd::knn
//                          <cukd::TrivialFloatPointTraits<floatN>>
//                          (d_stats,result,d_queries[tid],
// #if CUKD_IMPROVED_TRAVERSAL
//                           *d_bounds,
// #endif
//                           d_nodes,numNodes));
// }

// void knn50(float *d_results,
//            floatN *d_queries,
//            int numQueries,
// #if CUKD_IMPROVED_TRAVERSAL
//            const cukd::box_t<floatN> *d_bounds,
// #endif
//            floatN *d_nodes,
//            int numNodes,
//            float maxRadius)
// {
//   int bs = 128;
//   int nb = divRoundUp(numQueries,bs);
//   unsigned long long *d_stats = 0;
//   static bool firstTime = true;
//   if (firstTime) {
//     cudaMallocManaged((char **)&d_stats,sizeof(*d_stats));
//     *d_stats = 0;
//   }
//   d_knn50<<<nb,bs>>>(d_stats,d_results,d_queries,numQueries,
// #if CUKD_IMPROVED_TRAVERSAL
//                      d_bounds,
// #endif
//                      d_nodes,numNodes,maxRadius);
//   if (firstTime) {
//     cudaDeviceSynchronize();
//     std::cout << "KDTREE_STATS " << *d_stats << std::endl;
//     std::cout << "NICE_STATS " << prettyNumber(*d_stats) << std::endl;
//     cudaFree(d_stats);
//     firstTime = false;
//   }
// }

// ==================================================================
inline void verifyKNN(int pointID, int k, float maxRadius,
                      floatN *points, int numPoints,
                      floatN queryPoint,
                      float presumedResult)
{
  std::priority_queue<float> closest_k;
  for (int i=0;i<numPoints;i++) {
    float d = sqrtf(cukd::fSqrDistance(queryPoint,points[i]));
    if (d <= maxRadius)
      closest_k.push(d);
    if (closest_k.size() > k)
      closest_k.pop();
  }

  float actualResult = (closest_k.size() == k) ? closest_k.top() : maxRadius;


  // check if the top 21-ish bits are the same; this will allow the
  // compiler to produce slightly different results on host and device
  // (usually caused by it uses madd's on one and separate +/* on
  // t'other...
  bool closeEnough
    =  /* this catches result==inf:*/(actualResult == presumedResult)
    || /* this catches bit errors: */(fabsf(actualResult - presumedResult) <= 1e-6f);
  
  if (!closeEnough) {
    std::cout << "for point #" << pointID << ": "
              << "verify found max dist " << std::setprecision(10) << actualResult
              << " (bits " << (int*)(uint64_t)((uint32_t&)actualResult)
              << "), knn reported " << presumedResult
              << " (bits " << (int*)(uint64_t)((uint32_t&)presumedResult)
              << "), difference is " << (actualResult-presumedResult)
              << std::endl;
    throw std::runtime_error("verification failed");
  }
}

int main(int ac, const char **av)
{
  using namespace cukd::common;
  
  int nPoints = 173;
  bool verify = false;
  float maxQueryRadius = std::numeric_limits<float>::infinity();
  int nRepeats = 1;
  int k = 50;
  for (int i=1;i<ac;i++) {
    std::string arg = av[i];
    if (arg[0] != '-')
      nPoints = std::stoi(arg);
    else if (arg == "-v")
      verify = true;
    else if (arg == "-nr")
      nRepeats = atoi(av[++i]);
    else if (arg == "-r")
      maxQueryRadius = std::stof(av[++i]);
    else if (arg == "-k")
      k = std::stoi(av[++i]);
    else
      throw std::runtime_error("known cmdline arg "+arg);
  }
  
  // floatN *d_points = generatePoints(nPoints);
  floatN *d_points = loadPoints<floatN>("data_points",nPoints);//generatePoints(nPoints);
#if CUKD_IMPROVED_TRAVERSAL
  cukd::box_t<floatN> *d_bounds;
  cudaMalloc((void**)&d_bounds,sizeof(cukd::box_t<floatN>));
  cukd::computeBounds(d_bounds,d_points,nPoints);
#endif

  {
    double t0 = getCurrentTime();
    std::cout << "calling builder..." << std::endl;
    cukd::buildTree
      // <cukd::TrivialFloatPointTraits<floatN>>
      (d_points,nPoints);
    CUKD_CUDA_SYNC_CHECK();
    double t1 = getCurrentTime();
    std::cout << "done building tree, took " << prettyDouble(t1-t0) << "s" << std::endl;
  }

  size_t nQueries = 10*1000*1000;
  floatN *d_queries = loadPoints<floatN>("query_points",nQueries);
  // floatN *d_queries = generatePoints(nQueries);
  float  *d_results;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_results,nQueries*sizeof(float)));

  
  
  // ==================================================================
  std::cout << "running " << nRepeats << " sets of knn-" << k << " queries..." << std::endl;
  double t0 = getCurrentTime();
  for (int i=0;i<nRepeats;i++)
    if (k == 4)
      knn<FixedCandidateList<4>>(d_results,d_queries,nQueries,
#if CUKD_IMPROVED_TRAVERSAL
           d_bounds,
#endif
           d_points,nPoints,maxQueryRadius);
    else if (k == 8)
      knn<FixedCandidateList<8>>(d_results,d_queries,nQueries,
#if CUKD_IMPROVED_TRAVERSAL
           d_bounds,
#endif
           d_points,nPoints,maxQueryRadius);
    else if (k == 64)
      knn<HeapCandidateList<64>>(d_results,d_queries,nQueries,
#if CUKD_IMPROVED_TRAVERSAL
            d_bounds,
#endif
            d_points,nPoints,maxQueryRadius);
    else if (k == 20)
      knn<HeapCandidateList<20>>(d_results,d_queries,nQueries,
#if CUKD_IMPROVED_TRAVERSAL
            d_bounds,
#endif
            d_points,nPoints,maxQueryRadius);
    else if (k == 50)
      knn<HeapCandidateList<50>>(d_results,d_queries,nQueries,
#if CUKD_IMPROVED_TRAVERSAL
            d_bounds,
#endif
            d_points,nPoints,maxQueryRadius);
    else
      throw std::runtime_error("unsupported k for knn queries");


  
  // knn4(d_results,d_queries,nQueries,d_points,nPoints,maxQueryRadius);
  
  CUKD_CUDA_SYNC_CHECK();
  for (int i=0;i<14;i++) {
    int idx = nQueries-1-(1<<i);
    std::cout << "  result[" << idx << "] = " << d_results[idx] << std::endl;;
  }
  double sum = 0;
  for (int i=0;i<nQueries;i++)
    sum += d_results[i];
  std::cout << "CHECKSUM " << sum << std::endl;

  
  double t1 = getCurrentTime();
  std::cout << "done " << nRepeats << " iterations of knn query, took " << prettyDouble(t1-t0) << "s" << std::endl;
  std::cout << " that's " << prettyDouble((t1-t0)/nRepeats) << "s per query (avg)..." << std::endl;
  std::cout << " ... or " << prettyDouble(nQueries*nRepeats/(t1-t0)) << " queries/s" << std::endl;
  
  if (verify) {
    std::cout << "verifying result ..." << std::endl;
    for (int i=0;i<nQueries;i++)
      verifyKNN(i,k,maxQueryRadius,d_points,nPoints,d_queries[i],d_results[i]);
    std::cout << "verification passed ... " << std::endl;
  }
  // }

  //   // ==================================================================
  //   {
  //     std::cout << "running " << nRepeats << " sets of knn8 queries..." << std::endl;
  //     double t0 = getCurrentTime();
  //     for (int i=0;i<nRepeats;i++)
  //       knn8(d_results,d_queries,nQueries,d_points,nPoints,maxQueryRadius);
  //     CUKD_CUDA_SYNC_CHECK();
  //     double t1 = getCurrentTime();
  //     std::cout << "done " << nRepeats << " iterations of knn8 query, took " << prettyDouble(t1-t0) << "s" << std::endl;
  //     std::cout << " that's " << prettyDouble((t1-t0)/nRepeats) << "s per query (avg)..." << std::endl;
  //     std::cout << " ... or " << prettyDouble(nQueries*nRepeats/(t1-t0)) << " queries/s" << std::endl;

  //     if (verify) {
  //       std::cout << "verifying result ..." << std::endl;
  //       for (int i=0;i<nQueries;i++)
  //         verifyKNN(i,8,maxQueryRadius,d_points,nPoints,d_queries[i],d_results[i]);
  //       std::cout << "verification passed ... " << std::endl;
  //     }
  //   }
  // #endif
  
  //   // ==================================================================
  //   {
  //     std::cout << "running " << nRepeats << " sets of knn20 queries..." << std::endl;
  //     double t0 = getCurrentTime();
  //     for (int i=0;i<nRepeats;i++)
  //       knn20(d_results,d_queries,nQueries,d_points,nPoints,maxQueryRadius);
  //     CUKD_CUDA_SYNC_CHECK();
  //     double t1 = getCurrentTime();
  //     std::cout << "done " << nRepeats << " iterations of knn20 query, took " << prettyDouble(t1-t0) << "s" << std::endl;
  //     std::cout << " that's " << prettyDouble((t1-t0)/nRepeats) << "s per query (avg)..." << std::endl;
  //     std::cout << " ... or " << prettyDouble(nQueries*nRepeats/(t1-t0)) << " queries/s" << std::endl;

  //     if (verify) {
  //       std::cout << "verifying result ..." << std::endl;
  //       for (int i=0;i<nQueries;i++)
  //         verifyKNN(i,20,maxQueryRadius,d_points,nPoints,d_queries[i],d_results[i]);
  //       std::cout << "verification passed ... " << std::endl;
  //     }
  //   }

  //   // ==================================================================
  //   {
  //     std::cout << "running " << nRepeats << " sets of knn50 queries..." << std::endl;
  //     double t0 = getCurrentTime();
  //     for (int i=0;i<nRepeats;i++)
  //       knn50(d_results,d_queries,nQueries,d_points,nPoints,maxQueryRadius);
  //     CUKD_CUDA_SYNC_CHECK();
  //     double t1 = getCurrentTime();
  //     std::cout << "done " << nRepeats << " iterations of knn50 query, took " << prettyDouble(t1-t0) << "s" << std::endl;
  //     std::cout << " that's " << prettyDouble((t1-t0)/nRepeats) << "s per query (avg)..." << std::endl;
  //     std::cout << " ... or " << prettyDouble(nQueries*nRepeats/(t1-t0)) << " queries/s" << std::endl;

  //     if (verify) {
  //       std::cout << "verifying result ..." << std::endl;
  //       for (int i=0;i<nQueries;i++)
  //         verifyKNN(i,50,maxQueryRadius,d_points,nPoints,d_queries[i],d_results[i]);
  //       std::cout << "verification passed ... " << std::endl;
  //     }
  // }

}
#endif
