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

#include "cukd/builder.h"
// fcp = "find closest point" query
#include "cukd/fcp.h"
#include "cukd/knn.h"
#include <queue>
#include <iomanip>

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

using namespace cukd;


#if EXPLICIT_DIM
struct PointAndDim {
  floatN point;
  int    dim;
};

struct PointAndDim_traits : public cukd::default_node_traits<floatN> {
  enum { has_explicit_dim = true };
  using node_t = PointAndDim;
  
  static inline __both__ const point_t &get_point(const node_t &n) { return n.point; }
  static inline __both__ float get_coord(const PointAndDim &n, int d)
  { return cukd::get_coord(n.point,d); }
  
  static inline __both__ int  get_dim(const PointAndDim &p) 
  { return p.dim; }
	   
  static inline __both__ void set_dim(PointAndDim &p, int dim) 
  { p.dim = dim; }
};

using node_t = PointAndDim;
using node_traits = PointAndDim_traits;
#else
using node_t = floatN;
using node_traits = default_node_traits<floatN>;
#endif



__global__
void d_fcp(unsigned long long *d_stats,
           float   *d_results,
           floatN  *d_queries,
           int      numQueries,
#if CUKD_IMPROVED_TRAVERSAL
           const cukd::box_t<floatN> *d_bounds,
#endif
           node_t  *d_nodes,
           int      numNodes,
           float    cutOffRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  using point_t = floatN;
  point_t queryPos = d_queries[tid];
  FcpSearchParams params;
  params.cutOffRadius = cutOffRadius;
  int closestID
    = cukd::fcp
    <node_t,node_traits>
    (d_stats,queryPos,
#if CUKD_IMPROVED_TRAVERSAL
     *d_bounds,
#endif
     d_nodes,numNodes,params);
  
  d_results[tid]
    = (closestID < 0)
    ? INFINITY
    : distance(queryPos,node_traits::get_point(d_nodes[closestID]));
}




template<typename CandidateList>
__global__
void d_knn(unsigned long long *d_stats,
           float   *d_results,
           floatN  *d_queries,
           int      numQueries,
#if CUKD_IMPROVED_TRAVERSAL
           const cukd::box_t<floatN> *d_bounds,
#endif
           node_t  *d_nodes,
           int      numNodes,
           float    cutOffRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;
  
  CandidateList result(cutOffRadius);
  float sqrDist
    = cukd::knn(d_stats,result,d_queries[tid],
#if CUKD_IMPROVED_TRAVERSAL
                *d_bounds,
#endif
                d_nodes,numNodes);
  d_results[tid] = sqrtf(sqrDist);
}



void run_kernel(float  *d_results,
                floatN *d_queries,
                int     numQueries,
#if CUKD_IMPROVED_TRAVERSAL
                const cukd::box_t<floatN> *d_bounds,
#endif
                node_t *d_nodes,
                int     numNodes,
#if USE_KNN
                int k,
#endif
                float   cutOffRadius
                )
{
  int bs = 128;
  int nb = divRoundUp(numQueries,bs);
  unsigned long long *d_stats = 0;
  static bool firstTime = true;
  if (firstTime) {
    cudaMallocManaged((char **)&d_stats,sizeof(*d_stats));
    *d_stats = 0;
  }
#if USE_KNN
  if (k == 4)
    d_knn<FixedCandidateList<4>><<<nb,bs>>>
      (d_stats,d_results,d_queries,numQueries,
# if CUKD_IMPROVED_TRAVERSAL
       d_bounds,
# endif
       d_nodes,numNodes,cutOffRadius);
  else if (k == 8)
    d_knn<FixedCandidateList<8>><<<nb,bs>>>
      (d_stats,d_results,d_queries,numQueries,
# if CUKD_IMPROVED_TRAVERSAL
       d_bounds,
# endif
       d_nodes,numNodes,cutOffRadius);
  else if (k == 64)
    d_knn<HeapCandidateList<64>><<<nb,bs>>>
      (d_stats,d_results,d_queries,numQueries,
# if CUKD_IMPROVED_TRAVERSAL
       d_bounds,
# endif
       d_nodes,numNodes,cutOffRadius);
  else if (k == 20)
    d_knn<HeapCandidateList<20>><<<nb,bs>>>
      (d_stats,d_results,d_queries,numQueries,
# if CUKD_IMPROVED_TRAVERSAL
       d_bounds,
# endif
       d_nodes,numNodes,cutOffRadius);
  else if (k == 50)
    d_knn<HeapCandidateList<50>><<<nb,bs>>>
      (d_stats,d_results,d_queries,numQueries,
# if CUKD_IMPROVED_TRAVERSAL
       d_bounds,
# endif
       d_nodes,numNodes,cutOffRadius);
  else
    throw std::runtime_error("unsupported k for knn queries");
#else
  d_fcp<<<nb,bs>>>
    (d_stats,d_results,d_queries,numQueries,
# if CUKD_IMPROVED_TRAVERSAL
     d_bounds,
# endif
     d_nodes,numNodes,cutOffRadius);
#endif
  if (firstTime) {
    cudaDeviceSynchronize();
    std::cout << "KDTREE_STATS " << *d_stats << std::endl;
    cudaFree(d_stats);
    firstTime = false;
  }
}

#if EXPLICIT_DIM
__global__ void copyPoints(PointAndDim *d_points,
                           floatN *d_inputs,
                           int numPoints)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numPoints) return;
  d_points[tid].point = d_inputs[tid];
}
#endif

template<typename node_t, typename node_traits>
void verifyKNN(int pointID,
               int k,
               float maxRadius,
               floatN *points, int numPoints,
               floatN queryPoint,
               float reportedResult)
{
  using point_t = typename node_traits::point_t;
  std::priority_queue<float> closest_k;
  for (int i=0;i<numPoints;i++) {
    point_t point_i = node_traits::get_point(points[i]);
    float d = sqrDistance(queryPoint,point_i);
    if (d >= maxRadius*maxRadius)
      continue;
    
    closest_k.push(d);
    if (closest_k.size() > k)
      closest_k.pop();
  }
  
  float actualResult = (closest_k.size() == k) ? sqrtf(closest_k.top()) : maxRadius;
  
  // check if the top 21-ish bits are the same; this will allow the
  // compiler to produce slightly different results on host and device
  // (usually caused by it uses madd's on one and separate +/* on
  // t'other...
  bool closeEnough
    =  /* this catches result==inf:*/
    (actualResult == reportedResult)
    || /* this catches bit errors: */
    (fabsf(actualResult - reportedResult)/std::max(actualResult,reportedResult) <= 1e-6f);
  
  if (!closeEnough) {
    std::cout << "for point #" << pointID << ": "
              << "verify found max dist " << std::setprecision(10) << actualResult
              << " (bits " << (int*)(uint64_t)((uint32_t&)actualResult)
              << "), knn reported " << reportedResult
              << " (bits " << (int*)(uint64_t)((uint32_t&)reportedResult)
              << "), difference is " << (actualResult-reportedResult)
              << std::endl;
    throw std::runtime_error("verification failed");
  }
}


template<typename node_t, typename node_traits>
void verifyFCP(int pointID,
               float cutOffRadius,
               node_t *points, int numPoints,
               floatN queryPoint,
               float reportedResult)
{
  using point_t = typename node_traits::point_t;
  float actualResult = INFINITY;
  for (int i=0;i<numPoints;i++) {
    point_t point_i = node_traits::get_point(points[i]);
    float d = sqrDistance(queryPoint,point_i);
    if (d >= cutOffRadius*cutOffRadius)
      continue;

    actualResult = std::min(actualResult,sqrtf(d));
  }
  
  
  // check if the top 21-ish bits are the same; this will allow the
  // compiler to produce slightly different results on host and device
  // (usually caused by it uses madd's on one and separate +/* on
  // t'other...
  bool closeEnough
    =  /* this catches result==inf:*/
    (actualResult == reportedResult)
    || /* this catches bit errors: */
    (fabsf(actualResult - reportedResult)/std::max(actualResult,reportedResult) <= 1e-6f);
  
  if (!closeEnough) {
    std::cout << "for point #" << pointID << ": "
              << "verify found max dist " << std::setprecision(10) << actualResult
              << " (bits " << (int*)(uint64_t)((uint32_t&)actualResult)
              << "), knn reported " << reportedResult
              << " (bits " << (int*)(uint64_t)((uint32_t&)reportedResult)
              << "), difference is " << (actualResult-reportedResult)
              << std::endl;
    throw std::runtime_error("verification failed");
  }
}


int main(int ac, const char **av)
{
  using namespace cukd::common;

  int    numPoints = 173;
  bool   verify = false;
  int    nRepeats = 1;
  size_t numQueries = 10000000;
  float  cutOffRadius = std::numeric_limits<float>::infinity();
#if USE_KNN
  int    k = 50;
#endif
  for (int i=1;i<ac;i++) {
    std::string arg = av[i];
    if (arg[0] != '-')
      numPoints = std::stoi(arg);
    else if (arg == "-v")
      verify = true;
    else if (arg == "-nq")
      numQueries = atoi(av[++i]);
    else if (arg == "-nr")
      nRepeats = atoi(av[++i]);
    else if (arg == "-r")
      cutOffRadius = std::stof(av[++i]);
#if USE_KNN
    else if (arg == "-k")
      k = std::stoi(av[++i]);
#endif
    else
      throw std::runtime_error("known cmdline arg "+arg);
  }
  
  floatN *d_inputs = loadPoints<floatN>("data_points",numPoints);
#if EXPLICIT_DIM
  PointAndDim *d_points;
  cudaMallocManaged((void**)&d_points,numPoints*sizeof(*d_points));
  copyPoints<<<divRoundUp(numPoints,128),128>>>
    (d_points,d_inputs,numPoints);
  using node_t = PointAndDim;
#else
  floatN *d_points = d_inputs;
  using node_t = floatN;
#endif
  
#if CUKD_IMPROVED_TRAVERSAL
  cukd::box_t<floatN> *d_bounds;
  cudaMalloc((void**)&d_bounds,sizeof(cukd::box_t<floatN>));
  cukd::computeBounds<node_t,node_traits>
    (d_bounds,d_points,numPoints);
#endif
  {
    double t0 = getCurrentTime();
    std::cout << "calling builder..." << std::endl;
    cukd::buildTree<node_t,node_traits>
      (d_points,numPoints);
    CUKD_CUDA_SYNC_CHECK();
    double t1 = getCurrentTime();
    std::cout << "done building tree, took "
              << prettyDouble(t1-t0) << "s" << std::endl;
  }
  
  floatN *d_queries = loadPoints<floatN>("query_points",numQueries);
  float  *d_results;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_results,numQueries*sizeof(*d_results)));
  {
    double t0 = getCurrentTime();
    for (int i=0;i<nRepeats;i++) {
      run_kernel
        (d_results,d_queries,numQueries,
#if CUKD_IMPROVED_TRAVERSAL
         d_bounds,
#endif
         d_points,numPoints,
#if USE_KNN
         k,
#endif
         cutOffRadius);
    }
    CUKD_CUDA_SYNC_CHECK();
    double t1 = getCurrentTime();
    std::cout << "done " << nRepeats
              << " iterations of " << numQueries
              << " fcp queries, took " << prettyDouble(t1-t0)
              << "s" << std::endl;
    std::cout << "that is " << prettyDouble(numQueries*nRepeats/(t1-t0))
              << " queries/s" << std::endl;
  }
  
  if (verify) {
    std::cout << "verifying ..." << std::endl;
    for (int i=0;i<numQueries;i++) {
      floatN qp           = d_queries[i];
      float  reportedResult = d_results[i];
#if USE_KNN
      verifyKNN<node_t,node_traits>
        (i,k,cutOffRadius,d_points,numPoints,qp,reportedResult);
#else
      verifyFCP<node_t,node_traits>
        (i,cutOffRadius,d_points,numPoints,qp,reportedResult);
#endif          
    }
  }
  std::cout << "verification succeeded... done." << std::endl;
}
  
