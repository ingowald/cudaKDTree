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
#include <queue>
#include <iomanip>
#include <random>

using namespace cukd;

float3 *generatePoints(int N)
{
  static int g_seed = 100000;
  std::seed_seq seq{g_seed++};
  std::default_random_engine rd(seq);
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> dist(0,N);

  std::cout << "generating " << N << " uniform random points" << std::endl;
  float3 *d_points = 0;
  cudaMallocManaged((char **)&d_points,N*sizeof(*d_points));
  if (!d_points)
    throw std::runtime_error("could not allocate points mem...");
  
  for (int i=0;i<N;i++) {
    d_points[i].x = (float)dist(gen);
    d_points[i].y = (float)dist(gen);
    d_points[i].z = (float)dist(gen);
  }
  return d_points;
}


__global__
void d_fcp(float   *d_results,
           float3  *d_queries,
           int      numQueries,
           /*! the world bounding box computed by the builder */
           const cukd::box_t<float3> *d_bounds,
           float3  *d_nodes,
           int      numNodes,
           float    cutOffRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  using point_t = float3;
  point_t queryPos = d_queries[tid];
  FcpSearchParams params;
  params.cutOffRadius = cutOffRadius;
  int closestID
    = cukd::cct::fcp
    (queryPos,*d_bounds,d_nodes,numNodes,params);
  
  d_results[tid]
    = (closestID < 0)
    ? INFINITY
    : distance(queryPos,d_nodes[closestID]);
}




int main(int ac, const char **av)
{
  using namespace cukd::common;

  int    numPoints = 1000000;
  int    nRepeats = 1;
  size_t numQueries = 1000000;
  float  cutOffRadius = std::numeric_limits<float>::infinity();
  for (int i=1;i<ac;i++) {
    std::string arg = av[i];
    if (arg[0] != '-')
      numPoints = std::stoi(arg);
    else if (arg == "-nq")
      numQueries = atoi(av[++i]);
    else if (arg == "-nr")
      nRepeats = atoi(av[++i]);
    else if (arg == "-r")
      cutOffRadius = std::stof(av[++i]);
    else
      throw std::runtime_error("known cmdline arg "+arg);
  }
  
  // ==================================================================
  // create sample input point that we'll build the tree over
  // ==================================================================
  float3 *d_points = generatePoints(numPoints);

  // ==================================================================
  // allocate some memory for the world-space bounding box, so the
  // builder can compute and return that for our chosen traversal
  // method to use
  // ==================================================================
  cukd::box_t<float3> *d_bounds;
  cudaMallocManaged((void**)&d_bounds,sizeof(cukd::box_t<float3>));
  std::cout << "allocated memory for the world space bounding box ..." << std::endl;

  // ==================================================================
  // build the tree. this will also comptue the world-space boudig box
  // of all points
  // ==================================================================
  std::cout << "calling builder..." << std::endl;
  double t0 = getCurrentTime();
  cukd::buildTree(d_points,numPoints,d_bounds);
  CUKD_CUDA_SYNC_CHECK();
  double t1 = getCurrentTime();
  std::cout << "done building tree, took "
            << prettyDouble(t1-t0) << "s" << std::endl;

  // ==================================================================
  // create set of sample query points
  // ==================================================================
  float3 *d_queries
    = generatePoints(numQueries);
  // allocate memory for the results
  float  *d_results;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_results,numQueries*sizeof(*d_results)));


  // ==================================================================
  // and do some queryies - let's do the same ones in a loop so we cna
  // measure perf.
  // ==================================================================
  {
    double t0 = getCurrentTime();
    for (int i=0;i<nRepeats;i++) {
      int bs = 128;
      int nb = divRoundUp((int)numQueries,bs);
      d_fcp<<<nb,bs>>>
        (d_results,d_queries,numQueries,
         d_bounds,d_points,numPoints,cutOffRadius);
      cudaDeviceSynchronize();
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
  
}
  
