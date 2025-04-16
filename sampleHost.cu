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

void generatePoints(size_t N, std::vector<float3> &points)
{
  static int g_seed = 100000;
  std::seed_seq seq{g_seed++};
  std::default_random_engine rd(seq);
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> dist(0,N);

  std::cout << "generating " << N << " uniform random points" << std::endl;
  points.resize(N);

#ifdef OPENMP_FOUND
  #pragma omp parallel for
#endif  
  for (size_t i=0;i<N;i++) {
    points[i].x = (float)dist(gen);
    points[i].y = (float)dist(gen);
    points[i].z = (float)dist(gen);
  }
}

void fcp_host(float *results,
           float3  *queries,
           size_t   numQueries,
           /*! the world bounding box computed by the builder */
           const cukd::box_t<float3> *bounds,
           float3  *nodes,
           int      numNodes,
           float    cutOffRadius)
{
  using point_t = float3;

#ifdef OPENMP_FOUND
  #pragma omp parallel for
#endif  
  for (size_t tid = 0; tid < numQueries; tid++) {    
    point_t queryPos = queries[tid];
    FcpSearchParams params;
    params.cutOffRadius = cutOffRadius;
    int closestID
      = cukd::cct::fcp
      (queryPos,*bounds,nodes,numNodes,params);
    
    results[tid]
      = (closestID < 0)
      ? INFINITY
      : distance(queryPos,nodes[closestID]);
  }
}

int main(int ac, const char **av)
{
  using namespace cukd::common;

  size_t numPoints = 10000;
  int    nRepeats = 1;
  size_t numQueries = 10000;
  float  cutOffRadius = std::numeric_limits<float>::infinity();
  for (int i=1;i<ac;i++) {
    std::string arg = av[i];
    if (arg[0] != '-')
      numPoints = atoll(arg.c_str());
    else if (arg == "-nq")
      numQueries = atoll(av[++i]);
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
  std::vector<float3> points;
  generatePoints(numPoints, points);

  // ==================================================================
  // allocate some memory for the world-space bounding box, so the
  // builder can compute and return that for our chosen traversal
  // method to use
  // ==================================================================
  cukd::box_t<float3> bounds;
  std::cout << "allocated memory for the world space bounding box ..." << std::endl;

  // ==================================================================
  // build the tree. this will also comptue the world-space boudig box
  // of all points
  // ==================================================================
  std::cout << "calling builder..." << std::endl;
  double t0 = getCurrentTime();
  cukd::buildTree_host(points.data(),numPoints,&bounds);
  double t1 = getCurrentTime();
  std::cout << "done building tree, took "
            << prettyDouble(t1-t0) << "s" << std::endl;

  // ==================================================================
  // create set of sample query points
  // ==================================================================
  std::vector<float3> queries;
  generatePoints(numQueries, queries);
  // allocate memory for the results
  std::vector<float> results(numQueries);

  // ==================================================================
  // and do some queryies - let's do the same ones in a loop so we cna
  // measure perf.
  // ==================================================================
  {
    double t0 = getCurrentTime();
    for (int i=0;i<nRepeats;i++) {
      fcp_host
        (results.data(),queries.data(),numQueries,
         &bounds,points.data(),numPoints,cutOffRadius);
    }
    double t1 = getCurrentTime();
    std::cout << "done " << nRepeats
              << " iterations of " << numQueries
              << " fcp queries, took " << prettyDouble(t1-t0)
              << "s" << std::endl;
    std::cout << "that is " << prettyDouble(numQueries*nRepeats/(t1-t0))
              << " queries/s" << std::endl;
  }
  
}
  
