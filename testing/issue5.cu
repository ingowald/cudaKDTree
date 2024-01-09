// ======================================================================== //
// Copyright 2018-2024 Ingo Wald                                            //
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
#include <random>
#include "cukd/fcp.h"


#define AS_STRING(x) #x
#define TO_STRING(x) AS_STRING(x)

void generateRandomPoints(size_t nb_,
                          int seed_,
                          std::vector<float3> &destPts_)
{
  const double maxVal = 100.0;
  const double scale = 2.0f * maxVal / RAND_MAX;

  destPts_.resize( nb_ );
  std::srand( seed_ );
  
  for ( size_t i = 0; i < nb_; i++ ) {
    destPts_[ i ].x = static_cast< float >( std::rand() * scale - maxVal );
    destPts_[ i ].y = static_cast< float >( std::rand() * scale - maxVal );
    destPts_[ i ].z = static_cast< float >( std::rand() * scale - maxVal );
  }
}


__global__ void checkResult(float3 *data,
                            int numData,
                            float3 queryPoint,
                            cukd::FcpSearchParams params,
                            float expectedSqrDist)
{
  if (threadIdx.x != 0) return;
  
  int res = cukd::stackBased::fcp(queryPoint,data,numData,params);
  if (res < 0) {
    printf("no result!?\n");
    return;
  }
  float3 pt = data[res];
  float sqrDist = cukd::fSqrDistance(pt,queryPoint);
  
  printf("found res %i, pos %f %f %f sqrdist %f expected %f\n",
         res,pt.x,pt.y,pt.z,sqrDist,expectedSqrDist);
}

float distance(float3 a, float3 b)
{
  auto sqr = [&](float f) { return f*f; };
  float f = 0.f;
  f += sqr(a.x-b.x);
  f += sqr(a.y-b.y);
  f += sqr(a.z-b.z);
  return f;
}

int main(int, char **)
{
  std::vector<float3> points;
  // Point are generated like this (nb_= 90167, seed_= 33):
  int nb_= 90167, seed_= 33;
  generateRandomPoints(nb_,seed_,points);
  // It should start like this:
  // [0] {x=-99.1088562 y=-87.9879150 z=27.7626877 } float3
  // [1] {x=-38.3892326 y=31.5713978 z=-37.0891457 } float3
  // [2] {x=-22.0435200 y=-92.5473785 z=89.4833221 } float3
  // [3] {x=48.5274811 y=-94.0671997 z=80.3888092 } float3
  // [4] {x=-33.9030113 y=34.4157219 z=95.2085953 } float3
  for (int i=0;i<5;i++)
    printf("[%i] (%f %f %f)\n",i,points[i].x,points[i].y,points[i].z);
  
  cukd::box_t<float3> *worldBounds = 0;
  CUKD_CUDA_CALL(MallocManaged((void **)&worldBounds,sizeof(*worldBounds)));
  float3 *d_points = 0;
  CUKD_CUDA_CALL(MallocManaged((void **)&d_points,points.size()*sizeof(float3)));
  CUKD_CUDA_CALL(Memcpy(d_points,points.data(),points.size()*sizeof(float3),
                        cudaMemcpyDefault));
  
  cukd::BUILDER_TO_TEST
    (d_points,points.size(),worldBounds);
  
  std::cout << "world bounds is " << *worldBounds << std::endl;

  // The querry point is {x=-98.4496613 y=76.9219055 z=25.8888512 }
  float3 queryPoint = make_float3(-98.4496613, 76.9219055, 25.8888512);
  // The "cutOffRadius" is 5.0
  cukd::FcpSearchParams params;
  params.cutOffRadius = 5.f;
  
  // The closest point should be at squared distance of 2.8466301
  float expectedSqrDist = 2.8466301f;

  float closestDist = INFINITY;
  int   closest = -1;
  for (int i=0;i<points.size();i++) {
    float3 pt = points[i];
    float dist = distance(points[i],queryPoint);
    // float dist = cukd::fSqrDistance(points[i],queryPoint);
    if (dist >= closestDist) continue;
    closestDist = dist;
    closest = i;
  }
  float3 pt = points[closest];
  std::cout << "reference closest dist is " << pt.x << ", " << pt.y << ", " << pt.z
         << " at dist " << closestDist << std::endl;
  checkResult<<<1,32>>>(d_points,points.size(),
                        queryPoint,params,expectedSqrDist);
  CUKD_CUDA_SYNC_CHECK();
}


