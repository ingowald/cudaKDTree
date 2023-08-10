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
#include <random>

#define AS_STRING(x) #x
#define TO_STRING(x) AS_STRING(x)

namespace test_float3 {
  void test_simple()
  {
    std::cout << "testing `" << TO_STRING(BUILDER_TO_TEST)
              << "` on float3 array, 1000 uniform random points." << std::endl;
    
    int numPoints = 1000;
    
    float3 *points = 0;
    CUKD_CUDA_CALL(MallocManaged((void **)&points,numPoints*sizeof(float3)));
    
    std::default_random_engine rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.f,100.f);
    for (int i=0;i<numPoints;i++) {
      points[i].x = dist(gen);
      points[i].y = dist(gen);
      points[i].z = dist(gen);
    }
    // BUILDER_TO_TEST supplied by cmakefile:
    cukd::BUILDER_TO_TEST(points,numPoints);
    CUKD_CUDA_CALL(Free(points));
  }
}

namespace test_photon {
  /*! for those wondering what this test is for: have a look at Henrik
    Wan Jensen, "Realistic Image Synthesis using Photon Mapping"
    https://www.amazon.com/Realistic-Image-Synthesis-Photon-Mapping/dp/1568811470 */
  struct Photon {
    float3 position;
    float3 power;
    uint16_t normal_phi;
    uint8_t  normal_theta;
    uint8_t  splitDim;
  };

  struct Photon_traits {
    using point_t = float3;
    enum { has_explicit_dim = true };
    
    static inline __both__
    const point_t &get_point(const Photon &p)
    { return p.position; }
    
    static inline __both__ float get_coord(const Photon &p, int d)
    { return cukd::get_coord(p.position,d); }
    
    static inline __device__ int  get_dim(const Photon &p)
    { return p.splitDim; }
    
    static inline __device__ void set_dim(Photon &p, int d)
    { p.splitDim = d; }
  };
  
  void test_simple()
  {
    std::cout << "testing `" << AS_STRING(BUILDER_TO_TEST)
              << "` on 'Photons' array (float3 plus payload), 1000 random photons." << std::endl;

    int numPhotons = 1000;
    
    Photon *photons = 0;
    CUKD_CUDA_CALL(MallocManaged((void **)&photons,numPhotons*sizeof(Photon)));
    
    std::default_random_engine rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.f,100.f);
    for (int i=0;i<numPhotons;i++) {
      photons[i].position.x = dist(gen);
      photons[i].position.y = dist(gen);
      photons[i].position.z = dist(gen);
      photons[i].power = make_float3(0.f,0.f,0.f);
      photons[i].normal_theta = 0;
      photons[i].normal_phi = 0;
    }
    cukd::box_t<float3> *worldBounds = 0;
    CUKD_CUDA_CALL(MallocManaged((void **)&worldBounds,sizeof(*worldBounds)));
    
    cukd::buildTree_bitonic<Photon,Photon_traits>
      (photons,numPhotons,worldBounds);

    std::cout << "world bounds is " << *worldBounds << std::endl;
    CUKD_CUDA_CALL(Free(photons));
    CUKD_CUDA_CALL(Free(worldBounds));
  }
}

int main(int, const char **)
{
  test_float3::test_simple();
  CUKD_CUDA_SYNC_CHECK();

  test_photon::test_simple();
  CUKD_CUDA_SYNC_CHECK();

  return 0;
}

