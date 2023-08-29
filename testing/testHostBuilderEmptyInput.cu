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

#include "cukd/builder_host.h"
#include <random>

namespace test_float3 {
  void test_empty()
  {
    std::cout << "testing float3 array, empty input." << std::endl;
    
    // dummy arrays, just to get the types to force the right builder
    // instantiation:
    float3 *points = 0;
    int numPoints = 0;
    // BUILDER_TO_TEST supplied by cmakefile:
    cukd::buildTree_host(points,numPoints);
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
#if 1 
    enum { has_explicit_dim = false };
#else
    enum { has_explicit_dim = true };
    
    static inline __both__ int  get_dim(const Photon &p)
    { return p.splitDim; }
    
    static inline __both__ void set_dim(Photon &p, int d)
    { p.splitDim = d; }
#endif
    
    static inline __both__
    const point_t &get_point(const Photon &p)
    { return p.position; }
    
    static inline __both__ float get_coord(const Photon &p, int d)
    { return cukd::get_coord(p.position,d); }
  };
  
  void test_empty()
  {
    std::cout << "testing 'Photons' array (float3 plus payload), empty input." << std::endl;

    // dummy arrays, just to get the types to force the right builder
    // instantiation:
    Photon *points = 0;
    int numPoints = 0;
    // BUILDER_TO_TEST supplied by cmakefile:
    cukd::buildTree_host<Photon,Photon_traits>
      (points,numPoints);
  }
}

int main(int, const char **)
{
  test_float3::test_empty();
  test_photon::test_empty();
  return 0;
}

