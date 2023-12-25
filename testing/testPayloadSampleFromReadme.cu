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

namespace example1 {
  
  struct PointPlusPayload {
    float3 position;
    int    payload;
  };

  struct PointPlusPayload_traits
    : public cukd::default_data_traits<float3>
  {
    using point_t = float3;

    static inline __device__ __host__
    float3 get_point(const PointPlusPayload &data)
    { return data.position; }

    static inline __device__ __host__
    float  get_coord(const PointPlusPayload &data, int dim)
    { return cukd::get_coord(get_point(data),dim); }
  };

  void foo(PointPlusPayload *data, int numData) {
    cukd::buildTree
      </* type of the data: */PointPlusPayload,
                              /* traits for this data: */PointPlusPayload_traits>
      (data,numData);
  }

  void test()
  {
    std::cout << "testing `" << AS_STRING(BUILDER_TO_TEST)
              << "` on 'PointPlusPayloads' array (float3 plus payload), 1000 random data." << std::endl;

    int numPointPlusPayloads = 1000;
    
    PointPlusPayload *data = 0;
    CUKD_CUDA_CALL(MallocManaged((void **)&data,numPointPlusPayloads*sizeof(PointPlusPayload)));
    
    std::default_random_engine rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.f,100.f);
    for (int i=0;i<numPointPlusPayloads;i++) {
      data[i].position.x = dist(gen);
      data[i].position.y = dist(gen);
      data[i].position.z = dist(gen);
      data[i].payload    = i;
    }
    cukd::box_t<float3> *worldBounds = 0;
    CUKD_CUDA_CALL(MallocManaged((void **)&worldBounds,sizeof(*worldBounds)));
    
    // cukd::BUILDER_TO_TEST<PointPlusPayload,PointPlusPayload_traits>
    //   (data,numPointPlusPayloads,worldBounds);
    foo(data,numPointPlusPayloads);

    std::cout << "world bounds is " << *worldBounds << std::endl;
    CUKD_CUDA_CALL(Free(data));
    CUKD_CUDA_CALL(Free(worldBounds));
  }

}

int main(int, const char **)
{
  example1::test();
  CUKD_CUDA_SYNC_CHECK();

  return 0;
}
