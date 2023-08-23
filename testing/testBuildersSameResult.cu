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

#include "cukd/common.h"
#include "cukd/builder_host.h"
#include "cukd/builder_thrust.h"
#include "cukd/builder_bitonic.h"
#include "cukd/builder_inplace.h"
#include <random>

template<typename T, int D>
struct vecN {
  T v[D];
};
  
// template<typename T, int D>
// struct num_dims_of<vecN<T,D>> {
//   enum { value = D };
// };
  
// template<typename T, int D>
// struct scalar_type_of<vecN<T,D>> {
//   using type = T;
// };
  
template<typename T, int D>
inline __both__
T get_coord(const vecN<T,D> &v, int d) { return v.v[d]; }
  
template<typename T, int D>
inline __both__
void set_coord(vecN<T,D> &v, int d, T vv) { v.v[d] = vv; }
  
// template<typename T, int D>
// inline __both__
// T &get_coord(vecN<T,D> &v, int d) { return v.v[d]; }
  
// template<typename T, int D>
// inline __both__
// T &get_coord(vecN<T,D> &v, int d) { return v.v[d]; }
  
  
template<typename T, int D>
inline __both__
vecN<T,D> min(vecN<T,D> a, vecN<T,D> b)
{
  vecN<T,D> res;
  for (int d=0;d<D;d++) res.v[d] = ::min(a.v[d],b.v[d]);
  return res;
}
template<typename T, int D>
inline __both__
vecN<T,D> max(vecN<T,D> a, vecN<T,D> b)
{
  vecN<T,D> res;
  for (int d=0;d<D;d++) res.v[d] = ::max(a.v[d],b.v[d]);
  return res;
}

namespace cukd {
  template<typename T, int D>
  struct point_traits<vecN<T,D>> {
    using scalar_t = T;
    using point_t  = vecN<T,D>;
    enum { num_dims = D };
    
    static inline __both__
    T get_coord(const point_t &p, int d) { return p.v[d]; }
    static inline __both__
    T &get_coord(point_t &p, int d) { return p.v[d]; }
    static inline __both__
    void set_coord(point_t &p, int d, scalar_t vv) { p.v[d] = vv; }
  };
}

template<typename T> std::string typeToString();

template<> std::string typeToString<float>() { return "float"; }
template<> std::string typeToString<int>()   { return "int"; }

size_t computeHash(uint32_t *ptr, size_t numBytes)
{
  size_t hash = 0;
  assert((numBytes % 4) == 0);
  for (int i=0;i<numBytes/4;i++)
    hash = hash*17 ^ ((const uint32_t*)ptr)[i];
  return hash;
}


template<typename T, int D>
void test_pointOnly(size_t numPoints)
{
  std::cout << "testing for " << numPoints
            << " inputs, type = " << typeToString<T>() << D
            << std::endl;
    
  std::random_device rd;  // a seed source for the random number engine
  std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()

  // generate a vector of unique ints that we'll then randomize and
  // use as coordinates. using unique ints makes sure that there's no
  // duplicates - becuaes if there _are_ duplicates you can indeed up
  // with different trees that all valid, which would break this test
  std::vector<int> uniqueInts(numPoints*D);
  for (int i=0;i<numPoints*D;i++)
    uniqueInts[i] = i;
  // now scramle those
  for (int i=numPoints*D-1;i>0;--i) {
    std::uniform_int_distribution<> distrib(0,i);
    int partner = distrib(gen);
    if (partner == i) continue;
    std::swap(uniqueInts[i],uniqueInts[partner]);
  }
      
  using data_t = vecN<T,D>;
  std::vector<data_t> inputData(numPoints);
  for (int i=0;i<numPoints;i++)
    for (int d=0;d<D;d++)
      inputData[i].v[d] = T(uniqueInts[i*D+d]);
  
  data_t *d_data = 0;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_data,numPoints*sizeof(data_t)));
    
  // ------------------------------------------------------------------
  // host builder
  // ------------------------------------------------------------------
  std::vector<data_t> data_host = inputData;
  cukd::buildTree_host
    (data_host.data(),numPoints);

  size_t hash_host = computeHash((uint32_t*)data_host.data(),numPoints*sizeof(data_t));
  std::cout << "hash host:\t " << (int*)hash_host << std::endl;
  
  // ------------------------------------------------------------------
  CUKD_CUDA_CALL(Memcpy(d_data,inputData.data(),numPoints*sizeof(data_t),cudaMemcpyDefault));
  cukd::buildTree_thrust(d_data,numPoints);
  CUKD_CUDA_SYNC_CHECK();
  
  size_t hash_thrust = computeHash((uint32_t*)d_data,numPoints*sizeof(data_t));
  std::cout << "hash thrust:\t " << (int*)hash_thrust << std::endl;
  

  // ------------------------------------------------------------------
  CUKD_CUDA_CALL(Memcpy(d_data,inputData.data(),numPoints*sizeof(data_t),cudaMemcpyDefault));
  cukd::buildTree_bitonic(d_data,numPoints);
  CUKD_CUDA_SYNC_CHECK();

  size_t hash_bitonic = computeHash((uint32_t*)d_data,numPoints*sizeof(data_t));
  std::cout << "hash bitonic:\t " << (int*)hash_bitonic << std::endl;
  
  // ------------------------------------------------------------------
  CUKD_CUDA_CALL(Memcpy(d_data,inputData.data(),numPoints*sizeof(data_t),cudaMemcpyDefault));
  cukd::buildTree_inPlace(d_data,numPoints);
  CUKD_CUDA_SYNC_CHECK();

  size_t hash_inPlace = computeHash((uint32_t*)d_data,numPoints*sizeof(data_t));
  std::cout << "hash inPlace:\t " << (int*)hash_inPlace << std::endl;

  if (hash_thrust  != hash_host ||
      hash_bitonic != hash_host ||
      hash_inPlace  != hash_host)
    throw std::runtime_error("hashes do not match!");
}

template<typename T, int D>
struct PointWithPayload {
  vecN<T,D> position;
  int    payload;
  int    splitDim;
};

template<typename T, int D>
struct PointWithPayload_traits {
  using point_t = vecN<T,D>;
  using point_traits = ::cukd::point_traits<point_t>;
  using data_t = PointWithPayload<T,D>;
  using box_t  = ::cukd::box_t<point_t>;
    
  enum { has_explicit_dim = true };
  
  static inline __both__ point_t &get_point(data_t &data) { return data.position; }
  static inline __both__ const point_t &get_point(const data_t &data) { return data.position; }
  static inline __both__ int get_dim(const data_t &data) { return data.splitDim; }
  static inline __both__ void set_dim(data_t &data, int d) { data.splitDim = d; }
  static inline __both__ T get_coord(const data_t &data, int d)
  { return point_traits::get_coord(get_point(data),d); }
};


template<typename T, int D>
void printTree(PointWithPayload<T,D> *data, int numData)
{
  for (int i=0;i<numData;i++) {
    std::cout << "[(";
    for (int d=0;d<D;d++) {
      if (d) std::cout << ",";
      std::cout << data[i].position.v[d];
    }
    std::cout << "),d="<< data[i].splitDim;
    std::cout << ",p="<< data[i].payload;
    std::cout << "] ";
  }
  std::cout << std::endl;
}



template<typename T, int D>
void test_withPayload(int numPoints)
{
  std::cout << "testing for " << numPoints
            << " inputs, type = " << typeToString<T>() << D << "+payload"
            << std::endl;
    
  std::random_device rd;  // a seed source for the random number engine
  std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()

  // generate a vector of unique ints that we'll then randomize and
  // use as coordinates. using unique ints makes sure that there's no
  // duplicates - becuaes if there _are_ duplicates you can indeed up
  // with different trees that all valid, which would break this test
  std::vector<int> uniqueInts(numPoints*D);
  for (int i=0;i<numPoints*D;i++)
    uniqueInts[i] = i;
  // now scramle those
  for (int i=numPoints*D-1;i>0;--i) {
    std::uniform_int_distribution<> distrib(0,i);
    int partner = distrib(gen);
    if (partner == i) continue;
    std::swap(uniqueInts[i],uniqueInts[partner]);
  }
      
  using data_t = PointWithPayload<T,D>;
  std::vector<data_t> inputData(numPoints);
  for (int i=0;i<numPoints;i++) {
    for (int d=0;d<D;d++)
      inputData[i].position.v[d] = T(uniqueInts[i*D+d]);
    inputData[i].payload = i;
  }
  
  data_t *d_data = 0;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_data,numPoints*sizeof(data_t)));
  
  ::cukd::box_t<vecN<T,D>> *d_bounds = 0;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_bounds,sizeof(*d_bounds)));
  assert(d_bounds);
  
  // ------------------------------------------------------------------
  // host builder
  // ------------------------------------------------------------------
  std::vector<data_t> data_host = inputData;
  cukd::buildTree_host
    <PointWithPayload<T,D>,PointWithPayload_traits<T,D>>
    (data_host.data(),numPoints,d_bounds);

  size_t hash_host = computeHash((uint32_t*)data_host.data(),numPoints*sizeof(data_t));
  std::cout << "hash host:\t " << (int*)hash_host << std::endl;

  if (numPoints <= 10)
    printTree(data_host.data(),numPoints);

  // ------------------------------------------------------------------
  CUKD_CUDA_CALL(Memcpy(d_data,inputData.data(),numPoints*sizeof(data_t),cudaMemcpyDefault));
  cukd::buildTree_thrust
    <PointWithPayload<T,D>,PointWithPayload_traits<T,D>>
                 (d_data,numPoints,d_bounds);
  CUKD_CUDA_SYNC_CHECK();
  
  size_t hash_thrust = computeHash((uint32_t*)d_data,numPoints*sizeof(data_t));
  std::cout << "hash thrust:\t " << (int*)hash_thrust << std::endl;
  if (numPoints <= 10)
    printTree(d_data,numPoints);

  

  // ------------------------------------------------------------------
  CUKD_CUDA_CALL(Memcpy(d_data,inputData.data(),numPoints*sizeof(data_t),cudaMemcpyDefault));
  cukd::buildTree_bitonic
    <PointWithPayload<T,D>,PointWithPayload_traits<T,D>>
    (d_data,numPoints,d_bounds);
  CUKD_CUDA_SYNC_CHECK();

  size_t hash_bitonic = computeHash((uint32_t*)d_data,numPoints*sizeof(data_t));
  std::cout << "hash bitonic:\t " << (int*)hash_bitonic << std::endl;
  if (numPoints <= 10)
    printTree(d_data,numPoints);
  
  // ------------------------------------------------------------------
  CUKD_CUDA_CALL(Memcpy(d_data,inputData.data(),numPoints*sizeof(data_t),cudaMemcpyDefault));
  cukd::buildTree_inPlace
    <PointWithPayload<T,D>,PointWithPayload_traits<T,D>>
    (d_data,numPoints,d_bounds);
  CUKD_CUDA_SYNC_CHECK();

  size_t hash_inPlace = computeHash((uint32_t*)d_data,numPoints*sizeof(data_t));
  std::cout << "hash inPlace:\t " << (int*)hash_inPlace << std::endl;
  if (numPoints <= 10)
    printTree(d_data,numPoints);

  if (hash_thrust  != hash_host ||
      hash_bitonic != hash_host ||
      hash_inPlace  != hash_host)
    throw std::runtime_error("hashes do not match!");

  CUKD_CUDA_CALL(Free(d_data));
  CUKD_CUDA_CALL(Free(d_bounds));
  // std::uniform_int_distribution<> distrib(0,D);

  // using data_t = PointWithPayload<T,D>;
  // std::vector<data_t> inputData(numPoints);
  // for (auto &data : inputData)
  //   for (int i=0;i<D;i++)
  //     data.position.v[i] = distrib(gen);

  // // ------------------------------------------------------------------
  // std::vector<data_t> data_host = inputData;
  // cukd::buildTree_host
  //   <PointWithPayload<T,D>,PointWithPayload_traits<T,D>>
  //   (data_host.data(),numPoints);
}

template<int N>
void testN(int sizeToTest)
{
  test_pointOnly<float,N>(sizeToTest);
  test_pointOnly<int,N>(sizeToTest);

  test_withPayload<float,N>(sizeToTest);
  test_withPayload<int,N>(sizeToTest);
}

void testAll(int sizeToTest)
{
  testN<2>(sizeToTest);
  testN<3>(sizeToTest);
  testN<4>(sizeToTest);
  testN<8>(sizeToTest);
}

int main(int, const char **)
{
  std::vector<int> sizesToTest = { 4,10,1000,10000 };
  for (auto sizeToTest : sizesToTest)
    testAll(sizeToTest);
  return 0;
}

