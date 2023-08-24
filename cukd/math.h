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

/* copied from OWL project, and put into new namespace to avoid naming conflicts.*/

#pragma once

#include "cukd/common.h"

namespace cukd {

#ifdef __CUDA_ARCH__
  using ::min;
  using ::max;
  using std::abs;
#else
  using std::min;
  using std::max;
  using std::abs;
#endif

  // ==================================================================
  // default operators on cuda vector types:
  // ==================================================================

  /*! template interface for cuda vector types (such as float3, int4,
      etc), that allows for querying which scalar type this vec is
      defined over */
  template<typename cuda_vec_t> struct scalar_type_of;
  template<> struct scalar_type_of<float2> { using type = float; };
  template<> struct scalar_type_of<float3> { using type = float; };
  template<> struct scalar_type_of<float4> { using type = float; };
  template<> struct scalar_type_of<int2>   { using type = int; };
  template<> struct scalar_type_of<int3>   { using type = int; };
  template<> struct scalar_type_of<int4>   { using type = int; };
  
  /*! template interface for cuda vector types (such as float3, int4,
      etc), that allows for querying which scalar type this vec is
      defined over */
  template<typename cuda_vec_t> struct num_dims_of;
  template<> struct num_dims_of<float2> { enum { value = 2 }; };
  template<> struct num_dims_of<float3> { enum { value = 3 }; };
  template<> struct num_dims_of<float4> { enum { value = 4 }; };
  template<> struct num_dims_of<int2>   { enum { value = 2 }; };
  template<> struct num_dims_of<int3>   { enum { value = 3 }; };
  template<> struct num_dims_of<int4>   { enum { value = 4 }; };

  inline __both__ float get_coord(const float2 &v, int d) { return d?v.y:v.x; }
  inline __both__ float get_coord(const float3 &v, int d) { return (d==2)?v.z:(d?v.y:v.x); }
  inline __both__ float get_coord(const float4 &v, int d) { return (d>=2)?(d>2?v.w:v.z):(d?v.y:v.x); }
  
  inline __both__ float &get_coord(float2 &v, int d) { return d?v.y:v.x; }
  inline __both__ float &get_coord(float3 &v, int d) { return (d==2)?v.z:(d?v.y:v.x); }
  inline __both__ float &get_coord(float4 &v, int d) { return (d>=2)?(d>2?v.w:v.z):(d?v.y:v.x); }


  inline __both__ int get_coord(const int2 &v, int d) { return d?v.y:v.x; }
  inline __both__ int get_coord(const int3 &v, int d) { return (d==2)?v.z:(d?v.y:v.x); }
  inline __both__ int get_coord(const int4 &v, int d) { return (d>=2)?(d>2?v.w:v.z):(d?v.y:v.x); }
  
  inline __both__ int &get_coord(int2 &v, int d) { return d?v.y:v.x; }
  inline __both__ int &get_coord(int3 &v, int d) { return (d==2)?v.z:(d?v.y:v.x); }
  inline __both__ int &get_coord(int4 &v, int d) { return (d>=2)?(d>2?v.w:v.z):(d?v.y:v.x); }

  
  inline __both__ void set_coord(int2 &v, int d, int vv) { (d?v.y:v.x) = vv; }
  inline __both__ void set_coord(int3 &v, int d, int vv) { ((d==2)?v.z:(d?v.y:v.x)) = vv; }
  inline __both__ void set_coord(int4 &v, int d, int vv) { ((d>=2)?(d>2?v.w:v.z):(d?v.y:v.x)) = vv; }
  
  inline __both__ void set_coord(float2 &v, float d, float vv) { (d?v.y:v.x) = vv; }
  inline __both__ void set_coord(float3 &v, float d, float vv) { ((d==2)?v.z:(d?v.y:v.x)) = vv; }
  inline __both__ void set_coord(float4 &v, float d, float vv) { ((d>=2)?(d>2?v.w:v.z):(d?v.y:v.x)) = vv; }
  
  inline __both__ int32_t divRoundUp(int32_t a, int32_t b) { return (a+b-1)/b; }
  inline __both__ uint32_t divRoundUp(uint32_t a, uint32_t b) { return (a+b-1)/b; }
  inline __both__ int64_t divRoundUp(int64_t a, int64_t b) { return (a+b-1)/b; }
  inline __both__ uint64_t divRoundUp(uint64_t a, uint64_t b) { return (a+b-1)/b; }

  using ::sin; // this is the double version
  using ::cos; // this is the double version

  // ==================================================================
  // default operators on cuda vector types:
  // ==================================================================


  inline __both__ float2 operator-(float2 a, float2 b)
  { return make_float2(a.x-b.x,a.y-b.y); }
  inline __both__ float3 operator-(float3 a, float3 b)
  { return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
  inline __both__ float4 operator-(float4 a, float4 b)
  { return make_float4(a.x-b.x,a.y-b.y,a.z-b.z,a.w-b.w); }

  inline __both__ float dot(float2 a, float2 b)
  { return a.x*b.x+a.y*b.y; }
  inline __both__ float dot(float3 a, float3 b)
  { return a.x*b.x+a.y*b.y+a.z*b.z; }
  inline __both__ float dot(float4 a, float4 b)
  { return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w; }
  
  inline __both__ float2 min(float2 a, float2 b)
  { return make_float2(min(a.x,b.x),min(a.y,b.y)); }
  inline __both__ float3 min(float3 a, float3 b)
  { return make_float3(min(a.x,b.x),min(a.y,b.y),min(a.z,b.z)); }
  inline __both__ float4 min(float4 a, float4 b)
  { return make_float4(min(a.x,b.x),min(a.y,b.y),min(a.z,b.z),min(a.w,b.w)); }

  inline __both__ float2 max(float2 a, float2 b)
  { return make_float2(max(a.x,b.x),max(a.y,b.y)); }
  inline __both__ float3 max(float3 a, float3 b)
  { return make_float3(max(a.x,b.x),max(a.y,b.y),max(a.z,b.z)); }
  inline __both__ float4 max(float4 a, float4 b)
  { return make_float4(max(a.x,b.x),max(a.y,b.y),max(a.z,b.z),max(a.w,b.w)); }

  inline std::ostream &operator<<(std::ostream &o, float3 v)
  { o << "(" << v.x << "," << v.y << "," << v.z << ")"; return o; }

      
  // ==================================================================
  // for some tests: our own, arbitrary-dimensioal vector type
  // ==================================================================
  template<int N>
  struct vec_float {
    float v[N];
  };
  template<int N> struct scalar_type_of<vec_float<N>> { using type = float; };
  template<int N> struct num_dims_of<vec_float<N>> { enum { value = N }; };
  
  template<int N>
  inline __both__ float get_coord(const vec_float<N> &v, int d) { return v.v[d]; }
  template<int N>
  inline __both__ float &get_coord(vec_float<N> &v, int d) { return v.v[d]; }
  template<int N>
  inline __both__ void set_coord(vec_float<N> &v, int d, float vv) { v.v[d] = vv; }
  
  

  template<int N>
  inline __both__ vec_float<N> min(vec_float<N> a, vec_float<N> b)
  {
    vec_float<N> r;
    for (int i=0;i<N;i++) r.v[i] = min(a.v[i],b.v[i]);
    return r;
  }
  
  template<int N>
  inline __both__ vec_float<N> max(vec_float<N> a, vec_float<N> b)
  {
    vec_float<N> r;
    for (int i=0;i<N;i++) r.v[i] = max(a.v[i],b.v[i]);
    return r;
  }

  template<int N>
  inline __both__ float dot(vec_float<N> a, vec_float<N> b)
  {
    float sum = 0.f;
    for (int i=0;i<N;i++) sum += a.v[i] * b.v[i];
    return sum;
  }
  
  template<int N>
  inline __both__ vec_float<N> operator-(const vec_float<N> &a, const vec_float<N> &b)
  {
    vec_float<N> r;
    for (int i=0;i<N;i++) r.v[i] = a.v[i] - b.v[i];
    return r;
  }



  // ------------------------------------------------------------------
  /*! @{ helper function(s) to convert scalar of any type to float,
      with guarnateed round-to-zero mode, so functions like fSqrDist
      can reliably compute distance in float with conservative
      distance metric */

  template<typename T> inline __both__ float as_float_rz(T t);
  template<> inline __both__ float as_float_rz(float f) { return f; }
  template<> inline __device__ float as_float_rz(int i) { return __int2float_rz(i); }
  
  /*! @] */

  
  // ------------------------------------------------------------------
  /*! float-accuracy (with round-to-zero mode) of distance between two point_t's */
  template<typename point_t>
  inline __both__
  float fSqrDistance(const point_t &a, const point_t &b)
  {
    const point_t diff = b-a;
    return as_float_rz(dot(diff,diff));
  }

  template<typename point_t>
  inline __both__
  auto sqrDistance(const point_t &a, const point_t &b)
  { const point_t d = a-b; return dot(d,d); }

  // ------------------------------------------------------------------
  // scalar distance(point,point)
  // ------------------------------------------------------------------

  inline __both__ float square_root(float f) { return sqrtf(f); }
  
  template<typename point_t>
  inline __both__ auto distance(const point_t &a, const point_t &b)
  { return square_root(sqrDistance(a,b)); }
  
  // ------------------------------------------------------------------
  template<typename point_t>
  inline __both__ int arg_max(point_t p)
  {
    enum { num_dims = num_dims_of<point_t>::value };
    using scalar_t = typename scalar_type_of<point_t>::type;
    int best_dim = 0;
    scalar_t best_val = get_coord(p,0);
    for (int i=1;i<num_dims;i++) {
      scalar_t f = get_coord(p,i);
      if (f > best_val) {
        best_val = f;
        best_dim = i;
      }
    }
    return best_dim;
  }
  
  // ------------------------------------------------------------------
  inline std::ostream &operator<<(std::ostream &out,
                                  float2 v)
  {
    out << "(" << v.x << "," << v.y << ")";
    return out;
  }

  template <typename scalar_t>
  inline __device__ __host__
  auto sqr(scalar_t f) { return f * f; }

  template <typename scalar_t>
  inline __device__ __host__
  scalar_t sqrt(scalar_t f);

  template<> inline __device__ __host__
  float sqrt(float f) { return ::sqrtf(f); }




  


  
  template <typename point_traits_a, typename point_traits_b=point_traits_a>
  inline __device__ __host__
  auto sqrDistance(const typename point_traits_a::point_t& a,
                   const typename point_traits_b::point_t& b)
  {
    typename point_traits_a::scalar_t res = 0;
    for(int i=0; i<min(point_traits_a::numDims, point_traits_b::numDims); ++i) {
      const auto diff = point_traits_a::getCoord(a, i) - point_traits_b::getCoord(b, i);
      res += sqr(diff);
    }
    return res;
  }

  template <typename point_traits_a, typename point_traits_b=point_traits_a>
  inline __device__ __host__
  auto distance(const typename point_traits_a::point_t& a,
                const typename point_traits_b::point_t& b)
  {
    typename point_traits_a::scalar_t res = 0;
    for(int i=0; i<min(point_traits_a::numDims, point_traits_b::numDims); ++i) {
      const auto diff = point_traits_a::getCoord(a, i) - point_traits_b::getCoord(b, i);
      res += sqr(diff);
    }
    return sqrt(res);
  }




  template<typename T> struct point_traits;

  /*! point traits that describe our defaul tpoint type of cuda float3, int3, float4, etc.
    
    The four basic things a point_traits has to do for a given type are:
    
    - define the scalar_t that this point is built over
    
    - define the enum num_dims of dimensions that this point has
    
    - define a static function `get_coord(const point_t, int d)` that
    returns the given point's d'th coordiate
    
    - define a static function `set_coord(point_t &, int d, scalar_t
      v)` that sets the given point's d'the coordinate to the given
      value
   */
  template<typename cuda_t>
  struct point_traits {
    enum { num_dims = num_dims_of<cuda_t>::value };
    using scalar_t  = typename scalar_type_of<cuda_t>::type;

    /*! get the d'th coordindate - for our default cuda types we use
        the ::cukd::get_coord helpers we hvae for those types */
    static inline __both__
    scalar_t get_coord(const cuda_t &v, int d) { return ::cukd::get_coord(v,d); }

    static inline __both__
    scalar_t &get_coord(cuda_t &v, int d) { return ::cukd::get_coord(v,d); }
    
    static inline __both__
    void set_coord(cuda_t &v, int d, scalar_t vv) { ::cukd::set_coord(v,d,vv); }
  };




  
} // ::cukd
