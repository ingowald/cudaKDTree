// ======================================================================== //
// Copyright 2019-2023 Ingo Wald                                            //
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

#pragma once

#include "cukd/helpers.h"
#include "cukd/box.h"
#include "cukd/data.h"

#include <cuda.h>

namespace cukd {

  /*! helper function for swapping two elements - need to explcitly
      prefix this to avoid name clashed with/in thrust */
  template<typename T>
  inline __both__ void cukd_swap(T &a, T &b)
  { T c = a; a = b; b = c; }

  
  /*! helper class to allow for conditionally "dropping" calls to
    set_dim/get_dim for data that doesn't have those functions */
  template<typename data_t, typename data_traits, bool has_dim>
  struct if_has_dims;
  
  template<typename data_t, typename data_traits>
  struct if_has_dims<data_t,data_traits,false> {
    static inline __both__ void set_dim(data_t &t, int dim) {}
    static inline __both__ int get_dim(const data_t &t, int value_if_false)
    { return value_if_false; }
  };
  
  template<typename data_t, typename data_traits>
  struct if_has_dims<data_t,data_traits,true> {
    static inline __both__ void set_dim(data_t &t, int dim) {
      data_traits::set_dim(t,dim);
    }
    static inline __both__ int get_dim(const data_t &t, int /* ignore: value_if_false */) {
      return data_traits::get_dim(t);
    }
  };
  /*! @} */
  
  /*! helper function that computes the bounding box of a given set of
      points */
  template<typename data_t, 
           typename data_traits=default_data_traits<data_t>>
  void computeBounds(cukd::box_t<typename data_traits::point_t> *d_bounds,
                     const data_t *d_points,
                     int numPoints,
                     cudaStream_t stream=0);
  
  template<typename data_t, 
           typename data_traits=default_data_traits<data_t>>
  void host_computeBounds(cukd::box_t<typename data_traits::point_t> *d_bounds,
                          const data_t *d_points,
                          int numPoints);

  // ==================================================================
  // IMPLEMENTATION SECTION
  // ==================================================================

  template<typename data_t, typename data_traits>
  __global__
  void computeBounds_copyFirst(cukd::box_t<typename data_traits::point_t> *d_bounds,
                               const data_t *d_points)
  {
    if (threadIdx.x != 0) return;
    
    using point_t = typename data_traits::point_t;
    const point_t point = data_traits::get_point(d_points[0]);
    d_bounds->lower = d_bounds->upper = point;
  }

  inline __device__
  int atomicMin(int *addr, int value)
  { return ::atomicMin(addr,value); }
  
  inline __device__
  int atomicMax(int *addr, int value)
  { return ::atomicMax(addr,value); }
  
  inline __device__
  float atomicMin(float *addr, float value)
  {
    float old = *addr, assumed;
    if(old <= value) return old;
    do {
      assumed = old;
      old = __int_as_float(atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value)));
      value = min(value,old);
    } while(old!=assumed);
    return old;
  }

  inline __device__
  float atomicMax(float *addr, float value)
  {
    float old = *addr, assumed;
    if(old >= value) return old;
    do {
      assumed = old;
      old = __int_as_float(atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value)));
      value = max(value,old);
    } while(old!=assumed);
    return old;
  }

  template<typename data_t,
           typename data_traits>
  __global__
  void computeBounds_atomicGrow(cukd::box_t<typename data_traits::point_t> *d_bounds,
                                const data_t *d_points,
                                int numPoints)
  {
    using point_t = typename data_traits::point_t;
    using point_traits = ::cukd::point_traits<point_t>;//typename data_traits::point_traits;
    using scalar_t = typename point_traits::scalar_t;
    enum { num_dims = point_traits::num_dims };
    
    const int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numPoints) return;
    
    point_t point = data_traits::get_point(d_points[tid]);
#pragma unroll(num_dims)
    for (int d=0;d<num_dims;d++) {
      scalar_t &lo = point_traits::get_coord(d_bounds->lower,d);
      scalar_t &hi = point_traits::get_coord(d_bounds->upper,d);
      scalar_t f = point_traits::get_coord(point,d);
      atomicMin(&lo,f);
      atomicMax(&hi,f);
    }
  }

  /*! host-side helper function to compute bounding box of the data set */
  template<typename data_t, typename data_traits>
  void computeBounds(cukd::box_t<typename data_traits::point_t> *d_bounds,
                     const data_t *d_points,
                     int numPoints,
                     cudaStream_t s)
  {
    computeBounds_copyFirst<data_t,data_traits>
      <<<1,1,0,s>>>
      (d_bounds,d_points);
    computeBounds_atomicGrow<data_t,data_traits>
      <<<divRoundUp(numPoints,128),128,0,s>>>
      (d_bounds,d_points,numPoints);
  }

  /*! host-side helper function to compute bounding box of the data set */
  template<typename data_t, typename data_traits>
  void host_computeBounds(cukd::box_t<typename data_traits::point_t> *d_bounds,
                          const data_t *d_points,
                          int numPoints)
  {
    d_bounds->setEmpty();
    for (int i=0;i<numPoints;i++)
      d_bounds->grow(data_traits::get_point(d_points[i]));
  }
  

  /*! helper function that finds, for a given node in the tree, the
      bounding box of that subtree's domain; by walking _up_ the tree
      and applying all clipping planes to the world-space bounding
      box */
  template<typename data_t,typename data_traits>
  inline __both__
  cukd::box_t<typename data_traits::point_t>
  findBounds(int subtree,
             const cukd::box_t<typename data_traits::point_t> *d_bounds,
             data_t *d_nodes)
  {
    using point_t  = typename data_traits::point_t;
    using point_traits = ::cukd::point_traits<point_t>;
    using scalar_t = typename point_traits::scalar_t;
    enum { num_dims = point_traits::num_dims };
    
    cukd::box_t<typename data_traits::point_t> bounds = *d_bounds;
    int curr = subtree;
    while (curr > 0) {
      const int     parent = (curr+1)/2-1;
      const data_t &parent_node = d_nodes[parent];
      const int     parent_dim
        = if_has_dims<data_t,data_traits,data_traits::has_explicit_dim>
        ::get_dim(parent_node,/* if not: */BinaryTree::levelOf(parent) % num_dims);
      const scalar_t parent_split_pos
        = data_traits::get_coord(parent_node,parent_dim);
      
      if (curr & 1) {
        // curr is left child, set upper
        point_traits::set_coord(bounds.upper,parent_dim,
                                min(parent_split_pos,
                                    get_coord(bounds.upper,parent_dim)));
      } else {
        // curr is right child, set lower
        point_traits::set_coord(bounds.lower,parent_dim,
                                max(parent_split_pos,
                                    get_coord(bounds.lower,parent_dim)));
      }
      curr = parent;
    }
    
    return bounds;
  }
  

}
