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


  /*! helper function that computes the bounding box of a given set of
      points */
  template<typename data_t, 
           typename data_traits=default_data_traits<data_t>>
  void computeBounds(cukd::box_t<typename data_traits::point_t> *d_bounds,
                     const data_t *d_points,
                     int numPoints,
                     cudaStream_t stream=0);

  // ==================================================================
  // IMPLEMENTATION SECTION
  // ==================================================================

  template<typename data_t, typename data_traits>
  __global__
  void computeBounds_copyFirst(box_t<typename data_traits::point_t> *d_bounds,
                               const data_t *d_points)
  {
    if (threadIdx.x != 0) return;
    
    using point_t = typename data_traits::point_t;
    const point_t point = data_traits::get_point(d_points[0]);
    d_bounds->lower = d_bounds->upper = point;
  }

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
  void computeBounds_atomicGrow(box_t<typename data_traits::point_t> *d_bounds,
                                const data_t *d_points,
                                int numPoints)
  {
    using point_t = typename data_traits::point_t;
    enum { num_dims = num_dims_of<point_t>::value };
    
    const int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numPoints) return;
    
    using point_t = typename data_traits::point_t;
    point_t point = data_traits::get_point(d_points[tid]);
#pragma unroll(num_dims)
    for (int d=0;d<num_dims;d++) {
      float &lo = get_coord(d_bounds->lower,d);
      float &hi = get_coord(d_bounds->upper,d);
      float f = get_coord(point,d);
      atomicMin(&lo,f);
      atomicMax(&hi,f);
    }
  }

  /*! host-side helper function to compute bounding box of the data set */
  template<typename data_t, typename data_traits>
  void computeBounds(box_t<typename data_traits::point_t> *d_bounds,
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


  /*! helper function that finds, for a given node in the tree, the
      bounding box of that subtree's domain; by walking _up_ the tree
      and applying all clipping planes to the world-space bounding
      box */
  template<typename data_t,typename data_traits>
  inline __device__
  box_t<typename data_traits::point_t>
  findBounds(int subtree,
             const box_t<typename data_traits::point_t> *d_bounds,
             data_t *d_nodes)
  {
    using point_t  = typename data_traits::point_t;
    using scalar_t = typename scalar_type_of<point_t>::type;
    enum { num_dims = num_dims_of<point_t>::value };
    
    box_t<point_t> bounds = *d_bounds;
    int curr = subtree;
    // const bool dbg = false;
    while (curr > 0) {
      const int     parent = (curr+1)/2-1;
      const data_t &parent_node = d_nodes[parent];
      const int     parent_dim
        = data_traits::has_explicit_dim
        ? data_traits::get_dim(parent_node)
        : (BinaryTree::levelOf(parent) % num_dims);
      const scalar_t parent_split_pos
        = data_traits::get_coord(parent_node,parent_dim);
      
      if (curr & 1) {
        // curr is left child, set upper
        get_coord(bounds.upper,parent_dim)
          = min(parent_split_pos,
                get_coord(bounds.upper,parent_dim));
      } else {
        // curr is right child, set lower
        get_coord(bounds.lower,parent_dim)
          = max(parent_split_pos,
                get_coord(bounds.lower,parent_dim));
      }
      curr = parent;
    }
    
    return bounds;
  }
  

}
