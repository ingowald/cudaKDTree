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

#include "cukd/builder_thrust.h"

namespace cukd {

  // ==================================================================
  // INTERFACE SECTION
  // ==================================================================

  /*! Builds a left-balanced k-d tree over the given data points,
    using data_traits to describe the type of data points that this
    tree is being built over (i.e., how to separate a data item's
    positional coordinates from any potential payload (if such exists,
    e.g., in a 'photon' in photon mapping), what vector/point type to
    use for this coordinate data (e.g., float3), whether the data have
    a field to store an explicit split dimensional (for Bentley and
    Samet's 'optimized' trees, etc.

    Since a (point-)k-d tree's tree topology is implicit in the
    ordering of its data points this will re-arrange the data points
    to fulfill the balanced k-d tree criterion - ie, this WILL modify
    the data array: no individual entry will get changed, but their
    order might. If data_traits::has_explicit_dims is defined this
    builder will choose each node's split dimension based on the
    widest dimension of that node's subtree's domain; if not, it will
    chose the dimension in a round-robin style, where the root level
    is split along the 'x' coordinate, the next level in y, etc

    'worldBounds' is a pointer to device-writeable memory to store the
    world-space bounding box of the data points that the builder will
    compute. If data_traits::has_explicit_dims is true this memory
    _has_ to be provided to the builder, and the builder will fill it
    in; if data_traits::has_explicit_dims is false, this memory region
    is optional: the builder _will_ fill it in if provided, but will
    ignore it if isn't.

    *** Example 1: To build a 2D k-dtree over a CUDA int2 type (no other
    payload than the two coordinates):
      
    buildTree<int2>(....);

    In this case no data_traits need to be supplied beause these will
    be auto-computed for simple cuda vector types.
      
    *** Example 2: to build a 1D kd-tree over a data type of float4,
    where the first coordinate of each point is the dimension we
    want to build the kd-tree over, and the other three coordinate
    are arbitrary other payload data:
      
    struct float2_plus_payload_traits {
    using point_t = float2;
    static inline __both__ const point_t &get_point(const float4 &n) 
    { return make_float2(n.z,n.w); }
    }
    buildTree<float4,float2_plus_payload_traits>(...);
      
    *** Example 3: assuming you have a data type 'Photon' and a
    Photon_traits has Photon_traits::has_explicit_dim defined:
      
    cukd::box_t<float3> *d_worldBounds = <cudaMalloc>;
    buildTree<Photon,Photon_traits>(..., worldBounds, ...);
      
  */
  template<typename data_t,
           typename data_traits=default_data_traits<data_t>>
  void buildTree_host(/*! device-read/writeable array of data points */
                      data_t *d_points,
                      /*! number of data points */
                      int numPoints,
                      /*! device-writeable pointer to store the world-space
                        bounding box of all data points. if
                        data_traits::has_explicit_dim is false, this is
                        optionally allowed to be null */
                      box_t<typename data_traits::point_t> *worldBounds=0);


  // ==================================================================
  // IMPLEMENTATION SECTION
  // ==================================================================

  namespace builder_host {

    inline __both__ int firstNodeOnLevel(int L) { return (1<<L) - 1; }
    inline __both__ int numNodesOnLevel(int L) { return 1<<L; }
    inline __both__ int partnerOf(int n, int L_r, int L_b)
    {
      return (((n+1) ^ (1<<(L_r-L_b-1))))-1;
    }


    template<typename scalar_t, int side>
    inline __both__
    bool desiredOrder(scalar_t a, scalar_t b)
    {
      if (side) {
        return !(b < a);
      } else {
        return !(a < b); 
      }
    }

    template<typename data_t, typename data_traits, int side>
    inline __both__
    void trickleDownHeap(int n,
                         data_t *__restrict__ points,
                         int numPoints,
                         int dim)
    {
      const int input_n = n;
      using point_t  = typename data_traits::point_t;
      using scalar_t = typename scalar_type_of<point_t>::type;
      data_t point_n = points[n];
      scalar_t s_n = data_traits::get_coord(point_n,dim);
      while (true) {
        int l = 2*n+1;
        if (l >= numPoints)
          break;
        scalar_t s_l = data_traits::get_coord(points[l],dim);

        int c = l;
        scalar_t s_c = s_l;
      
        int r = l+1;
        if (r < numPoints) {
          scalar_t s_r = data_traits::get_coord(points[r],dim);
          if (!desiredOrder<scalar_t,side>(s_c,s_r)) {
            c = r;
            s_c = s_r;
          }
        }
        if (desiredOrder<scalar_t,side>(s_n,s_c)) 
          break;

        points[n] = points[c];
        n = c;
      }
      if (n != input_n) 
        points[n] = point_n;
    }
    
    template<typename data_t,
             typename data_traits>
    inline
    void computeBounds_host(box_t<typename data_traits::point_t> *d_bounds,
                            const data_t *d_points,
                            int numPoints)
    {
      using point_t = typename data_traits::point_t;
      enum { num_dims = num_dims_of<point_t>::value };
    
      box_t<typename data_traits::point_t> bb;
      bb.setEmpty();

      for (int tid=0;tid<numPoints;tid++) {
        using point_t = typename data_traits::point_t;
        point_t point = data_traits::get_point(d_points[tid]);
        bb.grow(point);
      }
      *d_bounds = bb;
    }

    template<typename data_t, typename data_traits>
    // __global__
    inline
    void h_buildHeaps(int tid,
                      /*! _heap_ root level */int L_h,
                      /*! _build_ root level */int L_b,
                      data_t *__restrict__ points,
                      int numPoints)
    {
      // int tid = threadIdx.x + blockIdx.x*blockDim.x;
      if (L_h == L_b+1)
        tid *= 1<<(L_h-L_b);
      int numNodesOnL_h = numNodesOnLevel(L_h);
      if (tid >= numNodesOnL_h)
        return;
    
      int n = firstNodeOnLevel(L_h)+tid;
      if (n >= numPoints) return;
                                           
      int partner = partnerOf(n,L_h,L_b);
      if (partner >= numPoints) return;

      if (partner < n)
        // only one of the two can do the work, or they'll race each
        // other - let's always pick the lower one.
        return;

      using point_t  = typename data_traits::point_t;
      enum { num_dims = num_dims_of<point_t>::value };

      const int     dim
        = data_traits::has_explicit_dim
        ? data_traits::get_dim(points[((n+1)>>(L_h-L_b))-1])
        : (L_b % num_dims);

      if (data_traits::get_coord(points[partner],dim)
          <
          data_traits::get_coord(points[n],dim)) {
        cukd::cukd_swap(points[n],points[partner]);
      } 
      while (1) {
        trickleDownHeap<data_t,data_traits,0>(n,points,numPoints,dim);
        trickleDownHeap<data_t,data_traits,1>(partner,points,numPoints,dim);
        if (data_traits::get_coord(points[partner],dim)
            <
            data_traits::get_coord(points[n],dim)) {
          cukd::cukd_swap(points[n],points[partner]);
          continue;
        } else
          break;
      }
    }


    template<typename data_t, typename data_traits>
    void buildHeaps(/*! _heap_ root level */
                    int          L_h,
                    /*! _build_ root level */
                    int          L_b,
                    data_t      *points,
                    int          numPoints)
    {
      int numNodesOnL_h = numNodesOnLevel(L_h);
      // int bs = 64;
      // int nb = divRoundUp(numNodesOnL_h,bs);
      for (int tid=0;tid<numNodesOnL_h;tid++)
        h_buildHeaps<data_t,data_traits>//<<<nb,bs,0,stream>>>
          (tid,L_h,L_b,points,numPoints);
    }



    template<typename data_t, typename data_traits>
    inline
    void h_selectDimsOnLevel(int tid,
                             int     L_b,
                             data_t *points,
                             int     numPoints,
                             box_t<typename data_traits::point_t> *worldBounds)
    {
      // int tid = threadIdx.x + blockIdx.x*blockDim.x;
      int numNodesOnL_b = numNodesOnLevel(L_b);
      if (tid >= numNodesOnL_b) {
        return;
      }

      int n = firstNodeOnLevel(L_b)+tid;
      if (n >= numPoints) {
        return;
      }
                                           
      using point_t  = typename data_traits::point_t;
      enum { num_dims = num_dims_of<point_t>::value };

      if (worldBounds) {
        box_t<typename data_traits::point_t> bounds
          = findBounds<data_t,data_traits>(n,worldBounds,points);
        data_traits::set_dim(points[n],arg_max(bounds.size()));
      } else {
        data_traits::set_dim(points[n],L_b % num_dims);
      }
    }

    template<typename data_t, typename data_traits>
    void selectDimsOnLevel(int          L_b,
                                data_t      *points,
                                int          numPoints,
                                box_t<typename data_traits::point_t> *worldBounds)
    {
      // std::cout << "selecting dims ..." << std::endl << std::flush;
      int numNodesOnL_b = numNodesOnLevel(L_b);
      // int bs = 64;
      // int nb = divRoundUp(numNodesOnL_b,bs);
      for (int tid=0;tid<numNodesOnL_b;tid++)
        h_selectDimsOnLevel<data_t,data_traits>//<<<nb,bs,0,stream>>>
          (tid,L_b,points,numPoints,worldBounds);
    }
  
  
    template<typename data_t, typename data_traits>
    // __global__
    inline
    void h_fixPivots(int tid,
                     /*! _build_ root level */
                     int     L_b,
                     data_t *points,
                     int     numPoints)
    {
      // int tid = threadIdx.x + blockIdx.x*blockDim.x;
      int numNodesOnL_b = numNodesOnLevel(L_b);
      if (tid >= numNodesOnL_b)
        return;

      int n = firstNodeOnLevel(L_b)+tid;
      if (n >= numPoints) return;
                                           
      int l = 2*n+1;
      int r = l+1;

      using point_t  = typename data_traits::point_t;
      using scalar_t = typename scalar_type_of<point_t>::type;
      enum { num_dims = num_dims_of<point_t>::value };
    
      const int  dim
        = data_traits::has_explicit_dim
        ? data_traits::get_dim(points[n])
        : (L_b % num_dims);
    
      scalar_t s_n = data_traits::get_coord(points[n],dim);
      if (l < numPoints && s_n < data_traits::get_coord(points[l],dim)) {
        cukd::cukd_swap(points[n],points[l]);
        // todo: trckle?
      } else if  (r < numPoints && data_traits::get_coord(points[r],dim) < s_n) {
        cukd::cukd_swap(points[n],points[r]);
        // todo: trckle?
      }
      if (data_traits::has_explicit_dim) 
        data_traits::set_dim(points[n],dim);
    }
  
    template<typename data_t, typename data_traits>
    inline void fixPivots(/*! _build_ root level */int L_b,
                          data_t *points,
                          int numPoints)
    {
      int numNodesOnL_b = numNodesOnLevel(L_b);
      // int bs = 64;
      // int nb = divRoundUp(numNodesOnL_b,bs);
      for (int tid=0;tid<numNodesOnL_b;tid++)
        h_fixPivots<data_t,data_traits>(tid,L_b,points,numPoints);
    }
  

    template<typename data_t, typename data_traits>
    void buildLevel(/*! level that we're ultimately _building_ */
                    int          L_b,
                    int          numLevels,
                    data_t      *d_points,
                    int          numPoints,
                    box_t<typename data_traits::point_t> *worldBounds)
    {
      if (data_traits::has_explicit_dim)
        selectDimsOnLevel<data_t,data_traits>
          (L_b,d_points,numPoints,worldBounds);
    
      for (int L_h = numLevels-1; L_h > L_b; --L_h)
        buildHeaps<data_t,data_traits>(L_h,L_b,d_points,numPoints);

      fixPivots<data_t,data_traits>(L_b,d_points,numPoints);
    }
    

  }
  
  template<typename data_t, typename data_traits>
  void buildTree_host(data_t *points,
                      int numPoints,
                      box_t<typename data_traits::point_t> *worldBounds)
  {
    if (numPoints <= 1)
      return;
    if (worldBounds) 
      builder_host::computeBounds_host<data_t,data_traits>
        (worldBounds,points,numPoints);
    else if  (data_traits::has_explicit_dim) 
      throw std::runtime_error
        ("cukd::builder_inplace: asked to build k-d tree over "
         "nodes with explicit dims, but no memory for world bounds provided");
    
    int numLevels = BinaryTree::numLevelsFor(numPoints);
    for (int L_b = 0; L_b < numLevels; L_b++)
      builder_host::buildLevel<data_t,data_traits>
        (L_b,numLevels,points,numPoints,worldBounds);
  }
  
}
