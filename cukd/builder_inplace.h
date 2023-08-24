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

#include "cukd/builder_common.h"

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
  void buildTree_inPlace(/*! device-read/writeable array of data points */
                         data_t      *points,
                         /*! number of data points */
                         int          numPoints,
                         /*! device-writeable pointer to store the world-space
                           bounding box of all data points. if
                           data_traits::has_explicit_dim is false, this is
                           optionally allowed to be null */
                         box_t<typename data_traits::point_t> *worldBounds=0,
                         /*! cuda stream to use for all kernels and mallocs
                           (the builder_thrust may _also_ do some global
                           device syncs) */
                         cudaStream_t stream = 0,
                         /*! memory resource that can be used to
                           control how memory allocations will be
                           implemented (eg, using Async allocs only
                           on CDUA > 11, or using managed vs device
                           mem) */
                         GpuMemoryResource &memResource=defaultGpuMemResource());

  // ==================================================================
  // IMPLEMENTATION SECTION
  // ==================================================================

  namespace inPlaceBuilder {
    
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
      const int input_n  = n;
      using point_t      = typename data_traits::point_t;
      using point_traits = ::cukd::point_traits<point_t>;
      using scalar_t     = typename point_traits::scalar_t;
      
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


    template<typename data_t, typename data_traits>
    __global__ void d_quickSwap(/*! _build_ root level */int L_b,
                                data_t *points,
                                int numPoints)
    {
      int n = threadIdx.x + blockIdx.x*blockDim.x;
      if (n >= numPoints) return;
    
      int L_n = BinaryTree::levelOf(n);
      if (L_n <= L_b) return;
    
      int partner = partnerOf(n,L_n,L_b);
      if (partner >= numPoints) return;

      if (partner < n)
        // only one of the two can do the work, or they'll race each
        // other - let's always pick the lower one.
        return;

    using point_t  = typename data_traits::point_t;
    using point_traits = ::cukd::point_traits<point_t>;
    using scalar_t = typename point_traits::scalar_t;
    enum { num_dims = point_traits::num_dims };

      const int     dim
        = data_traits::has_explicit_dim
        ? data_traits::get_dim(points[((n+1)>>(L_n-L_b))-1])
        : (L_b % num_dims);
    
      if (data_traits::get_coord(points[partner],dim)
          <
          data_traits::get_coord(points[n],dim)) {
        cukd::cukd_swap(points[n],points[partner]);
      } 
    }
  
    template<typename data_t, typename data_traits>
    void quickSwap(/*! _build_ root level */
                   int          L_b,
                   data_t      *points,
                   int          numPoints,
                   cudaStream_t stream)
    {
      // printTree<data_t,data_traits>(points,numPoints);
      // std::cout << "---- building heaps on " << L_h << ", root level " << L_b << std::endl << std::flush;
      int bs = 1024;
      int nb = divRoundUp(numPoints,bs);
      d_quickSwap<data_t,data_traits><<<nb,bs,0,stream>>>(L_b,points,numPoints);
    }

    template<typename data_t, typename data_traits>
    void printTree(data_t *points,int numPoints)
    {
      cudaDeviceSynchronize();

    using point_t  = typename data_traits::point_t;
    using point_traits = ::cukd::point_traits<point_t>;
    using scalar_t = typename point_traits::scalar_t;
    enum { num_dims = point_traits::num_dims };

      for (int L=0;true;L++) {
        int begin = firstNodeOnLevel(L);
        int end = std::min(numPoints,begin+numNodesOnLevel(L));
        if (end <= begin) break;
        printf("### level %i ###\n",L);
        for (int i=begin;i<end;i++) 
          printf("%5i.",i);
        printf("\n");
      
        for (int d=0;d<num_dims;d++) {
          for (int i=begin;i<end;i++) 
            printf("%5.3f ",(data_traits::get_coord(points[i],d)));
          // printf("%6i",int(data_traits::get_coord(points[i],d)));
          printf("\n");
        }
      }
    }
  

    template<typename data_t, typename data_traits>
    __global__ void d_buildHeaps(/*! _heap_ root level */int L_h,
                                 /*! _build_ root level */int L_b,
                                 data_t *__restrict__ points,
                                 int numPoints)
    {
      int tid = threadIdx.x + blockIdx.x*blockDim.x;
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
    using point_traits = ::cukd::point_traits<point_t>;
    using scalar_t = typename point_traits::scalar_t;
    enum { num_dims = point_traits::num_dims };

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
                    int          numPoints,
                    cudaStream_t stream)
    {
      int numNodesOnL_h = numNodesOnLevel(L_h);
      int bs = 64;
      int nb = divRoundUp(numNodesOnL_h,bs);
      d_buildHeaps<data_t,data_traits><<<nb,bs,0,stream>>>
        (L_h,L_b,points,numPoints);
    }



    template<typename data_t, typename data_traits>
    __global__
    void d_selectDimsOnLevel(int     L_b,
                             data_t *points,
                             int     numPoints,
                             box_t<typename data_traits::point_t> *worldBounds)
    {
      int tid = threadIdx.x + blockIdx.x*blockDim.x;
      int numNodesOnL_b = numNodesOnLevel(L_b);
      if (tid >= numNodesOnL_b)
        return;

      int n = firstNodeOnLevel(L_b)+tid;
      if (n >= numPoints) return;
                                           
    using point_t  = typename data_traits::point_t;
    using point_traits = ::cukd::point_traits<point_t>;
    using scalar_t = typename point_traits::scalar_t;
    enum { num_dims = point_traits::num_dims };

      if (worldBounds) {
        box_t<typename data_traits::point_t> bounds
          = findBounds<data_t,data_traits>(n,worldBounds,points);
        data_traits::set_dim(points[n],bounds.widestDimension());
      } else {
        data_traits::set_dim(points[n],L_b % num_dims);
      }
    }

    template<typename data_t, typename data_traits>
    void selectDimsOnLevel(int          L_b,
                           data_t      *points,
                           int          numPoints,
                           box_t<typename data_traits::point_t> *worldBounds,
                           cudaStream_t stream)
    {
      // std::cout << "selecting dims ..." << std::endl << std::flush;
      int numNodesOnL_b = numNodesOnLevel(L_b);
      int bs = 64;
      int nb = divRoundUp(numNodesOnL_b,bs);
      d_selectDimsOnLevel<data_t,data_traits><<<nb,bs,0,stream>>>
        (L_b,points,numPoints,worldBounds);
    }
  


    template<typename data_t, typename data_traits>
    __global__
    void d_fixPivots(/*! _build_ root level */
                     int     L_b,
                     data_t *points,
                     int     numPoints)
    {
      int tid = threadIdx.x + blockIdx.x*blockDim.x;
      int numNodesOnL_b = numNodesOnLevel(L_b);
      if (tid >= numNodesOnL_b)
        return;

      int n = firstNodeOnLevel(L_b)+tid;
      if (n >= numPoints) return;
                                           
      int l = 2*n+1;
      int r = l+1;


    using point_t  = typename data_traits::point_t;
    using point_traits = ::cukd::point_traits<point_t>;
    using scalar_t = typename point_traits::scalar_t;
    enum { num_dims = point_traits::num_dims };
    
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
    void fixPivots(/*! _build_ root level */int L_b,
                   data_t *points,
                   int numPoints,
                   cudaStream_t stream)
    {
      int numNodesOnL_b = numNodesOnLevel(L_b);
      int bs = 64;
      int nb = divRoundUp(numNodesOnL_b,bs);
      d_fixPivots<data_t,data_traits><<<nb,bs,0,stream>>>(L_b,points,numPoints);
    }

    template<typename data_t, typename data_traits>
    void buildLevel(/*! level that we're ultimately _building_ */
                    int          L_b,
                    int          numLevels,
                    data_t      *d_points,
                    int          numPoints,
                    box_t<typename data_traits::point_t> *worldBounds,
                    cudaStream_t stream)
    {
      if (data_traits::has_explicit_dim)
        selectDimsOnLevel<data_t,data_traits>
          (L_b,d_points,numPoints,worldBounds,stream);
    
      for (int L_h = numLevels-1; L_h > L_b; --L_h)
        buildHeaps<data_t,data_traits>(L_h,L_b,d_points,numPoints,stream);

      fixPivots<data_t,data_traits>(L_b,d_points,numPoints,stream);
    }
  } // ::cukd::inPlaceBuilder
  
  template<typename data_t, typename data_traits>
  void buildTree_inPlace(data_t      *points,
                         int          numPoints,
                         box_t<typename data_traits::point_t> *worldBounds,
                         cudaStream_t stream,
                         /*! memory resource that can be used to
                           control how memory allocations will be
                           implemented (eg, using Async allocs only
                           on CDUA > 11, or using managed vs device
                           mem) */
                         GpuMemoryResource &memResource)
  {
    if (numPoints <= 1)
      return;
    if (worldBounds) 
      computeBounds<data_t,data_traits>(worldBounds,points,numPoints,stream);
    else if  (data_traits::has_explicit_dim) 
      throw std::runtime_error
        ("cukd::builder_inplace: asked to build k-d tree over "
         "nodes with explicit dims, but no memory for world bounds provided");
    
    int numLevels = BinaryTree::numLevelsFor(numPoints);
    for (int L_b = 0; L_b < numLevels; L_b++)
      inPlaceBuilder::buildLevel<data_t,data_traits>
        (L_b,numLevels,points,numPoints,worldBounds,stream);
    
    cudaStreamSynchronize(stream);
  }
  
  /*! non-generalized direction tree build */
  template<typename data_t, typename data_traits>
  void buildTree_inPlace(data_t *points,
                         int numPoints,
                         cudaStream_t stream)
  {
    buildTree_inPlace<data_t,data_traits>(points,numPoints,nullptr,stream);
  }
  
}
