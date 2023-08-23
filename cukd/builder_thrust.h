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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

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
  void buildTree_thrust(/*! device-read/writeable array of data points */
                        data_t *d_points,
                        /*! number of data points */
                        int numPoints,
                        /*! device-writeable pointer to store the world-space
                          bounding box of all data points. if
                          data_traits::has_explicit_dim is false, this is
                          optionally allowed to be null */
                        cukd::box_t<typename data_traits::point_t> *worldBounds=0,
                        /*! cuda stream to use for all kernels and mallocs
                          (the builder_thrust may _also_ do some global
                          device syncs) */
                        cudaStream_t stream=0,
                        /*! memory resource that can be used to
                          control how memory allocations will be
                          implemented (eg, using Async allocs only
                          on CDUA > 11, or using managed vs device
                          mem) */
                        GpuMemoryResource &memResource=defaultGpuMemResource());

  /*! builds tree on the host, using host read/writeable data (using
    managed memory is fine) */
  template<typename data_t,
           typename data_traits=default_data_traits<data_t>>
  void buildTree_host(data_t *d_points,
                      int numPoints,
                      cukd::box_t<typename data_traits::point_t> *worldBounds=0);
  
  // ==================================================================
  // IMPLEMENTATION SECTION
  // ==================================================================

  namespace thrustSortBuilder {
    
    template<typename data_t, typename data_traits>
    struct ZipCompare {
      explicit ZipCompare(const int dim, const data_t *nodes)
        : dim(dim), nodes(nodes)
      {}

      /*! the actual comparison operator; will perform a
        'zip'-comparison in that the first element is the major sort
        order, and the second the minor one (for those of same major
        sort key) */
      inline __both__ bool operator()
      (const thrust::tuple<uint32_t, data_t> &a,
       const thrust::tuple<uint32_t, data_t> &b);

      const int dim;
      const data_t *nodes;
    };

    template<typename data_t,typename data_traits>
    __global__
    void chooseInitialDim(cukd::box_t<typename data_traits::point_t> *d_bounds,
                          data_t *d_nodes,
                          int numPoints)
    {
      const int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numPoints) return;

      int dim = d_bounds->widestDimension();//arg_max(d_bounds->size());
      data_traits::set_dim(d_nodes[tid],dim);
    }
  
    template<typename data_t,typename data_traits>
    void host_chooseInitialDim(cukd::box_t<typename data_traits::point_t> *d_bounds,
                               data_t *d_nodes,
                               int numPoints)
    {
      for (int tid=0;tid<numPoints;tid++) {
        int dim = d_bounds->widestDimension();//arg_max(d_bounds->size());
        data_traits::set_dim(d_nodes[tid],dim);
      }
    }
  
    /* performs the L-th step's tag update: each input tag refers to a
       subtree ID on level L, and - assuming all points and tags are in
       the expected sort order described inthe paper - this kernel will
       update each of these tags to either left or right child (or root
       node) of given subtree*/
    inline __both__
    void updateTag(int gid,
                   /*! array of tags we need to update */
                   uint32_t *tag,
                   /*! num elements in the tag[] array */
                   int numPoints,
                   /*! which step we're in             */
                   int L)
    {
      // const int gid = threadIdx.x+blockIdx.x*blockDim.x;
      // if (gid >= numPoints) return;

      const int numSettled = FullBinaryTreeOf(L).numNodes();
      if (gid < numSettled) return;

      // get the subtree that the given node is in - which is exactly
      // what the tag stores...
      int subtree = tag[gid];

      // computed the expected positoin of the pivot element for the
      // given subtree when using our speific array layout.
      const int pivotPos = ArrayLayoutInStep(L,numPoints).pivotPosOf(subtree);

      if (gid < pivotPos)
        // point is to left of pivot -> must be smaller or equal to
        // pivot in given dim -> must go to left subtree
        subtree = BinaryTree::leftChildOf(subtree);
      else if (gid > pivotPos)
        // point is to left of pivot -> must be bigger or equal to pivot
        // in given dim -> must go to right subtree
        subtree = BinaryTree::rightChildOf(subtree);
      else
        // point is _on_ the pivot position -> it's the root of that
        // subtree, don't change it.
        ;
      tag[gid] = subtree;
    }

    /* performs the L-th step's tag update: each input tag refers to a
       subtree ID on level L, and - assuming all points and tags are in
       the expected sort order described inthe paper - this kernel will
       update each of these tags to either left or right child (or root
       node) of given subtree*/
    __global__
    void updateTags(/*! array of tags we need to update */
                    uint32_t *tag,
                    /*! num elements in the tag[] array */
                    int numPoints,
                    /*! which step we're in             */
                    int L)
    {
      const int gid = threadIdx.x+blockIdx.x*blockDim.x;
      if (gid >= numPoints) return;

      updateTag(gid,tag,numPoints,L);
    }
    
    /* performs the L-th step's tag update: each input tag refers to a
       subtree ID on level L, and - assuming all points and tags are in
       the expected sort order described inthe paper - this kernel will
       update each of these tags to either left or right child (or root
       node) of given subtree*/
    inline void host_updateTags(/*! array of tags we need to update */
                                uint32_t *tag,
                                /*! num elements in the tag[] array */
                                int numPoints,
                                /*! which step we're in             */
                                int L)
    {
      for (int gid=0;gid<numPoints;gid++) 
        updateTag(gid,tag,numPoints,L);
    }
    

    /* performs the L-th step's tag update: each input tag refers to a
       subtree ID on level L, and - assuming all points and tags are in
       the expected sort order described inthe paper - this kernel will
       update each of these tags to either left or right child (or root
       node) of given subtree*/
    template<typename data_t, typename data_traits>
    inline __both__
    void updateTagAndSetDim(int gid,
                            /*! array of tags we need to update */
                            const cukd::box_t<typename data_traits::point_t> *d_bounds,
                            uint32_t  *tag,
                            data_t *d_nodes,
                            /*! num elements in the tag[] array */
                            int numPoints,
                            /*! which step we're in             */
                            int L)
    {
      using point_t = typename data_traits::point_t;
      using point_traits = typename ::cukd::point_traits<point_t>;
      
      const int numSettled = FullBinaryTreeOf(L).numNodes();
      if (gid < numSettled) return;

      // get the subtree that the given node is in - which is exactly
      // what the tag stores...
      int subtree = tag[gid];
      cukd::box_t<typename data_traits::point_t> bounds
        = findBounds<data_t,data_traits>(subtree,d_bounds,d_nodes);
      // computed the expected positoin of the pivot element for the
      // given subtree when using our speific array layout.
      const int pivotPos = ArrayLayoutInStep(L,numPoints).pivotPosOf(subtree);

      const int   pivotDim   = data_traits::get_dim(d_nodes[pivotPos]);
      const float pivotCoord = data_traits::get_coord(d_nodes[pivotPos],pivotDim);
    
      if (gid < pivotPos) {
        // point is to left of pivot -> must be smaller or equal to
        // pivot in given dim -> must go to left subtree
        subtree = BinaryTree::leftChildOf(subtree);
        point_traits::set_coord(bounds.upper,pivotDim,pivotCoord);
      } else if (gid > pivotPos) {
        // point is to left of pivot -> must be bigger or equal to pivot
        // in given dim -> must go to right subtree
        subtree = BinaryTree::rightChildOf(subtree);
        point_traits::set_coord(bounds.lower,pivotDim,pivotCoord);
      } else
        // point is _on_ the pivot position -> it's the root of that
        // subtree, don't change it.
        ;
      if (gid != pivotPos)
        data_traits::set_dim(d_nodes[gid],bounds.widestDimension());
      tag[gid] = subtree;
    }
  
    /* performs the L-th step's tag update: each input tag refers to a
       subtree ID on level L, and - assuming all points and tags are in
       the expected sort order described inthe paper - this kernel will
       update each of these tags to either left or right child (or root
       node) of given subtree*/
    template<typename data_t, typename data_traits>
    __global__
    void updateTagsAndSetDims(/*! array of tags we need to update */
                              const cukd::box_t<typename data_traits::point_t> *d_bounds,
                              uint32_t  *tag,
                              data_t *d_nodes,
                              /*! num elements in the tag[] array */
                              int numPoints,
                              /*! which step we're in             */
                              int L)
    {
      const int gid = threadIdx.x+blockIdx.x*blockDim.x;
      if (gid >= numPoints) return;
      
      updateTagAndSetDim<data_t,data_traits>
        (gid,
         /*! array of tags we need to update */
         d_bounds,
         tag,
         d_nodes,
         /*! num elements in the tag[] array */
         numPoints,
         /*! which step we're in             */
         L);
    }

    /* performs the L-th step's tag update: each input tag refers to a
       subtree ID on level L, and - assuming all points and tags are in
       the expected sort order described inthe paper - this kernel will
       update each of these tags to either left or right child (or root
       node) of given subtree*/
    template<typename data_t, typename data_traits>
    void host_updateTagsAndSetDims
    (/*! array of tags we need to update */
     const cukd::box_t<typename data_traits::point_t> *d_bounds,
     uint32_t  *tag,
     data_t *d_nodes,
     /*! num elements in the tag[] array */
     int numPoints,
     /*! which step we're in             */
     int L)
    {
      for (int gid=0;gid<numPoints;gid++) 
        updateTagAndSetDim<data_t,data_traits>
          (gid,
           /*! array of tags we need to update */
           d_bounds,
           tag,
           d_nodes,
           /*! num elements in the tag[] array */
           numPoints,
           /*! which step we're in             */
           L);
    }
    
    /*! the actual comparison operator; will perform a
      'zip'-comparison in that the first element is the major sort
      order, and the second the minor one (for those of same major
      sort key) */
    template<typename data_t, typename data_traits>
    inline __both__
    bool ZipCompare<data_t,data_traits>::operator()
      (const thrust::tuple<uint32_t, data_t> &a,
       const thrust::tuple<uint32_t, data_t> &b)
    {
      using point_t = typename data_traits::point_t;
      using point_traits = ::cukd::point_traits<point_t>;
      
      const auto tag_a = thrust::get<0>(a);
      const auto tag_b = thrust::get<0>(b);
      const auto pnt_a = thrust::get<1>(a);
      const auto pnt_b = thrust::get<1>(b);
      int dim
        = data_traits::has_explicit_dim
        ? data_traits::get_dim(pnt_a)
        : this->dim;
      const auto coord_a = data_traits::get_coord(pnt_a,dim);
      const auto coord_b = data_traits::get_coord(pnt_b,dim);
      const bool less =
        (tag_a < tag_b)
        ||
        ((tag_a == tag_b) && (coord_a < coord_b));

      return less;
    }

  } // ::cukd::thrustSortBuilder




  template<typename data_t, typename data_traits>
  void buildTree_thrust(data_t *d_points,
                        int numPoints,
                        box_t<typename data_traits::point_t> *worldBounds,
                        cudaStream_t stream,
                        /*! memory resource that can be used to
                          control how memory allocations will be
                          implemented (eg, using Async allocs only
                          on CDUA > 11, or using managed vs device
                          mem) */
                        GpuMemoryResource &memResource)
  {
    using namespace thrustSortBuilder;

    using point_t  = typename data_traits::point_t;
    using point_traits = ::cukd::point_traits<point_t>;
    using scalar_t = typename point_traits::scalar_t;
    enum { num_dims = point_traits::num_dims };
    
    /* thrust helper typedefs for the zip iterator, to make the code
       below more readable */
    typedef typename thrust::device_vector<uint32_t>::iterator tag_iterator;
    typedef typename thrust::device_vector<data_t>::iterator point_iterator;
    typedef thrust::tuple<tag_iterator,point_iterator> iterator_tuple;
    typedef thrust::zip_iterator<iterator_tuple> tag_point_iterator;

    // check for invalid input, and return gracefully if so
    if (numPoints < 1) return;

    /* the helper array  we use to store each node's subtree ID in */
    // TODO allocate in stream?
    thrust::device_vector<uint32_t> tags(numPoints);
    /* to kick off the build, every element is in the only
       level-0 subtree there is, namely subtree number 0... duh */
    thrust::fill(thrust::device.on(stream),tags.begin(),tags.end(),0);

    /* create the zip iterators we use for zip-sorting the tag and
       points array */
    thrust::device_ptr<data_t> points_begin(d_points);
    thrust::device_ptr<data_t> points_end(d_points+numPoints);
    tag_point_iterator begin = thrust::make_zip_iterator
      (thrust::make_tuple(tags.begin(),points_begin));
    tag_point_iterator end = thrust::make_zip_iterator
      (thrust::make_tuple(tags.end(),points_end));

    /* compute number of levels in the tree, which dicates how many
       construction steps we need to run */
    const int numLevels = BinaryTree::numLevelsFor(numPoints);
    const int deepestLevel = numLevels-1;
    
    using box_t = cukd::box_t<typename data_traits::point_t>;
    if (worldBounds) {
      computeBounds<data_t,data_traits>
        (worldBounds,d_points,numPoints,stream);
    }
    if (data_traits::has_explicit_dim) {
      if (!worldBounds)
        throw std::runtime_error
          ("cukd::builder_thrust: asked to build k-d tree over nodes"
           " with explicit dims, but no memory for world bounds provided");
      
      const int blockSize = 128;
      chooseInitialDim<data_t,data_traits>
        <<<divRoundUp(numPoints,blockSize),blockSize,0,stream>>>
        (worldBounds,d_points,numPoints);
      cudaStreamSynchronize(stream);
    }
    
    
    /* now build each level, one after another, cycling through the
       dimensoins */
    for (int level=0;level<deepestLevel;level++) {
      thrust::sort(thrust::device.on(stream),begin,end,
                   ZipCompare<data_t,data_traits>
                   ((level)%num_dims,d_points));

      const int blockSize = 128;
      if (data_traits::has_explicit_dim) {
        updateTagsAndSetDims<data_t,data_traits>
          <<<divRoundUp(numPoints,blockSize),blockSize,0,stream>>>
          (worldBounds,thrust::raw_pointer_cast(tags.data()),
           d_points,numPoints,level);
      } else {
        updateTags
          <<<divRoundUp(numPoints,blockSize),blockSize,0,stream>>>
          (thrust::raw_pointer_cast(tags.data()),numPoints,level);
      }
      cudaStreamSynchronize(stream);
    }
    
    /* do one final sort, to put all elements in order - by now every
       element has its final (and unique) nodeID stored in the tag[]
       array, so the dimension we're sorting in really won't matter
       any more */
    thrust::sort(thrust::device.on(stream),begin,end,
                 ZipCompare<data_t,data_traits>
                 ((deepestLevel)%num_dims,d_points));
    cudaStreamSynchronize(stream);
  }
      
  template<typename data_t, typename data_traits>
  void buildTree_host(data_t *d_points,
                      int numPoints,
                      cukd::box_t<typename data_traits::point_t> *worldBounds)
  {
    using namespace thrustSortBuilder;

    using point_t      = typename data_traits::point_t;
    using point_traits = ::cukd::point_traits<point_t>;
    using scalar_t    = typename point_traits::scalar_t;
    enum { num_dims   = point_traits::num_dims };
    
    /* thrust helper typedefs for the zip iterator, to make the code
       below more readable */
#if 1
    typedef uint32_t *tag_iterator;
    typedef data_t   *point_iterator;
#else
    typedef typename thrust::device_vector<uint32_t>::iterator tag_iterator;
    typedef typename thrust::device_vector<data_t>::iterator point_iterator;
#endif
    typedef thrust::tuple<tag_iterator,point_iterator> iterator_tuple;
    typedef thrust::zip_iterator<iterator_tuple> tag_point_iterator;

    // check for invalid input, and return gracefully if so
    if (numPoints < 1) return;

    /* the helper array  we use to store each node's subtree ID in */
    // TODO allocate in stream?
    std::vector<uint32_t> tags(numPoints);
    /* to kick off the build, every element is in the only
       level-0 subtree there is, namely subtree number 0... duh */
    thrust::fill(thrust::host,tags.begin(),tags.end(),0);

    /* create the zip iterators we use for zip-sorting the tag and
       points array */
    thrust::device_ptr<data_t> points_begin(d_points);
    thrust::device_ptr<data_t> points_end(d_points+numPoints);
    tag_point_iterator begin = thrust::make_zip_iterator
      (thrust::make_tuple(tags.data(),d_points));
    tag_point_iterator end = thrust::make_zip_iterator
      (thrust::make_tuple(tags.data()+numPoints,d_points+numPoints));

    /* compute number of levels in the tree, which dicates how many
       construction steps we need to run */
    const int numLevels = BinaryTree::numLevelsFor(numPoints);
    const int deepestLevel = numLevels-1;
    
    using box_t = cukd::box_t<point_t>;
    if (worldBounds) {
      host_computeBounds<data_t,data_traits>
        (worldBounds,d_points,numPoints);
    }
    if (data_traits::has_explicit_dim) {
      if (!worldBounds)
        throw std::runtime_error
          ("cukd::builder_host: asked to build k-d tree over nodes"
           " with explicit dims, but no memory for world bounds provided");
      
      host_chooseInitialDim<data_t,data_traits>
        (worldBounds,d_points,numPoints);
    }
    
    /* now build each level, one after another, cycling through the
       dimensoins */
    for (int level=0;level<deepestLevel;level++) {
      thrust::sort(thrust::host,begin,end,
                   ZipCompare<data_t,data_traits>
                   ((level)%num_dims,d_points));
      
      if (data_traits::has_explicit_dim) {
        host_updateTagsAndSetDims<data_t,data_traits>
          (worldBounds,thrust::raw_pointer_cast(tags.data()),
           d_points,numPoints,level);
      } else {
        host_updateTags
          (thrust::raw_pointer_cast(tags.data()),numPoints,level);
      }
    }
    
    /* do one final sort, to put all elements in order - by now every
       element has its final (and unique) nodeID stored in the tag[]
       array, so the dimension we're sorting in really won't matter
       any more */
    thrust::sort(thrust::host,begin,end,
                 ZipCompare<data_t,data_traits>
                 ((deepestLevel)%num_dims,d_points)); 
  }

} // ::cukd
