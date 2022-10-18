// ======================================================================== //
// Copyright 2019-2021 Ingo Wald                                            //
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

#include "helpers.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <device_launch_parameters.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

namespace cukd {

  typedef uint32_t tag_t;
  
  // ==================================================================
  // INTERFACE SECTION
  // ==================================================================

  /*! Trivial implementation of the point interface for those kinds of
    point types where the first K elements are the K-dimensional
    coordinates that we buid the K-d tree over; the point_t struct
    may contain additional data at the end, too (ie, you can build,
    for exapmle, a 2-d tree over a float4 point type - in this case
    the x and y coordinates are the point coordinates, and z and w
    are any other payload that does not get considered during the
    (2-d) construction) */
  template<typename _point_t, typename scalar_t>
  struct TrivialPointInterface
  {
    typedef _point_t point_t;
    inline static __device__
    scalar_t get(const point_t &p, int dim) { return ((scalar_t*)&p)[dim]; }
  };
  
  /*! builds a regular, "round-robin" style k-d tree over the given
    (device-side) array of points. Round-robin in this context means
    that the first dimension is sorted along x, the second along y,
    etc, going through all dimensions x->y->z... in round-robin
    fashion. point_t can be any arbitrary struct, and is assumed to
    have at least 'numDims' coordinates of type 'scalar_t', plus
    whatever other payload data is desired. To customize where or
    how the point_t stores its numDims coordinates the 'GetElement'
    class can be used by providing a class that implements a static
    scalar_t=GetElement::get(point_t,dim) method. The default
    TrivialPointInterface simply assumes that whatever type point_t
    is used, its nuMDims coordinates are stored at the beginning of
    this class, in array order. 

    Example 1: To build a 2D k-dtree over a CUDA int2 type (no other
    payload than the two coordinates):

    buildKDTree<int2,int>(....);

    Example 2: to build a 1D kd-tree over a data type of float4,
    where the first coordinate of each point is the dimension we
    want to build the kd-tree over, and the other three coordinate
    are arbitrary other payload data:

    buildKDTree<float4,float,1>(...);
  */
  template<typename point_t,
           typename scalar_t,
           int      numDims=sizeof(point_t)/sizeof(scalar_t),
           typename GetElement = TrivialPointInterface<point_t,scalar_t>>
  void buildTree(point_t *d_points, int numPoints);

  // ==================================================================
  // IMPLEMENTATION SECTION
  // ==================================================================

  template<typename PointInterface>
  struct ZipCompare {
    ZipCompare(const int dim) : dim(dim) {}

    /*! the actual comparison operator; will perform a
      'zip'-comparison in that the first element is the major sort
      order, and the second the minor one (for those of same major
      sort key) */
    inline __device__ bool operator()
    (const thrust::tuple<tag_t, typename PointInterface::point_t> &a,
     const thrust::tuple<tag_t, typename PointInterface::point_t> &b);
    
    const int dim;
  };

  /* performs the L-th step's tag update: each input tag refers to a
     subtree ID on level L, and - assuming all points and tags are in
     the expected sort order described inthe paper - this kernel will
     update each of these tags to either left or right child (or root
     node) of given subtree*/
  __global__
  /*inline*/ void updateTag(/*! array of tags we need to update */
                        tag_t *tag,
                        /*! num elements in the tag[] array */
                        int numPoints,
                        /*! which step we're in             */
                        int L)
  {
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if (gid >= numPoints) return;
    
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


#if KDTREE_BUILDER_LOGGING
  void print(const char *txt,
             int step,
             int numPoints,
             tag_t *d_tags,
             int1 *d_points)
  {
    std::vector<int> points(numPoints), tags(numPoints);
    cudaMemcpy(points.data(),d_points,numPoints*sizeof(int),cudaMemcpyDefault);
    cudaMemcpy(tags.data(),d_tags,numPoints*sizeof(int),cudaMemcpyDefault);
    printf("-----------\n");
    printf(txt,step);
    printf("arry:");
    for (int i=0;i<numPoints;i++)
      printf("%6i",i);
    printf("\n");
    printf("tags:");
    for (int i=0;i<numPoints;i++)
      printf("%6i",tags[i]);
    printf("\n");
    printf("pnts:");
    for (int i=0;i<numPoints;i++)
      printf("%6i",points[i]);
    printf("\n");
  }
#endif
  
  template<typename point_t,
           typename scalar_t,
           int      numDims,
           typename GetElement>
  void buildTree(point_t *d_points,
                 int numPoints)
  {
    /* thrust helper typedefs for the zip iterator, to make the code
       below more readable */
    typedef typename thrust::device_vector<tag_t>::iterator tag_iterator;
    typedef typename thrust::device_vector<point_t>::iterator point_iterator;
    typedef thrust::tuple<tag_iterator,point_iterator> iterator_tuple;
    typedef thrust::zip_iterator<iterator_tuple> tag_point_iterator;

    // check for invalid input, and return gracefully if so
    if (numPoints < 1) return;
    
    /* the helper array  we use to store each node's subtree ID in */
    thrust::device_vector<tag_t> tags(numPoints);
    /* to kick off the build, every element is in the only
       level-0 subtree there is, namely subtree number 0... duh */
    thrust::fill(tags.begin(),tags.end(),0);

    /* create the zip iterators we use for zip-sorting the tag and
       points array */
    thrust::device_ptr<point_t> points_begin(d_points);
    thrust::device_ptr<point_t> points_end(d_points+numPoints);
    tag_point_iterator begin = thrust::make_zip_iterator
      (thrust::make_tuple(tags.begin(),points_begin));
    tag_point_iterator end = thrust::make_zip_iterator
      (thrust::make_tuple(tags.end(),points_end));

    /* compute number of levels in the tree, which dicates how many
       construction steps we need to run */
    const int numLevels = BinaryTree::numLevelsFor(numPoints);
    const int deepestLevel = numLevels-1;

#if KDTREE_BUILDER_LOGGING
    print("init\n",-1,numPoints,thrust::raw_pointer_cast(tags.data()),d_points);
#endif
  
    /* now build each level, one after another, cycling through the
       dimensoins */
    for (int level=0;level<deepestLevel;level++) {
      thrust::sort(thrust::device,begin,end,
                   ZipCompare<GetElement>((level)%numDims));
      
#if KDTREE_BUILDER_LOGGING
    print("step %i sort\n",level,numPoints,thrust::raw_pointer_cast(tags.data()),d_points);
#endif
      const int blockSize = 32;
      const int numSettled = FullBinaryTreeOf(level).numNodes();
      updateTag<<<common::divRoundUp(numPoints,blockSize),blockSize>>>
        (thrust::raw_pointer_cast(tags.data()),numPoints,level);
      
#if KDTREE_BUILDER_LOGGING
    print("step %i tags updated\n",level,numPoints,thrust::raw_pointer_cast(tags.data()),d_points);
#endif
    }
    /* do one final sort, to put all elements in order - by now every
       element has its final (and unique) nodeID stored in the tag[]
       array, so the dimension we're sorting in really won't matter
       any more */
    thrust::sort(thrust::device,begin,end,
                 ZipCompare<GetElement>((deepestLevel)%numDims));
#if KDTREE_BUILDER_LOGGING
    print("final sort\n",-1,numPoints,thrust::raw_pointer_cast(tags.data()),d_points);
#endif
  }
  
  /*! the actual comparison operator; will perform a
    'zip'-comparison in that the first element is the major sort
    order, and the second the minor one (for those of same major
    sort key) */
  template<typename PointInterface>
  inline __device__
  bool ZipCompare<PointInterface>::operator()
    (const thrust::tuple<tag_t, typename PointInterface::point_t> &a,
     const thrust::tuple<tag_t, typename PointInterface::point_t> &b)
  {
    auto tag_a = thrust::get<0>(a);
    auto tag_b = thrust::get<0>(b);
    auto pnt_a = thrust::get<1>(a);
    auto pnt_b = thrust::get<1>(b);
    auto dim_a = PointInterface::get(pnt_a,dim);
    auto dim_b = PointInterface::get(pnt_b,dim);
    bool less =
      (tag_a < tag_b)
      ||
      ((tag_a == tag_b) && (dim_a < dim_b));

    return less;
  }
}

