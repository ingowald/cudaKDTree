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
#include <cuda.h>
#include <device_launch_parameters.h>

namespace cukd {
  
  /*! defines an abstract interface to what a 'node' in a k-d tree
    is. This needs to define the follwing:

    - node_traits::scalar_t: the scalar type of each point member (eg,
    float for a float3 node_t)

    - enum node_traits::num_dims: the number of dimensions of the
    data; e.g., a k-d tree build over float4 4d points would define
    tihs to '4'; a kd tree built over a struct htat has 3d position
    and some other additional payload would use '3'.

    - scalar_t node_traits::get(node_t &, int d) : return a
    reference to the 'd'th positional coordinate of the given node

    - enum node_traits::has_explicit_dim : whether that node type
    has a field to store an explicit split dimensoin in each
    node. If not, the k-d tree builder and traverse _have_ to use
    round-robin for split distance; otherwise, it will alwyas
    split the widest dimension

    - enum node_traits::set_dim(node_t &, int) and
    node_traits::get_dim(node_t &
  */
  template<typename node_t> struct default_node_traits {

    // ------------------------------------------------------------------
    /* part I : describes the _types_ of node of the tree, position,
       scalar, dimnensionaltiy, etc */
    // ------------------------------------------------------------------

    /*! the *logical* type used for mathematical things like distance
      computations, specifiing the location of a data point,
      etc. this defines number of dimensions, scalar type, etc, but
      leaves the node to define its own data layout */
    using point_t = node_t;

    // ------------------------------------------------------------------
    /* part II : how to extract a point or coordinate from an actual
       data struct */
    // ------------------------------------------------------------------

    /*! return a reference to the 'd'th positional coordinate of the
      given node */
    static inline __both__ const point_t &get_point(const node_t &n) { return n; }
    
    /*! return a reference to the 'd'th positional coordinate of the
      given node */
    static inline __both__
    typename scalar_type_of<point_t>::type get_coord(const node_t &n, int d)
    { return cukd::get_coord(get_point(n),d); }
    
    // ------------------------------------------------------------------
    /* part III : whether the data struct has a way of storing a split
       dimension for non-round robin paritioning, and if so, how to
       store (for building) and read (for traversing) that split
       dimensional in/from a node */
    // ------------------------------------------------------------------

    /* whether that node type has a field to store an explicit split
       dimensoin in each node. If not, the k-d tree builder and
       traverse _have_ to use round-robin for split distance;
       otherwise, it will alwyas split the widest dimensoin */
    enum { has_explicit_dim = false };
    
    /*! !{ just defining this for completeness, get/set_dim should never
      get called for this type becaues we have set has_explicit_dim
      set to false. note traversal should ONLY ever call this
      function for node_t's that define has_explicit_dim to true */
    static inline __device__ int  get_dim(const node_t &) { return -1; }
    static inline __device__ void set_dim(node_t &, int) {}
    /*! @} */
  };


  /*! defines default node traits for our own vec_float<N> vector type */
  template<int N> struct default_node_traits<vec_float<N>> {
    using node_t   = vec_float<N>;
    using scalar_t = float;
    using point_t  = node_t;
    
    enum { has_explicit_dim = false };
    
    static inline __both__ const point_t &get_point(const node_t &n) { return n; }
    static inline __both__ scalar_t get_coord(const node_t &n, int d) { return n.v[d]; }
    static inline __both__ int  get_dim(const node_t &n) { return -1; }
    static inline __both__ void set_dim(node_t &n, int dim) {}
  };
  

  
  
  // ==================================================================
  // INTERFACE SECTION
  // ==================================================================

  /*! builds a regular, "round-robin" style k-d tree over the given
    (device-side) array of points. Round-robin in this context means
    that the first dimension is sorted along x, the second along y,
    etc, going through all dimensions x->y->z... in round-robin
    fashion. point_t can be any arbitrary struct, and is assumed to
    have at least 'numDims' coordinates of type 'scalar_t', plus
    whatever other payload data is desired.

    Example 1: To build a 2D k-dtree over a CUDA int2 type (no other
    payload than the two coordinates):

    buildKDTree<int2>(....);

    Example 2: to build a 1D kd-tree over a data type of float4,
    where the first coordinate of each point is the dimension we
    want to build the kd-tree over, and the other three coordinate
    are arbitrary other payload data:

    buildKDTree<float4>(...);
  */
  template<typename node_t,
           typename node_traits=default_node_traits<node_t>>
  void buildTree(node_t *d_points,
                 int numPoints,
                 cudaStream_t stream = 0);

  /*! build a k-d over given set of points, but can build both
      round-robin-style and "generalized" k-d trees where the split
      dimension for each subtree is chosen based on the dimension
      where that subtree's domain is widest. If the
      node_traits::has_explicit_dim field is true, the latter type of
      k-d tree is build; if it is false, this function build a regular
      round-robin k-d tree instead
 */
  template<typename node_t,
           typename node_traits=default_node_traits<node_t>>
  void buildTree(node_t      *points,
                 int          numPoints,
                 box_t<typename node_traits::point_t> *worldBounds,
                 cudaStream_t stream = 0);

  /*! helper function that computes the bounding box of a given set of
      points */
  template<typename node_t, 
  typename node_traits=default_node_traits<node_t>>
  void computeBounds(cukd::box_t<typename node_traits::point_t> *d_bounds,
                     const node_t *d_points,
                     int numPoints,
                     cudaStream_t stream=0);

  // ==================================================================
  // IMPLEMENTATION SECTION
  // ==================================================================

  template<typename node_t, typename node_traits>
  __global__
  void computeBounds_copyFirst(box_t<typename node_traits::point_t> *d_bounds,
                               const node_t *d_points)
  {
    if (threadIdx.x != 0) return;
    
    using point_t = typename node_traits::point_t;
    const point_t point = node_traits::get_point(d_points[0]);
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

  template<typename node_t,
           typename node_traits>
  __global__
  void computeBounds_atomicGrow(box_t<typename node_traits::point_t> *d_bounds,
                                const node_t *d_points,
                                int numPoints)
  {
    using point_t = typename node_traits::point_t;
    enum { num_dims = num_dims_of<point_t>::value };
    
    const int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numPoints) return;
    
    using point_t = typename node_traits::point_t;
    point_t point = node_traits::get_point(d_points[tid]);
#pragma unroll(num_dims)
    for (int d=0;d<num_dims;d++) {
      float &lo = get_coord(d_bounds->lower,d);
      float &hi = get_coord(d_bounds->upper,d);
      float f = get_coord(point,d);
      atomicMin(&lo,f);
      atomicMax(&hi,f);
    }
  }

  template<typename node_t, typename node_traits>
  void computeBounds(box_t<typename node_traits::point_t> *d_bounds,
                     const node_t *d_points,
                     int numPoints,
                     cudaStream_t s)
  {
    computeBounds_copyFirst<node_t,node_traits>
      <<<1,1,0,s>>>
      (d_bounds,d_points);
    computeBounds_atomicGrow<node_t,node_traits>
      <<<divRoundUp(numPoints,128),128,0,s>>>
      (d_bounds,d_points,numPoints);
  }


  template<typename node_t,typename node_traits>
  inline __device__
  box_t<typename node_traits::point_t>
  findBounds(int subtree,
             const box_t<typename node_traits::point_t> *d_bounds,
             node_t *d_nodes)
  {
    using point_t  = typename node_traits::point_t;
    using scalar_t = typename scalar_type_of<point_t>::type;
    enum { num_dims = num_dims_of<point_t>::value };
    
    box_t<point_t> bounds = *d_bounds;
    int curr = subtree;
    // const bool dbg = false;
    while (curr > 0) {
      const int     parent = (curr+1)/2-1;
      const node_t &parent_node = d_nodes[parent];
      const int     parent_dim
        = node_traits::has_explicit_dim
        ? node_traits::get_dim(parent_node)
        : (BinaryTree::levelOf(parent) % num_dims);
      const scalar_t parent_split_pos
        = node_traits::get_coord(parent_node,parent_dim);
      
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
  
  inline __both__ int firstNodeOnLevel(int L) { return (1<<L) - 1; }
  inline __both__ int numNodesOnLevel(int L) { return 1<<L; }
  inline __both__ int partnerOf(int n, int L_r, int L_b)
  {
    return (((n+1) ^ (1<<(L_r-L_b-1))))-1;
  }

  template<typename T>
  inline __both__ void swap(T &a, T &b)
  { T c = a; a = b; b = c; }


  template<typename scalar_t, int side>
  inline __device__
  bool desiredOrder(scalar_t a, scalar_t b)
  {
    if (side) {
      return !(b < a);
    } else {
      return !(a < b); 
    }
  }

  template<typename node_t, typename node_traits, int side>
  inline __device__
  void trickleDownHeap(int n,
                       node_t *__restrict__ points,
                       int numPoints,
                       int dim)
  {
    const int input_n = n;
    using point_t  = typename node_traits::point_t;
    using scalar_t = typename scalar_type_of<point_t>::type;
    node_t point_n = points[n];
    scalar_t s_n = node_traits::get_coord(point_n,dim);
    while (true) {
      int l = 2*n+1;
      if (l >= numPoints)
        break;
      scalar_t s_l = node_traits::get_coord(points[l],dim);

      int c = l;
      scalar_t s_c = s_l;
      
      int r = l+1;
      if (r < numPoints) {
        scalar_t s_r = node_traits::get_coord(points[r],dim);
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


  template<typename node_t, typename node_traits>
  __global__ void d_quickSwap(/*! _build_ root level */int L_b,
                              node_t *points,
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

    using point_t  = typename node_traits::point_t;
    enum { num_dims = num_dims_of<point_t>::value };

    const int     dim
      = node_traits::has_explicit_dim
      ? node_traits::get_dim(points[((n+1)>>(L_n-L_b))-1])
      : (L_b % num_dims);
    
    if (node_traits::get_coord(points[partner],dim)
        <
        node_traits::get_coord(points[n],dim)) {
      swap(points[n],points[partner]);
    } 
  }
  
  template<typename node_t, typename node_traits>
  void quickSwap(/*! _build_ root level */
                 int          L_b,
                 node_t      *points,
                 int          numPoints,
                 cudaStream_t stream)
  {
    // printTree<node_t,node_traits>(points,numPoints);
    // std::cout << "---- building heaps on " << L_h << ", root level " << L_b << std::endl << std::flush;
    int bs = 1024;
    int nb = divRoundUp(numPoints,bs);
    d_quickSwap<node_t,node_traits><<<nb,bs,0,stream>>>(L_b,points,numPoints);
  }

  template<typename node_t, typename node_traits>
  void printTree(node_t *points,int numPoints)
  {
    cudaDeviceSynchronize();
    using point_t  = typename node_traits::point_t;
    using scalar_t = typename scalar_type_of<point_t>::type;
    enum { num_dims = num_dims_of<point_t>::value };
    
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
          printf("%5.3f ",(node_traits::get_coord(points[i],d)));
        // printf("%6i",int(node_traits::get_coord(points[i],d)));
        printf("\n");
      }
    }
  }
  

  template<typename node_t, typename node_traits>
  __global__ void d_buildHeaps(/*! _heap_ root level */int L_h,
                               /*! _build_ root level */int L_b,
                               node_t *__restrict__ points,
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

    using point_t  = typename node_traits::point_t;
    enum { num_dims = num_dims_of<point_t>::value };

    const int     dim
      = node_traits::has_explicit_dim
      ? node_traits::get_dim(points[((n+1)>>(L_h-L_b))-1])
      : (L_b % num_dims);

    if (node_traits::get_coord(points[partner],dim)
        <
        node_traits::get_coord(points[n],dim)) {
      swap(points[n],points[partner]);
    } 
    while (1) {
      trickleDownHeap<node_t,node_traits,0>(n,points,numPoints,dim);
      trickleDownHeap<node_t,node_traits,1>(partner,points,numPoints,dim);
      if (node_traits::get_coord(points[partner],dim)
          <
          node_traits::get_coord(points[n],dim)) {
        swap(points[n],points[partner]);
        continue;
      } else
        break;
    }
  }


  template<typename node_t, typename node_traits>
  void buildHeaps(/*! _heap_ root level */
                  int          L_h,
                  /*! _build_ root level */
                  int          L_b,
                  node_t      *points,
                  int          numPoints,
                  cudaStream_t stream)
  {
    int numNodesOnL_h = numNodesOnLevel(L_h);
    int bs = 64;
    int nb = divRoundUp(numNodesOnL_h,bs);
    d_buildHeaps<node_t,node_traits><<<nb,bs,0,stream>>>
      (L_h,L_b,points,numPoints);
  }



  template<typename node_t, typename node_traits>
  __global__
  void d_selectDimsOnLevel(int     L_b,
                           node_t *points,
                           int     numPoints,
                           box_t<typename node_traits::point_t> *worldBounds)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int numNodesOnL_b = numNodesOnLevel(L_b);
    if (tid >= numNodesOnL_b)
      return;

    int n = firstNodeOnLevel(L_b)+tid;
    if (n >= numPoints) return;
                                           
    using point_t  = typename node_traits::point_t;
    enum { num_dims = num_dims_of<point_t>::value };

    if (worldBounds) {
      box_t<typename node_traits::point_t> bounds
        = findBounds<node_t,node_traits>(n,worldBounds,points);
      node_traits::set_dim(points[n],arg_max(bounds.size()));
    } else {
      node_traits::set_dim(points[n],L_b % num_dims);
    }
  }

  template<typename node_t, typename node_traits>
  void selectDimsOnLevel(int          L_b,
                         node_t      *points,
                         int          numPoints,
                         box_t<typename node_traits::point_t> *worldBounds,
                         cudaStream_t stream)
  {
    // std::cout << "selecting dims ..." << std::endl << std::flush;
    int numNodesOnL_b = numNodesOnLevel(L_b);
    int bs = 64;
    int nb = divRoundUp(numNodesOnL_b,bs);
    d_selectDimsOnLevel<node_t,node_traits><<<nb,bs,0,stream>>>
      (L_b,points,numPoints,worldBounds);
  }
  


  template<typename node_t, typename node_traits>
  __global__
  void d_fixPivots(/*! _build_ root level */
                   int     L_b,
                   node_t *points,
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

    using point_t  = typename node_traits::point_t;
    using scalar_t = typename scalar_type_of<point_t>::type;
    enum { num_dims = num_dims_of<point_t>::value };
    
    const int  dim
      = node_traits::has_explicit_dim
      ? node_traits::get_dim(points[n])
      : (L_b % num_dims);
    
    scalar_t s_n = node_traits::get_coord(points[n],dim);
    if (l < numPoints && s_n < node_traits::get_coord(points[l],dim)) {
      swap(points[n],points[l]);
      // todo: trckle?
    } else if  (r < numPoints && node_traits::get_coord(points[r],dim) < s_n) {
      swap(points[n],points[r]);
      // todo: trckle?
    }
    if (node_traits::has_explicit_dim) 
      node_traits::set_dim(points[n],dim);
  }
  
  template<typename node_t, typename node_traits>
  void fixPivots(/*! _build_ root level */int L_b,
                 node_t *points,
                 int numPoints,
                 cudaStream_t stream)
  {
    int numNodesOnL_b = numNodesOnLevel(L_b);
    int bs = 64;
    int nb = divRoundUp(numNodesOnL_b,bs);
    d_fixPivots<node_t,node_traits><<<nb,bs,0,stream>>>(L_b,points,numPoints);
  }

  template<typename node_t, typename node_traits>
  void buildLevel(/*! level that we're ultimately _building_ */
                  int          L_b,
                  int          numLevels,
                  node_t      *d_points,
                  int          numPoints,
                  box_t<typename node_traits::point_t> *worldBounds,
                  cudaStream_t stream)
  {
    if (node_traits::has_explicit_dim)
      selectDimsOnLevel<node_t,node_traits>
        (L_b,d_points,numPoints,worldBounds,stream);
    
    for (int L_h = numLevels-1; L_h > L_b; --L_h)
      buildHeaps<node_t,node_traits>(L_h,L_b,d_points,numPoints,stream);

    fixPivots<node_t,node_traits>(L_b,d_points,numPoints,stream);
  }
  
  template<typename node_t, typename node_traits>
  void buildTree(node_t      *points,
                 int          numPoints,
                 box_t<typename node_traits::point_t> *worldBounds,
                 cudaStream_t stream)
  {
    if (worldBounds) 
      computeBounds<node_t,node_traits>(worldBounds,points,numPoints,stream);
    
    int numLevels = BinaryTree::numLevelsFor(numPoints);
    for (int L_b = 0; L_b < numLevels; L_b++)
      buildLevel<node_t,node_traits>(L_b,numLevels,points,numPoints,worldBounds,stream);
    
    cudaStreamSynchronize(stream);
  }
  
  /*! non-generalized direction tree build */
  template<typename node_t, typename node_traits>
  void buildTree(node_t *points,
                 int numPoints,
                 cudaStream_t stream)
  {
    buildTree<node_t,node_traits>(points,numPoints,nullptr,stream);
  }
  
}
