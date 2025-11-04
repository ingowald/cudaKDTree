// ======================================================================== //
// Copyright 2018-2024 Ingo Wald                                            //
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

#include "builder.h"
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <limits.h>
#include <float.h>

namespace cukd {

  // Add near the beginning of your namespace
  inline __host__ __device__ int32_t float_as_int(float f)
  {
  #ifdef __CUDA_ARCH__
      // Use CUDA intrinsic in device code
      return __float_as_int(f);
  #else
      // Host version using type punning through union
      union {
          float f;
          int32_t i;
      } converter;
      converter.f = f;
      return converter.i;
  #endif
  }

  inline __host__ __device__ float int_as_float(int32_t i)
  {
  #ifdef __CUDA_ARCH__
      // Use CUDA intrinsic in device code
      return __int_as_float(i);
  #else
      // Host version using type punning through union
      union {
          int32_t i;
          float f;
      } converter;
      converter.i = i;
      return converter.f;
  #endif
  }  

  /*! A _spatial_ kd-tree that stores actual (axis-aligned) split
    planes, and leaves of primitives. This needs somewhat more memory
    than the other k-d tree variants because it does need to store
    arrays of explicit planes and primitives IDs (regular balanced
    k-tree, in contrast, only re-order points), but is often
    faster. Also unlike the non-spatial k-d trees this will _not_
    modifiy the points[] array */
  template<typename data_t,
           typename data_traits=default_data_traits<data_t>>
  struct SpatialKDTree {
    using point_t  = typename data_traits::point_t;
    using scalar_t = typename scalar_type_of<point_t>::type;
    using box_t = cukd::box_t<point_t>;

    struct Node {
      /*! split position - which coordinate the plane is at in chosen dim */
      scalar_t pos;
      
      /*! ID of first child node (if inner node), or offset into
        primIDs[] array, if leaf */
      uint32_t offset;

      /*! number of prims in the leaf (if > 0) or 0 (if inner node) */
      uint16_t count;

      /*! split dimension - which dimension the plane is
        subdividing, if inner node */
      int16_t  dim;
    };

    box_t     bounds;
    Node     *nodes;
    uint32_t *primIDs;
    const data_t *data;
    int       numPrims;
    int       numNodes;
  };

  struct BuildConfig {
    /*! threshold below which the builder should make a leaf, no
      matter what the prims in the subtree look like. A value of 0
      means "leave it to the builder" */
    int makeLeafThreshold = 0;
  };

  /*! builds a _spatial_ kd-tree (ie, one that allocates and stores
      explicit planes and leaves, not the kind of balanced k-tree that
      only re-orders points). Unlike a balanced k-d tree this variant
      does have to allocate gpu memory for nodes and primID lists, and
      thus the user _has_ to 'free()' this tree after use. (Also, it's
      memory usage will obviously be higher!) */
  template<typename data_t,
           typename data_traits=default_data_traits<data_t>>
  void buildTree(SpatialKDTree<data_t,data_traits> &tree,
                 data_t *d_points,
                 int numPrims,
                 BuildConfig buildConfig = {},
                 cudaStream_t stream = 0,
                 GpuMemoryResource &memResource=defaultGpuMemResource());

  template<typename data_t,
           typename data_traits=default_data_traits<data_t>>
  void free(SpatialKDTree<data_t,data_traits> &tree,
            cudaStream_t stream = 0,
            GpuMemoryResource &memResource=defaultGpuMemResource());
  

  // ==================================================================
  // IMPLEMENTATION
  // ==================================================================

  namespace spatial {

    template<typename point_t>
    struct AtomicBox {
      using point_traits = ::cukd::point_traits<point_t>;
      
      enum { num_dims = point_traits::num_dims };
      
      inline __host__ __device__ void set_empty();
      inline __host__ __device__ float get_center(int dim) const;
      inline __host__ __device__ box_t<point_t> make_box() const;

      inline __host__ __device__ float get_lower(int dim) const { return decode(lower[dim]); }
      inline __host__ __device__ float get_upper(int dim) const { return decode(upper[dim]); }

      int32_t lower[num_dims];
      int32_t upper[num_dims];

      inline static __host__ __device__ int32_t encode(float f);
      inline static __host__ __device__ float   decode(int32_t bits);
    };
    
    template<typename point_t>
    inline __host__ __device__ float AtomicBox<point_t>::get_center(int dim) const
    {
      return 0.5f*(decode(lower[dim])+decode(upper[dim]));
    }

    template<typename point_t>
    inline __host__ __device__ box_t<point_t> AtomicBox<point_t>::make_box() const
    {
      box_t<point_t> box;
#pragma unroll
      for (int d=0;d<num_dims;d++) {
        point_traits::set_coord(box.lower,d,decode(lower[d]));
        point_traits::set_coord(box.upper,d,decode(upper[d]));
      }
      return box;
    }
    
    template<typename point_t>
    inline __host__ __device__ int32_t AtomicBox<point_t>::encode(float f)
    {
      const int32_t sign = 0x80000000;
      int32_t bits = float_as_int(f);
      if (bits & sign) bits ^= 0x7fffffff;
      return bits;
    }
      
    template<typename point_t>
    inline __host__ __device__ float AtomicBox<point_t>::decode(int32_t bits)
    {
      const int32_t sign = 0x80000000;
      if (bits & sign) bits ^= 0x7fffffff;
      return int_as_float(bits);
    }
    
    template<typename point_t>
    inline __host__ __device__ void AtomicBox<point_t>::set_empty()
    {
#pragma unroll
      for (int d=0;d<num_dims;d++) {
        lower[d] = encode(+FLT_MAX);
        upper[d] = encode(-FLT_MAX);
      }
    }
    template<typename point_t>
    inline __host__ __device__ void atomic_grow(AtomicBox<point_t> &abox, const box_t<point_t> &other)
    {
#pragma unroll
      for (int d=0;d<abox.num_dims;d++) {
        const int32_t enc_lower = AtomicBox<point_t>::encode(other.get_lower(d));
        const int32_t enc_upper = AtomicBox<point_t>::encode(other.get_upper(d));
        if (enc_lower < abox.lower[d]) atomicMin(&abox.lower[d],enc_lower);
        if (enc_upper > abox.upper[d]) atomicMax(&abox.upper[d],enc_upper);
      }
    }

    template<typename point_t> inline __host__ __device__
    void atomic_grow(AtomicBox<point_t> &abox, const point_t &other)
    {
#pragma unroll
      for (int d=0;d<abox.num_dims;d++) {
        const int32_t enc = AtomicBox<point_t>::encode(get_coord(other,d));
        if (enc < abox.lower[d]) ::atomicMin(&abox.lower[d],enc);
        if (enc > abox.upper[d]) ::atomicMax(&abox.upper[d],enc);
      }
    }
    
    struct BuildState {
      uint32_t  numNodes;
    };
    
    template<typename T, typename count_t>
    inline void _ALLOC(GpuMemoryResource &memResource,
                       T *&ptr, count_t count, cudaStream_t s)
    { memResource.malloc((void **)&ptr,count*sizeof(T),s); }
    
    template<typename T>
    inline void _FREE(GpuMemoryResource &memResource,
                      T *&ptr, cudaStream_t s)
    { memResource.free(ptr,s); }
    
    typedef enum : int8_t { OPEN_BRANCH, OPEN_NODE, DONE_NODE } NodeState;

    struct PrimState {
      union {
        /* careful with this order - this is intentionally chosen such
           that all item with nodeID==-1 will end up at the end of the
           list; and all others will be sorted by nodeID */
        struct {
          uint64_t primID:31; //!< prim we're talking about
          uint64_t done  : 1;
          uint64_t nodeID:32; //!< node the given prim is (currently) in.
        };
        uint64_t bits;
      };
    };

    template<typename point_t>
    struct TempNode {
      using box_t = box_t<point_t>;
      union {
        struct {
          AtomicBox<point_t> centBounds;
          uint32_t         count;
          uint32_t         unused;
        } openBranch;
        struct {
          uint32_t offset;
          int      dim;
          volatile uint32_t tieBreaker;
          float    pos;
        } openNode;
        struct {
          uint32_t offset;
          uint32_t count;
          int      dim;
          float    pos;
        } doneNode;
      };
    };

    template<typename data_t,
             typename data_traits>
    __global__
    void initState(BuildState      *buildState,
                   NodeState       *nodeStates,
                   TempNode<typename data_traits::point_t> *nodes)
    {
      buildState->numNodes = 2;
      
      nodeStates[0]             = OPEN_BRANCH;
      nodes[0].openBranch.count = 0;
      nodes[0].openBranch.centBounds.set_empty();

      nodeStates[1]            = DONE_NODE;
      nodes[1].doneNode.offset = 0;
      nodes[1].doneNode.count  = 0;
    }

    template<typename data_t,
             typename data_traits>
    __global__
    void initPrims(TempNode<typename data_traits::point_t> *nodes,
                   PrimState       *primState,
                   const data_t    *prims,
                   uint32_t         numPrims)
    {
      const int primID = threadIdx.x+blockIdx.x*blockDim.x;
      if (primID >= numPrims) return;
      
      auto &me = primState[primID];
      me.primID = primID;
                                                    
      const data_t prim = prims[primID];
      me.nodeID = 0;
      me.done   = false;
      // this could be made faster by block-reducing ...
      atomicAdd(&nodes[0].openBranch.count,1);
      atomic_grow(nodes[0].openBranch.centBounds,data_traits::get_point(prim));
    }

    template<typename data_t,
             typename data_traits>
    __global__
    void selectSplits(BuildState      *buildState,
                      NodeState       *nodeStates,
                      TempNode<typename data_traits::point_t> *nodes,
                      uint32_t         numNodes,
                      BuildConfig      buildConfig)
    {
      enum { num_dims = num_dims_of<typename data_traits::point_t>::value };
      
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID >= numNodes) return;

      NodeState &nodeState = nodeStates[nodeID];
      if (nodeState == DONE_NODE)
        // this node was already closed before
        return;
      
      if (nodeState == OPEN_NODE) {
        // this node was open in the last pass, can close it.
        nodeState   = DONE_NODE;
        int offset  = nodes[nodeID].openNode.offset;
        int dim     = nodes[nodeID].openNode.dim;
        float pos   = nodes[nodeID].openNode.pos;
        auto &done  = nodes[nodeID].doneNode;
        done.count  = 0;
        done.offset = offset;
        done.dim    = dim;
        done.pos    = pos;
        return;
      }
      
      auto in = nodes[nodeID].openBranch;
      if (in.count <= buildConfig.makeLeafThreshold) {
        auto &done  = nodes[nodeID].doneNode;
        done.count  = in.count;
        // set this to max-value, so the prims can later do atomicMin
        // with their position ion the leaf list; this value is
        // greater than any prim position.
        done.offset = (uint32_t)-1;
        nodeState   = DONE_NODE;
      } else {
        float widestWidth = 0.f;
        int   widestDim   = -1;
#pragma unroll
        for (int d=0;d<num_dims;d++) {
          float width = in.centBounds.get_upper(d) - in.centBounds.get_lower(d);
          if (width <= widestWidth)
            continue;
          widestWidth = width;
          widestDim   = d;
        }
      
        auto &open = nodes[nodeID].openNode;
        if (widestDim >= 0) {
          open.pos = in.centBounds.get_center(widestDim);
          if (open.pos == in.centBounds.get_lower(widestDim)) {
            // this is the (rare!) case where lower and upper are not
            // "empty" (in the sense that lower.coord==upper.coord),
            // but lower and upper coord differ in only one mantissa
            // bit, so the 'center' rounds to lower. The only way to
            // properly handle that is to define that everything ">="
            // (not ">"!!!) is on the right side during traversal, and
            // set plane pos to right side during build
            open.pos = in.centBounds.get_upper(widestDim);
          }
        }
        open.dim = widestDim;
        if (open.dim < 0) {
          open.pos = in.centBounds.get_upper(0);
        }
        // if (open.dim == -1)
        //   printf("WARNING - ZERO-SPLITS NOT SUPPORTED FOR KD-TREE BUILDER\n");
        
        // this will be epensive - could make this faster by block-reducing
        open.offset = atomicAdd(&buildState->numNodes,2);
#pragma unroll
        for (int side=0;side<2;side++) {
          const int childID = open.offset+side;
          auto &child = nodes[childID].openBranch;
          child.centBounds.set_empty();
          child.count         = 0;
          nodeStates[childID] = OPEN_BRANCH;
        }
        nodeState = OPEN_NODE;
      }
    }

    template<typename data_t,
             typename data_traits>
    __global__
    void updatePrims(NodeState       *nodeStates,
                     TempNode<typename data_traits::point_t> *nodes,
                     PrimState       *primStates,
                     const data_t    *prims,
                     int numPrims)
    {
      const int primID = threadIdx.x+blockIdx.x*blockDim.x;
      if (primID >= numPrims) return;

      auto &me = primStates[primID];
      if (me.done) return;
      
      auto ns = nodeStates[me.nodeID];
      if (ns == DONE_NODE) {
        // node became a leaf, we're done.
        me.done = true;
        return;
      }

      auto &split = nodes[me.nodeID].openNode;
      // const box_t<T,D> primBox = primBoxes[me.primID];
      const typename data_traits::point_t point = data_traits::get_point(prims[me.primID]);
      int side = 0;
      if (split.dim == -1) {
        // could block-reduce this, but will likely not happen often, anyway
        side = (atomicAdd((int*)&split.tieBreaker,1) & 1);
      } else {
        const float center = get_coord(point,split.dim);
        side = (center >= split.pos);
      }
      int newNodeID = split.offset+side;
      auto &myBranch = nodes[newNodeID].openBranch;
      atomicAdd(&myBranch.count,1);
      atomic_grow(myBranch.centBounds,point);
      me.nodeID = newNodeID;
    }
    /* given a sorted list of {nodeID,primID} pairs, this kernel does
       two things: a) it extracts the 'primID's and puts them into the
       bvh's primIDs[] array; and b) it writes, for each leaf nod ein
       the nodes[] array, the node.offset value to point to the first
       of this nodes' items in that bvh.primIDs[] list. */
    template<typename data_t,
             typename data_traits>
    __global__
    void writePrimsAndLeafOffsets(TempNode<typename data_traits::point_t> *nodes,
                                  uint32_t        *bvhItemList,
                                  PrimState       *primStates,
                                  int              numPrims)
    {
      const int offset = threadIdx.x+blockIdx.x*blockDim.x;
      if (offset >= numPrims) return;

      auto &ps = primStates[offset];
      bvhItemList[offset] = ps.primID;
      
      if ((int)ps.nodeID < 0)
        /* invalid prim, just skip here */
        return;
      auto &node = nodes[ps.nodeID];
      ::atomicMin(&node.doneNode.offset,offset);
    }


    template<typename data_t,
             typename data_traits>
    __global__
    void saveBounds(box_t<typename data_traits::point_t> *returnedBounds,
                    TempNode<typename data_traits::point_t> *nodes)
    {
      const int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid > 0) return;
      *returnedBounds = nodes[0].openBranch.centBounds.make_box();
    }
    
    /* writes main phase's temp nodes into final bvh.nodes[]
       layout. actual bounds of that will NOT yet bewritten */
    template<typename data_t,
             typename data_traits>
    __global__
    void writeNodes(typename SpatialKDTree<data_t,data_traits>::Node *finalNodes,
                    TempNode<typename data_traits::point_t>  *tempNodes,
                    int        numNodes)
    {
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID >= numNodes) return;

      finalNodes[nodeID].offset = tempNodes[nodeID].doneNode.offset;
      finalNodes[nodeID].count  = tempNodes[nodeID].doneNode.count;
      finalNodes[nodeID].dim    = tempNodes[nodeID].doneNode.dim;
      finalNodes[nodeID].pos    = tempNodes[nodeID].doneNode.pos;
      if (finalNodes[nodeID].dim == -1) {
        finalNodes[nodeID].dim = 0;
      }
    }
    
    template<typename data_t,
             typename data_traits>
    void builder(SpatialKDTree<data_t,data_traits> &tree,
                 const data_t *prims,
                 int numPrims,
                 BuildConfig buildConfig,
                 cudaStream_t s,
                 GpuMemoryResource &memResource)
    {
      if (buildConfig.makeLeafThreshold == 0)
        buildConfig.makeLeafThreshold = 8;

      tree.data = prims;
      tree.numPrims = numPrims;
      
      // ==================================================================
      // do build on temp nodes
      // ==================================================================
      TempNode<typename data_traits::point_t>   *tempNodes = 0;
      NodeState  *nodeStates = 0;
      PrimState  *primStates = 0;
      BuildState *buildState = 0;
      _ALLOC(memResource,tempNodes,2*numPrims,s);
      _ALLOC(memResource,nodeStates,2*numPrims,s);
      _ALLOC(memResource,primStates,numPrims,s);
      _ALLOC(memResource,buildState,1,s);
      initState<data_t,data_traits>
        <<<1,1,0,s>>>(buildState,
                      nodeStates,
                      tempNodes);
      initPrims<data_t,data_traits>
        <<<divRoundUp(numPrims,1024),1024,0,s>>>
        (tempNodes,
         primStates,prims,numPrims);
      CUKD_CUDA_CALL(StreamSynchronize(s));
      box_t<typename data_traits::point_t> *savedBounds;
      _ALLOC(memResource,savedBounds,sizeof(*savedBounds),s);
      
      saveBounds<data_t,data_traits>
        <<<divRoundUp(numPrims,1024),1024,0,s>>>
        (savedBounds,tempNodes);
      CUKD_CUDA_CALL(StreamSynchronize(s));
      CUKD_CUDA_CALL(Memcpy(&tree.bounds,savedBounds,sizeof(tree.bounds),cudaMemcpyDefault));
      CUKD_CUDA_CALL(StreamSynchronize(s));
      _FREE(memResource,savedBounds,s);
      
      int numDone = 0;
      int numNodes;
      // ------------------------------------------------------------------      
      while (true) {
        CUKD_CUDA_CALL(MemcpyAsync(&numNodes,&buildState->numNodes,
                                   sizeof(numNodes),cudaMemcpyDeviceToHost,s));
        CUKD_CUDA_CALL(StreamSynchronize(s));
        if (numNodes == numDone)
          break;

        selectSplits<data_t,data_traits>
          <<<divRoundUp(numNodes,1024),1024,0,s>>>
          (buildState,
           nodeStates,tempNodes,numNodes,
           buildConfig);

        CUKD_CUDA_CALL(StreamSynchronize(s));
        
        numDone = numNodes;
        updatePrims<data_t,data_traits>
          <<<divRoundUp(numPrims,1024),1024,0,s>>>
          (nodeStates,tempNodes,
           primStates,prims,numPrims);

        CUKD_CUDA_CALL(StreamSynchronize(s));
      }
      // ==================================================================
      // sort {item,nodeID} list
      // ==================================================================
      
      // set up sorting of prims
      uint8_t *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;
      PrimState *sortedPrimStates;
      _ALLOC(memResource,sortedPrimStates,numPrims,s);
      cub::DeviceRadixSort::SortKeys((void*&)d_temp_storage, temp_storage_bytes,
                                     (uint64_t*)primStates,
                                     (uint64_t*)sortedPrimStates,
                                     numPrims,32,64,s);
      _ALLOC(memResource,d_temp_storage,temp_storage_bytes,s);
      cub::DeviceRadixSort::SortKeys((void*&)d_temp_storage, temp_storage_bytes,
                                     (uint64_t*)primStates,
                                     (uint64_t*)sortedPrimStates,
                                     numPrims,32,64,s);
      CUKD_CUDA_CALL(StreamSynchronize(s));
      _FREE(memResource,d_temp_storage,s);
      // ==================================================================
      // allocate and write BVH item list, and write offsets of leaf nodes
      // ==================================================================

      tree.numPrims = numPrims;
      _ALLOC(memResource,tree.primIDs,numPrims,s);
      writePrimsAndLeafOffsets<data_t,data_traits>
        <<<divRoundUp(numPrims,1024),1024,0,s>>>
        (tempNodes,tree.primIDs,sortedPrimStates,numPrims);

      // ==================================================================
      // allocate and write final nodes
      // ==================================================================
      tree.numNodes = numNodes;
      _ALLOC(memResource,tree.nodes,numNodes,s);
      writeNodes<data_t,data_traits>
        <<<divRoundUp(numNodes,1024),1024,0,s>>>
        (tree.nodes,tempNodes,numNodes);
      CUKD_CUDA_CALL(StreamSynchronize(s));
      _FREE(memResource,sortedPrimStates,s);
      _FREE(memResource,tempNodes,s);
      _FREE(memResource,nodeStates,s);
      _FREE(memResource,primStates,s);
      _FREE(memResource,buildState,s);
    }
  } // ::cukd::spatial




  template<typename data_t,
           typename data_traits>
  void free(SpatialKDTree<data_t,data_traits> &tree,
            cudaStream_t stream,
            GpuMemoryResource &memResource)
  {
    memResource.free(tree.nodes,stream);
    tree.nodes = 0;
    tree.numNodes = 0;
    memResource.free(tree.primIDs,stream);
    tree.primIDs = 0;
    tree.numPrims = 0;
  }
    
  template<typename data_t,
           typename data_traits>
  void buildTree(SpatialKDTree<data_t,data_traits> &tree,
                 data_t *d_points,
                 int numPrims,
                 BuildConfig buildConfig,
                 cudaStream_t s,
                 GpuMemoryResource &memResource)
  {
    spatial::builder<data_t,data_traits>
      (tree,d_points,numPrims,buildConfig,s,memResource);
    CUKD_CUDA_CALL(StreamSynchronize(s));
    
  }
}
