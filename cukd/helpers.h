// ======================================================================== //
// Copyright 2019-2024 Ingo Wald                                            //
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

#include "cukd/common.h"
#include "cukd/cukd-math.h"

namespace cukd {

  // ------------------------------------------------------------------
  /*! defines a 'memory resource' that can be used for allocating gpu
      memory; this allows the user to switch between usign
      cudaMallocAsync (where avialble) vs regular cudaMalloc (where
      not), or to use their own memory pool, to use managed memory,
      etc. All memory allocatoins done during construction will use
      the memory resource passed to the respective build function. */
  struct GpuMemoryResource {
    virtual cudaError_t malloc(void** ptr, size_t size, cudaStream_t s) = 0;
    virtual cudaError_t free(void* ptr, cudaStream_t s) = 0;
  };

  struct ManagedMemMemoryResource : public GpuMemoryResource {
    cudaError_t malloc(void** ptr, size_t size, cudaStream_t s) override
    {
      cudaStreamSynchronize(s);
      return cudaMallocManaged(ptr,size);
    }
    cudaError_t free(void* ptr, cudaStream_t s) override
    {
      cudaStreamSynchronize(s);
      return cudaFree(ptr);
    }
  };

  /* by default let's use cuda malloc async, which is much better and
     faster than regular malloc; but that's available on cuda 11, so
     let's add a fall back for older cuda's, too */
#if CUDART_VERSION >= 11020
  struct AsyncGpuMemoryResource final : GpuMemoryResource {
    cudaError_t malloc(void** ptr, size_t size, cudaStream_t s) override {
      return cudaMallocAsync(ptr, size, s);
    }
    cudaError_t free(void* ptr, cudaStream_t s) override {
      return cudaFreeAsync(ptr, s);
    }
  };

  inline GpuMemoryResource &defaultGpuMemResource() {
    static AsyncGpuMemoryResource memResource;
    return memResource;
  }
#else
  inline GpuMemoryResource &defaultGpuMemResource() {
    static ManagedMemMemoryResource memResource;
    return memResource;
  }
#endif

  /*! helper functions for a generic, arbitrary-size binary tree -
    mostly to compute level of a given node in that tree, and child
    IDs, parent IDs, etc */
  struct BinaryTree {
    inline static __host__ __device__ int rootNode() { return 0; }
    inline static __host__ __device__ int parentOf(int nodeID) { return (nodeID-1)/2; }
    inline static __host__ __device__ int isLeftSibling(int nodeID) { return (nodeID & 1); }
    inline static __host__ __device__ int leftChildOf (int nodeID) { return 2*nodeID+1; }
    inline static __host__ __device__ int rightChildOf(int nodeID) { return 2*nodeID+2; }
    inline static __host__ __device__ int firstNodeInLevel(int L) { return (1<<L)-1; }
  
    inline static __host__ __device__ int levelOf(int nodeID)
    {
#ifdef __CUDA_ARCH__
      int k = 63 - __clzll(nodeID+1);
#elif defined(_MSC_VER)
      unsigned long bs;
      _BitScanReverse(&bs, nodeID + 1);
      int k = bs;
#else
      int k = 63 - __builtin_clzll(nodeID+1);
#endif
      return k;
    }
  
    inline static __host__ __device__ int numLevelsFor(int numPoints)
    {
      return levelOf(numPoints-1)+1;
    }
  
    inline __host__ __device__ int numSiblingsToLeftOf(int n)
    {
      int levelOf_n = BinaryTree::levelOf(n);
      return n - BinaryTree::firstNodeInLevel(levelOf_n);
    }
  };

  /*! helper class for all expressions operating on a full binary tree
      of a given number of levels */
  struct FullBinaryTreeOf
  {
    inline __host__ __device__ FullBinaryTreeOf(int numLevels) : numLevels(numLevels) {}
  
    // tested, works for any numLevels >= 0
    inline __host__ __device__ int numNodes() const { return (1<<numLevels)-1; }
    inline __host__ __device__ int numOnLastLevel() const { return (1<<(numLevels-1)); }
  
    const int numLevels;
  };

  /*! helper class for all kind of values revolving around a given
      subtree in full binary tree of a given number of levels. Allos
      us to compute the number of nodes in a given subtree, the first
      and last node of a given subtree, etc */
  struct SubTreeInFullTreeOf
  {
    inline __host__ __device__
    SubTreeInFullTreeOf(int numLevelsTree, int subtreeRoot)
      : numLevelsTree(numLevelsTree),
        subtreeRoot(subtreeRoot),
        levelOfSubtree(BinaryTree::levelOf(subtreeRoot)),
        numLevelsSubtree(numLevelsTree - levelOfSubtree)
    {}
    inline __host__ __device__
    int lastNodeOnLastLevel() const
    {
      // return ((subtreeRoot+2) << (numLevelsSubtree-1)) - 2;
      int first = (subtreeRoot+1)<<(numLevelsSubtree-1);
      int onLast = (1<<(numLevelsSubtree-1)) - 1;
      return first+onLast;
    }
    inline __host__ __device__
    int numOnLastLevel() const { return FullBinaryTreeOf(numLevelsSubtree).numOnLastLevel(); }
    inline __host__ __device__
    int numNodes()            const { return FullBinaryTreeOf(numLevelsSubtree).numNodes(); }
  
    const int numLevelsTree;
    const int subtreeRoot;
    const int levelOfSubtree;
    const int numLevelsSubtree;
  };

  inline __host__ __device__ int clamp(int val, int lo, int hi)
  { return max(min(val,hi),lo); }

                                       
  /*! helper functions for a binary tree of exactly N nodes. For this
      paper, all we need to be able to compute is the size of any
      given subtree in this tree */
  struct ArbitraryBinaryTree {
    inline __host__ __device__ ArbitraryBinaryTree(int numNodes)
      : numNodes(numNodes) {}
    inline __host__ __device__ int numNodesInSubtree(int n)
    {
      auto fullSubtree
        = SubTreeInFullTreeOf(BinaryTree::numLevelsFor(numNodes),n);
      const int lastOnLastLevel
        = fullSubtree.lastNodeOnLastLevel();
      const int numMissingOnLastLevel
        = clamp(lastOnLastLevel - numNodes, 0, fullSubtree.numOnLastLevel());
      const int result = fullSubtree.numNodes() - numMissingOnLastLevel;
      return result;
    }
  
    const int numNodes;
  };

  // ==================================================================
  // helper functions for our N-step data ordering
  // ==================================================================

  /*! helper class for the array layout that this method is based upon
      (please see accompanying paper): in the L'th construction step,
      this array layout first stores all the first L levels' nodes in
      proper KD-tree order, then has, for each level-L subtree on this
      L'th level, first all nodes from the first subtree on this
      level, then those for the second, etc. */
  struct ArrayLayoutInStep {
    inline __host__ __device__ 
    ArrayLayoutInStep(int step, /* num nodes in three: */int numPoints)
      : numLevelsDone(step), numPoints(numPoints)
    {}

    /*! number of nodes already settled to their final position in all
      previous steps; if we start counting steps at L=0 for the
      first step, then 'L' is also the number of binary tree levels
      that have already been built. */
    inline __host__ __device__ int numSettledNodes() const
    { return FullBinaryTreeOf(numLevelsDone).numNodes(); }

    /*! given a node ID 'n' *on* (!) the current level 'L' (ie, a
      subtree), computes the number of nodes in the subtree under (and
      including) node n */
    inline __host__ __device__ int segmentBegin(int subtreeOnLevel)
    {
      int numSettled = FullBinaryTreeOf(numLevelsDone).numNodes();
      int numLevelsTotal = BinaryTree::numLevelsFor(numPoints);
      int numLevelsRemaining = numLevelsTotal-numLevelsDone;
    
      int firstNodeInThisLevel = FullBinaryTreeOf(numLevelsDone).numNodes();
      int numEarlierSubtreesOnSameLevel = subtreeOnLevel-firstNodeInThisLevel;

      int numToLeftIfFull
        = numEarlierSubtreesOnSameLevel
        * FullBinaryTreeOf(numLevelsRemaining).numNodes();

      int numToLeftOnLastIfFull
        = numEarlierSubtreesOnSameLevel
        * FullBinaryTreeOf(numLevelsRemaining).numOnLastLevel();

      int numTotalOnLastLevel
        = numPoints - FullBinaryTreeOf(numLevelsTotal-1).numNodes();

      int numReallyToLeftOnLast
        = min(numTotalOnLastLevel,numToLeftOnLastIfFull);
      int numMissingOnLast
        = numToLeftOnLastIfFull - numReallyToLeftOnLast;

      int result = numSettled + numToLeftIfFull - numMissingOnLast;
      return result;
    }

    inline __host__ __device__
    int pivotPosOf(int subtree)
    {
      int segBegin = segmentBegin(subtree);
      int pivotPos = segBegin + sizeOfLeftSubtreeOf(subtree);
      return pivotPos;
    }

    inline __host__ __device__
    int sizeOfLeftSubtreeOf(int subtree)
    {
      int leftChildRoot = BinaryTree::leftChildOf(subtree);
      if (leftChildRoot >= numPoints) return 0;
      return ArbitraryBinaryTree(numPoints).numNodesInSubtree(leftChildRoot);
    }
    
    inline __host__ __device__
    int sizeOfSegment(int n) const
    { return ArbitraryBinaryTree(numPoints).numNodesInSubtree(n); }

  
    const int numLevelsDone;
    const int numPoints;
  };

}

