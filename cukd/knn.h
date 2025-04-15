// ======================================================================== //
// Copyright 2022-2023 Ingo Wald                                            //
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
#include "cukd/helpers.h"
#include "cukd/fcp.h"

// ==================================================================
// INTERFACE SECTION
// ==================================================================
namespace cukd {
  struct BaseCandidateList {
  protected:
    inline __device__ uint64_t encode(float f, int i);
    inline __device__ float decode_dist2(uint64_t v) const;
    inline __device__ int   decode_pointID(uint64_t v) const;
  };
  
  /*! ABSTRACT interface to a candidate list. a candidate list is a
    list of the at-the-time k-nearest elements encountered during
    traversal. A user initiates a query by creating a _actual_ (ie,
    not abstract) candidate list (either a HeapCnadidateList or
    FixedCandidateList, see below), and passing that to a knn
    traversal kernel. The kernel will traverse the tree, present the
    respective candidate list with any closer candidates it can find
    during traversal, and thereby fill in this list. After traversal
    is complete the user can then 'interrogate' this list of the IDs
    and distances of the final k-nearest points. 

    Note that FixedCandidateList and HeapCandidateList will have
    different perormance implications for different k, but will also
    slightly differ in the order in which candidates are
    returned. I.e., in the FixedCandidateList the result candidates
    will be stored in ascending order (and in positions 0..j if j<k
    get found!); while the HeapCandidateList does not guarantee
    tiehr of these condisions (for heap, if j<k items get found some
    of the k slots will return an ID of -1) */
  template<int k>
  struct CandidateList : public BaseCandidateList {
    // ------------------------------------------------------------------
    // interface fcts with which _user_ can read results of query:
    // ------------------------------------------------------------------
    inline __device__ CandidateList(float cutOffRadius) {}
    
    /*! returns _square_ of maximum radius of any found point, if k
      points were found. if less than k points were found, this
      returns the square of the max query radius/cut-off radius */
    inline __device__ float maxRadius2() const /* abstract */;
    
    /*! returns _square_ of distance to i'th found point. points will
      be sorted by distance in FixedCandidateList, but will _not_ be
      sorted in HeapCandidateList */
    inline __device__ float get_dist2(int i) const;
    
    /*! returns ID of i'th found k-nearest data point */
    inline __device__ int   get_pointID(int i) const;

    using BaseCandidateList::encode;
    using BaseCandidateList::decode_dist2;
    using BaseCandidateList::decode_pointID;
    
    /*! storage for k elements; we encode those float:int pairs as a
        single int64 to make reading/writing/swapping faster */
    uint64_t entry[k];
    enum { num_k = k };
  };

  namespace stackBased {
    /*! default, stack-based kNN traversal kernel, with simple
      point-to-plane-distance test for culling subtrees
      
      \returns square of maximum found distance (if k elements got
      round) of square of max search radius used to initialze
      candidatelist (if j < k were found)
    */
    template<
      /*! type of object to manage the k-nearest objects*/
      typename CandidateList,
      /*! type of data in the underlying tree */
      typename data_t,
      /*! traits of data in the underlying tree */
      typename data_traits=default_data_traits<data_t>>
    inline __device__
    float knn(CandidateList &result,
              typename data_traits::point_t queryPoint,
              // const box_t<typename data_traits::point_t> worldBounds,
              const data_t *d_nodes,
              int N);
    template<
      /*! type of object to manage the k-nearest objects*/
      typename CandidateList,
      /*! type of data in the underlying tree */
      typename data_t,
      /*! traits of data in the underlying tree */
      typename data_traits=default_data_traits<data_t>>
    inline __device__
    float knn(CandidateList &result,
              typename data_traits::point_t queryPoint,
              const box_t<typename data_traits::point_t> worldBounds,
              const data_t *d_nodes,
              int N)
    {
      /* TODO: add early-out if distance to worldbounds is >= max query dist */
      return knn<CandidateList,data_t,data_traits>
        (result,queryPoint,d_nodes,N);
    }


    /* the same, for a _spatial_ k-d tree */
    template<typename CandidateList,
             typename data_t,
             typename data_traits=default_data_traits<data_t>>
    inline __device__
    float knn(CandidateList &result,
              const SpatialKDTree<data_t,data_traits> &tree,
              typename data_traits::point_t queryPoint);
  } // ::cukd::stackBased

  namespace stackFree {
    /*! kNN traversal kernel that uses stack-free traversal. this also
      uses simple point-to-plane-distacne test for culling subtrees,
      but uses a stack-free rather than a stack-based traversal. Will
      need a few more traversal steps than the stack-based variant,
      but doesn't need to maintain a traversal stack (nor incur the
      memory overhead for that) */
    template<
      /*! type of object to manage the k-nearest objects*/
      typename CandidateList,
      /*! type of data in the underlying tree */
      typename data_t,
      /*! traits of data in the underlying tree */
      typename data_traits=default_data_traits<data_t>>
    inline __device__
    float knn(CandidateList &result,
              typename data_traits::point_t queryPoint,
              // const box_t<typename data_traits::point_t> worldBounds,
              const data_t *d_nodes,
              int N);
    template<typename CandidateList,
             typename data_t,
             typename data_traits=default_data_traits<data_t>>
    inline __device__
    float knn(CandidateList &result,
              typename data_traits::point_t queryPoint,
              const box_t<typename data_traits::point_t> worldBounds,
              const data_t *d_nodes,
              int N)
    {
      /* TODO: add early-out if distance to worldbounds is >= max query dist */
      return knn<CandidateList,data_t,data_traits>
        (result,queryPoint,d_nodes,N);
    }
      
  } // ::cukd::stackFree
  
  namespace cct {
    /*! kNN kernel using special 'closest-corner-tracking' traversal
      code; this traversal uses a stack just like the stackBased::knn
      (in fact, its stack footprint is even larger), but is much
      better at culling data in particular for non-uniform input data
      and unbounded queries */
    template<
      /*! type of object to manage the k-nearest objects*/
      typename CandidateList,
      /*! type of data in the underlying tree */
      typename data_t,
      /*! traits of data in the underlying tree */
      typename data_traits=default_data_traits<data_t>>
    inline __device__
    float knn(CandidateList &result,
              typename data_traits::point_t queryPoint,
              const box_t<typename data_traits::point_t> worldBounds,
              const data_t *d_nodes,
              int N);

    /* the same, for a _spatial_ k-d tree */
    template<typename CandidateList,
             typename data_t,
             typename data_traits=default_data_traits<data_t>>
    inline __device__
    float knn(CandidateList &result,
              const SpatialKDTree<data_t,data_traits> &tree,
              typename data_traits::point_t queryPoint);
  } // ::cukd::cct


  
  // ------------------------------------------------------------------
  // actual candidate list implementation(s) to use in queries
  // ------------------------------------------------------------------
  
  /*! candidate list (see above) in which the k-nearest points are
    stored in a linear sorted list.  Inserting into this list costs
    O(k), and will thus be (very) expensive for large k; but for
    smaller k can probably be unrolled and kept in registers (unlike
    the heap based variant below). This should be  faster for small
    k than using the heav variant - but it'll be painfully slow for
    larger k. */
  template<int k>
  struct FixedCandidateList : public CandidateList<k>
  {
    using CandidateList<k>::entry;
    using CandidateList<k>::encode;
    
    // ------------------------------------------------------------------
    // interface fcts with which _user_ can read results of query:
    // ------------------------------------------------------------------
    inline __device__ FixedCandidateList(float cutOffRadius);
    inline __device__ float maxRadius2() const;
    // ------------------------------------------------------------------
    // interface for traversal/query routines to interact with this
    // ------------------------------------------------------------------
    inline __device__ float returnValue() const;
    inline __device__ float processCandidate(int candPrimID, float candDist2);
    inline __device__ float initialCullDist2() const;
    inline __device__ void  push(float dist, int pointID);
  };

  /*! candidate list (see above) that uses a heap to organize the
    points. problem with a heap is that the memory accesses to this
    heap cannot be predicted, thus cannot be unrolled or kep in
    registers, and thus _have_ to go to memory (at best polluting
    the cache); but insertion cost is O(*log* k), so much better for
    large k than using insertion sort in a sorted list */
  template<int k>
  struct HeapCandidateList : public CandidateList<k>
  {
    using CandidateList<k>::entry;
    using CandidateList<k>::encode;
    
    // ------------------------------------------------------------------
    // interface fcts with which _user_ can read results of query:
    // ------------------------------------------------------------------
    inline __device__ HeapCandidateList(float cutOffRadius);
    inline __device__ float maxRadius2() const;
    // ------------------------------------------------------------------
    // interface for traversal/query routines to interact with this
    // ------------------------------------------------------------------
    inline __device__ float returnValue() const;
    inline __device__ float processCandidate(int candPrimID, float candDist2);
    inline __device__ float initialCullDist2() const;
    inline __device__ void  push(float dist, int pointID);
  };

  /*! a _flexible_ heap candidate list is a list of candidates that
      uses a heap structure to store these candidates, but unlike the
      HeapCandidateList the number of elements is NOT a compile-tiem
      paramter, but a runtime parameters. Hence, the memory for the
      list is also not part of this class, but needs to be provided
      externally through a pointer */
  struct FlexHeapCandidateList : public BaseCandidateList
  {
    /*! initialize this struct from given cancidate list pointer and
      'list length' k. If radius >= 0 use this to initialize the list;
      if radius < 0.f do NOT initialize the list, and use as is,
      assuming it is a valid heap list */
    inline __device__ FlexHeapCandidateList(uint64_t *entryMem,
                                            int k,
                                            float cutOffRadius);
    inline __device__ float maxRadius2() const;
    // ------------------------------------------------------------------
    // interface for traversal/query routines to interact with this
    // ------------------------------------------------------------------
    inline __device__ float returnValue() const;
    inline __device__ float processCandidate(int candPrimID, float candDist2);
    inline __device__ float initialCullDist2() const;
    inline __device__ void  push(float dist, int pointID);
    
    using BaseCandidateList::encode;
    uint64_t *const entry;
    int const k;
  };
}

// ==================================================================
// IMPLEMENTATION SECTION
// ==================================================================

namespace cukd {


  // ------------------------------------------------------------------
  // parent CandidateList
  // ------------------------------------------------------------------

  template<int k>
  inline __device__
  float CandidateList<k>::get_dist2(int i) const
  { return decode_dist2(entry[i]); }
  
  template<int k>
  inline __device__
  int CandidateList<k>::get_pointID(int i) const
  { return decode_pointID(entry[i]); }
    
  inline __device__
  uint64_t BaseCandidateList::encode(float f, int i)
  { return (uint64_t(__float_as_uint(f)) << 32) | uint32_t(i); }

  inline __device__
  float BaseCandidateList::decode_dist2(uint64_t v) const
  { return __uint_as_float(v >> 32); }
  
  inline __device__
  int BaseCandidateList::decode_pointID(uint64_t v) const
  { return int(uint32_t(v)); }


  // ------------------------------------------------------------------
  // FlexHeapCandidateList
  // ------------------------------------------------------------------

  inline __device__
  FlexHeapCandidateList::FlexHeapCandidateList(uint64_t *entryMem,
                                               int k,
                                               float cutOffRadius)
    : entry(entryMem),
      k(k)
  {
    if (cutOffRadius >= 0.f) {
      for (int i=0;i<k;i++)
        entry[i] = encode(cutOffRadius*cutOffRadius,-1);
    }
  }
  
  inline __device__
  float FlexHeapCandidateList::maxRadius2() const
  { return decode_dist2(entry[0]); }
  
  inline __device__
  float FlexHeapCandidateList::returnValue() const
  { return maxRadius2(); }
  
  inline __device__
  float FlexHeapCandidateList::processCandidate(int candPrimID, float candDist2)
  {
    push(candDist2,candPrimID);
    return maxRadius2();
  }
  
  inline __device__
  float FlexHeapCandidateList::initialCullDist2() const
  { return maxRadius2(); }
  
  inline __device__ void  FlexHeapCandidateList::push(float dist, int pointID)
  {
    uint64_t e = encode(dist,pointID);
    if (e >= entry[0]) return;

    int pos = 0;
    while (true) {
      uint64_t largestChildValue = uint64_t(-1);
      int firstChild = 2*pos+1;
      int largestChild = k;
      if (firstChild < k) {
        largestChild = firstChild;
        largestChildValue = entry[firstChild];
      }
        
      int secondChild = firstChild+1;
      if (secondChild < k && entry[secondChild] > largestChildValue) {
        largestChild = secondChild;
        largestChildValue = entry[secondChild];
      }

      if (largestChild == k || largestChildValue < e) {
        entry[pos] = e;
        break;
      } else {
        entry[pos] = largestChildValue;
        pos = largestChild;
      }
    }
  }
  
  // ------------------------------------------------------------------
  // HeapCandidateList
  // ------------------------------------------------------------------

  template<int k>
  inline __device__
  HeapCandidateList<k>::HeapCandidateList(float cutOffRadius)
    : CandidateList<k>(cutOffRadius)
  {
#pragma unroll
    for (int i=0;i<k;i++)
      entry[i] = encode(cutOffRadius*cutOffRadius,-1);
  }

  template<int k>
  inline __device__
  float HeapCandidateList<k>::returnValue() const
  { return maxRadius2(); }
  
  template<int k>
  inline __device__
  float HeapCandidateList<k>::processCandidate(int candPrimID,
                                               float candDist2)
  {
    push(candDist2,candPrimID);
    return maxRadius2();
  }
  
  template<int k>
  inline __device__
  float HeapCandidateList<k>::initialCullDist2() const
  { return maxRadius2(); }
    
  template<int k>
  inline __device__
  void HeapCandidateList<k>::push(float dist,
                                  int pointID)
  {
    uint64_t e = encode(dist,pointID);
    if (e >= entry[0]) return;

    int pos = 0;
    while (true) {
      uint64_t largestChildValue = uint64_t(-1);
      int firstChild = 2*pos+1;
      int largestChild = k;
      if (firstChild < k) {
        largestChild = firstChild;
        largestChildValue = entry[firstChild];
      }
        
      int secondChild = firstChild+1;
      if (secondChild < k && entry[secondChild] > largestChildValue) {
        largestChild = secondChild;
        largestChildValue = entry[secondChild];
      }

      if (largestChild == k || largestChildValue < e) {
        entry[pos] = e;
        break;
      } else {
        entry[pos] = largestChildValue;
        pos = largestChild;
      }
    }
  }
    
  template<int k>
  inline __device__
  float HeapCandidateList<k>::maxRadius2() const
  { return decode_dist2(entry[0]); }
    




  // ------------------------------------------------------------------
  // FixedCandidateList
  // ------------------------------------------------------------------

  template<int k>
  inline __device__
  FixedCandidateList<k>::FixedCandidateList(float cutOffRadius)
    : CandidateList<k>(cutOffRadius)
  {
#pragma unroll
    for (int i=0;i<k;i++)
      entry[i] = encode(cutOffRadius*cutOffRadius,-1);
  }

  template<int k>
  inline __device__
  float FixedCandidateList<k>::returnValue() const
  { return maxRadius2(); }
  
  template<int k>
  inline __device__
  float FixedCandidateList<k>::processCandidate(int candPrimID,
                                                float candDist2)
  {
    push(candDist2,candPrimID);
    return maxRadius2();
  }
  
  template<int k>
  inline __device__
  float FixedCandidateList<k>::initialCullDist2() const
  { return maxRadius2(); }

  template<int k>
  inline __device__ void FixedCandidateList<k>::push(float dist, int pointID)
  {
    uint64_t v = encode(dist,pointID);
#pragma unroll
    for (int i=0;i<k;i++) {
      uint64_t vmax = max(entry[i],v);
      uint64_t vmin = min(entry[i],v);
      entry[i] = vmin;
      v = vmax;
    }
  }

  template<int k>
  inline __device__
  float FixedCandidateList<k>::maxRadius2() const
  { return decode_dist2(entry[k-1]); }
    
  namespace cct {
    template<typename CandidateList,
             typename data_t,
             typename data_traits>
    inline __device__
    float knn(CandidateList &result,
              typename data_traits::point_t queryPoint,
              const box_t<typename data_traits::point_t> worldBounds,
              const data_t *d_nodes,
              int N)
    {
      traverse_cct<CandidateList,data_t,data_traits>
        (result,queryPoint,worldBounds,d_nodes,N);
      return result.returnValue();
    }

    template<typename CandidateList,
             typename data_t,
             typename data_traits>
    inline __device__
    float knn(CandidateList &result,
              const SpatialKDTree<data_t,data_traits> &tree,
              typename data_traits::point_t queryPoint)
    {
      using node_t     = typename SpatialKDTree<data_t,data_traits>::Node;
      using point_t    = typename data_traits::point_t;
      using point_traits = ::cukd::point_traits<point_t>;
      using scalar_t   = typename point_traits::scalar_t;
      enum { num_dims  = point_traits::num_dims };
    
      scalar_t cullDist = result.initialCullDist2();

      /* can do at most 2**30 points... */
      struct StackEntry {
        int   nodeID;
        point_t closestCorner;
      };
      enum{ stack_depth = 50 };
      StackEntry stackBase[stack_depth];
      StackEntry *stackPtr = stackBase;

      /*! current node in the tree we're traversing */
      int nodeID = 0;
      point_t closestPointOnSubtreeBounds = project(tree.bounds,queryPoint);
      if (sqrDistance(queryPoint,closestPointOnSubtreeBounds) > cullDist)
        return result.returnValue();
      node_t node;
      while (true) {
        while (true) {
          CUKD_STATS(if (cukd::g_traversalStats) ::atomicAdd(cukd::g_traversalStats,1));
          node = tree.nodes[nodeID];
          if (node.count)
            // this is a leaf...
            break;
          const auto query_coord = get_coord(queryPoint,node.dim);
        
          const bool leftIsClose = query_coord < node.pos;
          const int  lChild = node.offset+0;
          const int  rChild = node.offset+1;

          const int closeChild = leftIsClose?lChild:rChild;
          const int farChild   = leftIsClose?rChild:lChild;

          auto farSideCorner = closestPointOnSubtreeBounds;
          point_traits::set_coord(farSideCorner,node.dim,node.pos);
        
          if (sqrDistance(farSideCorner,queryPoint) < cullDist) {
            stackPtr->closestCorner = farSideCorner;
            stackPtr->nodeID  = farChild;
            ++stackPtr;
            if ((stackPtr - stackBase) >= stack_depth) {
              printf("STACK OVERFLOW %i\n",int(stackPtr - stackBase));
              return -1;
            }
          }
          nodeID = closeChild;
        }

        for (int i=0;i<node.count;i++) {
          int primID = tree.primIDs[node.offset+i];
          CUKD_STATS(if (cukd::g_traversalStats) ::atomicAdd(cukd::g_traversalStats,1));
          const auto sqrDist = sqrDistance(data_traits::get_point(tree.data[primID]),queryPoint);
          cullDist = result.processCandidate(primID,sqrDist);
        }
      
        while (true) {
          if (stackPtr == stackBase) 
            return result.returnValue();
          --stackPtr;
          closestPointOnSubtreeBounds = stackPtr->closestCorner;
          if (sqrDistance(closestPointOnSubtreeBounds,queryPoint) >= cullDist)
            continue;
          nodeID = stackPtr->nodeID;
          break;
        }
      }
    }
  } // ::cukd::cct

  namespace stackFree {
    template<typename CandidateList,
             typename data_t,
             typename data_traits>
    inline __device__
    float knn(CandidateList &result,
              typename data_traits::point_t queryPoint,
              const data_t *d_nodes,
              int N)
    {
      traverse_stack_free<CandidateList,data_t,data_traits>
        (result,queryPoint,d_nodes,N);
      return result.returnValue();
    }
  } // ::cukd::stackFree

  namespace stackBased {
    template<typename CandidateList,
             typename data_t,
             typename data_traits>
    inline __device__
    float knn(CandidateList &result,
              typename data_traits::point_t queryPoint,
              const data_t *d_nodes,
              int N)
    {
      traverse_default<CandidateList,data_t,data_traits>
        (result,queryPoint,d_nodes,N);
      return result.returnValue();
    }
  
    template<typename CandidateList,
             typename data_t,
             typename data_traits>
    inline __device__
    float knn(CandidateList &result,
              const SpatialKDTree<data_t,data_traits> &tree,
              typename data_traits::point_t queryPoint)
    {
      using node_t     = typename SpatialKDTree<data_t,data_traits>::Node;
      using point_t    = typename data_traits::point_t;
      using scalar_t   = typename scalar_type_of<point_t>::type;
      enum { num_dims  = num_dims_of<point_t>::value };
    
      scalar_t cullDist = result.initialCullDist2();

      /* can do at most 2**30 points... */
      struct StackEntry {
        int   nodeID;
        float sqrDist;
      };
      enum{ stack_depth = 50 };
      StackEntry stackBase[stack_depth];
      StackEntry *stackPtr = stackBase;

      /*! current node in the tree we're traversing */
      int nodeID = 0;
      node_t node;
      while (true) {
        while (true) {
          CUKD_STATS(if (cukd::g_traversalStats) ::atomicAdd(cukd::g_traversalStats,1));
          node = tree.nodes[nodeID];
          if (node.count)
            // this is a leaf...
            break;
          const auto query_coord = get_coord(queryPoint,node.dim);
          const bool leftIsClose = query_coord < node.pos;
          const int  lChild = node.offset+0;
          const int  rChild = node.offset+1;

          const int closeChild = leftIsClose?lChild:rChild;
          const int farChild   = leftIsClose?rChild:lChild;
        
          const float sqrDistToPlane = sqr(query_coord - node.pos);
          if (sqrDistToPlane < cullDist) {
            stackPtr->nodeID  = farChild;
            stackPtr->sqrDist = sqrDistToPlane;
            ++stackPtr;
            if ((stackPtr - stackBase) >= stack_depth) {
              printf("STACK OVERFLOW %i\n",int(stackPtr - stackBase));
              return -1;
            }
          }
          nodeID = closeChild;
        }

        for (int i=0;i<node.count;i++) {
          int primID = tree.primIDs[node.offset+i];
          CUKD_STATS(if (cukd::g_traversalStats) ::atomicAdd(cukd::g_traversalStats,1));
          const auto sqrDist = sqrDistance(data_traits::get_point(tree.data[primID]),queryPoint);
          cullDist = result.processCandidate(primID,sqrDist);
        }
      
        while (true) {
          if (stackPtr == stackBase) 
            return result.returnValue();
          --stackPtr;
          if (stackPtr->sqrDist >= cullDist)
            continue;
          nodeID = stackPtr->nodeID;
          break;
        }
      }
    }
  } // ::cukd::stackBased

} // :: cukd
