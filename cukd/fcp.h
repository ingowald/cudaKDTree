// ======================================================================== //
// Copyright 2018-2022 Ingo Wald                                            //
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

/* copied from OWL project, and put into new namespace to avoid naming conflicts.*/

#pragma once

#include "cukd/common.h"
#include "cukd/helpers.h"
#include "cukd/data.h"
#include "cukd/spatial-kdtree.h"

// ==================================================================
// INTERFACE SECTION
// ==================================================================
namespace cukd {

  /*! Structure of parameters to control the behavior of the FCP
    search.  By default FCP will perform an exact nearest neighbor
    search, but the following parameters can be set to cut some
    corners and make the search approximate in favor of speed. */
  struct FcpSearchParams {
    /*! Controls how many "far branches" of the tree will be
      searched. If set to 0 the algorithm will only go down the tree
      once following the nearest branch each time. Kernels may ignore
      this value. */
    int far_node_inspect_budget = INT_MAX;

    /*! will only search for elements that are BELOW (i.e., NOT
      including) this radius. This in particular allows for cutting
      down on the number of branches to be visited during
      traversal */
    float cutOffRadius = INFINITY;
  };

  namespace stackBased {
    /*! default, stack-based find-closest point kernel, with simple
      point-to-plane-distacne test for culling subtrees 
      
      \returns the ID of the point that's closest to the query point,
      or -1 if none could be found within the given serach radius
    */
    template<
      /*! type of data point(s) that the tree is built over (e.g., float3) */
      typename data_t,
      /*! traits that describe these points (float3 etc have working defaults */
      typename data_traits=default_data_traits<data_t>>
    inline __device__
    int fcp(typename data_traits::point_t queryPoint,
            // /*! the world-space bounding box of all data points */
            // const box_t<typename data_traits::point_t> worldBounds,
            /*! device(!)-side array of data point, ordered in the
              right way as produced by buildTree()*/
            const data_t *dataPoints,
            /*! number of data points in the tree */
            int numDataPoints,
            /*! paramteres to fine-tune the search */
            FcpSearchParams params = FcpSearchParams{});
  } // ::cukd::stackBased

  namespace stackFree {
    /*! stack-free version of the default find-cloest-point kernel
      that lso uses simple point-to-plane-distacne test for culling
      subtrees, but uses a stack-free rather than a stack-based
      traversal. Will need a few more traversal steps than the
      stack-based variant, but doesn't need to maintain a traversal
      stack (nor incur the memory overhead for that) */
    template<typename data_t,
             typename data_traits=default_data_traits<data_t>>
    inline __device__
    int fcp(typename data_traits::point_t queryPoint,
            // /*! the world-space bounding box of all data points */
            // const box_t<typename data_traits::point_t> worldBounds,
            /*! device(!)-side array of data point, ordered in the
              right way as produced by buildTree()*/
            const data_t *dataPoints,
            /*! number of data points in the tree */
            int numDataPoints,
            /*! paramteres to fine-tune the search */
            FcpSearchParams params = FcpSearchParams{});

    // the same, for a _spatial_ k-d tree 
    template<typename data_t,
             typename data_traits=default_data_traits<data_t>>
    inline __device__
    int fcp(const SpatialKDTree<data_t,data_traits> &tree,
            typename data_traits::point_t queryPoint,
            FcpSearchParams params = FcpSearchParams{});
  } // ::cukd::stackFree
  
  namespace cct {
    /*! find-closest-point (fcp) kernel using specal
      'closest-corner-tracking' traversal code; this traversal uses
      a stack just like the stackBased::fcp (in fact, its stack
      footprint is even larger), but is much better at culling data
      in particular for non-uniform input data and unbounded
      queries */
    template<typename data_t,
             typename data_traits=default_data_traits<data_t>>
    inline __device__
    int fcp(typename data_traits::point_t queryPoint,
            /*! the world-space bounding box of all data points */
            const box_t<typename data_traits::point_t> worldBounds,
            /*! device(!)-side array of data point, ordered in the
              right way as produced by buildTree()*/
            const data_t *dataPoints,
            /*! number of data points in the tree */
            int numDataPoints,
            /*! paramteres to fine-tune the search */
            FcpSearchParams params = FcpSearchParams{});
    
    // the same, for a _spatial_ k-d tree 
    template<typename data_t,
             typename data_traits=default_data_traits<data_t>>
    inline __device__
    int fcp(const SpatialKDTree<data_t,data_traits> &tree,
            typename data_traits::point_t queryPoint,
            FcpSearchParams params = FcpSearchParams{});
  } // ::cukd::cct
  
} // ::cukd


  // ==================================================================
  // IMPLEMENTATION SECTION
  // ==================================================================

#include "traverse-default-stack-based.h"
#include "traverse-cct.h"
#include "traverse-stack-free.h"

namespace cukd {

  /*! helper struct to hold the current-best results of a fcp kernel during traversal */
  struct FCPResult {
    inline __device__ float initialCullDist2() const
    { return closestDist2; }
    
    inline __device__ float clear(float initialDist2)
    {
      closestDist2 = initialDist2;
      closestPrimID = -1;
      return closestDist2;
    }
    
    /*! process a new candidate with given ID and (square) distance;
      and return square distance to be used for subsequent
      queries */
    inline __device__ float processCandidate(int candPrimID, float candDist2)
    {
      if (candDist2 < closestDist2) {
        closestDist2 = candDist2;
        closestPrimID = candPrimID;
      }
      return closestDist2;
    }

    inline __device__ int returnValue() const
    { return closestPrimID; }
    
    int   closestPrimID;
    float closestDist2;
  };


  template<typename data_t,
           typename data_traits=default_data_traits<data_t>>
  inline __device__
  int cct::fcp(typename data_traits::point_t queryPoint,
               const box_t<typename data_traits::point_t> worldBounds,
               const data_t *d_nodes,
               int N,
               FcpSearchParams params)
  {
    FCPResult result;
    result.clear(sqr(params.cutOffRadius));
    traverse_cct<FCPResult,data_t,data_traits>
      (result,queryPoint,worldBounds,d_nodes,N);
    return result.returnValue();
  }

  template<typename data_t,
           typename data_traits=default_data_traits<data_t>>
  inline __device__
  int stackFree::fcp(typename data_traits::point_t queryPoint,
                     const data_t *d_nodes,
                     int N,
                     FcpSearchParams params)
  {
    FCPResult result;
    result.clear(sqr(params.cutOffRadius));
    traverse_stack_free<FCPResult,data_t,data_traits>
      (result,queryPoint,d_nodes,N);
    return result.returnValue();
  }

  template<typename data_t,
           typename data_traits=default_data_traits<data_t>>
  inline __device__
  int stackBased::fcp(typename data_traits::point_t queryPoint,
                      const data_t *d_nodes,
                      int N,
                      FcpSearchParams params)
  {
    FCPResult result;
    result.clear(sqr(params.cutOffRadius));
    traverse_default<FCPResult,data_t,data_traits>
      (result,queryPoint,d_nodes,N);
    return result.returnValue();
  }

  template<typename data_t,
           typename data_traits=default_data_traits<data_t>>
  inline __device__
  int cct::fcp(const SpatialKDTree<data_t,data_traits> &tree,
               typename data_traits::point_t queryPoint,
               FcpSearchParams params)
  {
    FCPResult result;
    result.clear(sqr(params.cutOffRadius));

    using node_t     = typename SpatialKDTree<data_t,data_traits>::Node;
    using point_t    = typename data_traits::point_t;
    using scalar_t   = typename scalar_type_of<point_t>::type;
    enum { num_dims  = num_dims_of<point_t>::value };
    
    scalar_t cullDist = result.initialCullDist2();

    /* can do at most 2**30 points... */
    struct StackEntry {
      int   nodeID;
      point_t closestCorner;
    };
    enum{ stack_depth = 50 };
    StackEntry stackBase[stack_depth];
    StackEntry *stackPtr = stackBase;

    int numSteps = 0;
    /*! current node in the tree we're traversing */
    int nodeID = 0;
    point_t closestPointOnSubtreeBounds = project(tree.bounds,queryPoint);
    if (sqrDistance(queryPoint,closestPointOnSubtreeBounds) > cullDist)
      return result.returnValue();
    node_t node;
    while (true) {
      while (true) {
        numSteps++;
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
          
        get_coord(farSideCorner,node.dim) = node.pos;

        const float farSideDist2 = sqrDistance(farSideCorner,queryPoint);
        if (farSideDist2 < cullDist) {
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
        auto dp = data_traits::get_point(tree.data[primID]);
          
        const auto sqrDist = sqrDistance(data_traits::get_point(tree.data[primID]),queryPoint);
        cullDist = result.processCandidate(primID,sqrDist);
      }
      
      while (true) {
        if (stackPtr == stackBase)  {
          return result.returnValue();
        }
        --stackPtr;
        closestPointOnSubtreeBounds = stackPtr->closestCorner;
        if (sqrDistance(closestPointOnSubtreeBounds,queryPoint) >= cullDist)
          continue;
        nodeID = stackPtr->nodeID;
        break;
      }
    }
  }

  template<typename data_t,
           typename data_traits=default_data_traits<data_t>>
  inline __device__
  int stackBased::fcp(const SpatialKDTree<data_t,data_traits> &tree,
                      typename data_traits::point_t queryPoint,
                      FcpSearchParams params = FcpSearchParams{})
  {
    FCPResult result;
    result.clear(sqr(params.cutOffRadius));

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
    int numSteps = 0;
    while (true) {
      while (true) {
        CUKD_STATS(if (cukd::g_traversalStats) ::atomicAdd(cukd::g_traversalStats,1));
        node = tree.nodes[nodeID];
        ++numSteps;
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
        const auto sqrDist = sqrDistance(data_traits::get_point(tree.data[primID]),queryPoint);
        CUKD_STATS(if (cukd::g_traversalStats) ::atomicAdd(cukd::g_traversalStats,1));
        cullDist = result.processCandidate(primID,sqrDist);
      }
      
      while (true) {
        if (stackPtr == stackBase)  {
          return result.returnValue();
        }
        --stackPtr;
        if (stackPtr->sqrDist >= cullDist)
          continue;
        nodeID = stackPtr->nodeID;
        break;
      }
    }
  }
// #endif
} // :: cukd
