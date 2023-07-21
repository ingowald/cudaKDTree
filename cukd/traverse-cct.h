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

/* traversal with 'closest-corner-tracking' - somewhat better for some
   input distributions, by tracking the (N-dimensional) closest point
   in the given subtree's domain, rather than just always comparing
   only to the 1-dimensoinal plane */
#pragma once

namespace cukd {
//   template<int numBytes>
//   struct auto_align {
//     enum { value
//            = ((numBytes % 16) == 0) ? 16
//            : (((numBytes % 8) == 0) ? 8 : 4) };
//   };

// #define _MAX(a,b) ((a) > (b) ? (a) : (b))
  
  template<typename result_t,
           typename node_t,
           typename node_traits=default_node_traits<node_t>>
  inline __device__
  void traverse_cct(result_t &result,
                    CUKD_STATS_ARG(unsigned long long *d_stats,)
                    typename node_traits::point_t queryPoint,
                    const box_t<typename node_traits::point_t> d_bounds,
                    const node_t *d_nodes,
                    int numPoints)
  {
    using point_t  = typename node_traits::point_t;
    using scalar_t = typename scalar_type_of<point_t>::type;
    enum { num_dims = num_dims_of<point_t>::value };
    
    scalar_t cullDist = result.initialCullDist2();

    struct// __align__(_MAX(alignof(point_t),auto_align<sizeof(int)+sizeof(point_t)>::value))
      StackEntry {
      int     nodeID;
      point_t closestCorner;
    };
    /* can do at most 2**30 points... */
    StackEntry  stackBase[30];
    StackEntry *stackPtr = stackBase;

    int nodeID = 0;
    point_t closestPointOnSubtreeBounds = project(d_bounds,queryPoint);
    if (sqrDistance(queryPoint,closestPointOnSubtreeBounds) > cullDist)
      return;

    while (true) {
      
      if (nodeID >= numPoints) {
        while (true) {
          if (stackPtr == stackBase)
            return;
          --stackPtr;
          closestPointOnSubtreeBounds = stackPtr->closestCorner;
          if (sqrDistance(closestPointOnSubtreeBounds,queryPoint) >= cullDist)
            continue;
          nodeID = stackPtr->nodeID;
          break;
        }
      }
      CUKD_STATS(if (d_stats)
                   atomicAdd(d_stats,1));
      const auto &node  = d_nodes[nodeID];
      const point_t nodePoint = node_traits::get_point(node);
      {
        const auto sqrDist = sqrDistance(nodePoint,queryPoint);
        cullDist = result.processCandidate(nodeID,sqrDist);
      }
      
      const int  dim
        = node_traits::has_explicit_dim
        ? node_traits::get_dim(d_nodes[nodeID])
        : (BinaryTree::levelOf(nodeID) % num_dims);
      const auto node_dim   = get_coord(nodePoint,dim);
      const auto query_dim  = get_coord(queryPoint,dim);
      const bool  leftIsClose = query_dim < node_dim;
      const int   lChild = 2*nodeID+1;
      const int   rChild = lChild+1;

      auto farSideCorner = closestPointOnSubtreeBounds;
      const int farChild = leftIsClose?rChild:lChild;
      get_coord(farSideCorner,dim) = node_dim;
      // set_coord(farSideCorner,dim,node_dim);
      if (farChild < numPoints && sqrDistance(farSideCorner,queryPoint) < cullDist) {
        stackPtr->closestCorner = farSideCorner;
        stackPtr->nodeID = farChild;
        stackPtr++;
      }

      nodeID = leftIsClose?lChild:rChild;
    }
  }


}
