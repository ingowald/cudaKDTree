#pragma once

namespace cukd {
  template <typename math_point_traits_t,
            typename node_point_traits_t,
            typename result_t>
  inline __device__
  void traverse_sb_imp(result_t &result,
                       unsigned long long *d_stats,
                       typename math_point_traits_t::point_t queryPoint,
                       const common::box_t<typename math_point_traits_t::point_t> d_bounds,
                       const typename node_point_traits_t::point_t *d_nodes,
                       int numPoints)
  {
    using scalar_t = typename math_point_traits_t::scalar_t;
    scalar_t cullDist = result.initialCullDist2();

    struct StackEntry {
      typename math_point_traits_t::point_t closestCorner;
      int          nodeID;
    };
    /* can do at most 2**30 points... */
    StackEntry  stackBase[30];
    StackEntry *stackPtr = stackBase;

    // int tid = threadIdx.x + blockIdx.x*blockDim.x;
    // bool dbg = 0;//tid == 73;
    
    int nodeID = 0;
    auto closestPointOnSubtreeBounds = project<math_point_traits_t>(d_bounds,queryPoint);
    if (sqrDistance<math_point_traits_t>(queryPoint,closestPointOnSubtreeBounds) > cullDist)
      return;

    // if (dbg) printf("-----------\nquery %f %f, close %f %f ...\n",queryPoint.x,queryPoint.y,
    //                 closestPointOnSubtreeBounds.x,
    //                 closestPointOnSubtreeBounds.y);
    while (true) {
      
      if (nodeID >= numPoints) {
        while (true) {
          if (stackPtr == stackBase)
            return;// closestID;
          --stackPtr;
          closestPointOnSubtreeBounds = stackPtr->closestCorner;
          if (sqrDistance<math_point_traits_t>(closestPointOnSubtreeBounds,queryPoint) >= cullDist)
            continue;
          nodeID = stackPtr->nodeID;
          break;
        }
      }
      if (d_stats)
        atomicAdd(d_stats,1);
      const auto node  = d_nodes[nodeID];
      {
        const auto sqrDist = sqrDistance<node_point_traits_t,math_point_traits_t>(node,queryPoint);
        // if (dbg)
        //   printf("node %i (%f %f) dist %f close %f %f\n",
        //          nodeID,
        //          node.x,
        //          node.y,
        //          sqrtf(sqrDist),
        //          closestPointOnSubtreeBounds.x,
        //          closestPointOnSubtreeBounds.y);
        cullDist = result.processCandidate(nodeID,sqrDist);
        // if (sqrDist < cullDist) {
        //   cullDist  = sqrDist;
        //   closestID = nodeID;
        // }
      }
        
      
      const int dim    = BinaryTree::levelOf(nodeID) % math_point_traits_t::numDims;
      const auto node_dim   = node_point_traits_t::getCoord(node,dim);
      const auto query_dim  = math_point_traits_t::getCoord(queryPoint,dim);
      const bool  leftIsClose = query_dim < node_dim;
      const int   lChild = 2*nodeID+1;
      const int   rChild = lChild+1;

      auto farSideCorner = closestPointOnSubtreeBounds;
      const int farChild = leftIsClose?rChild:lChild;
      math_point_traits_t::setCoord(farSideCorner,dim,node_dim);
      if (farChild < numPoints && sqrDistance<math_point_traits_t>(farSideCorner,queryPoint) < cullDist) {
        stackPtr->closestCorner = farSideCorner;
        stackPtr->nodeID = farChild;
        stackPtr++;
      }

      nodeID = leftIsClose?lChild:rChild;
    }
  }


}
