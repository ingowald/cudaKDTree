#pragma once

namespace cukd {
  template <typename math_point_traits_t,
            typename node_point_traits_t,
            typename result_t>
  inline __device__
  void traverse_sb_reg(result_t &result,
                       unsigned long long *d_stats,
                       typename math_point_traits_t::point_t queryPoint,
                       // const common::box_t<typename math_point_traits_t::point_t> *d_bounds,
                       const typename node_point_traits_t::point_t *d_nodes,
                       int numPoints)
  {
    using scalar_t = typename math_point_traits_t::scalar_t;
    scalar_t cullDist = result.initialCullDist2();
    // scalar_t cullDist = sqr(params.max_far_node_search_radius);
    // int   closestID = -1;

    /* can do at most 2**30 points... */
    struct StackEntry {
      int   nodeID;
      float sqrDist;
    };
    StackEntry stackBase[30];
    StackEntry *stackPtr = stackBase;

    int curr = 0;
    
    // int tid = threadIdx.x + blockIdx.x*blockDim.x;
    // bool dbg = 0;//tid == 73;
    
    // if (dbg) printf("-----------\nquery ...\n");
    while (true) {
      while (curr < numPoints) {
        if (d_stats)
          atomicAdd(d_stats,1ull);
        const int  curr_dim    = BinaryTree::levelOf(curr) % math_point_traits_t::numDims;
        const auto curr_node  = d_nodes[curr];
        const auto sqrDist = sqrDistance<node_point_traits_t,math_point_traits_t>(curr_node,queryPoint);
        // if (dbg)
        //   printf("node %i (%f %f) dist %f\n",
        //          curr,
        //          curr_node.x,
        //          curr_node.y,
        //          sqrtf(sqrDist));
      

        // if (sqrDist < cullDist) {
        //   cullDist  = sqrDist;
        //   closestID = curr;
        // }
        cullDist = result.processCandidate(curr,sqrDist);

        const auto node_dim   = node_point_traits_t::getCoord(curr_node,curr_dim);
        const auto query_dim  = math_point_traits_t::getCoord(queryPoint,curr_dim);
        const bool  leftIsClose = query_dim < node_dim;
        const int   lChild = 2*curr+1;
        const int   rChild = lChild+1;

        const int closeChild = leftIsClose?lChild:rChild;
        const int farChild   = leftIsClose?rChild:lChild;
        
        const float curr_dim_dist = query_dim - node_dim;//(&queryPoint.x)[curr_dim] - (&curr_node.x)[curr_dim];

        
        if (curr_dim_dist*curr_dim_dist <= cullDist && farChild < numPoints) {
          stackPtr->nodeID  = farChild;
          stackPtr->sqrDist = curr_dim_dist*curr_dim_dist;
          ++stackPtr;
        }
        curr = closeChild;
      }

      while (true) {
        if (stackPtr == stackBase) 
          return;
        --stackPtr;
        if (stackPtr->sqrDist >= cullDist)
          continue;
        curr = stackPtr->nodeID;
        break;
      }
    }
  }
  
}
