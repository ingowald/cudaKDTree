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

#include "cukd/helpers.h"

namespace cukd {

  /*! traverse k-d tree with default, stack-based (sb) traversal */
  template<typename result_t,
           typename node_t,
           typename node_traits=default_node_traits<node_t>>
  inline __device__
  void traverse_default(result_t &result,
                        typename node_traits::point_t queryPoint,
                        const node_t *d_nodes,
                        int numPoints)
  {
    using point_t  = typename node_traits::point_t;
    using scalar_t = typename scalar_type_of<point_t>::type;
    enum { num_dims = num_dims_of<point_t>::value };
    
    scalar_t cullDist = result.initialCullDist2();

    /* can do at most 2**30 points... */
    struct StackEntry {
      int   nodeID;
      float sqrDist;
    };
    StackEntry stackBase[30];
    StackEntry *stackPtr = stackBase;

    /*! current node in the tree we're traversing */
    int curr = 0;
    
    while (true) {
      while (curr < numPoints) {
        const int  curr_dim
          = node_traits::has_explicit_dim
          ? node_traits::get_dim(d_nodes[curr])
          : (BinaryTree::levelOf(curr) % num_dims);
        const node_t &curr_node  = d_nodes[curr];
        const auto sqrDist = sqrDistance(node_traits::get_point(curr_node),queryPoint);
        
        cullDist = result.processCandidate(curr,sqrDist);

        const auto node_coord   = node_traits::get_coord(curr_node,curr_dim);
        const auto query_coord  = get_coord(queryPoint,curr_dim);
        const bool  leftIsClose = query_coord < node_coord;
        const int   lChild = 2*curr+1;
        const int   rChild = lChild+1;

        const int closeChild = leftIsClose?lChild:rChild;
        const int farChild   = leftIsClose?rChild:lChild;
        
        const float sqrDistToPlane = sqr(query_coord - node_coord);
        if (sqrDistToPlane < cullDist && farChild < numPoints) {
          stackPtr->nodeID  = farChild;
          stackPtr->sqrDist = sqrDistToPlane;
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
