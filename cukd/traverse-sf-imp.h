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

#pragma once

namespace cukd {

  template<typename data_t,
           typename data_traits=default_data_traits<data_t>>
  inline __device__
  box_t<typename data_traits::point_t>
  recomputeBounds(int curr,
                  box_t<typename data_traits::point_t> bounds,
                  const data_t *d_nodes
                  )
  {
    using point_t  = typename data_traits::point_t;
    using scalar_t = typename scalar_type_of<point_t>::type;
    enum { num_dims = num_dims_of<point_t>::value };
    
    while (true) {
      if (curr == 0) break;
      const int parent = (curr+1)/2-1;

      const auto &parent_node = d_nodes[parent];
      CUKD_STATS(if (cukd::g_traversalStats) ::atomicAdd(cukd::g_traversalStats,1));
      const int   parent_dim
        = data_traits::has_explicit_dim
        ? data_traits::get_dim(parent_node)
        : (BinaryTree::levelOf(parent) % num_dims);
      const float parent_split_pos = data_traits::get_coord(parent_node,parent_dim);
      
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
    };
    return bounds;
  }
  
  template<typename result_t,
           typename data_t,
           typename data_traits=default_data_traits<data_t>>
  inline __device__
  void traverse_sf_imp(result_t &result,
                       typename data_traits::point_t queryPoint,
                       const box_t<typename data_traits::point_t> worldBounds,
                       const data_t *d_nodes,
                       int numPoints)
  {
    using point_t  = typename data_traits::point_t;
    using scalar_t = typename scalar_type_of<point_t>::type;
    enum { num_dims = num_dims_of<point_t>::value };

    float cullDist = result.initialCullDist2();
    
    
    int prev = -1;
    int curr = 0;

    box_t<point_t> bounds = worldBounds;
    
    while (true) {
      if (curr == -1)
        // this can only (and will) happen if and only if we come from a
        // child, arrive at the root, and decide to go to the parent of
        // the root ... while means we're done.
        return;// closest_found_so_far;

      bounds = recomputeBounds<data_t,data_traits>
        (curr,worldBounds,d_nodes);
      const int parent = (curr+1)/2-1;
      
      point_t closestPointOnSubtreeBounds = project(bounds,queryPoint);
      if (sqrDistance(closestPointOnSubtreeBounds,queryPoint) >= cullDist) {
        prev = curr;
        curr = parent;
        continue;
      }


      if (curr >= numPoints) {
        // in some (rare) cases it's possible that below traversal
        // logic will go to a "close child", but may actually only
        // have a far child. In that case it's easiest to fix this
        // right here, pretend we've done that (non-existent) close
        // child, and let parent pick up traversal as if it had been
        // done.
        prev = curr;
        curr = parent;

        continue;
      }
      CUKD_STATS(if (cukd::g_traversalStats) ::atomicAdd(cukd::g_traversalStats,1));
      const auto &curr_node = d_nodes[curr];
      const int  child = 2*curr+1;
      const bool from_child = (prev >= child);
      if (!from_child) {
        const auto dist_sqr =
          sqrDistance(queryPoint,data_traits::get_point(curr_node));
        cullDist = result.processCandidate(curr,dist_sqr);
      }

      const int  curr_dim
        = data_traits::has_explicit_dim
        ? data_traits::get_dim(d_nodes[curr])
        : (BinaryTree::levelOf(curr) % num_dims);
      const float curr_split_pos = data_traits::get_coord(curr_node,curr_dim);
      const float curr_dim_dist = get_coord(queryPoint,curr_dim) - curr_split_pos;
      const int   curr_side = curr_dim_dist > 0.f;
      const int   curr_close_child = 2*curr + 1 + curr_side;
      const int   curr_far_child   = 2*curr + 2 - curr_side;

      int next = -1;
      if (prev == curr_close_child) {
        // if we came from the close child, we may still have to check
        // the far side - but only if this exists, and if far half of
        // current space if even within search radius.
        if ((curr_far_child<numPoints)
            &&
            (curr_dim_dist * curr_dim_dist < cullDist)
            )
          {
            next = curr_far_child;
            if (curr_side == 1) {
              get_coord(bounds.lower,curr_dim) = curr_split_pos;
            } else {
              get_coord(bounds.upper,curr_dim) = curr_split_pos;
            }
          }
        else
          {
            next = parent;
          }
      } else if (prev == curr_far_child) {
        // if we did come from the far child, then both children are
        // done, and we can only go up.
        next = parent;
      } else {
        // we didn't come from any child, so must be coming from a
        // parent... we've already been processed ourselves just now,
        // so next stop is to look at the children (unless there
        // aren't any). this still leaves the case that we might have
        // a child, but only a far child, and this far child may or
        // may not be in range ... we'll fix that by just going to
        // near child _even if_ only the far child exists, and have
        // that child do a dummy traversal of that missing child, then
        // pick up on the far-child logic when we return.
        // next
        
        if (child < numPoints) {
          next = curr_close_child;
          if (curr_side == 1) {
            get_coord(bounds.upper,curr_dim) = curr_split_pos;
          } else {
            get_coord(bounds.lower,curr_dim) = curr_split_pos;
          }
        } else {
          next = parent;
        }
      }

      if (next == -1)
        // this can only (and will) happen if and only if we come from a
        // child, arrive at the root, and decide to go to the parent of
        // the root ... while means we're done.
        return;

      prev = curr;
      curr = next;
    }
  }
}
