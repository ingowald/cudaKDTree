#pragma once

namespace cukd {

  template<typename node_t,
           typename node_traits=default_node_traits<node_t>>
  inline __device__
  box_t<typename node_traits::point_t>
  recomputeBounds(int curr,
                  box_t<typename node_traits::point_t> bounds,
                  const node_t *d_nodes
                  )
  {
    using point_t  = typename node_traits::point_t;
    using scalar_t = typename scalar_type_of<point_t>::type;
    enum { num_dims = num_dims_of<point_t>::value };
    
    while (true) {
      if (curr == 0) break;
      const int parent = (curr+1)/2-1;

      const auto &parent_node = d_nodes[parent];
      const int   parent_dim
        = node_traits::has_explicit_dim
        ? node_traits::get_dim(parent_node)
        : (BinaryTree::levelOf(parent) % num_dims);
      const float parent_split_pos = node_traits::get_coord(parent_node,parent_dim);
      
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
           typename node_t,
           typename node_traits=default_node_traits<node_t>>
  inline __device__
  void traverse_sf_imp(result_t &result,
                       typename node_traits::point_t queryPoint,
                       const box_t<typename node_traits::point_t> worldBounds,
                       const node_t *d_nodes,
                       int numPoints)
  {
    using point_t  = typename node_traits::point_t;
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

      bounds = recomputeBounds<node_t,node_traits>
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
      const auto &curr_node = d_nodes[curr];
      const int  child = 2*curr+1;
      const bool from_child = (prev >= child);
      if (!from_child) {
        const auto dist_sqr =
          sqrDistance(queryPoint,node_traits::get_point(curr_node));
        cullDist = result.processCandidate(curr,dist_sqr);
        // if (dist_sqr < cullDist) {
        //   cullDist = dist_sqr;
        //   closest_found_so_far          = curr;
        // }
      }

      const int  curr_dim
        = node_traits::has_explicit_dim
        ? node_traits::get_dim(d_nodes[curr])
        : (BinaryTree::levelOf(curr) % num_dims);
      const float curr_split_pos = node_traits::get_coord(curr_node,curr_dim);
      const float curr_dim_dist = get_coord(queryPoint,curr_dim) - curr_split_pos;
      const int   curr_side = curr_dim_dist > 0.f;
      const int   curr_close_child = 2*curr + 1 + curr_side;
      const int   curr_far_child   = 2*curr + 2 - curr_side;

      int next = -1;
      if (prev == curr_close_child) {
        // if we came from the close child, we may still have to check
        // the far side - but only if this exists, and if far half of
        // current space if even within search radius.
        // next
        //   = ((curr_far_child<N) && ((curr_dim_dist * curr_dim_dist) * epsErr < min(max_far_node_search_radius_sqr, cullDist)) && (--params.far_node_inspect_budget>=0))
        //   ? curr_far_child
        //   : parent;

        if ((curr_far_child<numPoints)
            &&
            (curr_dim_dist * curr_dim_dist < cullDist)
            // && (--params.far_node_inspect_budget>=0))
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
