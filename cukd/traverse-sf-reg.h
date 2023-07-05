#pragma once

namespace cukd {

  template<
    typename math_point_traits_t,
    typename node_point_traits_t,
    typename result_t>
    // typename math_point_traits_t,
    // typename node_point_traits_t=math_point_traits_t>
  inline __device__
  void traverse_sf_reg(result_t &result,
                      unsigned long long *d_stats,
                      typename math_point_traits_t::point_t queryPoint,
                      const typename node_point_traits_t::point_t *d_nodes,
                      int N)
  {
    using scalar_t = typename math_point_traits_t::scalar_t;
    scalar_t cullDist = result.initialCullDist2();
    // const auto max_far_node_search_radius_sqr
    //   = params.max_far_node_search_radius
    //   * params.max_far_node_search_radius;
    // const auto epsErr = 1 + params.eps;

    // int   closest_found_so_far = -1;
    // float closest_sqrDist_found_so_far = CUDART_INF;

    int prev = -1;
    int curr = 0;

    while (true) {
      const int parent = (curr+1)/2-1;
      if (curr >= N) {
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
      if (d_stats)
        atomicAdd(d_stats,1ull);
      const auto &curr_node = d_nodes[curr];
      const int  child = 2*curr+1;
      const bool from_child = (prev >= child);
      if (!from_child) {
        const auto sqrDist =
          sqrDistance<math_point_traits_t,node_point_traits_t>(queryPoint,curr_node);
        cullDist = result.processCandidate(curr,sqrDist);
        
        // if (sqrDist < closest_sqrDist_found_so_far) {
        //   closest_sqrDist_found_so_far = sqrDist;
        //   closest_found_so_far          = curr;
        // }
      }

      const int   curr_dim = BinaryTree::levelOf(curr) % math_point_traits_t::numDims;
      const float curr_dim_dist = (&queryPoint.x)[curr_dim] - (&curr_node.x)[curr_dim];
      const int   curr_side = curr_dim_dist > 0.f;
      const int   curr_close_child = 2*curr + 1 + curr_side;
      const int   curr_far_child   = 2*curr + 2 - curr_side;

      int next = -1;
      if (prev == curr_close_child)
        // if we came from the close child, we may still have to check
        // the far side - but only if this exists, and if far half of
        // current space if even within search radius.
        next
          = ((curr_far_child<N) && (curr_dim_dist * curr_dim_dist < cullDist))
          ? curr_far_child
          : parent;
      else if (prev == curr_far_child)
        // if we did come from the far child, then both children are
        // done, and we can only go up.
        next = parent;
      else
        // we didn't come from any child, so must be coming from a
        // parent... we've already been processed ourselves just now,
        // so next stop is to look at the children (unless there
        // aren't any). this still leaves the case that we might have
        // a child, but only a far child, and this far child may or
        // may not be in range ... we'll fix that by just going to
        // near child _even if_ only the far child exists, and have
        // that child do a dummy traversal of that missing child, then
        // pick up on the far-child logic when we return.
        next
          = (child<N)
          ? curr_close_child
          : parent;

      if (next == -1)
        // if (curr == 0 && from_child)
        // this can only (and will) happen if and only if we come from a
        // child, arrive at the root, and decide to go to the parent of
        // the root ... while means we're done.
        return;

      prev = curr;
      curr = next;
    }
  }


}
