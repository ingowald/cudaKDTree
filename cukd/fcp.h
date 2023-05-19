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

namespace cukd {

  template <typename scalar_t>
  inline __device__ __host__
  auto sqr(scalar_t f) { return f * f; }

  template <typename point_traits_a, typename point_traits_b=point_traits_a>
  inline __device__ __host__
  auto sqrDistance(const typename point_traits_a::point_t& a,
                   const typename point_traits_b::point_t& b)
  {
    typename point_traits_a::scalar_t res = 0;
#pragma unroll
    for(int i=0; i<min(point_traits_a::numDims, point_traits_b::numDims); ++i) {
      const auto diff = point_traits_a::getCoord(a, i) - point_traits_b::getCoord(b, i);
      res += sqr(diff);
    }
    return res;
  }

  // Structure of parameters to control the behavior of the FCP search.
  // By default FCP will perform an exact nearest neighbor search, but the
  // following parameters can be set to cut some corners and make the search
  // approximate in favor of speed.
  struct FcpSearchParams {
    // Controls how many "far branches" of the tree will be searched. If set to
    // 0 the algorithm will only go down the tree once following the nearest
    // branch each time.
    int far_node_inspect_budget = INT_MAX;

    // Controls when to go down the far branch: only follow a far branch if
    // (1+eps) * D is within the search radius, where D is the distance to the
    // far node. Similar to FLANN eps parameter.
    float eps = 0.f;

    // Controls when to go down the far branch: only go down the far branch if
    // the distance to the far node is larger than this search radius.
    float max_far_node_search_radius = 1e9f;
  };

  template<
    typename math_point_traits_t,
    typename node_point_traits_t=math_point_traits_t>
  inline __device__
  int fcp(typename math_point_traits_t::point_t queryPoint,
          const typename node_point_traits_t::point_t *d_nodes,
          int N,
          FcpSearchParams params = FcpSearchParams{})
  {
    using scalar_t = typename math_point_traits_t::scalar_t;
    const auto max_far_node_search_radius_sqr
      = params.max_far_node_search_radius
      * params.max_far_node_search_radius;
    const auto epsErr = 1 + params.eps;

    int   closest_found_so_far = -1;
    float closest_dist_sqr_found_so_far = CUDART_INF;

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
      const auto &curr_node = d_nodes[curr];
      const int  child = 2*curr+1;
      const bool from_child = (prev >= child);
      if (!from_child) {
        const auto dist_sqr =
          sqrDistance<math_point_traits_t,node_point_traits_t>(queryPoint,curr_node);
        if (dist_sqr < closest_dist_sqr_found_so_far) {
          closest_dist_sqr_found_so_far = dist_sqr;
          closest_found_so_far          = curr;
        }
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
          = ((curr_far_child<N) && ((curr_dim_dist * curr_dim_dist) * epsErr < min(max_far_node_search_radius_sqr, closest_dist_sqr_found_so_far)) && (--params.far_node_inspect_budget>=0))
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
        return closest_found_so_far;

      prev = curr;
      curr = next;
    }
  }

  /*1 project a point into a boundinx box */
  template <typename math_point_traits_t, typename node_point_traits_t>
  inline __device__
  int fcp(typename math_point_traits_t::point_t queryPoint,
          const common::box_t<typename math_point_traits_t::point_t> *d_bounds,
          const typename node_point_traits_t::point_t *d_nodes,
          int numPoints,
          FcpSearchParams params = FcpSearchParams{})
  {
    using scalar_t = typename math_point_traits_t::scalar_t;
    scalar_t cullDist = sqr(params.max_far_node_search_radius);
    int   closestID = -1;

    struct StackEntry {
      typename math_point_traits_t::point_t closestCorner;
      int          nodeID;
    };
    /* can do at most 2**30 points... */
    StackEntry  stackBase[30];
    StackEntry *stackPtr = stackBase;

    int nodeID = 0;
    auto closestPointOnSubtreeBounds = project<math_point_traits_t>(*d_bounds,queryPoint);
    if (sqrDistance<math_point_traits_t>(queryPoint,closestPointOnSubtreeBounds) > cullDist)
      return closestID;


    while (true) {
      if (nodeID >= numPoints) {
        while (true) {
          if (stackPtr == stackBase)
            return closestID;
          --stackPtr;
          closestPointOnSubtreeBounds = stackPtr->closestCorner;
          if (sqrDistance<math_point_traits_t>(closestPointOnSubtreeBounds,queryPoint) > cullDist)
            continue;
          nodeID = stackPtr->nodeID;
          break;
        }
      }
      const int dim    = BinaryTree::levelOf(nodeID) % math_point_traits_t::numDims;
      const auto node  = d_nodes[nodeID];
      const auto sqrDist = sqrDistance<node_point_traits_t,math_point_traits_t>(node,queryPoint);
      if (sqrDist < cullDist) {
        cullDist  = sqrDist;
        closestID = nodeID;
      }

      const auto node_dim   = node_point_traits_t::getCoord(node,dim);
      const auto query_dim  = math_point_traits_t::getCoord(queryPoint,dim);
      const bool  leftIsClose = query_dim < node_dim;
      const int   lChild = 2*nodeID+1;
      const int   rChild = lChild+1;

      auto farSideCorner = closestPointOnSubtreeBounds;
      const int farChild = leftIsClose?rChild:lChild;
      math_point_traits_t::setCoord(farSideCorner,dim,node_dim);
      if (farChild < numPoints && sqrDistance<math_point_traits_t>(farSideCorner,queryPoint) <= cullDist) {
        stackPtr->closestCorner = farSideCorner;
        stackPtr->nodeID = farChild;
        stackPtr++;
      }

      nodeID = leftIsClose?lChild:rChild;
    }
  }
} // ::cukd

