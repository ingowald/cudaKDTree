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

  inline __device__ __host__
  float dot(float4 a, float4 b) { return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w; }

  inline __device__ __host__
  float4 sub(float4 a, float4 b) { return make_float4(a.x-b.x,a.y-b.y,a.z-b.z,a.w-b.w); }

  inline __host__ __device__
  float sqr_distance(float4 a, float4 b)
  {
    return dot(sub(a,b),sub(a,b));
  }

  inline __host__ __device__
  float distance(float4 a, float4 b)
  {
    return sqrtf(sqr_distance(a,b));
  }

  inline __device__ __host__
  float dot(float3 a, float3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }

  inline __device__ __host__
  float3 sub(float3 a, float3 b) { return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }

  inline __host__ __device__
  float sqr_distance(float3 a, float3 b)
  {
    return dot(sub(a,b),sub(a,b));
  }

  inline __host__ __device__
  float distance(float3 a, float3 b)
  {
    return sqrtf(sqr_distance(a,b));
  }

  template<
    typename query_point_t = float4,
    typename node_point_t = float4,
    typename scalar_t = float,
    typename QueryPointInterface = common::TrivialPointInterface<query_point_t,scalar_t>,
    typename NodePointInterface = common::TrivialPointInterface<node_point_t,scalar_t>>
  inline __device__
  int fcp(query_point_t queryPoint,
          const node_point_t *d_nodes,
          int N)
  {
    int   closest_found_so_far = -1;
    float closest_dist_found_so_far = CUDART_INF;

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
      const int  child = 2*curr+1;
      const bool from_child = (prev >= child);
      if (!from_child) {
        query_point_t nodePosition
          = common::extractPosition<query_point_t,node_point_t,scalar_t,
                                    QueryPointInterface,NodePointInterface>(d_nodes[curr]);
        float dist = distance(queryPoint,nodePosition);
        if (dist < closest_dist_found_so_far) {
          closest_dist_found_so_far = dist;
          closest_found_so_far      = curr;
        }
      }

      const auto &curr_node = d_nodes[curr];
      const int   curr_dim = BinaryTree::levelOf(curr) % common::point_traits<query_point_t>::numDims;
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
          = ((curr_far_child<N) && (fabsf(curr_dim_dist) < closest_dist_found_so_far))
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

} // ::cukd

