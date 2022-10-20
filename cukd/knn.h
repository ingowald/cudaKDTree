// ======================================================================== //
// Copyright 2022-2022 Ingo Wald                                            //
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

#include "cukd/fcp.h"

namespace cukd {

  template<int k>
  struct FixedCandidateList
  {
    inline __device__ uint64_t encode(float f, int i)
    {
      return (uint64_t(__float_as_uint(f)) << 32) | uint32_t(i);
    }
    
    inline __device__ FixedCandidateList(float maxQueryDist)
    {
#pragma unroll
      for (int i=0;i<k;i++)
        entry[i] = encode(maxQueryDist*maxQueryDist,-1);
    }

    inline __device__ void push(float dist, int pointID)
    {
      uint64_t v = encode(dist,pointID);
#pragma unroll
      for (int i=0;i<k;i++) {
        uint64_t vmax = max(entry[i],v);
        uint64_t vmin = min(entry[i],v);
        entry[i] = vmin;
        v = vmax;
      }
    }

    inline __device__ float maxRadius2()
    { return decode_dist2(entry[k-1]); }

    inline __device__ float decode_dist2(uint64_t v)
    { return __uint_as_float(v >> 32); }
    inline __device__ int decode_pointID(uint64_t v)
    { return int(v); }

    uint64_t entry[k];
  };

  template<int k>
  struct HeapCandidateList
  {
    inline __device__ uint64_t encode(float f, int i)
    {
      return (uint64_t(__float_as_uint(f)) << 32) | uint32_t(i);
    }
    
    inline __device__ HeapCandidateList(float maxRange)
    {
#pragma unroll
      for (int i=0;i<k;i++)
        entry[i] = encode(maxRange*maxRange,-1);
    }

    inline __device__ void push(float dist, int pointID)
    {
      uint64_t e = encode(dist,pointID);
      if (e >= entry[0]) return;

      int pos = 0;
      while (true) {
        uint64_t largestChildValue = 0;//encode(CUDART_INF,-1);
        int firstChild = 2*pos+1;
        int largestChild = k;
        if (firstChild < k) {
          largestChild = firstChild;
          largestChildValue = entry[firstChild];
        }
        
        int secondChild = firstChild+1;
        if (secondChild < k && entry[secondChild] < largestChildValue) {
          largestChild = secondChild;
          largestChildValue = entry[secondChild];
        }

        if (e > largestChildValue) {
          entry[pos] = e;
          return;
        } else {
          entry[pos] = largestChildValue;
          pos = largestChild;
        }
      }
    }
    
    inline __device__ void maxQueryRange()
    { return decode_dist2(entry[0]); }
    
    inline __device__ int decode_dist2(uint64_t v)
    { return __uint_as_float(v >> 32); }
    inline __device__ int decode_pointID(uint64_t v)
    { return int(v); }

    uint64_t entry[k];
  };
  
  template<typename CandidateList>
  inline __device__
  float knn(CandidateList &currentlyClosest,
            float4 queryPoint,
            const float4 *d_nodes,
            int N)
  {
    float maxRadius2 = currentlyClosest.maxRadius2();

    int prev = -1;
    int curr = 0;

    // if (dbg)
    //   printf("starting query %f %f %f %f\n",
    //          queryPoint.x,
    //          queryPoint.y,
    //          queryPoint.z,
    //          queryPoint.w);
    
    while (true) {
      const int parent = (curr+1)/2-1;
      // if (dbg)
      //   printf("- at node %i, prev %i, parent %i\n",curr,prev,parent);
      if (curr >= N) {
        // in some (rare) cases it's possible that below traversal
        // logic will go to a "close child", but may actually only
        // have a far child. In that case it's easiest to fix this
        // right here, pretend we've done that (non-existent) close
        // child, and let parent pick up traversal as if it had been
        // done.
        prev = curr;
        curr = parent;
        // if (dbg)
        //   printf("==> NONEXISTENT NODE!\n");
        
        continue;
      }
      const int  child = 2*curr+1;
      const bool from_child = (prev >= child);
      if (!from_child) {
        float dist2 = sqr_distance(queryPoint,d_nodes[curr]);
        // if (dbg)
        //   printf(" ==> PROCESSING %i, dist = %f\n",curr,dist);
        if (dist2 <= maxRadius2) {
          currentlyClosest.push(dist2,curr);
          maxRadius2 = currentlyClosest.maxRadius2();
          // if (dbg)
          //   printf("### DID find closer point at dist %f\n",dist);
        }
      }

      const auto &curr_node = d_nodes[curr];
      const int   curr_dim = BinaryTree::levelOf(curr)%4;
      const float curr_dim_dist = (&queryPoint.x)[curr_dim] - (&curr_node.x)[curr_dim];
      const int   curr_side = curr_dim_dist > 0.f;
      const int   curr_close_child = 2*curr + 1 + curr_side;
      const int   curr_far_child   = 2*curr + 2 - curr_side;
      // if (dbg)
      //   printf("  qp %f plane %f -> children close %i far %i\n",
      //          (&queryPoint.x)[curr_dim],
      //          (&curr_node.x)[curr_dim],
      //          curr_close_child,curr_far_child);
      
      int next = -1;
      if (prev == curr_close_child)
        // if we came from the close child, we may still have to check
        // the far side - but only if this exists, and if far half of
        // current space if even within search radius.
        next
          = ((curr_far_child<N) && ((curr_dim_dist*curr_dim_dist) <= maxRadius2))
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
        return maxRadius2;
    
      prev = curr;
      curr = next;
    }
  }
  
} // ::cukd

