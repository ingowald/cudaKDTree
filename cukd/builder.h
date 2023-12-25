// ======================================================================== //
// Copyright 2019-2023 Ingo Wald                                            //
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
#include "cukd/box.h"
#include <cuda.h>

/* This is a single include file from which

  Builder variants "cheat sheet"

  builder_thrust:
  - temporary memory overhead for N points: N ints + order 2N points 
    (ie, total mem order 3x that of input data!)
  - perf 100K float3s (4090) :   ~4ms
  - perf   1M float3s (4090) :  ~20ms
  - perf  10M float3s (4090) : ~200ms
  
  builder_bitonic:
  - temporary memory overhead for N points: N ints 
    (ie, ca 30% mem overhead for float3)
  - perf 100K float3s (4090) :  ~10ms
  - perf   1M float3s (4090) :  ~27ms
  - perf  10M float3s (4090) : ~390ms

  builder_inplace:
  - temporary memory overhead for N points: nada, nil, zilch.
  - perf 100K float3s (4090) :  ~10ms
  - perf   1M float3s (4090) : ~220ms
  - perf  10M float3s (4090) : ~4.3s

 */

#include "cukd/builder_thrust.h"
#include "cukd/builder_bitonic.h"
#include "cukd/builder_inplace.h"

namespace cukd {
  /*! Builds a left-balanced k-d tree over the given data points,
    using data_traits to describe the type of data points that this
    tree is being built over (i.e., how to separate a data item's
    positional coordinates from any potential payload (if such exists,
    e.g., in a 'photon' in photon mapping), what vector/point type to
    use for this coordinate data (e.g., float3), whether the data have
    a field to store an explicit split dimensional (for Bentley and
    Samet's 'optimized' trees, etc.

    Since a (point-)k-d tree's tree topology is implicit in the
    ordering of its data points this will re-arrange the data points
    to fulfill the balanced k-d tree criterion - ie, this WILL modify
    the data array: no individual entry will get changed, but their
    order might. If data_traits::has_explicit_dims is defined this
    builder will choose each node's split dimension based on the
    widest dimension of that node's subtree's domain; if not, it will
    chose the dimension in a round-robin style, where the root level
    is split along the 'x' coordinate, the next level in y, etc

    'worldBounds' is a pointer to device-writeable memory to store the
    world-space bounding box of the data points that the builder will
    compute. If data_traits::has_explicit_dims is true this memory
    _has_ to be provided to the builder, and the builder will fill it
    in; if data_traits::has_explicit_dims is false, this memory region
    is optional: the builder _will_ fill it in if provided, but will
    ignore it if isn't.

    *** Example 1: To build a 2D k-dtree over a CUDA int2 type (no other
    payload than the two coordinates):
      
    buildTree<int2>(....);

    In this case no data_traits need to be supplied beause these will
    be auto-computed for simple cuda vector types.
      
    *** Example 2: to build a 2D kd-tree over a data type of float4,
    where the first 2 coordinates of each point is the dimension we
    want to build the kd-tree over, and the other 2 coordinates
    are arbitrary other payload data:
      
    struct float2_plus_payload_traits {
       using point_t = float2;
       static inline __both__ point_t get_point(const float4 &n)
       { return make_float2(n.x, n.y); }
    };
    buildTree<float4,float2_plus_payload_traits>(...);
      
    *** Example 3: assuming you have a data type 'Photon' and a
    Photon_traits has Photon_traits::has_explicit_dim defined:
      
    cukd::box_t<float3> *d_worldBounds = <cudaMalloc>;
    buildTree<Photon,Photon_traits>(..., worldBounds, ...);
      
  */
  template<typename data_t, typename data_traits=default_data_traits<data_t>>
  void buildTree(/*! device-read/writeable array of data points */
                 data_t *d_points,
                 /*! number of data points */
                 int numPoints,
                 /*! device-writeable pointer to store the world-space
                     bounding box of all data points. if
                     data_traits::has_explicit_dim is false, this is
                     optionally allowed to be null */
                 box_t<typename data_traits::point_t> *worldBounds=0,
                 /*! cuda stream to use for all kernels and mallocs
                     (the builder_thrust may _also_ do some global
                     device syncs) */
                 cudaStream_t stream=0,
                 GpuMemoryResource &memResource=defaultGpuMemResource())
  {
#if defined(CUKD_BUILDER_INPLACE)
/* this is a _completely_ in-place builder; it will not allocate a
   single byte of additional memory during building (or at any other
   time); the downside is that for large array's it can be 10x-20x
   slower . For refernece: for 10M float3 poitns, builder_inplace
   takes about 4.3 seconds; builder_thrust will take about 200ms,
   builder_bitonic will take about 390ms */
    buildTree_inPlace<data_t,data_traits>
      (d_points,numPoints,worldBounds,stream,memResource);

#elif defined(CUKD_BUILDER_BITONIC)
/* this builder uses our tag-update algorithm, but uses bitonic sort
   instead of thrust for soring. it doesn't require thrust, and
   doesn't require additional memory other than 1 int for the tag, but
   for large arrays (10M-ish points) is about 2x slwoer than than the
   thrust variant */
    buildTree_bitonic<data_t,data_traits>
      (d_points,numPoints,worldBounds,stream,memResource);
#else
/* this builder uses our tag-update algorithm, and uses thrust for
    sorting the tag:node pairs. This is our fastest builder, but has
    the downside that thrust's sort will not properly work in a
    stream, and will, in parituclar, have to allocate (quite a bit
    of!) temporary memory during sorting */
    buildTree_thrust<data_t,data_traits>
      (d_points,numPoints,worldBounds,stream,memResource);
#endif
  }
}

