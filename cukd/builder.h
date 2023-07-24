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

/*

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
  - perf  10M float3s (4090) : ~4.3ms

 */

#if 0
/* this builder uses our tag-update algorithm, and uses thrust for
    sorting the tag:node pairs. This is our fastest builder, but has
    the downside that thrust's sort will not properly work in a
    stream, and will, in parituclar, have to allocate (quite a bit
    of!) temporary memory during sorting */
# include "cukd/builder_thrust.h"
#elif 1
/* this builder uses our tag-update algorithm, but uses bitonic sort
   instead of thrust for soring. it doesn't require thrust, and
   doesn't require additional memory other than 1 int for the tag, but
   for large arrays (10M-ish points) is about 2x slwoer than than the
   thrust variant */
# include "cukd/builder_bitonic.h"
#else
/* this is a _completely_ in-place builder; it will not allocate a
   single byte of additional memory during building (or at any other
   time); the downside is that for large array's it can be 10x-20x
   slower . For refernece: for 10M float3 poitns, builder_inplace
   takes about 4.3 seconds; builder_thrust will take about 200ms,
   builder_bitonic will take about 390ms */
# include "cukd/builder_inplace.h"
#endif
