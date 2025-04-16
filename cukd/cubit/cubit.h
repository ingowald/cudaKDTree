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

/*! this file is from https://github.com/ingowald/cudaBitonic

  It implements a bitonic sorter that, unlike most other sample codes
  for this algorithm, can also do non-power of two sizes of data, as
  well as both key and key-value sorts. it is templated over both key
  and (where appropriate) value type, so should work for a wide range
  of types; all it requires is that there's a operator< defined for
  the key_t. 

  Memory usage: Unlike most other sort algorithms bitonic sorting (and
  this implementation thereof) operates exclusively by swapping
  elements; consequently this memory requires exactly zero bytes of
  additionsal "temporary" memory.

  Note re performance: this implementation aims as effectively using
  shared memory whereever it can; however; however, for large data
  this algorithm will require a fair amount of meomry
  bandwidth. Except it to be about roughly as fast as cub::Sort and
  cub::SortPairs for small to modest-sized data sets (say, <=100k),
  but for larger data sets you will see performance drop to about 5x
  to 10x less than that of the cub radix sort variants
*/

#pragma once

#include "../cubit/common.h"
#include <cuda_runtime.h>

namespace cubit {
  using namespace cubit::common;
  
  enum { block_size = 1024 };

  template<typename key_t>
  inline static __host__ __device__ void shm_sort(bool  *const __restrict__ valid,
                                         key_t *const __restrict__ keys,
                                         uint32_t a,
                                         uint32_t b)
  {
    key_t key_a = keys[a];
    key_t key_b = keys[b];
    const bool keepAsIs
      = valid[a] & (!valid[b] | (key_a < key_b));
      // = valid[a] && (!valid[b] || (key_a < key_b));
    keys[a] = (keepAsIs)?key_a:key_b;
    keys[b] = (keepAsIs)?key_b:key_a;
  }

  template<typename key_t, typename val_t>
  inline static __host__ __device__ void shm_sort(bool  *const __restrict__ valid,
                                         key_t *const __restrict__ keys,
                                         val_t *const __restrict__ vals,
                                         uint32_t a,
                                         uint32_t b)
  {
    key_t key_a = keys[a];
    key_t key_b = keys[b];
    val_t val_a = vals[a];
    val_t val_b = vals[b];
    const bool keepAsIs
      = valid[a] & (!valid[b] | (key_a < key_b));
      // = valid[a] && (!valid[b] || (key_a < key_b));
    keys[a] = (keepAsIs)?key_a:key_b;
    keys[b] = (keepAsIs)?key_b:key_a;
    vals[a] = (keepAsIs)?val_a:val_b;
    vals[b] = (keepAsIs)?val_b:val_a;
  }

  template<typename key_t>
  inline static __host__ __device__ void gmem_sort(key_t *const keys,
                                          uint32_t a,
                                          uint32_t b)
  {
    key_t key_a = keys[a];
    key_t key_b = keys[b];
    if (key_b < key_a) {
      keys[a] = key_b;
      keys[b] = key_a;
    }
  }
  
  template<typename key_t, typename val_t>
  inline static __host__ __device__ void gmem_sort(key_t *const keys,
                                          val_t *const vals,
                                          uint32_t a,
                                          uint32_t b)
  {
    key_t key_a = keys[a];
    key_t key_b = keys[b];
    if (key_b < key_a) {
      keys[a] = key_b;
      keys[b] = key_a;

      val_t val_a = vals[a];
      val_t val_b = vals[b];
      vals[a] = val_b;
      vals[b] = val_a;
    }
  }
  

  template<typename key_t>
  __global__ void block_sort_up(key_t *const __restrict__ g_keys, uint32_t _N)
  {
    __shared__ key_t keys[2*1024];
    __shared__ bool  valid[2*1024];
    uint32_t blockStart = blockIdx.x*(2*1024);
    if (blockStart+threadIdx.x < _N) {
      keys [threadIdx.x] = g_keys[blockStart+threadIdx.x];
      valid[threadIdx.x] = true;
    } else {
      valid[threadIdx.x] = false;
    }
    if (1024+blockStart+threadIdx.x < _N) {
      keys [1024+threadIdx.x] = g_keys[1024+blockStart+threadIdx.x];
      valid[1024+threadIdx.x] = true;
    } else {
      keys [1024+threadIdx.x] = key_t(0);// doesn't matter, just to make sure it's initialized
      valid[1024+threadIdx.x] = false;
    }
    __syncthreads();
    
    int l, r, s;
    // ======== seq size 1 ==========
    {
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,l,r);
    }

    // ======== seq size 2 ==========
    {
      s    = (int)threadIdx.x & -2;
      l    = threadIdx.x+s;
      r    = l ^ (4-1);
      shm_sort(valid,keys,l,r);
      
      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,l,r);
    }

    // ======== seq size 4 ==========
    {
      s    = (int)threadIdx.x & -4;
      l    = threadIdx.x+s;
      r    = l ^ (8-1);
      shm_sort(valid,keys,l,r);
      
      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,l,r);
    }

    // ======== seq size 8 ==========
    {
      s    = (int)threadIdx.x & -8;
      l    = threadIdx.x+s;
      r    = l ^ (16-1);
      shm_sort(valid,keys,l,r);
      
      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,l,r);
    }

    // ======== seq size 16 ==========
    {
      // __syncthreads();
      s    = (int)threadIdx.x & -16;
      l    = threadIdx.x+s;
      r    = l ^ (32-1);
      shm_sort(valid,keys,l,r);
      
      // ------ down seq size 8 ---------
      // __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,l,r);
    }

    // ======== seq size 32 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -32;
      l    = threadIdx.x+s;
      r    = l ^ (64-1);
      shm_sort(valid,keys,l,r);
      __syncthreads();
      
      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 8 ---------
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,l,r);
    }

    // ======== seq size 64 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -64;
      l    = threadIdx.x+s;
      r    = l ^ (128-1);
      shm_sort(valid,keys,l,r);
      __syncthreads();
      
      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(valid,keys,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 8 ---------
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,l,r);
    }

    // ======== seq size 128 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -128;
      l    = threadIdx.x+s;
      r    = l ^ (256-1);
      shm_sort(valid,keys,l,r);
      __syncthreads();
      
      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(valid,keys,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(valid,keys,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,l,r);
    }

    // ======== seq size 256 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -256;
      l    = threadIdx.x+s;
      r    = l ^ (512-1);
      shm_sort(valid,keys,l,r);
      __syncthreads();
      
      // ------ down seq size 128 ---------
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      shm_sort(valid,keys,l,r);
      __syncthreads();

      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(valid,keys,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(valid,keys,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,l,r);
    }

    // ======== seq size 512 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -512;
      l    = threadIdx.x+s;
      r    = l ^ (1024-1);
      shm_sort(valid,keys,l,r);
      __syncthreads();
      
      // ------ down seq size 256 ---------
      l = threadIdx.x+((int)threadIdx.x & -256);
      r = l + 256;
      shm_sort(valid,keys,l,r);
      __syncthreads();
      
      // ------ down seq size 128 ---------
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      shm_sort(valid,keys,l,r);
      __syncthreads();

      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(valid,keys,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(valid,keys,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,l,r);
    }

    // ======== seq size 1024 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -1024;
      l    = threadIdx.x+s;
      r    = l ^ (2048-1);
      shm_sort(valid,keys,l,r);
      __syncthreads();
      
      // ------ down seq size 512 ---------
      l = threadIdx.x+((int)threadIdx.x & -512);
      r = l + 512;
      shm_sort(valid,keys,l,r);
      __syncthreads();
      
      // ------ down seq size 256 ---------
      l = threadIdx.x+((int)threadIdx.x & -256);
      r = l + 256;
      shm_sort(valid,keys,l,r);
      __syncthreads();
      
      // ------ down seq size 128 ---------
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      shm_sort(valid,keys,l,r);
      __syncthreads();

      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(valid,keys,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(valid,keys,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 8 ---------
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,l,r);
    }

    __syncthreads();
    if (blockStart+threadIdx.x < _N) g_keys[blockStart+threadIdx.x] = keys[threadIdx.x];
    if (1024+blockStart+threadIdx.x < _N) g_keys[1024+blockStart+threadIdx.x] = keys[1024+threadIdx.x];
  }




  template<typename key_t, typename val_t>
  __global__ void block_sort_up(key_t *const __restrict__ g_keys,
                                val_t *const __restrict__ g_vals,
                                uint32_t _N)
  {
    __shared__ key_t keys[2*1024];
    __shared__ val_t vals[2*1024];
    __shared__ bool  valid[2*1024];
    uint32_t blockStart = blockIdx.x*(2*1024);
    if (blockStart+threadIdx.x < _N) {
      keys [threadIdx.x] = g_keys[blockStart+threadIdx.x];
      vals [threadIdx.x] = g_vals[blockStart+threadIdx.x];
      valid[threadIdx.x] = true;
    } else {
      // keys[threadIdx.x] = max_value<key_t>();
      keys [threadIdx.x] = key_t(0);// doesn't matter, just to make sure it's initialized
      valid[threadIdx.x] = false;
    }
    if (1024+blockStart+threadIdx.x < _N) {
      keys [1024+threadIdx.x] = g_keys[1024+blockStart+threadIdx.x];
      vals [1024+threadIdx.x] = g_vals[1024+blockStart+threadIdx.x];
      valid[1024+threadIdx.x] = true;
    } else {
      keys [1024+threadIdx.x] = key_t(0);
      valid[1024+threadIdx.x] = false;
    }
    __syncthreads();
    
    int l, r, s;
    // ======== seq size 1 ==========
    {
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,vals,l,r);
    }

    // ======== seq size 2 ==========
    {
      s    = (int)threadIdx.x & -2;
      l    = threadIdx.x+s;
      r    = l ^ (4-1);
      shm_sort(valid,keys,vals,l,r);
      
      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,vals,l,r);
    }

    // ======== seq size 4 ==========
    {
      s    = (int)threadIdx.x & -4;
      l    = threadIdx.x+s;
      r    = l ^ (8-1);
      shm_sort(valid,keys,vals,l,r);
      
      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,vals,l,r);
    }

    // ======== seq size 8 ==========
    {
      s    = (int)threadIdx.x & -8;
      l    = threadIdx.x+s;
      r    = l ^ (16-1);
      shm_sort(valid,keys,vals,l,r);
      
      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,vals,l,r);
    }

    // ======== seq size 16 ==========
    {
      // __syncthreads();
      s    = (int)threadIdx.x & -16;
      l    = threadIdx.x+s;
      r    = l ^ (32-1);
      shm_sort(valid,keys,vals,l,r);
      
      // ------ down seq size 8 ---------
      // __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,vals,l,r);
    }

    // ======== seq size 32 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -32;
      l    = threadIdx.x+s;
      r    = l ^ (64-1);
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 8 ---------
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,vals,l,r);
    }

    // ======== seq size 64 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -64;
      l    = threadIdx.x+s;
      r    = l ^ (128-1);
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 8 ---------
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,vals,l,r);
    }

    // ======== seq size 128 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -128;
      l    = threadIdx.x+s;
      r    = l ^ (256-1);
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,vals,l,r);
    }

    // ======== seq size 256 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -256;
      l    = threadIdx.x+s;
      r    = l ^ (512-1);
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 128 ---------
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,vals,l,r);
    }

    // ======== seq size 512 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -512;
      l    = threadIdx.x+s;
      r    = l ^ (1024-1);
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 256 ---------
      l = threadIdx.x+((int)threadIdx.x & -256);
      r = l + 256;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 128 ---------
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,vals,l,r);
    }

    // ======== seq size 1024 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -1024;
      l    = threadIdx.x+s;
      r    = l ^ (2048-1);
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 512 ---------
      l = threadIdx.x+((int)threadIdx.x & -512);
      r = l + 512;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 256 ---------
      l = threadIdx.x+((int)threadIdx.x & -256);
      r = l + 256;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 128 ---------
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 8 ---------
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,vals,l,r);
    }

    __syncthreads();
    if (blockStart+threadIdx.x < _N) {
      g_keys[blockStart+threadIdx.x] = keys[threadIdx.x];
      g_vals[blockStart+threadIdx.x] = vals[threadIdx.x];
    }
    if (1024+blockStart+threadIdx.x < _N) {
      g_keys[1024+blockStart+threadIdx.x] = keys[1024+threadIdx.x];
      g_vals[1024+blockStart+threadIdx.x] = vals[1024+threadIdx.x];
    }
  }

  
  
  template<typename key_t>
  __global__ void block_sort_down(key_t *const __restrict__ g_keys,
                                  uint32_t _N)
  {
    __shared__ bool  valid[2*1024];
    __shared__ key_t keys[2*1024];
    uint32_t blockStart = blockIdx.x*(2*1024);
    if (blockStart+threadIdx.x < _N) {
      keys [threadIdx.x] = g_keys[blockStart+threadIdx.x];
      valid[threadIdx.x] = true;
    } else {
      keys [threadIdx.x] = key_t(0);
      valid[threadIdx.x] = false;
    }
    if (1024+blockStart+threadIdx.x < _N) {
      keys [1024+threadIdx.x] = g_keys[1024+blockStart+threadIdx.x];
      valid[1024+threadIdx.x] = true;
    } else {
      keys [1024+threadIdx.x] = key_t(0);
      valid[1024+threadIdx.x] = false;
    }
    __syncthreads();
    
    int l, r;
    // ======== seq size 1024 ==========
    {
      // ------ down seq size 1024 ---------
      l = threadIdx.x+((int)threadIdx.x & -1024);
      r = l + 1024;
      shm_sort(valid,keys,l,r);
      __syncthreads();
      
      // ------ down seq size 512 ---------
      l = threadIdx.x+((int)threadIdx.x & -512);
      r = l + 512;
      shm_sort(valid,keys,l,r);
      __syncthreads();
      
      // ------ down seq size 256 ---------
      l = threadIdx.x+((int)threadIdx.x & -256);
      r = l + 256;
      shm_sort(valid,keys,l,r);
      __syncthreads();
      
      // ------ down seq size 128 ---------
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      shm_sort(valid,keys,l,r);
      __syncthreads();

      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(valid,keys,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(valid,keys,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 8 ---------
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,l,r);
    }

    __syncthreads();
    if (blockStart+threadIdx.x < _N) g_keys[blockStart+threadIdx.x] = keys[threadIdx.x];
    if (1024+blockStart+threadIdx.x < _N) g_keys[1024+blockStart+threadIdx.x] = keys[1024+threadIdx.x];
  }


  template<typename key_t, typename val_t>
  __global__ void block_sort_down(key_t *const __restrict__ g_keys,
                                  val_t *const __restrict__ g_vals,
                                  uint32_t _N)
  {
    __shared__ bool  valid[2*1024];
    __shared__ key_t keys[2*1024];
    __shared__ val_t vals[2*1024];
    uint32_t blockStart = blockIdx.x*(2*1024);
    if (blockStart+threadIdx.x < _N) {
      keys [threadIdx.x] = g_keys[blockStart+threadIdx.x];
      vals [threadIdx.x] = g_vals[blockStart+threadIdx.x];
      valid[threadIdx.x] = true;
    } else {
      keys [threadIdx.x] = key_t(0);
      valid[threadIdx.x] = false;
    }
    if (1024+blockStart+threadIdx.x < _N) {
      keys [1024+threadIdx.x] = g_keys[1024+blockStart+threadIdx.x];
      vals [1024+threadIdx.x] = g_vals[1024+blockStart+threadIdx.x];
      valid[1024+threadIdx.x] = true;
    } else {
      keys [1024+threadIdx.x] = key_t(0);
      valid[1024+threadIdx.x] = false;
    }
    __syncthreads();
    
    int l, r;
    // ======== seq size 1024 ==========
    {
      // ------ down seq size 1024 ---------
      l = threadIdx.x+((int)threadIdx.x & -1024);
      r = l + 1024;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 512 ---------
      l = threadIdx.x+((int)threadIdx.x & -512);
      r = l + 512;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 256 ---------
      l = threadIdx.x+((int)threadIdx.x & -256);
      r = l + 256;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 128 ---------
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(valid,keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 8 ---------
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(valid,keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(valid,keys,vals,l,r);
    }

    __syncthreads();
    if (blockStart+threadIdx.x < _N) {
      g_keys[blockStart+threadIdx.x] = keys[threadIdx.x];
      g_vals[blockStart+threadIdx.x] = vals[threadIdx.x];
    }
    if (1024+blockStart+threadIdx.x < _N) {
      g_keys[1024+blockStart+threadIdx.x] = keys[1024+threadIdx.x];
      g_vals[1024+blockStart+threadIdx.x] = vals[1024+threadIdx.x];
    }
  }

  template<typename key_t>
  __global__ void big_down(key_t *const __restrict__ keys,
                           uint32_t N,
                           int seqLen)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;

    int s    = tid & -seqLen;
    int l    = tid+s;
    int r    = l + seqLen;

    if (r < N)
      gmem_sort(keys,l,r);
  }

  template<typename key_t, typename val_t>
  __global__ void big_down(key_t *const __restrict__ keys,
                           val_t *const __restrict__ vals,
                           uint32_t N,
                           int seqLen)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;

    int s    = tid & -seqLen;
    int l    = tid+s;
    int r    = l + seqLen;

    if (r < N)
      gmem_sort(keys,vals,l,r);
  }

  template<typename key_t>
  __global__ void big_up(key_t *const __restrict__ keys,
                         uint32_t N,
                         int seqLen)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= N) return;
    
    int s    = tid & -seqLen;
    int l    = tid+s;
    int r    = l ^ (2*seqLen-1);

    if (r < N) {
      gmem_sort(keys,l,r);
    }
  }
  template<typename key_t, typename val_t>
  __global__ void big_up(key_t *const __restrict__ keys,
                         val_t *const __restrict__ vals,
                         uint32_t N,
                         int seqLen)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= N) return;
    
    int s    = tid & -seqLen;
    int l    = tid+s;
    int r    = l ^ (2*seqLen-1);

    if (r < N) {
      gmem_sort(keys,vals,l,r);
    }
  }
  
  template<typename key_t>
  inline void sort(key_t *const __restrict__ d_values,
                   size_t numValues,
                   cudaStream_t stream=0)

  {
    int bs = 1024;
    int numValuesPerBlock = 2*bs;

    // ==================================================================
    // first - sort all blocks of 2x1024 using per-block sort
    // ==================================================================
    int nb = divRoundUp((int)numValues,numValuesPerBlock);
    block_sort_up<<<nb,bs,0,stream>>>(d_values,numValues);

    int _nb = divRoundUp(int(numValues),1024);
    for (int upLen=numValuesPerBlock;upLen<numValues;upLen+=upLen) {
      big_up<<<_nb,bs,0,stream>>>(d_values,numValues,upLen);
      for (int downLen=upLen/2;downLen>1024;downLen/=2) {
        big_down<<<_nb,bs,0,stream>>>(d_values,numValues,downLen);
      }
      block_sort_down<<<nb,bs,0,stream>>>(d_values,numValues);
    }
  }
  
  template<typename key_t, typename val_t>
  inline void sort(key_t *const __restrict__ d_keys,
                   val_t *const __restrict__ d_values,
                   size_t numValues,
                   cudaStream_t stream=0)

  {
    int bs = 1024;
    int numValuesPerBlock = 2*bs;

    // ==================================================================
    // first - sort all blocks of 2x1024 using per-block sort
    // ==================================================================
    int nb = divRoundUp((int)numValues,numValuesPerBlock);
    block_sort_up<<<nb,bs,0,stream>>>(d_keys,d_values,numValues);

    int _nb = divRoundUp(int(numValues),1024);
    for (int upLen=numValuesPerBlock;upLen<numValues;upLen+=upLen) {
      big_up<<<_nb,bs,0,stream>>>(d_keys,d_values,numValues,upLen);
      for (int downLen=upLen/2;downLen>1024;downLen/=2) {
        big_down<<<_nb,bs,0,stream>>>(d_keys,d_values,numValues,downLen);
      }
      block_sort_down<<<nb,bs,0,stream>>>(d_keys,d_values,numValues);
    }
  }
}

