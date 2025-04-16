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

#include "../cubit/common.h"
#include <cuda_runtime.h>

namespace cubit {

  enum { block_size = 1024 };

  /*! helper function to compute block count for given number of
      thread and block size; by dividing and rounding up */
  inline int divRoundUp(int a, int b) { return (a+b-1)/b; }

  template<typename T>
  inline __host__ __device__ T max_value();

  template<>
  inline __host__ __device__ uint32_t max_value() { return UINT_MAX; };
  template<>
  inline __host__ __device__ int32_t max_value() { return INT_MAX; };
  template<>
  inline __host__ __device__ uint64_t max_value() { return ULONG_MAX; };
  template<>
  inline __host__ __device__ float max_value() { return INFINITY; };
  template<>
  inline __host__ __device__ double max_value() { return INFINITY; };
  
  template<typename key_t>
  inline static __host__ __device__ void putInOrder(key_t *const __restrict__ keys,
                                           uint32_t N,
                                           uint32_t a,
                                           uint32_t b)
  {
    if (b >= N) return;
    key_t key_a = keys[a];
    key_t key_b = keys[b];
    if (key_a > key_b) {
      keys[a] = key_b;
      keys[b] = key_a;
    }
  }
    
  template<typename key_t>
  inline static __host__ __device__ void putInOrder(key_t *const __restrict__ keys,
                                           uint64_t N,
                                           uint64_t a,
                                           uint64_t b)
  {
    if (b >= N) return;
    key_t key_a = keys[a];
    key_t key_b = keys[b];
    if (key_a > key_b) {
      keys[a] = key_b;
      keys[b] = key_a;
    }
  }

  template<typename key_t>
  inline static __host__ __device__ void shm_sort(key_t *const __restrict__ keys,
                                         uint32_t a,
                                         uint32_t b)
  {
    key_t key_a = keys[a];
    key_t key_b = keys[b];
    keys[a] = (key_a<key_b)?key_a:key_b;
    keys[b] = (key_a<key_b)?key_b:key_a;
  }

  template<typename key_t, typename val_t>
  inline static __host__ __device__ void shm_sort(key_t *const __restrict__ keys,
                                         val_t *const __restrict__ vals,
                                         uint32_t a,
                                         uint32_t b)
  {
    key_t key_a = keys[a];
    key_t key_b = keys[b];
    val_t val_a = vals[a];
    val_t val_b = vals[b];
    keys[a] = (key_a<key_b)?key_a:key_b;
    keys[b] = (key_a<key_b)?key_b:key_a;
    vals[a] = (key_a<key_b)?val_a:val_b;
    vals[b] = (key_a<key_b)?val_b:val_a;
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
    uint32_t blockStart = blockIdx.x*(2*1024);
    if (blockStart+threadIdx.x < _N)
      keys[threadIdx.x] = g_keys[blockStart+threadIdx.x];
    else
      keys[threadIdx.x] = max_value<key_t>();
    if (1024+blockStart+threadIdx.x < _N)
      keys[1024+threadIdx.x] = g_keys[1024+blockStart+threadIdx.x];
    else
      keys[1024+threadIdx.x] = max_value<key_t>();
    __syncthreads();
    
    int l, r, s;
    // ======== seq size 1 ==========
    {
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 2 ==========
    {
      s    = (int)threadIdx.x & -2;
      l    = threadIdx.x+s;
      r    = l ^ (4-1);
      shm_sort(keys,l,r);
      
      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 4 ==========
    {
      s    = (int)threadIdx.x & -4;
      l    = threadIdx.x+s;
      r    = l ^ (8-1);
      shm_sort(keys,l,r);
      
      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 8 ==========
    {
      s    = (int)threadIdx.x & -8;
      l    = threadIdx.x+s;
      r    = l ^ (16-1);
      shm_sort(keys,l,r);
      
      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 16 ==========
    {
      // __syncthreads();
      s    = (int)threadIdx.x & -16;
      l    = threadIdx.x+s;
      r    = l ^ (32-1);
      shm_sort(keys,l,r);
      
      // ------ down seq size 8 ---------
      // __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 32 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -32;
      l    = threadIdx.x+s;
      r    = l ^ (64-1);
      shm_sort(keys,l,r);
      __syncthreads();
      
      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(keys,l,r);

      // ------ down seq size 8 ---------
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 64 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -64;
      l    = threadIdx.x+s;
      r    = l ^ (128-1);
      shm_sort(keys,l,r);
      __syncthreads();
      
      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(keys,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(keys,l,r);

      // ------ down seq size 8 ---------
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 128 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -128;
      l    = threadIdx.x+s;
      r    = l ^ (256-1);
      shm_sort(keys,l,r);
      __syncthreads();
      
      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(keys,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(keys,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(keys,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 256 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -256;
      l    = threadIdx.x+s;
      r    = l ^ (512-1);
      shm_sort(keys,l,r);
      __syncthreads();
      
      // ------ down seq size 128 ---------
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      shm_sort(keys,l,r);
      __syncthreads();

      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(keys,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(keys,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(keys,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 512 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -512;
      l    = threadIdx.x+s;
      r    = l ^ (1024-1);
      shm_sort(keys,l,r);
      __syncthreads();
      
      // ------ down seq size 256 ---------
      l = threadIdx.x+((int)threadIdx.x & -256);
      r = l + 256;
      shm_sort(keys,l,r);
      __syncthreads();
      
      // ------ down seq size 128 ---------
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      shm_sort(keys,l,r);
      __syncthreads();

      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(keys,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(keys,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(keys,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,l,r);
    }

    // ======== seq size 1024 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -1024;
      l    = threadIdx.x+s;
      r    = l ^ (2048-1);
      shm_sort(keys,l,r);
      __syncthreads();
      
      // ------ down seq size 512 ---------
      l = threadIdx.x+((int)threadIdx.x & -512);
      r = l + 512;
      shm_sort(keys,l,r);
      __syncthreads();
      
      // ------ down seq size 256 ---------
      l = threadIdx.x+((int)threadIdx.x & -256);
      r = l + 256;
      shm_sort(keys,l,r);
      __syncthreads();
      
      // ------ down seq size 128 ---------
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      shm_sort(keys,l,r);
      __syncthreads();

      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(keys,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(keys,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(keys,l,r);

      // ------ down seq size 8 ---------
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,l,r);
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
    uint32_t blockStart = blockIdx.x*(2*1024);
    if (blockStart+threadIdx.x < _N) {
      keys[threadIdx.x] = g_keys[blockStart+threadIdx.x];
      vals[threadIdx.x] = g_vals[blockStart+threadIdx.x];
    } else
      keys[threadIdx.x] = max_value<key_t>();
    if (1024+blockStart+threadIdx.x < _N) {
      keys[1024+threadIdx.x] = g_keys[1024+blockStart+threadIdx.x];
      vals[1024+threadIdx.x] = g_vals[1024+blockStart+threadIdx.x];
    } else
      keys[1024+threadIdx.x] = max_value<key_t>();
    __syncthreads();
    
    int l, r, s;
    // ======== seq size 1 ==========
    {
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,vals,l,r);
    }

    // ======== seq size 2 ==========
    {
      s    = (int)threadIdx.x & -2;
      l    = threadIdx.x+s;
      r    = l ^ (4-1);
      shm_sort(keys,vals,l,r);
      
      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,vals,l,r);
    }

    // ======== seq size 4 ==========
    {
      s    = (int)threadIdx.x & -4;
      l    = threadIdx.x+s;
      r    = l ^ (8-1);
      shm_sort(keys,vals,l,r);
      
      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,vals,l,r);
    }

    // ======== seq size 8 ==========
    {
      s    = (int)threadIdx.x & -8;
      l    = threadIdx.x+s;
      r    = l ^ (16-1);
      shm_sort(keys,vals,l,r);
      
      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,vals,l,r);
    }

    // ======== seq size 16 ==========
    {
      // __syncthreads();
      s    = (int)threadIdx.x & -16;
      l    = threadIdx.x+s;
      r    = l ^ (32-1);
      shm_sort(keys,vals,l,r);
      
      // ------ down seq size 8 ---------
      // __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,vals,l,r);
    }

    // ======== seq size 32 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -32;
      l    = threadIdx.x+s;
      r    = l ^ (64-1);
      shm_sort(keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 8 ---------
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,vals,l,r);
    }

    // ======== seq size 64 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -64;
      l    = threadIdx.x+s;
      r    = l ^ (128-1);
      shm_sort(keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 8 ---------
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,vals,l,r);
    }

    // ======== seq size 128 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -128;
      l    = threadIdx.x+s;
      r    = l ^ (256-1);
      shm_sort(keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,vals,l,r);
    }

    // ======== seq size 256 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -256;
      l    = threadIdx.x+s;
      r    = l ^ (512-1);
      shm_sort(keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 128 ---------
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      shm_sort(keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,vals,l,r);
    }

    // ======== seq size 512 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -512;
      l    = threadIdx.x+s;
      r    = l ^ (1024-1);
      shm_sort(keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 256 ---------
      l = threadIdx.x+((int)threadIdx.x & -256);
      r = l + 256;
      shm_sort(keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 128 ---------
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      shm_sort(keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,vals,l,r);
    }

    // ======== seq size 1024 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -1024;
      l    = threadIdx.x+s;
      r    = l ^ (2048-1);
      shm_sort(keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 512 ---------
      l = threadIdx.x+((int)threadIdx.x & -512);
      r = l + 512;
      shm_sort(keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 256 ---------
      l = threadIdx.x+((int)threadIdx.x & -256);
      r = l + 256;
      shm_sort(keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 128 ---------
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      shm_sort(keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 8 ---------
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,vals,l,r);
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
    __shared__ key_t keys[2*1024];
    uint32_t blockStart = blockIdx.x*(2*1024);
    if (blockStart+threadIdx.x < _N)
      keys[threadIdx.x] = g_keys[blockStart+threadIdx.x];
    else
      keys[threadIdx.x] = max_value<key_t>();
    if (1024+blockStart+threadIdx.x < _N)
      keys[1024+threadIdx.x] = g_keys[1024+blockStart+threadIdx.x];
    else
      keys[1024+threadIdx.x] = max_value<key_t>();
    __syncthreads();
    
    int l, r;
    // ======== seq size 1024 ==========
    {
      // ------ down seq size 1024 ---------
      l = threadIdx.x+((int)threadIdx.x & -1024);
      r = l + 1024;
      shm_sort(keys,l,r);
      __syncthreads();
      
      // ------ down seq size 512 ---------
      l = threadIdx.x+((int)threadIdx.x & -512);
      r = l + 512;
      shm_sort(keys,l,r);
      __syncthreads();
      
      // ------ down seq size 256 ---------
      l = threadIdx.x+((int)threadIdx.x & -256);
      r = l + 256;
      shm_sort(keys,l,r);
      __syncthreads();
      
      // ------ down seq size 128 ---------
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      shm_sort(keys,l,r);
      __syncthreads();

      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(keys,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(keys,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(keys,l,r);

      // ------ down seq size 8 ---------
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,l,r);
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
    __shared__ key_t keys[2*1024];
    __shared__ val_t vals[2*1024];
    uint32_t blockStart = blockIdx.x*(2*1024);
    if (blockStart+threadIdx.x < _N) {
      keys[threadIdx.x] = g_keys[blockStart+threadIdx.x];
      vals[threadIdx.x] = g_vals[blockStart+threadIdx.x];
    } else
      keys[threadIdx.x] = max_value<key_t>();
    if (1024+blockStart+threadIdx.x < _N) {
      keys[1024+threadIdx.x] = g_keys[1024+blockStart+threadIdx.x];
      vals[1024+threadIdx.x] = g_vals[1024+blockStart+threadIdx.x];
    } else
      keys[1024+threadIdx.x] = max_value<key_t>();
    __syncthreads();
    
    int l, r;
    // ======== seq size 1024 ==========
    {
      // ------ down seq size 1024 ---------
      l = threadIdx.x+((int)threadIdx.x & -1024);
      r = l + 1024;
      shm_sort(keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 512 ---------
      l = threadIdx.x+((int)threadIdx.x & -512);
      r = l + 512;
      shm_sort(keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 256 ---------
      l = threadIdx.x+((int)threadIdx.x & -256);
      r = l + 256;
      shm_sort(keys,vals,l,r);
      __syncthreads();
      
      // ------ down seq size 128 ---------
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      shm_sort(keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 64 ---------
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      shm_sort(keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 32 ---------
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      shm_sort(keys,vals,l,r);
      __syncthreads();

      // ------ down seq size 16 ---------
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 8 ---------
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      shm_sort(keys,vals,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      shm_sort(keys,vals,l,r);
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

