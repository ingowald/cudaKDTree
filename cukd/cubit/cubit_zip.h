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

  cubit_zip.h: This file implemnets a "zip"-sort, in the sense that
  there are two separate arrays that _together_ make the key value
  that is being sorted by.
*/

#pragma once

#include "../cubit/common.h"
#include <cuda_runtime.h>

namespace cubit {
  using namespace cubit::common;

  template<typename U, typename V>
  struct tuple {
    U u;
    V v;
  };
  
  namespace zip {
    
    // enum { block_size = 1024 };

    template<typename U, typename V, typename Less>
    inline static __device__ void shm_sort(bool  *const __restrict__ valid,
                                           U *const __restrict__ us,
                                           V *const __restrict__ vs,
                                           uint32_t a,
                                           uint32_t b,
                                           Less less)
    {
      U ua = us[a];
      U ub = us[b];
      V va = vs[a];
      V vb = vs[b];
      const bool keepAsIs
        = valid[a] && (!valid[b] || less(tuple<U,V>{ua,va},tuple<U,V>{ub,vb}));
      us[a] = (keepAsIs)?ua:ub;
      us[b] = (keepAsIs)?ub:ua;
      vs[a] = (keepAsIs)?va:vb;
      vs[b] = (keepAsIs)?vb:va;
    }

    template<typename U, typename V, typename Less>
    inline static __device__ void gmem_sort(U *const us,
                                            V *const vs,
                                            uint32_t a,
                                            uint32_t b,
                                            Less less)
    {
      U ua = us[a];
      U ub = us[b];
      
      V va = vs[a];
      V vb = vs[b];
      
      if (less({ub,vb},{ua,va})) { //key_b < key_a) {
        us[a] = ua;
        us[b] = ub;
        
        vs[a] = va;
        vs[b] = vb;
      }
    }
  

    template<typename U, typename V, typename Less, int BLOCK_SIZE>
    __global__ void block_sort_up(U *const __restrict__ g_us,
                                  V *const __restrict__ g_vs,
                                  uint32_t _N,
                                  Less less
                                  )
    {
      __shared__ U us[2*BLOCK_SIZE];
      __shared__ V vs[2*BLOCK_SIZE];
      __shared__ bool  valid[2*BLOCK_SIZE];
      uint32_t blockStart = blockIdx.x*(2*BLOCK_SIZE);
      if (blockStart+threadIdx.x < _N) {
        us [threadIdx.x] = g_us[blockStart+threadIdx.x];
        vs [threadIdx.x] = g_vs[blockStart+threadIdx.x];
        valid[threadIdx.x] = true;
      } else {
        // us[threadIdx.x] = max_value<U>();
        us [threadIdx.x] = U{};// doesn't matter, just to make sure it's initialized
        vs [threadIdx.x] = V{};// doesn't matter, just to make sure it's initialized
        valid[threadIdx.x] = false;
      }
      if (BLOCK_SIZE+blockStart+threadIdx.x < _N) {
        us [BLOCK_SIZE+threadIdx.x] = g_us[BLOCK_SIZE+blockStart+threadIdx.x];
        vs [BLOCK_SIZE+threadIdx.x] = g_vs[BLOCK_SIZE+blockStart+threadIdx.x];
        valid[BLOCK_SIZE+threadIdx.x] = true;
      } else {
        us [BLOCK_SIZE+threadIdx.x] = U{};
        vs [BLOCK_SIZE+threadIdx.x] = V{};
        valid[BLOCK_SIZE+threadIdx.x] = false;
      }
      __syncthreads();
    
      int l, r, s;
      // ======== seq size 1 ==========
      {
        l = threadIdx.x+threadIdx.x;
        r = l + 1;
        shm_sort(valid,us,vs,l,r,less);
      }

      // ======== seq size 2 ==========
      {
        s    = (int)threadIdx.x & -2;
        l    = threadIdx.x+s;
        r    = l ^ (4-1);
        shm_sort(valid,us,vs,l,r,less);
      
        // ------ down seq size 1 ---------
        l = threadIdx.x+threadIdx.x;
        r = l + 1;
        shm_sort(valid,us,vs,l,r,less);
      }

      // ======== seq size 4 ==========
      {
        s    = (int)threadIdx.x & -4;
        l    = threadIdx.x+s;
        r    = l ^ (8-1);
        shm_sort(valid,us,vs,l,r,less);
      
        // ------ down seq size 2 ---------
        l = threadIdx.x+((int)threadIdx.x & -2);
        r = l + 2;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 1 ---------
        l = threadIdx.x+threadIdx.x;
        r = l + 1;
        shm_sort(valid,us,vs,l,r,less);
      }

      // ======== seq size 8 ==========
      {
        s    = (int)threadIdx.x & -8;
        l    = threadIdx.x+s;
        r    = l ^ (16-1);
        shm_sort(valid,us,vs,l,r,less);
      
        // ------ down seq size 4 ---------
        l = threadIdx.x+((int)threadIdx.x & -4);
        r = l + 4;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 2 ---------
        l = threadIdx.x+((int)threadIdx.x & -2);
        r = l + 2;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 1 ---------
        l = threadIdx.x+threadIdx.x;
        r = l + 1;
        shm_sort(valid,us,vs,l,r,less);
      }

      // ======== seq size 16 ==========
      {
        // __syncthreads();
        s    = (int)threadIdx.x & -16;
        l    = threadIdx.x+s;
        r    = l ^ (32-1);
        shm_sort(valid,us,vs,l,r,less);
      
        // ------ down seq size 8 ---------
        // __syncthreads();
        l = threadIdx.x+((int)threadIdx.x & -8);
        r = l + 8;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 4 ---------
        l = threadIdx.x+((int)threadIdx.x & -4);
        r = l + 4;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 2 ---------
        l = threadIdx.x+((int)threadIdx.x & -2);
        r = l + 2;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 1 ---------
        l = threadIdx.x+threadIdx.x;
        r = l + 1;
        shm_sort(valid,us,vs,l,r,less);
      }

      // ======== seq size 32 ==========
      {
        __syncthreads();
        s    = (int)threadIdx.x & -32;
        l    = threadIdx.x+s;
        r    = l ^ (64-1);
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();
      
        // ------ down seq size 16 ---------
        l = threadIdx.x+((int)threadIdx.x & -16);
        r = l + 16;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 8 ---------
        l = threadIdx.x+((int)threadIdx.x & -8);
        r = l + 8;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 4 ---------
        l = threadIdx.x+((int)threadIdx.x & -4);
        r = l + 4;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 2 ---------
        l = threadIdx.x+((int)threadIdx.x & -2);
        r = l + 2;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 1 ---------
        l = threadIdx.x+threadIdx.x;
        r = l + 1;
        shm_sort(valid,us,vs,l,r,less);
      }

      // ======== seq size 64 ==========
      {
        __syncthreads();
        s    = (int)threadIdx.x & -64;
        l    = threadIdx.x+s;
        r    = l ^ (128-1);
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();
      
        // ------ down seq size 32 ---------
        l = threadIdx.x+((int)threadIdx.x & -32);
        r = l + 32;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();

        // ------ down seq size 16 ---------
        l = threadIdx.x+((int)threadIdx.x & -16);
        r = l + 16;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 8 ---------
        l = threadIdx.x+((int)threadIdx.x & -8);
        r = l + 8;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 4 ---------
        l = threadIdx.x+((int)threadIdx.x & -4);
        r = l + 4;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 2 ---------
        l = threadIdx.x+((int)threadIdx.x & -2);
        r = l + 2;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 1 ---------
        l = threadIdx.x+threadIdx.x;
        r = l + 1;
        shm_sort(valid,us,vs,l,r,less);
      }

      // ======== seq size 128 ==========
      {
        __syncthreads();
        s    = (int)threadIdx.x & -128;
        l    = threadIdx.x+s;
        r    = l ^ (256-1);
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();
      
        // ------ down seq size 64 ---------
        l = threadIdx.x+((int)threadIdx.x & -64);
        r = l + 64;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();

        // ------ down seq size 32 ---------
        l = threadIdx.x+((int)threadIdx.x & -32);
        r = l + 32;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();

        // ------ down seq size 16 ---------
        l = threadIdx.x+((int)threadIdx.x & -16);
        r = l + 16;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 8 ---------
        __syncthreads();
        l = threadIdx.x+((int)threadIdx.x & -8);
        r = l + 8;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 4 ---------
        l = threadIdx.x+((int)threadIdx.x & -4);
        r = l + 4;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 2 ---------
        l = threadIdx.x+((int)threadIdx.x & -2);
        r = l + 2;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 1 ---------
        l = threadIdx.x+threadIdx.x;
        r = l + 1;
        shm_sort(valid,us,vs,l,r,less);
      }

      // ======== seq size 256 ==========
      {
        __syncthreads();
        s    = (int)threadIdx.x & -256;
        l    = threadIdx.x+s;
        r    = l ^ (512-1);
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();
      
        // ------ down seq size 128 ---------
        l = threadIdx.x+((int)threadIdx.x & -128);
        r = l + 128;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();

        // ------ down seq size 64 ---------
        l = threadIdx.x+((int)threadIdx.x & -64);
        r = l + 64;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();

        // ------ down seq size 32 ---------
        l = threadIdx.x+((int)threadIdx.x & -32);
        r = l + 32;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();

        // ------ down seq size 16 ---------
        __syncthreads();
        l = threadIdx.x+((int)threadIdx.x & -16);
        r = l + 16;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 8 ---------
        __syncthreads();
        l = threadIdx.x+((int)threadIdx.x & -8);
        r = l + 8;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 4 ---------
        l = threadIdx.x+((int)threadIdx.x & -4);
        r = l + 4;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 2 ---------
        l = threadIdx.x+((int)threadIdx.x & -2);
        r = l + 2;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 1 ---------
        l = threadIdx.x+threadIdx.x;
        r = l + 1;
        shm_sort(valid,us,vs,l,r,less);
      }

      // ======== seq size 512 ==========
      {
        if (BLOCK_SIZE == 1024) {
          __syncthreads();
          s    = (int)threadIdx.x & -512;
          l    = threadIdx.x+s;
          r    = l ^ (1024-1);
          shm_sort(valid,us,vs,l,r,less);
        }
        __syncthreads();
      
        // ------ down seq size 256 ---------
        l = threadIdx.x+((int)threadIdx.x & -256);
        r = l + 256;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();
      
        // ------ down seq size 128 ---------
        l = threadIdx.x+((int)threadIdx.x & -128);
        r = l + 128;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();

        // ------ down seq size 64 ---------
        l = threadIdx.x+((int)threadIdx.x & -64);
        r = l + 64;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();

        // ------ down seq size 32 ---------
        l = threadIdx.x+((int)threadIdx.x & -32);
        r = l + 32;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();

        // ------ down seq size 16 ---------
        l = threadIdx.x+((int)threadIdx.x & -16);
        r = l + 16;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 8 ---------
        __syncthreads();
        l = threadIdx.x+((int)threadIdx.x & -8);
        r = l + 8;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 4 ---------
        l = threadIdx.x+((int)threadIdx.x & -4);
        r = l + 4;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 2 ---------
        l = threadIdx.x+((int)threadIdx.x & -2);
        r = l + 2;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 1 ---------
        l = threadIdx.x+threadIdx.x;
        r = l + 1;
        shm_sort(valid,us,vs,l,r,less);
      }

      // ======== seq size 1024 ==========
      {
        if (BLOCK_SIZE == 1024) {
          __syncthreads();
          s    = (int)threadIdx.x & -1024;
          l    = threadIdx.x+s;
          r    = l ^ (2048-1);
          shm_sort(valid,us,vs,l,r,less);
          __syncthreads();
      
          // ------ down seq size 512 ---------
          l = threadIdx.x+((int)threadIdx.x & -512);
          r = l + 512;
          shm_sort(valid,us,vs,l,r,less);
        }
        __syncthreads();
      
        // ------ down seq size 256 ---------
        l = threadIdx.x+((int)threadIdx.x & -256);
        r = l + 256;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();
      
        // ------ down seq size 128 ---------
        l = threadIdx.x+((int)threadIdx.x & -128);
        r = l + 128;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();

        // ------ down seq size 64 ---------
        l = threadIdx.x+((int)threadIdx.x & -64);
        r = l + 64;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();

        // ------ down seq size 32 ---------
        l = threadIdx.x+((int)threadIdx.x & -32);
        r = l + 32;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();

        // ------ down seq size 16 ---------
        l = threadIdx.x+((int)threadIdx.x & -16);
        r = l + 16;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 8 ---------
        l = threadIdx.x+((int)threadIdx.x & -8);
        r = l + 8;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 4 ---------
        l = threadIdx.x+((int)threadIdx.x & -4);
        r = l + 4;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 2 ---------
        l = threadIdx.x+((int)threadIdx.x & -2);
        r = l + 2;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 1 ---------
        l = threadIdx.x+threadIdx.x;
        r = l + 1;
        shm_sort(valid,us,vs,l,r,less);
      }

      __syncthreads();
      if (blockStart+threadIdx.x < _N) {
        g_us[blockStart+threadIdx.x] = us[threadIdx.x];
        g_vs[blockStart+threadIdx.x] = vs[threadIdx.x];
      }
      if (BLOCK_SIZE+blockStart+threadIdx.x < _N) {
        g_us[BLOCK_SIZE+blockStart+threadIdx.x] = us[BLOCK_SIZE+threadIdx.x];
        g_vs[BLOCK_SIZE+blockStart+threadIdx.x] = vs[BLOCK_SIZE+threadIdx.x];
      }
    }

  
  
    template<typename U, typename V, typename Less, int BLOCK_SIZE>
    __global__ void block_sort_down(U *const __restrict__ g_us,
                                    V *const __restrict__ g_vs,
                                    uint32_t _N,
                                    Less less)
    {
      __shared__ bool  valid[2*BLOCK_SIZE];
      __shared__ U us[2*BLOCK_SIZE];
      __shared__ V vs[2*BLOCK_SIZE];
      uint32_t blockStart = blockIdx.x*(2*BLOCK_SIZE);
      if (blockStart+threadIdx.x < _N) {
        us   [threadIdx.x] = g_us[blockStart+threadIdx.x];
        vs   [threadIdx.x] = g_vs[blockStart+threadIdx.x];
        valid[threadIdx.x] = true;
      } else {
        us   [threadIdx.x] = U(0);
        valid[threadIdx.x] = false;
      }
      if (BLOCK_SIZE+blockStart+threadIdx.x < _N) {
        us   [BLOCK_SIZE+threadIdx.x] = g_us[BLOCK_SIZE+blockStart+threadIdx.x];
        vs   [BLOCK_SIZE+threadIdx.x] = g_vs[BLOCK_SIZE+blockStart+threadIdx.x];
        valid[BLOCK_SIZE+threadIdx.x] = true;
      } else {
        us   [BLOCK_SIZE+threadIdx.x] = U(0);
        valid[BLOCK_SIZE+threadIdx.x] = false;
      }
      __syncthreads();
    
      int l, r;
      // ======== seq size 1024 ==========
      {
        if (BLOCK_SIZE == 1024) {
          // ------ down seq size 1024 ---------
          l = threadIdx.x+((int)threadIdx.x & -1024);
          r = l + 1024;
          shm_sort(valid,us,vs,l,r,less);
          __syncthreads();
          
          // ------ down seq size 512 ---------
          l = threadIdx.x+((int)threadIdx.x & -512);
          r = l + 512;
          shm_sort(valid,us,vs,l,r,less);
          __syncthreads();
        }
        
        // ------ down seq size 256 ---------
        l = threadIdx.x+((int)threadIdx.x & -256);
        r = l + 256;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();
      
        // ------ down seq size 128 ---------
        l = threadIdx.x+((int)threadIdx.x & -128);
        r = l + 128;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();

        // ------ down seq size 64 ---------
        l = threadIdx.x+((int)threadIdx.x & -64);
        r = l + 64;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();

        // ------ down seq size 32 ---------
        l = threadIdx.x+((int)threadIdx.x & -32);
        r = l + 32;
        shm_sort(valid,us,vs,l,r,less);
        __syncthreads();

        // ------ down seq size 16 ---------
        l = threadIdx.x+((int)threadIdx.x & -16);
        r = l + 16;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 8 ---------
        l = threadIdx.x+((int)threadIdx.x & -8);
        r = l + 8;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 4 ---------
        l = threadIdx.x+((int)threadIdx.x & -4);
        r = l + 4;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 2 ---------
        l = threadIdx.x+((int)threadIdx.x & -2);
        r = l + 2;
        shm_sort(valid,us,vs,l,r,less);

        // ------ down seq size 1 ---------
        l = threadIdx.x+threadIdx.x;
        r = l + 1;
        shm_sort(valid,us,vs,l,r,less);
      }

      __syncthreads();
      if (blockStart+threadIdx.x < _N) {
        g_us[blockStart+threadIdx.x] = us[threadIdx.x];
        g_vs[blockStart+threadIdx.x] = vs[threadIdx.x];
      }
      if (BLOCK_SIZE+blockStart+threadIdx.x < _N) {
        g_us[BLOCK_SIZE+blockStart+threadIdx.x] = us[BLOCK_SIZE+threadIdx.x];
        g_vs[BLOCK_SIZE+blockStart+threadIdx.x] = vs[BLOCK_SIZE+threadIdx.x];
      }
    }

    template<typename U, typename V, typename Less>
    __global__ void big_down(U *const __restrict__ us,
                             V *const __restrict__ vs,
                             uint32_t N,
                             int seqLen,
                             Less less)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;

      int s    = tid & -seqLen;
      int l    = tid+s;
      int r    = l + seqLen;

      if (r < N)
        gmem_sort(us,vs,l,r,less);
    }

    // template<typename U>
    // __global__ void big_up(U *const __restrict__ us,
    //                        uint32_t N,
    //                        int seqLen)
    // {
    //   int tid = threadIdx.x+blockIdx.x*blockDim.x;
    //   if (tid >= N) return;
    
    //   int s    = tid & -seqLen;
    //   int l    = tid+s;
    //   int r    = l ^ (2*seqLen-1);

    //   if (r < N) {
    //     gmem_sort(us,l,r);
    //   }
    // }
    template<typename U, typename V, typename Less>
    __global__ void big_up(U *const __restrict__ us,
                           V *const __restrict__ vs,
                           uint32_t N,
                           int seqLen,
                           Less less)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= N) return;
    
      int s    = tid & -seqLen;
      int l    = tid+s;
      int r    = l ^ (2*seqLen-1);

      if (r < N) {
        gmem_sort(us,vs,l,r,less);
      }
    }
  
    // template<typename U>
    // inline void sort(U *const __restrict__ d_values,
    //                  size_t numValues,
    //                  cudaStream_t stream=0)

    // {
    //   int bs = 1024;
    //   int numValuesPerBlock = 2*bs;

    //   // ==================================================================
    //   // first - sort all blocks of 2x1024 using per-block sort
    //   // ==================================================================
    //   int nb = divRoundUp((int)numValues,numValuesPerBlock);
    //   block_sort_up<<<nb,bs,0,stream>>>(d_values,numValues);

    //   int _nb = divRoundUp(int(numValues),1024);
    //   for (int upLen=numValuesPerBlock;upLen<numValues;upLen+=upLen) {
    //     big_up<<<_nb,bs,0,stream>>>(d_values,numValues,upLen);
    //     for (int downLen=upLen/2;downLen>1024;downLen/=2) {
    //       big_down<<<_nb,bs,0,stream>>>(d_values,numValues,downLen);
    //     }
    //     block_sort_down<<<nb,bs,0,stream>>>(d_values,numValues);
    //   }
    // }
  }


  /*! valid values for block_size are 1024 and 256;everything else is
    undefined behavior */
  template<typename U, typename V, typename Less, int BLOCK_SIZE=1024>
  inline void zip_sort(U *const __restrict__ us,
                       V *const __restrict__ vs,
                       size_t numValues,
                       Less less,
                       cudaStream_t stream=0)

  {
    int bs = BLOCK_SIZE;
    int numValuesPerBlock = 2*bs;

    // ==================================================================
    // first - sort all blocks of 2x1024 using per-block sort
    // ==================================================================
    int nb = divRoundUp((int)numValues,numValuesPerBlock);
    zip::block_sort_up<U,V,Less,BLOCK_SIZE>
      <<<nb,bs,0,stream>>>(us,vs,numValues,less);

    int _nb = divRoundUp(int(numValues),BLOCK_SIZE);
    for (int upLen=numValuesPerBlock;upLen<numValues;upLen+=upLen) {
      zip::big_up
        <<<_nb,bs,0,stream>>>(us,vs,numValues,upLen,less);
      for (int downLen=upLen/2;downLen>BLOCK_SIZE;downLen/=2) {
        zip::big_down
          <<<_nb,bs,0,stream>>>(us,vs,numValues,downLen,less);
      }
      zip::block_sort_down<U,V,Less,BLOCK_SIZE>
        <<<nb,bs,0,stream>>>(us,vs,numValues,less);
    }
  }
}

