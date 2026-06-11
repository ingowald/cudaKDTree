// ======================================================================== //
// Copyright 2024 Ingo Wald                                                  //
// Copyright (c) 2026 Advanced Micro Devices, Inc.                          //
// Author: Jeff Daily <jeff.daily@amd.com>                                  //
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

// Single CUDA->HIP compatibility shim. On AMD/ROCm it aliases the CUDA
// runtime spellings this library uses to their HIP equivalents and pulls in
// the HIP runtime; on NVIDIA it is a plain include of the CUDA runtime
// headers. Every other file keeps the unqualified CUDA spelling, so this is
// the only header that knows about HIP.

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

// True during the device-code compilation pass. CUDA marks this with
// __CUDA_ARCH__; HIP/clang uses __HIP_DEVICE_COMPILE__ (and does NOT define
// __CUDA_ARCH__, which would falsely imply an NVIDIA target). Code that picks
// a device intrinsic vs a host fallback keys off CUKD_DEVICE_CODE.
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#define CUKD_DEVICE_CODE 1
#endif

// --- error handling -----------------------------------------------------
#define cudaError_t            hipError_t
#define cudaSuccess            hipSuccess
#define cudaGetLastError       hipGetLastError
#define cudaGetErrorString     hipGetErrorString

// --- memory -------------------------------------------------------------
#define cudaMalloc             hipMalloc
#define cudaFree               hipFree
#define cudaMallocManaged      hipMallocManaged
#define cudaMallocAsync        hipMallocAsync
#define cudaFreeAsync          hipFreeAsync
#define cudaMemcpy             hipMemcpy
#define cudaMemcpyAsync        hipMemcpyAsync
#define cudaMemsetAsync        hipMemsetAsync
#define cudaMemcpyDefault      hipMemcpyDefault
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost

// --- streams / sync / device -------------------------------------------
#define cudaStream_t           hipStream_t
#define cudaStreamSynchronize  hipStreamSynchronize
#define cudaDeviceSynchronize  hipDeviceSynchronize
#define cudaSetDevice          hipSetDevice

// --- symbols (stats path) ----------------------------------------------
#define cudaGetSymbolAddress   hipGetSymbolAddress

// CUB -> hipCUB (spatial-kdtree.h uses cub::DeviceRadixSort). hipCUB mirrors
// the cub:: API, so aliasing the namespace keeps the call sites unchanged.
#define CUKD_CUB_INCLUDE       <hipcub/hipcub.hpp>
#define cub                    hipcub

#else // CUDA

#include <cuda_runtime.h>
#include <cuda.h>

#define CUKD_CUB_INCLUDE       <cub/cub.cuh>

#if defined(__CUDA_ARCH__)
#define CUKD_DEVICE_CODE 1
#endif

#endif
