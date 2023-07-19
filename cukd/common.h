// ======================================================================== //
// Copyright 2018-2023 Ingo Wald                                            //
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

#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
#endif
#include <math.h> // using cmath causes issues under Windows
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <assert.h>
#include <string>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <sstream>
#ifdef __GNUC__
#include <execinfo.h>
#include <sys/time.h>
#endif
#include <fstream>
#include <iostream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

#if !defined(WIN32)
#include <signal.h>
#endif

#if defined(_MSC_VER)
#  define CUKD_DLL_EXPORT __declspec(dllexport)
#  define CUKD_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#  define CUKD_DLL_EXPORT __attribute__((visibility("default")))
#  define CUKD_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define CUKD_DLL_EXPORT
#  define CUKD_DLL_IMPORT
#endif

# define CUKD_INTERFACE /* nothing - currently not building any special 'owl.dll' */
#if defined(_MSC_VER)
#  define __PRETTY_FUNCTION__ __FUNCTION__
#endif


#ifndef PRINT
# define PRINT(var) std::cout << #var << "=" << var << std::endl;
#ifdef __WIN32__
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__ << std::endl;
#else
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__ << std::endl;
#endif
#endif

#if defined(__CUDA_ARCH__)
# define __owl_device   __device__
# define __owl_host     __host__
#else
# define __owl_device   /* ignore */
# define __owl_host     /* ignore */
#endif

# define __both__   __owl_host __owl_device


#ifdef __GNUC__
#define MAYBE_UNUSED __attribute__((unused))
#else
#define MAYBE_UNUSED
#endif

#define CUKD_NOTIMPLEMENTED throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" not implemented")

#ifdef WIN32
# define CUKD_TERMINAL_RED ""
# define CUKD_TERMINAL_GREEN ""
# define CUKD_TERMINAL_LIGHT_GREEN ""
# define CUKD_TERMINAL_YELLOW ""
# define CUKD_TERMINAL_BLUE ""
# define CUKD_TERMINAL_LIGHT_BLUE ""
# define CUKD_TERMINAL_RESET ""
# define CUKD_TERMINAL_DEFAULT CUKD_TERMINAL_RESET
# define CUKD_TERMINAL_BOLD ""

# define CUKD_TERMINAL_MAGENTA ""
# define CUKD_TERMINAL_LIGHT_MAGENTA ""
# define CUKD_TERMINAL_CYAN ""
# define CUKD_TERMINAL_LIGHT_RED ""
#else
# define CUKD_TERMINAL_RED "\033[0;31m"
# define CUKD_TERMINAL_GREEN "\033[0;32m"
# define CUKD_TERMINAL_LIGHT_GREEN "\033[1;32m"
# define CUKD_TERMINAL_YELLOW "\033[1;33m"
# define CUKD_TERMINAL_BLUE "\033[0;34m"
# define CUKD_TERMINAL_LIGHT_BLUE "\033[1;34m"
# define CUKD_TERMINAL_RESET "\033[0m"
# define CUKD_TERMINAL_DEFAULT CUKD_TERMINAL_RESET
# define CUKD_TERMINAL_BOLD "\033[1;1m"

# define CUKD_TERMINAL_MAGENTA "\e[35m"
# define CUKD_TERMINAL_LIGHT_MAGENTA "\e[95m"
# define CUKD_TERMINAL_CYAN "\e[36m"
# define CUKD_TERMINAL_LIGHT_RED "\033[1;31m"
#endif

#ifdef _MSC_VER
# define CUKD_ALIGN(alignment) __declspec(align(alignment))
#else
# define CUKD_ALIGN(alignment) __attribute__((aligned(alignment)))
#endif



namespace cukd {
  namespace common {

#ifdef __WIN32__
#  define osp_snprintf sprintf_s
#else
#  define osp_snprintf snprintf
#endif

    /*! added pretty-print function for large numbers, printing 10000000 as "10M" instead */
    inline std::string prettyDouble(const double val) {
      const double absVal = abs(val);
      char result[1000];

      if      (absVal >= 1e+18f) osp_snprintf(result,1000,"%.1f%c",float(val/1e18f),'E');
      else if (absVal >= 1e+15f) osp_snprintf(result,1000,"%.1f%c",float(val/1e15f),'P');
      else if (absVal >= 1e+12f) osp_snprintf(result,1000,"%.1f%c",float(val/1e12f),'T');
      else if (absVal >= 1e+09f) osp_snprintf(result,1000,"%.1f%c",float(val/1e09f),'G');
      else if (absVal >= 1e+06f) osp_snprintf(result,1000,"%.1f%c",float(val/1e06f),'M');
      else if (absVal >= 1e+03f) osp_snprintf(result,1000,"%.1f%c",float(val/1e03f),'k');
      else if (absVal <= 1e-12f) osp_snprintf(result,1000,"%.1f%c",float(val*1e15f),'f');
      else if (absVal <= 1e-09f) osp_snprintf(result,1000,"%.1f%c",float(val*1e12f),'p');
      else if (absVal <= 1e-06f) osp_snprintf(result,1000,"%.1f%c",float(val*1e09f),'n');
      else if (absVal <= 1e-03f) osp_snprintf(result,1000,"%.1f%c",float(val*1e06f),'u');
      else if (absVal <= 1e-00f) osp_snprintf(result,1000,"%.1f%c",float(val*1e03f),'m');
      else osp_snprintf(result,1000,"%f",(float)val);

      return result;
    }


    /*! return a nicely formatted number as in "3.4M" instead of
      "3400000", etc, using mulitples of thousands (K), millions
      (M), etc. Ie, the value 64000 would be returned as 64K, and
      65536 would be 65.5K */
    inline std::string prettyNumber(const size_t s)
    {
      char buf[1000];
      if (s >= (1000LL*1000LL*1000LL*1000LL)) {
        osp_snprintf(buf, 1000,"%.2fT",s/(1000.f*1000.f*1000.f*1000.f));
      } else if (s >= (1000LL*1000LL*1000LL)) {
        osp_snprintf(buf, 1000, "%.2fG",s/(1000.f*1000.f*1000.f));
      } else if (s >= (1000LL*1000LL)) {
        osp_snprintf(buf, 1000, "%.2fM",s/(1000.f*1000.f));
      } else if (s >= (1000LL)) {
        osp_snprintf(buf, 1000, "%.2fK",s/(1000.f));
      } else {
        osp_snprintf(buf,1000,"%zi",s);
      }
      return buf;
    }

    /*! return a nicely formatted number as in "3.4M" instead of
      "3400000", etc, using mulitples of 1024 as in kilobytes,
      etc. Ie, the value 65534 would be 64K, 64000 would be 63.8K */
    inline std::string prettyBytes(const size_t s)
    {
      char buf[1000];
      if (s >= (1024LL*1024LL*1024LL*1024LL)) {
        osp_snprintf(buf, 1000,"%.2fT",s/(1024.f*1024.f*1024.f*1024.f));
      } else if (s >= (1024LL*1024LL*1024LL)) {
        osp_snprintf(buf, 1000, "%.2fG",s/(1024.f*1024.f*1024.f));
      } else if (s >= (1024LL*1024LL)) {
        osp_snprintf(buf, 1000, "%.2fM",s/(1024.f*1024.f));
      } else if (s >= (1024LL)) {
        osp_snprintf(buf, 1000, "%.2fK",s/(1024.f));
      } else {
        osp_snprintf(buf,1000,"%zi",s);
      }
      return buf;
    }

    inline double getCurrentTime()
    {
#ifdef _WIN32
      SYSTEMTIME tp; GetSystemTime(&tp);
      /*
        Please note: we are not handling the "leap year" issue.
      */
      size_t numSecsSince2020
        = tp.wSecond
        + (60ull) * tp.wMinute
        + (60ull * 60ull) * tp.wHour
        + (60ull * 60ul * 24ull) * tp.wDay
        + (60ull * 60ul * 24ull * 365ull) * (tp.wYear - 2020);
      return double(numSecsSince2020 + tp.wMilliseconds * 1e-3);
#else
      struct timeval tp; gettimeofday(&tp,nullptr);
      return double(tp.tv_sec) + double(tp.tv_usec)/1E6;
#endif
    }

    inline bool hasSuffix(const std::string &s, const std::string &suffix)
    {
      return s.substr(s.size()-suffix.size()) == suffix;
    }
  } // ::common

  template<typename T>
  inline T *loadPoints(std::string fileName, size_t &count)
  {
    // size_t count;
    std::cout << "loading points from " << fileName << std::endl;
    std::ifstream in(fileName,std::ios::binary);
    in.read((char*)&count,sizeof(count));
    // numPoints = count;
    std::cout << "loading " << count <<  " points" << std::endl;
    T *d_points = 0;
    cudaMallocManaged((void**)&d_points,count*sizeof(T));
    in.read((char*)d_points,count*sizeof(T));
    return d_points;
  }
    
  template<typename T>
  inline T *loadPoints(std::string fileName, int &count)
  {
    size_t count64;
    T *t = loadPoints<T>(fileName, count64);
    count = count64;
    return t;
  }

  // template<typename scalar_t>
  // inline __device__ scalar_t clamp(scalar_t v, scalar_t lo, scalar_t hi)
  // { return min(max(v,lo),hi); }

} // ::cukd


#define CUKD_CUDA_CHECK( call )                                         \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      throw std::runtime_error("fatal cuda error");                     \
    }                                                                   \
  }

#define CUKD_CUDA_CALL(call) CUKD_CUDA_CHECK(cuda##call)

#define CUKD_CUDA_CHECK2( where, call )                                 \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if(rc != cudaSuccess) {                                             \
      if (where)                                                        \
        fprintf(stderr, "at %s: CUDA call (%s) "                        \
                "failed with code %d (line %d): %s\n",                  \
                where,#call, rc, __LINE__, cudaGetErrorString(rc));     \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      throw std::runtime_error("fatal cuda error");                     \
    }                                                                   \
  }

#define CUKD_CUDA_SYNC_CHECK()                                  \
  {                                                             \
    cudaDeviceSynchronize();                                    \
    cudaError_t rc = cudaGetLastError();                        \
    if (rc != cudaSuccess) {                                    \
      fprintf(stderr, "error (%s: line %d): %s\n",              \
              __FILE__, __LINE__, cudaGetErrorString(rc));      \
      throw std::runtime_error("fatal cuda error");             \
    }                                                           \
  }



#define CUKD_CUDA_CHECK_NOTHROW( call )                                 \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      exit(2);                                                          \
    }                                                                   \
  }

#define CUKD_CUDA_CALL_NOTHROW(call) CUKD_CUDA_CHECK_NOTHROW(cuda##call)

#define CUKD_CUDA_CHECK2_NOTHROW( where, call )                         \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if(rc != cudaSuccess) {                                             \
      if (where)                                                        \
        fprintf(stderr, "at %s: CUDA call (%s) "                        \
                "failed with code %d (line %d): %s\n",                  \
                where,#call, rc, __LINE__, cudaGetErrorString(rc));     \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      exit(2);                                                          \
    }                                                                   \
  }


/* is supplied externally (from cmake) this adds a "int *stats"
   paramater to all query functions, and makes the traversal routines
   do atomic counting of traversal steps */
#if CUKD_ENABLE_STATS
# define CUKD_STATS(a) a
# define CUKD_STATS_ARG(a,b) a,
#else
# define CUKD_STATS(a) /* nothing */
# define CUKD_STATS_ARG(a,b) /* nothing */
#endif
