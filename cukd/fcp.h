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

  template <typename scalar_t>
  inline __device__ __host__
  scalar_t sqrt(scalar_t f);

  template<> inline __device__ __host__
  float sqrt(float f) { return ::sqrtf(f); }

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

  template <typename point_traits_a, typename point_traits_b=point_traits_a>
  inline __device__ __host__
  auto distance(const typename point_traits_a::point_t& a,
                const typename point_traits_b::point_t& b)
  {
    typename point_traits_a::scalar_t res = 0;
#pragma unroll
    for(int i=0; i<min(point_traits_a::numDims, point_traits_b::numDims); ++i) {
      const auto diff = point_traits_a::getCoord(a, i) - point_traits_b::getCoord(b, i);
      res += sqr(diff);
    }
    return sqrt(res);
  }

  
  // Structure of parameters to control the behavior of the FCP search.
  // By default FCP will perform an exact nearest neighbor search, but the
  // following parameters can be set to cut some corners and make the search
  // approximate in favor of speed.
  struct FcpSearchParams {
    // Controls how many "far branches" of the tree will be searched. If set to
    // 0 the algorithm will only go down the tree once following the nearest
    // branch each time. Kernels may ignore this value.
    int far_node_inspect_budget = INT_MAX;

    /*! will only search for elements that are BELOW (i.e., NOT
        including) this radius. This in particular allows for cutting
        down on the number of branches to be visited during
        traversal */
    float cutOffRadius = INFINITY;
  };





  struct FCPResult {
    inline __device__ float initialCullDist2() const
    { return closestDist2; }
    
    inline __device__ float clear(float initialDist2)
    {
      closestDist2 = initialDist2;
      closestPrimID = -1;
      return closestDist2;
    }
    
    /*! process a new candidate with given ID and (square) distance;
        and return square distance to be used for subsequent
        queries */
    inline __device__ float processCandidate(int candPrimID, float candDist2)
    {
      if (candDist2 < closestDist2) {
        closestDist2 = candDist2;
        closestPrimID = candPrimID;
      }
      return closestDist2;
    }

    inline __device__ int returnValue() const
    { return closestPrimID; }
    
    int   closestPrimID;
    float closestDist2;
  };
  

} // ::cukd





#if CUKD_IMPROVED_TRAVERSAL
# if CUKD_STACK_FREE
// stack-free, improved traversal -- only for refernce, this doesn't make sense in practice
#  include "traverse-sf-imp.h"
namespace cukd {
  template<typename node_t,
           typename node_traits=default_node_traits<node_t>>
  inline __device__
  int fcp(unsigned long long *d_stats,
          typename node_traits::point_t queryPoint,
          const box_t<typename node_traits::point_t> worldBounds,
          const node_t *d_nodes,
          int N,
          FcpSearchParams params = FcpSearchParams{})
  {
    FCPResult result;
    result.clear(sqr(params.cutOffRadius));
    traverse_sf_imp<FCPResult,node_t,node_traits>
      (result,d_stats,queryPoint,worldBounds,d_nodes,N);
    return result.returnValue();
  }
} // :: cukd

# else
// stack-free, improved traversal
#  include "traverse-cct.h"
namespace cukd {
  template<typename node_t,
           typename node_traits=default_node_traits<node_t>>
  inline __device__
  int fcp(unsigned long long *d_stats,
          typename node_traits::point_t queryPoint,
          const box_t<typename node_traits::point_t> worldBounds,
          const node_t *d_nodes,
          int N,
          FcpSearchParams params = FcpSearchParams{})
  {
    FCPResult result;
    result.clear(sqr(params.cutOffRadius));
    traverse_cct<FCPResult,node_t,node_traits>
      (result,d_stats,queryPoint,worldBounds,d_nodes,N);
    return result.returnValue();
  }
} // :: cukd

# endif
#else
# if CUKD_STACK_FREE
// stack-free, regular traversal
#  include "traverse-stack-free.h"
namespace cukd {
  template<typename node_t,
           typename node_traits=default_node_traits<node_t>>
  inline __device__
  int fcp(unsigned long long *d_stats,
          typename node_traits::point_t queryPoint,
          const node_t *d_nodes,
          int N,
          FcpSearchParams params = FcpSearchParams{})
  {
    FCPResult result;
    result.clear(sqr(params.cutOffRadius));
    traverse_stack_free<FCPResult,node_t,node_traits>
      (result,d_stats,queryPoint,d_nodes,N);
    return result.returnValue();
  }
} // :: cukd
# else
// default stack-based traversal
#  include "traverse-default-stack-based.h"
namespace cukd {
  template<typename node_t,
           typename node_traits=default_node_traits<node_t>>
  inline __device__
  int fcp(unsigned long long *d_stats,
          typename node_traits::point_t queryPoint,
          const node_t *d_nodes,
          int N,
          FcpSearchParams params = FcpSearchParams{})
  {
    FCPResult result;
    result.clear(sqr(params.cutOffRadius));
    traverse_default<FCPResult,node_t,node_traits>
      (result,d_stats,queryPoint,d_nodes,N);
    return result.returnValue();
  }
} // :: cukd

# endif
#endif

