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

/*! \file cukd/data.h Describes (abstract) data types (that k-d trees
    can be built over, and data type traits that describe this data 

    In particular, all the kernels and builders in this library assume
    that there's the following "things" to interact with whatever data
    a tree is going to be built over, or traversed traversed for:

    - a `scalar_type_o<
    
*/

#pragma once

#include "cukd/math.h"

namespace cukd {

  
  /*! defines an abstract interface to what a 'node' in a k-d tree
    is. This needs to define the following:

    - data_traits::scalar_t: the scalar type of each point member (eg,
    float for a float3 data_t)

    NOTE THIS IS NOT USED ANYWHERE
    - enum data_traits::num_dims: the number of dimensions of the
    data; e.g., a k-d tree build over float4 4d points would define
    tihs to '4'; a kd tree built over a struct htat has 3d position
    and some other additional payload would use '3'.

    - scalar_t data_traits::get_coord(const data_t &, int d) : return
    the 'd'th positional coordinate of the given node

    - enum data_traits::has_explicit_dim : whether that node type
    has a field to store an explicit split dimension in each
    node. If not, the k-d tree builder and traverse _have_ to use
    round-robin for split distance; otherwise, it will always
    split the widest dimension

    - data_traits::set_dim(data_t &, int) and data_traits::get_dim(data_t &)
  */
  template<typename data_t> struct default_data_traits {

    // ------------------------------------------------------------------
    /* part I : describes the _types_ of node of the tree, position,
       scalar, dimnensionaltiy, etc */
    // ------------------------------------------------------------------

    /*! the *logical* type used for mathematical things like distance
      computations, specifiing the location of a data point,
      etc. this defines number of dimensions, scalar type, etc, but
      leaves the node to define its own data layout */
    using point_t = data_t;

    using scalar_t = typename scalar_type_of<point_t>::type;

    // ------------------------------------------------------------------
    /* part II : how to extract a point or coordinate from an actual
       data struct */
    // ------------------------------------------------------------------

    /*! return a reference to the point in the given node */
    static inline __both__ const point_t &get_point(const data_t &n) { return n; }
   
    /*! return the 'd'th positional coordinate of the given node */
    static inline __both__
    scalar_t get_coord(const data_t &n, int d)
    { return cukd::get_coord(get_point(n),d); }
    
    // ------------------------------------------------------------------
    /* part III : whether the data struct has a way of storing a split
       dimension for non-round robin paritioning, and if so, how to
       store (for building) and read (for traversing) that split
       dimensional in/from a node */
    // ------------------------------------------------------------------

    /* whether that node type has a field to store an explicit split
       dimensoin in each node. If not, the k-d tree builder and
       traverse _have_ to use round-robin for split distance;
       otherwise, it will alwyas split the widest dimensoin */
    enum { has_explicit_dim = false };
    
    /*! !{ just defining this for completeness, get/set_dim should never
      get called for this type because we have set has_explicit_dim
      to false. note traversal should ONLY ever call this
      function for data_t's that define has_explicit_dim to true */
    static inline __device__ int  get_dim(const data_t &) { return -1; }
    static inline __device__ void set_dim(data_t &, int) {}
    /*! @} */
  };


  /*! defines default node traits for our own vec_float<N> vector type */
  template<int N> struct default_data_traits<vec_float<N>> {
    using data_t   = vec_float<N>;
    using scalar_t = float;
    using point_t  = data_t;
    
    enum { has_explicit_dim = false };
    
    static inline __both__ const point_t &get_point(const data_t &n) { return n; }
    static inline __both__ scalar_t get_coord(const data_t &n, int d) { return n.v[d]; }
    static inline __both__ int  get_dim(const data_t &n) { return -1; }
    static inline __both__ void set_dim(data_t &n, int dim) {}
  };
  
}


