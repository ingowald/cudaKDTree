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
#include "cukd/box.h"

namespace cukd {

  
  /*! defines an abstract interface to what a 'data point' in a k-d
    tree is -- which is some ort of actual D-dimensional point of
    scalar coordinates, plus potentially some payload, and potentially
    a means of storing the split dimension). This needs to define the
    follwing:

    - data_traits::point_t: the actual point type that stores the
    coordinates of this data point

    - enum data_traits::has_explicit_dim : whether that node type has
    a field to store an explicit split dimension in each node. If not,
    the k-d tree builder and traverse _have_ to use round-robin for
    split distance; otherwise, it will alwyas split the widest
    dimension. 

    - enum data_traits::set_dim(data_t &, int) and
    data_traits::get_dim(data_t &) to read and write dimensions. For
    data_t's that don't actually have any explicit split dmensoin
    these function may be dummies that don't do anything (they'll
    never get called in that case), but they have to be defined to
    make the compiler happy.

    The _default_ data point for this library is just the point_t
    itself: no payload, no means of storing any split dimension (ie,
    always doing round-robin dimensions), and the coordinates just
    stored as the point itself.
  */
  template<typename _point_t> struct default_data_traits {
    using data_t = _point_t;
    
    // ------------------------------------------------------------------
    /* part I : describes the _types_ of node of the tree, position,
       scalar, dimnensionaltiy, etc */
    // ------------------------------------------------------------------
    
    /*! the *logical* type used for mathematical things like distance
      computations, specifiing the location of a data point,
      etc. this defines number of dimensions, scalar type, etc, but
      leaves the node to define its own data layout */
    using point_t      = _point_t;
    using point_traits = point_traits<point_t>;
    using box_t        = cukd::box_t<point_t>;
    
    using scalar_t  = typename point_traits::scalar_t;
    enum { num_dims = point_traits::num_dims };
    
    // ------------------------------------------------------------------
    /* part II : how to extract a point or coordinate from an actual
       data struct */
    // ------------------------------------------------------------------

    /*! return a reference to the 'd'th positional coordinate of the
      given node - for the default simple 'data==point' case we can
      simply return a reference to the point itself */
    static inline __both__ const point_t &get_point(const data_t &n) { return n; }
    
    /*! return a reference to the 'd'th positional coordinate of the
      given node */
    static inline __both__
    scalar_t get_coord(const data_t &n, int d)
    { return point_traits::get_coord(get_point(n),d); }
    
    static inline __both__
    void set_coord(data_t &n, int d, scalar_t vv)
    { return point_traits::set_coord(get_point(n),d,vv); }
    
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
      get called for this type becaues we have set has_explicit_dim
      set to false. note traversal should ONLY ever call this
      function for data_t's that define has_explicit_dim to true */
    static inline __device__ int  get_dim(const data_t &) { return -1; }
    static inline __device__ void set_dim(data_t &, int) {}
    /*! @} */
  };


  // /*! defines default node traits for our own vec_float<N> vector type */
  // template<int N> struct default_data_traits<vec_float<N>> {
  //   using data_t   = vec_float<N>;
  //   using scalar_t = float;
  //   using point_t  = data_t;
    
  //   enum { has_explicit_dim = false };
    
  //   static inline __both__ const point_t &get_point(const data_t &n) { return n; }
  //   static inline __both__ scalar_t get_coord(const data_t &n, int d) { return n.v[d]; }
  //   static inline __both__ int  get_dim(const data_t &n) { return -1; }
  //   static inline __both__ void set_dim(data_t &n, int dim) {}
  // };
  
}


