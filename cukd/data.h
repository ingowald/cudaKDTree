// ======================================================================== //
// Copyright 2018-2024 Ingo Wald                                            //
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
    can be built over, and data type traits that describe this data.
*/

#pragma once

#include "cukd/cukd-math.h"
#include "cukd/box.h"

namespace cukd {


  /*! defines an abstract interface to what a 'data point' in a k-d
    tree is -- which is some sort of actual D-dimensional point of
    scalar coordinates, plus potentially some payload, and potentially
    a means of storing the split dimension). This needs to define the
    following:

    - data_traits::point_t: the actual point type that stores the
    coordinates of this data point

    - enum data_traits::has_explicit_dim : whether that node type has
    a field to store an explicit split dimension in each node. If not,
    the k-d tree builder and traverse _have_ to use round-robin for
    split distance; otherwise, it will always split the widest
    dimension.

    - enum data_traits::set_dim(data_t &, int) and
    data_traits::get_dim(const data_t &) to read and write dimensions. For
    data_t's that don't actually have any explicit split dimension
    these function may be dummies that don't do anything (they'll
    never get called in that case), but they have to be defined to
    make the compiler happy.

    The _default_ data point for this library is just the point_t
    itself: no payload, no means of storing any split dimension (ie,
    always doing round-robin dimensions), and the coordinates just
    stored as the point itself.
  */
  template<typename _point_t,
           typename _point_traits=cukd::point_traits<_point_t>>
  struct default_data_traits {
    // ------------------------------------------------------------------
    /* part I : describes the _types_ of d-dimensional point data that
       the tree will be built over */
    // ------------------------------------------------------------------
    using point_t      = _point_t;
    using point_traits = _point_traits;

    // ------------------------------------------------------------------
    /* part II : describes the type of _data_ (which can be more than
       just a point).   */
    // ------------------------------------------------------------------

    using data_t = _point_t;

    // ------------------------------------------------------------------
    /* part III : how to extract a point or coordinate from an actual
       data struct */
    // ------------------------------------------------------------------
  private:
    // this doesn't _need_ to be defined in a data_traits, but makes some of
    // the blow code cleaner to read
    using scalar_t  = typename point_traits::scalar_t;
  public:    
    /*! return a reference to the 'd'th positional coordinate of the
      given node - for the default simple 'data==point' case we can
      simply return a reference to the point itself */
    static inline __both__ const point_t &get_point(const data_t &n) { return n; }

    /*! return the 'd'th positional coordinate of the given node */
    static inline __both__
    scalar_t get_coord(const data_t &n, int d)
    { return point_traits::get_coord(get_point(n),d); }

    // ------------------------------------------------------------------
    /* part IV : whether the data has a way of storing a split
       dimension for non-round robin paritioning, and if so, how to
       store (for building) and read (for traversing) that split
       dimensional in/from a node */
    // ------------------------------------------------------------------

    /* whether that node type has a field to store an explicit split
       dimension in each node. If not, the k-d tree builder and
       traverse _have_ to use round-robin for split distance;
       otherwise, it will alwyas split the widest dimensoin */
    enum { has_explicit_dim = false };

    /*! !{ just defining this for completeness, get/set_dim should never
      get called for this type because we have set has_explicit_dim
      set to false. note traversal should ONLY ever call this
      function for data_t's that define has_explicit_dim to true */
    static inline __device__ int  get_dim(const data_t &) { return -1; }
    static inline __device__ void set_dim(data_t &, int) {}
    /*! @} */
  };

}

