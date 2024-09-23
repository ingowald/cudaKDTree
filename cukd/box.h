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

/* copied from OWL project, and put into new namespace to avoid naming conflicts.*/

#pragma once

#include "cukd/cukd-math.h"

namespace cukd {
  
  template<typename T> inline __both__ T empty_box_lower();
  template<typename T> inline __both__ T empty_box_upper();

  template<> inline __both__ float empty_box_lower<float>() { return +INFINITY; }
  template<> inline __both__ float empty_box_upper<float>() { return -INFINITY; }
  template<> inline __both__ int empty_box_lower<int>() { return INT_MAX; }
  template<> inline __both__ int empty_box_upper<int>() { return INT_MIN; }
  

  template<typename point_t>
  struct box_t {
    using point_traits = ::cukd::point_traits<point_t>;
    using scalar_t = typename point_traits::scalar_t;
    
    // inline __both__ point_t size() const { return upper - lower; }

    /*! returns the dimension in which the box has the widest extent */
    inline __both__ int widestDimension() const;
    
    inline __both__ bool contains(const point_t &p) const
    {
      enum { num_dims = num_dims_of<point_t>::value };
      for (int d=0;d<num_dims;d++) {
        if (point_traits::get_coord(p,d) < point_traits::get_coord(lower,d)) return false;
        if (point_traits::get_coord(p,d) > point_traits::get_coord(upper,d)) return false;
      }
      return true;
    }

    inline __both__ void grow(const point_t &p)
    {
      lower = min(lower,p);
      upper = max(upper,p);
    }
    
    inline __both__ void setEmpty()
    {
      for (int d=0;d<point_traits::num_dims;d++) {
        // get_coord(lower,d) = empty_box_lower<typename scalar_type_of<point_t>::type>();
        // get_coord(upper,d) = empty_box_upper<typename scalar_type_of<point_t>::type>();
        point_traits::set_coord(lower,d,empty_box_lower<scalar_t>());
        point_traits::set_coord(upper,d,empty_box_upper<scalar_t>());
      }
    }

    /*! set to an infinitely _open_ box */
    inline __both__ void setInfinite()
    {
      for (int d=0;d<point_traits::num_dims;d++) {
        // get_coord(lower,d) = empty_box_upper<typename scalar_type_of<point_t>::type>();
        // get_coord(upper,d) = empty_box_lower<typename scalar_type_of<point_t>::type>();
        point_traits::set_coord(lower,d,empty_box_upper<scalar_t>());
        point_traits::set_coord(upper,d,empty_box_lower<scalar_t>());
      }
    }

    point_t lower, upper;
  };

  /*! helper function for printf debugging */
  template<typename T>
  inline std::ostream &operator<<(std::ostream &o, const box_t<T> &b)
  {
    o << "{" << b.lower << "," << b.upper << "}";
    return o;
  }
  
  /*! computes the closest point to 'point' that's within the given
    box; if point itself is inside that box it'll be the point
    itself, otherwise it'll be a point on the outside surface of the
    box */
  template<typename point_t>
  inline __device__
  point_t project(const cukd::box_t<point_t>  &box,
                  const point_t               &point)
  {
    return min(max(point,box.lower),box.upper);
  }

  // ------------------------------------------------------------------
  template<typename point_t>
  inline __device__
  auto sqrDistance(const box_t<point_t> &box, const point_t &point)
  { return cukd::sqrDistance(project(box,point),point); }

  template<typename point_t>
  /*! returns the dimension in which the box has the widest extent */
  inline __both__ int box_t<point_t>::widestDimension() const
  {
    enum { num_dims = point_traits::num_dims };
      
    int d_best = 0;
    scalar_t w_best = scalar_t(0);
    for (int d=0;d<num_dims;d++) {
      scalar_t w_d = get_coord(upper,d) - get_coord(lower,d);
      if (w_d < w_best) continue;
      w_best = w_d;
      d_best = d;
    }
    return d_best;
  }
  
} // ::cukd
