// ======================================================================== //
// Copyright 2019-2023 Ingo Wald                                            //
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

#include "builder.h"

/*! if you are looking for a "struct KDTree" or the like: the
    _default_ kd-tree in cukd is one where the the tree is entirely
    _implicit_ in the order of the data points; i.e., there _is_ no
    separate dedicated data type for a k-d tree - it's simply an array
    of points (e.g., float3's, float2s, some type of Photons for
    photon-mapping, etc), and the builder will simply re-arrange those
    data points in the array */


