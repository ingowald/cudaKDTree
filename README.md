# cudaKDTree - A Library for Building and Querying Left-Balanced (point-)k-d Trees in CUDA

Quick note: This repository contains a set of CUDA based routines for
efficiently building and performing queries in k-d trees. If k-d trees
are what you're looking for then there's a good chance that this is
what you've been looking for - after all a lot of people seem to be
using it, and seem to be quite happy with it. 

However - if your ultimate goal is to simply perform device (i.e.,
GPU) based queries for tasks such as find-closest-point, kNN, etc,
then "might I suggest" to please also have a look at my "cuBQL"
repository (now hosted on https://github.com/NVIDIA/cuBQL). Unlike
this repo cuBQL is based on BVHes, and though I don't want to get into
the weeks of which of those is better on which metric my personal
conviction is that cuBQL is most likely the more easy and more
flexible route to whatever you want to do.

# What is this Repo/Codebase?

This repository contains a set of CUDA based routines for efficiently
building and performing queries in k-d trees. It supports building
over many different (customizable) input data types, and allows for
buildling on both host and device. For device-side builds we support
three different builders, with different performance/temporary-memory
use trade-offs:


```
  Builder variants "cheat sheet"

  builder_thrust:
  - temporary memory overhead for N points: N ints + order 2N points 
    (ie, total mem order 3x that of input data!)
  - perf 100K float3s (NVIDIA RTX 4090) :   ~4ms
  - perf   1M float3s (NVIDIA RTX 4090) :  ~20ms
  - perf  10M float3s (NVIDIA RTX 4090) : ~200ms
  
  builder_bitonic:
  - temporary memory overhead for N points: N ints 
    (ie, ca 30% mem overhead for float3)
  - perf 100K float3s (NVIDIA RTX 4090) :  ~10ms
  - perf   1M float3s (NVIDIA RTX 4090) :  ~27ms
  - perf  10M float3s (NVIDIA RTX 4090) : ~390ms

  builder_inplace:
  - temporary memory overhead for N points: nada, nil, zilch.
  - perf 100K float3s (NVIDIA RTX 4090) :  ~10ms
  - perf   1M float3s (NVIDIA RTX 4090) : ~220ms
  - perf  10M float3s (NVIDIA RTX 4090) : ~4.3s

```

Note (iw, Nov 2025): Note these numbers were run on a NVIDIA
RTX 4090. Still a pretty decent GPU, but not latest and greatest any
more.

# Introduction

K-d trees are versatile data structures for organizing (and then
performing queries in) data sets of k-dimensional point data.  K-d
trees can come in many forms, including "spatial" k-d trees where
split planes can be at arbitrary locations; and the more commonly
"Bentley-"k-d trees, where all split planes have to pass *through* a
corresponding data point. The reason Bentley-style k-d trees are so
useful is that when built in a "left-balance and complete" form they
can be stored very compactly, by simply re-arranding the data points,
and without any additional memory for storing pointers or other admin
data.

This library is intended to help with efficiently building and
traversing such trees. In particular, it supports:

- both GPU- and host-side k-d tree construction

- support for both spatial k-d trees and Bentley-style (balanced) k-d trees

- support for very general types of data, including both point-only
  and point-plus-payload data

- support for what Samet calls "optimized" trees where each split
  plane's split dimension is chosen adaptively based on widest extent
  of that subtree's domain
  
- different traversal routines for the different types of trees; and
  in particular, various exemplary queries for `fcp` (find closest
  point) and `knn` (k-nearest neighbor) that should be easily
  adaptable to other types of queries

To make it easier for users to use this library for their own specific
data types we have made an effort to template all build- and traversal
routines in a way that they can be used on pretty much any type of
input data and point/vector types for math. This library has built-in
support for the regular CUDA vector types `float2/3/4` and `int2/3/4`
(`double` etc should be easy to add, but hasn't been requested yet);
for those templates should all default automatically, so building
a tree over an array of `float3`s is as simple as

``` CUDA
#include "cukd/builder.h"
...
void foo(float3 *data, int numData) {
  ...
  cukd::buildTree(data,numData);
  ...
}
```

Once built, a `fcp` query could, for example, be done like this:
``` CUDA
__global__ void myKernel(...) { 
   float3 queryPoint = ...;
   int IDofClosestDataPoint = cukd::stackBased::fcp(queryPoint,data,numData);
   ...
}
```


# *Building* Trees

This library makes heavy use of templating to allow users to 
use pretty much any form of input data, as long as it is properly
"described" to the builder through what we call "data traits".

For simple CUDA vector types (e.g., float3), all of templates should
all default to useful values, so building a tree over an array of
`float3`s is as simple as

``` CUDA
#include "cukd/builder.h"
...
void foo(float3 *data, int numData) {
  ...
  cukd::buildTree(data,numData);
  ...
}
```

A specific builder, such as for example the bitonic-sort based one,
one also be chosen directly:

``` CUDA
  cukd::buildTree_bitonic(data,numData);
```

## Support for Non-Default Data Types

The templating mechanism of this library will automatically handle
input data for CUDA vector/point data; however, actual data often
comes with additional "payload" data, in arbitrary user-defined
types. To support such data all builders are templated over both the
`data_t` (the user's actual CUDA/C++ struct for that data), and a
second `data_traits` template parameter that *describes* how to interact with this data.

In particular, to be able to build (and traverse) trees for a given
arbitrary user data struct these `data_traits` have to describe:

- the logical point/vector type to do math with. 

- how to get the actual k-dimensional position that this data point is located in

- how to get a specific one of the k coordinates of that point 

- whether or not that `data_t` allows for storing and reading a
  per-node split plane dimension (and if so, how to do that)
  
- how to read a data point's given k'th coordinate

*Note that item '3' - getting a specific coordinate - seems redundant given that the
preceding item can already deliver all k coordinates (from which we could then select
one), but for efficiency reasons it's useful to have that (most cases one needs
only one of the coordiantes, not all k), and I haven't yet found a 'clean' way of 
having this be declared only optionally, so right now it's required to have this method.*

### Example: Float3+payload, no explicit split dimension

As an example, let us consider a simple case where the user's data
contains a 3D postion, and a one-int-per-data payload, as follows:

``` C++
struct PointPlusPayload {
  float3 position;
  int    payload;
};
```
To properly describe this to this library, one can define the following
matching `data_traits` struct:

``` C++
struct PointPlusPayload_traits
  : public cukd::default_data_traits<float3>
{
  using point_t = float3;
  static inline __host__ __device__
  float3 &get_point(PointPlusPayload &data) { return data.position; }
   
  static inline __host__ __device__
  float  get_coord(const PointPlusPayload &data, int dim)
  { return cukd::get_coord(get_point(data),dim); }
};
```

In this example we have subclassed `cukd::default_data_traits<float3>`
to save us from having to define anything about split planes etc that
we are not using in this simple type. 

Using these traits, we can now call our builder on this data by simply
passing these traits as second template argument:

``` C++
void foo(PointPlusPayload *data, int numData) {
  ...
  cukd::buildTree
    </* type of the data: */PointPlusPayload,
     /* traits for this data: */PointPlusPayload_traits>
    (data,numData);
  ...
```

### Example 2: Point plus payload, within existing CUDA type

To slightly modify the above example consider an application where the
user also uses 3D float data and a single (float) payload item, but
for performance/alignment reasons wants to store those in a CUDA
`float4` vector. If we simply passed a `float4` typed array to
`buildTree()`, the builder would by default assume this to be 4D float
positions - which would be wrong - but we can once again use traits to
properly define this:
``` C++
// assume user uses a CUDA float4 in which x,y,z are position, 
// and w stores a payload
struct PackedPointPlusPayload_traits
  : public cukd::default_data_traits<float3>
{
   using point_t = float3;
   static inline __host__ __device__
   float3 &get_point(float4 &packedPointAndPayload) 
   { return make_float3(packedPointAndPayload.x,....); }
};

void foo(float4 *packedPointsAndPayloads, int count) {
   ...
   cukd::buildTree
       </* type of user data */float4,
        /* how this actually looks like */PackedPointPlusPayload_traits>
       (packedPointsAndPayloads,count)
```

### Example 3: Trees with Support "arbitrary dimension" splits

In many cases the builder can produce better trees if the split
dimension does not have to be chosen round-robin and can instead be
chosen to always subdivide the given subtree's domain where it is
widest (what Samet calls "optimized" k-d trees). To do this, however,
the builder needs a way of *storing* which dimension it picked for a
given node (so the traverser can later on retrieve this value and do
the right thing). To do this, the user's `data_t` has to have some
bits to store this value, and the corresponding `data_traits` has to
describe how the builder and traverser can read and write this data.

As an example, consider the following `Photon` data type as it could
be encountere in photon mapping:

``` C++
struct Photon { 
   // the actual photon data:
   float3  position;
   float3  power;
   // 3 bytes for quantized normal
   uint8_t quantized_normal[3];
   // 1 byte for split dimension
   uint8_t split_dim; 
};

struct Photon_traits
: /* inherit scalar_t and num dims from float3 */
  public default_point_traits<float3> 
{
   enum { has_explicit_dim = true };

   static inline __host__ __device__
   float3 &get_point(Photon &photon) 
   { return photon.position; }

   static inline __device__ int  get_dim(const Photon &p) 
   { return p.split_dim }

   static inline __device__ void set_dim(Photon &p, int dim) 
   { p.split_dim = dim; }
};
```

## Support for Non-Default *Point/Vector* Types

The main goal of using templates in this library is to allow
users to support their own data types; preferably this is done by
using the `data_traits` as described above, and using CUDA vector
types for the actual point/vector math. E.g., a user could use
an arbitrary class `Photon`, but still use the builtin `float3`
type for the actual positions.

It is in fact possible to use one's own vector/point types with this
library (by defining some suitable `points_traits<>`), 
and there are some examples and test cases that do this. However,
this functionality should only be used by users that are quite
comfortable with templating; we strongly suggest to only customize
the `data_t` and `data_traits`, and use `float3` etc for point types
where possible.


# *Querying* Trees

Whereas *building* a tree over a given set of data points is very
well-defined operation, querying is not---different users want
different queries (fcp, knn, sdf, etcpp), often with different
variants (different cut-off radius, different k for knn, vaious
short-cuts or approximations, etc). For that reason we few the two
queries that come with this library---`fcp` for find-closest-point and
`knn` for k-nearest-neighbors---more sa *samples* of how to write
other queries.

Throughout those query routines we use the same `data_traits`
templating mechanism as above, allowing the query routines to properly
interpret the data they are operating on. We provide several diferent
query routines, including the "default" stack-based k-d tree
traversal, as well as a stack-free variant, and one that is closer to
Bentley's original "ball-overlaps-domain" variant (which we call
closest-corner-tracking, or `cct` for short) that is often much(!)
better for higher-dimensional and/or highly clustered data.

## Stack-Free Traversal and Querying

This repo also contains both a stack-based and a stack-free traversal
code for doing "shrinking-radius range-queries" (i.e., radius range
queries where the radius can shrink during traversal). This traversal
code is used in two examples: *fcp* (for find-closest-point) and *knn*
(for k-nearest neighbors).

For the *fcp* example, you can, for example (assuming that `points[]`
and `numPoints` describe a balanced k-d tree that was built as described
above), be done as such

``` C++
__global__ void myKernel(float3 *points, int numPoints, ... ) {
   ...
   float3 queryPoint = ...;
   int idOfClosestPoint
     = cukd::stackBased::fcp(queryPoint,points,numPoints)
   ...
```

Similarly, a knn query for k=4 elements can be done via

``` C++
cukd::FixedCandidateList<4> closest(maxRadius);
float sqrDistOfFurthestOneInClosest
    = cukd::stackBased::knn(closest,queryPoint,points,numPoints));
```

... or for a large number of, for example, k=50 via

``` C++
cukd::HeapCandidateList<50> closest(maxRadius);
float sqrDistOfFurthestOneInClosest
    = cukd::stackBased::knn(closest,queryPoint,points,numPoints));
```

As shown in the last two examples, the `knn` code can be templated
over a "container" used for storing the k-nearest points. One such
container provided by this library is the `FixedCandidateList`, which
implements a linear, sorted priority queue in which insertion cost is
linear in the number of elements stored in the list, but where all
list elements *should* end up in registers, without having to go to
gmem; use this for small `k`s. Alternatively, the `HeapCandidateList`
organizes the closest k in a heap that has O(log k) insertion cost (vs
O(k) for the fixed one), but which gets register-indirectly accessed
and will thus generate some gmem traffic; use this for larger k where
the savings from the O(log k) will actually pay off. Also note that in
the fixed list all k elements will always be stored in ascending
order, in the heap list this is not the case.


For some query routines it is required to also pass a
`cukd::box_t<point_t>` that contains the world-space bounding box of
the data. All builders provide a means of computing this "on the fly"
during building; all that has to be done is to provide a pointer to
writeable memory where the builder can store this:

``` C++
cukd::box_t<float3> *d_boundingBox;
cudaMalloc(...);
cukd::buildTree(data,numData,d_boundingBox);
```
