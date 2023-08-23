# cudaKDTree - A Library for Building and Querying Left-Balanced (point-)k-d Trees in CUDA

Left-balanced k-d trees can be used to store and query k-dimensional
data points; their main advantage over other data structures is that
they can be stored without any pointers or other admin data - which
makes them very useful for large data sets, and/or where you want to
be able to predict how much memory you are going to use.

This repository contains CUDA code two kinds of operations: *building* such
trees, and *querying* them.

## Building Left-balanced k-d Trees

This repository contains three different methods for building left-balanced
and complete k-d trees. All three variants are templated over the data
type contained in the tree; often this is simply a vector/point type
like cuda `float3`, `int4`, etc; but the templating mechanism used
also allow for specify more complex data types such as points carrying
a certain payload (e.g., `struct { int3 position, int pointID };`), or
even data points that allow for specifying a split dimension (to build
what Samet calls 'generalized' k-d trees, and what Bentley originally
called 'optimized' k-d trees - those where each node can choose which
dimension it wants to split in).

The three builders all offer exactly the same caller interface, and
will all build *exactly* the same trees, but they offer different
trade-offs regarding build speed vs temporary memory used during
building. `cubit/builder_thrust` is the fastest, but relies on
`thrust`, and requires up to 3x as much memory during building as the
input data itself. `cubit/builder_bitonic` doesn't need thrust, runs
better in a stream, and in terms of temporary memory during building needs
only exactly one int per input point. Finally, `cubit/builder_inplace`
requires zero additional memory during building, but for large arrays
(> 1M points) will be about an order of magnitude slower: 

```
  Builder variants "cheat sheet"

  builder_thrust:
  - temporary memory overhead for N points: N ints + order 2N points 
    (ie, total mem order 3x that of input data!)
  - perf 100K float3s (4090) :   ~4ms
  - perf   1M float3s (4090) :  ~20ms
  - perf  10M float3s (4090) : ~200ms
  
  builder_bitonic:
  - temporary memory overhead for N points: N ints 
    (ie, ca 30% mem overhead for float3)
  - perf 100K float3s (4090) :  ~10ms
  - perf   1M float3s (4090) :  ~27ms
  - perf  10M float3s (4090) : ~390ms

  builder_inplace:
  - temporary memory overhead for N points: nada, nil, zilch.
  - perf 100K float3s (4090) :  ~10ms
  - perf   1M float3s (4090) : ~220ms
  - perf  10M float3s (4090) : ~4.3s

```

### Building over `float3` and similar built-in CUDA types

In its simplest way, a k-d tree is just built over a CUDA vector type
such as float3, float4, etc. In this case, the builder can be called
as simply as 

    cukd::buildTree<float3>(points[],numPoints);
	
(or in fact, if `points[]` is of type `float3 *` the compiler will
even be able to auto-deduce that). This simple variant should work for
all of `float2`, `float3`, and `float4` (and probably even `int2`
etc - if not it's trivially simple to add).


### Building over user types with payload data

For data points with 'payload' you can specify a so-called
`node_traits` struct that describes how the builder can interact with
each data item. Note with 'node' we mean _both_ the node of the k-d
tree, _and_ the type of each item in the input data array---since our
balanced k-d trees are built by simply re-arranging these elements it
does not make any sense to differentiate between these types. However,
this is conceptually different from what we call a `point_t` for the
given `node_t`: the `node_t` is the struct the user uses for each
element of his input array/tree, the other is what we call the
`logical` CUDA type that this would correspond to. For example, a user
might want to store his points as

    struct my_data {
       float pos_x, nor_x;
       float pos_y, nor_y;
       float pos_z, nor_z;
    };
   
and then ask the builder to build a k-d tree over an array of such
`my_data`s. In that case, the _logical_ point type would be a `float3`
(because the tree would be built over what is logically 3-dimensional
floating point data points), but the `node_t would be this `my_data`.

To "explain" that to the builder, one would, for this example, define the following:

    struct my_data_traits : public default_node_traits<float3> {
	  /* inheriting from default_node_traits will
	     define scalar_t, node_t, point_t, num_dims ...*/
		  
      static inline __device__ float3 get_point(const my_data &n) 
	  { return make_float3(n.pos_x,...); }
    
	  static inline __device__ float get_coord(const my_data &n, int d)
	  { if (d==0) return n.pos_x; ... }
	};

Using this description of how to interact with a node, the builder
can then be called by

    buildTree<my_data,my_data_traits>(...)

### Building with "split-in-widest-dimension"

In particular for photon mapping, it has been shown that faster
queries can be achieved if for each node in the k-d tree we choose the
split dimension not in a round-robin way, but rather, by always
splitting in the widest dimension of that node's associated sub-tree.

Both our builder and our traversal allow for doing this; however,
since this requires each node to be able to "somehow" store and
retrieve a split dimension this requires to define one's own node
type, and then define a `node_traits` for this that also defines some
`set_dim` and `get_dim` members. For example, for a typical photon
mapping example one could use:

    struct Photon { 
	   // the actual photon data:
	   float3  position;
	   float3  power;
	   // 3 bytes for quantized normal
	   uint8_t quantized_normal[3];
	   // 1 byte for split dimension
	   uint8_t split_dim; 
	};
	
	struct Photon_traits : public default_point_traits<float3> {
	   enum { has_explicit_dim = true };
	   
       static inline __device__ int  get_dim(const Photon &p) 
	   { return p.split_dim }
	   
       static inline __device__ void set_dim(Photon &p, int dim) 
	   { p.split_dim = dim; }
	};





## Stack-Free Traversal and Querying

This repo also contains both a stack-based and a stack-free traversal
code for doing "shrinking-radius range-queries" (i.e., radius range
queries where the radius can shrink during traversal). This traversal
code is used in two examples: *fcp* (for find-closest-point) and *knn*
(for k-nearest neighbors).

For the *fcp* example, you can, for example (assuming that `points[]`
and `numPoints` describe a balanced k-d tree that was built as described
above), be done as such

    __global__ void myKernel(float3 *points, int numPoints, ... ) {
	   ...
	   float3 queryPoint = ...;
	   int idOfClosestPoint = fcp(queryPoint,points,numPoints)
	   ...
	   
Similarly, a knn query for k=4 elements can be done via

     cukd::FixedCandidateList<4> closest(maxRadius);
	 float sqrDistOfFurthestOneInClosest
	    = cukd::knn(closest,queryPoint,points,numPoints));

... or for a large number of, for example, k=50 via

     cukd::HeapCandidateList<50> closest(maxRadius);
	 float sqrDistOfFurthestOneInClosest
	    = cukd::knn(closest,queryPoint,points,numPoints));

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




	
