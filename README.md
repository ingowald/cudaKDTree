# cudaKDTree - A Library for Building and Querying Left-Balanced (point-)kd-Trees in CUDA

Left-balanced kd-trees can be used to store and query k-dimensional
data points; their main advantage over other data structures is that
they can be stored without any pointers or other admin data - which
makes them very useful for large data sets, and/or where you want to
be able to predict how much memory you are going to use.

This repo contains CUDA code two kinds of operations: *building* such
trees, and *querying* them.

## Building Left-balanced KD-Trees

The main builder provide dy this repo is one for those that are
left-balanced, and where the split dimension in each level of the tree
is chosen in a round-robin manner; i.e., for a builder over float3
data, the root would split in x coordinate, the next level in y, then
z, then the fourth level is back to x, etc. I also have a builder
where the split dimension gets chosen based on the widest extent of
the given subtree, but that one isn't included yet - let me know if
you need it.

To allow tihs library to build k-d trees over many different types of
input data types, formats, etc, the entire builder - and most of the
traversal examples - are templated in a way that allows for expressing
different dimensionality of the input data (ie, float2 vs float3
points), different unerlying scalar types (ie, float3 vs int3 point),
whether or not there is additional payload with each data point, etc.

### Building over `float3` and similar built-in CUDA types

In its simplest way, a k-d tree is just built over a CUDA vector type
such as float3, float4, etc. In this case, the builder can be called
as simply as 

    cukd::buildTree<float3>(points[],numPoints);
	
(or in fact, if `points[]` is of type `float3 *` the compiler will
even be able to auto-deduce that). This simple variant shold work for
all of `float2`, `float3`, and `float4` (and probably even `int2`
etc - if not it's trivially simpel to add).


### Building over user types with payload data

For data points with 'payload' you can specify a so-called
`node_traits` struct that desribes how the builder can interact with
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
   
and then ask the builder to build a kd-tree over an array of such
`my_data`s. In that case, the _logical_ point type would be a `float3`
(because the tree would be built over what is logically 3-dimensional
floating point data points), but the `node_t would be tihs `my_data`.

To "explain" that to the builder, one would, for this example, define the following:

    struct my_data_traits : public default_node_traits<float3> {
	   /* inheriting from default_node_traits will
	      define scalar_t, node_t, point_t, num_dims ...*/
		  
        static inline __device__ 
		const float3 get_point(const my_data &n) 
		{ return make_float3(n.pos_x,...); }
    
		static inline __device__
		float get_coord(const my_data &n, int d)
		{ if (d==0) return n.pos_x; ... }
	};

Using this description of how to interact with a node, the builder
can then be called by

    buildTree<my_data,my_data_traits>(...)

### Building with "split-in-widest-dimension"

In particular for photon mapping, it has been shown that faster
queries can be achieved if for each node in the k-d tree we choose the
split dimensoin not in a round-robin way, but rather, by always
splitting in the widest dimension of that node's associated sub-tree.

Both our builder and our traversal allow for doing tihs; however,
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
code is used in two examples: *fcp* (for find-closst-point) and *knn*
(for k-nearest neighbors).

For the *fct* example, you can, for example (assuming that `points[]`
and `numPoints` describe a balanced kd-tree that was built as described
abvoe), be done as such

    __global__ void myKernel(float3 *points, int numPoints, ... ) {
	   ...
	   float3 queryPoint = ...;
	   int idOfClosestPoint = fcp(queryPoints,points,numPoints)
	   ...
	   
Similarly, a knn query for k=4 elements can be done via

     cukd::FixedCandidateList<4> closest(maxRadius);
	 float sqrDistOfFuthestOneInClosest
	    = cukd::knn(closest,queryPoints,points,numPoints));

... or for a large number of, for example, k=50 via

     cukd::HeapCandidateList<50> closest(maxRadius);
	 float sqrDistOfFuthestOneInClosest
	    = cukd::knn(closest,queryPoints,points,numPoints));

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




	
