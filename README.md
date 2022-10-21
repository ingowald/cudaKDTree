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

The builder is templated over the type of data points; to use it, for
example, on float3 data, use thefollwing

    cukd::buildTree<float3,float,3>(points[],numPoints);
	
To do the me on float4 data, use 

    cukd::buildTree<float4,float,4>(points[],numPoints);
	
More interestingly, if you want to use a data type where you have
three float coordinates per point, and one extra 32-bit value as
"payload", you can, for example, use a `float4` for that point type,
store each point's payload value in its `float4::w` member, and then
build as follows:
	
    cukd::buildTree<float4,float,3>(points[],numPoints);
	
In this case, the biulder known that the structs provided by the user
are `float4`, but that the actual *points* are only *three* floats.

The builder included in this repo makes use of thrust for sorting; it
runs entirely on the GPU, with complexity O(N log^2 N) (and parallel
complexity O(N/k log^2 N), where K is the number of processors/cores);
it also needs only one int per data point in temp data storage during
build (plus however much thrust::sort is using, which is out of my
control).

## Stack-Free Traversal and Querying

This repo also contains a stack-free traversal code for doing
"shrinking-radius range-queries" (i.e., radius range queries where the
radius can shrink during traversal). This traversal code is used in
two examples: *fcp* (for find-closst-point) and *knn* (for k-nearest
neighbors).

For the *fct* example, you can, for example (assuming that points[]
and numPoints describe a balanced kd-tree that was built as described
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
list elements *should* end up in registers, whithout having to go to
gmem; use this for small k's. Alternatively, the `HeapCandidateList`
organizes the closest k in a heap that has O(log k) insertion cost (vs
O(k) for the fixed one), but which gets register-indirectly accessed
and will thus generate some gmem traffic; use this for larger k where
the savings from the O(log k) will actually pay off. Also note that in
the fixed list all k elements will always be stored in ascending
order, in the heap list this is not the case.




	
