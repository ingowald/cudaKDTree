
# sample created from reported issue #34. this sample uses a float3
# spatialKDTree with ~250K points and ~250k query points, and performs
# a set of queries on that, using a different set of points and query
# points in each run.
add_executable(knn-float3-spatialkdtree knn-float3-spatialkdtree.cu)
target_link_libraries(knn-float3-spatialkdtree PRIVATE cudaKDTree)

