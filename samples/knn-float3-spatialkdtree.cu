/* sample created from reported issue #34. this sample uses a float3
 spatialKDTree with ~250K points and ~250k query points, and performs
 a set of queries on that, using a different set of points and query
 points in each run.
*/

#include <cuda_runtime.h>
#include <cukd/builder.h>
#include <cukd/knn.h>
#include <random>

#define FIXED_K 16

using data_t = float3;
using data_traits = cukd::default_data_traits<float3>;

// CUDA KNN Kernel
__global__ void KnnKernel(
    const float3* d_queries, int numQueries,
    const cukd::SpatialKDTree<float3, data_traits> tree,
    float3* d_results, int k, float radius)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numQueries) return;

    cukd::HeapCandidateList<FIXED_K> result(radius); // Fixed at 16, for generalization make template

    cukd::stackBased::knn<decltype(result), float3, data_traits>
      (result, tree, d_queries[tid]);

    for (int i = 0; i < k; i++) {
      int ID = result.get_pointID(i);
      d_results[tid * k + i]
        = ID < 0
        ? make_float3(0.f,0.f,0.f)
        : tree.data[ID];
    }
}

float3* knnSearchCuda(const float3* points, const int numPoints,
                      const float3* queries, const int numQueries,
                      const int k, const float radius) {

    // Allocate managed memory for points, queries, and results
    float3* d_points;
    cudaMallocManaged(&d_points, numPoints * sizeof(float3));
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    std::cout << "Allocated " << numPoints << " points at " << d_points << std::endl;
    cudaMemcpy(d_points, points, numPoints * sizeof(float3), cudaMemcpyHostToDevice);

    float3* d_queries;
    cudaMallocManaged(&d_queries, numQueries * sizeof(float3));
    cudaMemcpy(d_queries, queries, numQueries * sizeof(float3), cudaMemcpyHostToDevice);

    // Build Spatial KD-Tree (managed memory)
    cukd::SpatialKDTree<float3, data_traits> tree;
    cukd::BuildConfig buildConfig{};
    buildTree(tree,d_points,numPoints,buildConfig);
    
    CUKD_CUDA_SYNC_CHECK();

    // Results
    float3* d_results;
    cudaMallocManaged(&d_results, numQueries * k * sizeof(float3));

    int threadsPerBlock = 256;
    int numBlocks = (numQueries + threadsPerBlock - 1) / threadsPerBlock;

    KnnKernel<<<numBlocks, threadsPerBlock>>>(d_queries, numQueries, tree, d_results, k, radius);
    cudaDeviceSynchronize();

    // Copy back results
    float3* neighbors = new float3[numQueries * k];
    cudaMemcpy(neighbors, d_results, numQueries * k * sizeof(float3), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_points);
    cudaFree(d_queries);
    cudaFree(d_results);
    cukd::free(tree);

    return neighbors;
}

int main(int, char **) {
  std::random_device rd;  // a seed source for the random number engine
  std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> rng(0.f,+1.f);

  for (int r=0;r<10;r++) {
    std::vector<float3> points;
    {
      int N = 240000+int(20000*rng(gen));
      for (int i=0;i<N;i++) {
        points.push_back(make_float3(rng(gen),rng(gen),rng(gen)));
      }
    }
    std::vector<float3> queries;
    {
      int N = 240000+int(20000*rng(gen));
      for (int i=0;i<N;i++) {
        queries.push_back(make_float3(rng(gen),rng(gen),rng(gen)));
      }
    }
    std::cout << "running knn query on " << points.size()
              << " points" << std::endl;
    float3 *result
      = knnSearchCuda(points.data(),points.size(),
                      queries.data(),queries.size(),
                      FIXED_K,2.f);
    delete[] result;
  }
}


