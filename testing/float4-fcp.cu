#include "cukd/builder.h"
// fcp = "find closest point" query
#include "cukd/fcp.h"

float4 *generatePoints(int N)
{
  std::cout << "generating " << N <<  " points" << std::endl;
  float4 *d_points = 0;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_points,N*sizeof(float4)));
  for (int i=0;i<N;i++) {
    d_points[i].x = (float)drand48();
    d_points[i].y = (float)drand48();
    d_points[i].z = (float)drand48();
    d_points[i].w = (float)drand48();
  }
  return d_points;
}

__global__ void d_fcp(int *d_results,
                    float4 *d_queries,
                    int numQueries,
                    float4 *d_nodes,
                    int numNodes)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  d_results[tid] = cukd::fcp(d_queries[tid],d_nodes,numNodes);
}

void fcp(int *d_results,
         float4 *d_queries,
         int numQueries,
         float4 *d_nodes,
         int numNodes)
{
  int bs = 128;
  int nb = cukd::common::divRoundUp(numQueries,bs);
  d_fcp<<<nb,bs>>>(d_results,d_queries,numQueries,d_nodes,numNodes);
}

bool noneBelow(float4 *d_points, int N, int curr, int dim, float value)
{
  if (curr >= N) return true;
  return
    ((&d_points[curr].x)[dim] >= value)
    && noneBelow(d_points,N,2*curr+1,dim,value)
    && noneBelow(d_points,N,2*curr+2,dim,value);
}

bool noneAbove(float4 *d_points, int N, int curr, int dim, float value)
{
  if (curr >= N) return true;
  return
    ((&d_points[curr].x)[dim] <= value)
    && noneAbove(d_points,N,2*curr+1,dim,value)
    && noneAbove(d_points,N,2*curr+2,dim,value);
}

bool checkTree(float4 *d_points, int N, int curr=0)
{
  if (curr >= N) return true;

  int dim = cukd::BinaryTree::levelOf(curr)%4;
  float value = (&d_points[curr].x)[dim];
  
  if (!noneAbove(d_points,N,2*curr+1,dim,value))
    return false;
  if (!noneBelow(d_points,N,2*curr+2,dim,value))
    return false;
  
  return
    checkTree(d_points,N,2*curr+1)
    &&
    checkTree(d_points,N,2*curr+2);
}

int main(int ac, const char **av)
{
  using namespace cukd::common;
  
  int N = 173;
  if (ac > 1) N = std::stoi(av[1]);
  
  float4 *d_points = generatePoints(N);

  {
    double t0 = getCurrentTime();
    std::cout << "calling builder..." << std::endl;
    cukd::buildTree<float4,float>(d_points,N);
    CUKD_CUDA_SYNC_CHECK();
    double t1 = getCurrentTime();
    std::cout << "done building tree, took " << prettyDouble(t1-t0) << "s" << std::endl;
  }

  std::cout << "checking tree..." << std::endl;
  if (!checkTree(d_points,N))
    throw std::runtime_error("not a valid kd-tree!?");
  else
    std::cout << "... passed" << std::endl;

  float4 *d_queries = generatePoints(N);
  int    *d_results;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_results,N*sizeof(int)));
  {
    double t0 = getCurrentTime();
    fcp(d_results,d_queries,N,d_points,N);
    CUKD_CUDA_SYNC_CHECK();
    double t1 = getCurrentTime();
    std::cout << "done fcp query, took " << prettyDouble(t1-t0) << "s" << std::endl;
  }

  std::cout << "verifying ..." << std::endl;
  for (int i=0;i<N;i++) {
    if (d_results[i] == -1) continue;
    
    float4 qp = d_queries[i];
    float reportedDist = cukd::distance(qp,d_points[d_results[i]]);
    for (int j=0;j<N;j++) {
      float dist_j = cukd::distance(qp,d_points[j]);
      if (dist_j < reportedDist) {
        printf("for query %i: found offending point %i (%f %f %f %f) with dist %f (vs %f)\n",
               i,
               j,
               d_points[j].x,
               d_points[j].y,
               d_points[j].z,
               d_points[j].w,
               dist_j,
               reportedDist);
               
        throw std::runtime_error("verification failed ...");
      }
    }
  }
  std::cout << "verification succeeded... done." << std::endl;
}
