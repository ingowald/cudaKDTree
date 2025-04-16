// ======================================================================== //
// Copyright 2025-2025 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this fle except in compliance with the License.         //
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

#include "cukd/cukd-math.h"
#include "cukd/traverse-stack-free.h"
#include "cukd/knn.h"
#include <mpi.h>
#include <stdexcept>

#define CUKD_MPI_CALL(fctCall)                                          \
  { int rc = MPI_##fctCall;                                             \
    if (rc != MPI_SUCCESS)                                              \
      throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+#fctCall); }

using cukd::divRoundUp;

struct MPIComm {
  MPIComm(MPI_Comm comm)
    : comm(comm)
  {
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
  }
  MPI_Comm comm;
  int rank, size;
};

template<typename T>
std::vector<T> readFilePortion(std::string inFileName,
                               int rank, int size,
                               size_t *pBegin = 0,
                               size_t *pNumTotal = 0
                               )
{
  std::ifstream in(inFileName.c_str(),std::ios::binary);
  in.seekg(0,std::ios::end);
  size_t numBytes = in.tellg();
  in.seekg(0,std::ios::beg);

  size_t numData = numBytes / sizeof(T);
  if (pNumTotal) *pNumTotal = numData;
  size_t begin = numData * (rank+0)/size;
  if (pBegin) *pBegin = begin;
  size_t end   = numData * (rank+1)/size;
  in.seekg(begin*sizeof(T),std::ios::beg);
  
  std::vector<T> result(end-begin);
  in.read((char *)result.data(),(end-begin)*sizeof(T));
  return result;
}


void usage(const std::string &error)
{
  std::cerr << "Error: " << error << std::endl << std::endl;
  std::cerr << "./mpiHugeQuery -k <k> [-r <maxRadius>] in.float3s -o out.dat" << std::endl;
  exit(error.empty()?0:1);
}



__global__ void runQuery(float3 *tree, int N,
                         uint64_t *candidateLists, int k, float maxRadius,
                         float3 *queries, int numQueries,
                         int round)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  float3 qp = queries[tid];
  cukd::FlexHeapCandidateList cl(candidateLists+k*tid,k,
                                 round == 0 ? maxRadius : -1.f);
  cukd::stackFree::knn(cl,qp,tree,N);
}

__global__ void extractFinalResult(float *d_finalResults,
                                   int numPoints,
                                   int k,
                                   uint64_t *candidateLists)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numPoints) return;

  cukd::FlexHeapCandidateList cl(candidateLists+k*tid,k,-1.f);
  float result = cl.returnValue();
  if (!isinf(result))
    result = sqrtf(result);

  d_finalResults[tid] = result;
 }
  
int main(int ac, char **av)
{
  MPI_Init(&ac,&av);
  float maxRadius = std::numeric_limits<float>::infinity();
  int   k = 0;
  int   gpuAffinityCount = 0;
  std::string inFileName;
  std::string outFileName;

  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "-o")
      outFileName = av[++i];
    else if (arg[0] != '-')
      inFileName = arg;
    else if (arg == "-r")
      maxRadius = std::atof(av[++i]);
    else if (arg == "-g")
      gpuAffinityCount = std::atoi(av[++i]);
    else if (arg == "-k")
      k = std::atoi(av[++i]);
    else
      usage("unknown cmdline arg '"+arg+"'");
  }

  if (inFileName.empty())
    usage("no input file name specified");
  if (outFileName.empty())
    usage("no output file name specified");
  if (k < 1)
    usage("no k specified, or invalid k value");

  MPIComm mpi(MPI_COMM_WORLD);
  if (gpuAffinityCount) {
    int deviceID = mpi.rank % gpuAffinityCount;
    std::cout << "#" << mpi.rank << "/" << mpi.size
              << "setting active GPU #" << deviceID << std::endl;
    CUKD_CUDA_CALL(SetDevice(deviceID));
  }

  size_t begin = 0;
  size_t numPointsTotal = 0;
  std::vector<float3> myPoints
    = readFilePortion<float3>(inFileName,mpi.rank,mpi.size,&begin,&numPointsTotal);
  std::cout << "#" << mpi.rank << "/" << mpi.size
            << ": got " << myPoints.size() << " points to work on"
            << std::endl;

  float3 *d_tree = 0;
  float3 *d_tree_recv = 0;
  int N = myPoints.size();
  // alloc N+1 so we can store one more if anytoher rank gets oen more point
  CUKD_CUDA_CALL(Malloc((void **)&d_tree,(N+1)*sizeof(myPoints[0])));
  CUKD_CUDA_CALL(Malloc((void **)&d_tree_recv,(N+1)*sizeof(myPoints[0])));
  CUKD_CUDA_CALL(Memcpy(d_tree,myPoints.data(),N*sizeof(myPoints[0]),
                        cudaMemcpyDefault));
  cukd::buildTree(d_tree,N);

  float3   *d_queries;
  int numQueries = myPoints.size();
  uint64_t *d_cand;
  CUKD_CUDA_CALL(Malloc((void **)&d_queries,N*sizeof(float3)));
  CUKD_CUDA_CALL(Memcpy(d_queries,myPoints.data(),N*sizeof(float3),cudaMemcpyDefault));
  CUKD_CUDA_CALL(Malloc((void **)&d_cand,N*k*sizeof(uint64_t)));

  // -----------------------------------------------------------------------------
  // now, do the queries and cycling:
  // -----------------------------------------------------------------------------
  for (int round=0;round<mpi.size;round++) {
    
    if (round == 0) {
      // nothing to do , we already have our own tree
    } else {
      MPI_Request requests[2];
      int sendCount = N;
      int recvCount = 0;
      int sendPeer = (mpi.rank+1)%mpi.size;
      int recvPeer = (mpi.rank+mpi.size-1)%mpi.size;
      CUKD_MPI_CALL(Irecv(&recvCount,1*sizeof(int),MPI_BYTE,recvPeer,0,
                        mpi.comm,&requests[0]));
      CUKD_MPI_CALL(Isend(&sendCount,1*sizeof(int),MPI_BYTE,sendPeer,0,
                        mpi.comm,&requests[1]));
      CUKD_MPI_CALL(Waitall(2,requests,MPI_STATUSES_IGNORE));
      
      CUKD_MPI_CALL(Irecv(d_tree_recv,recvCount*sizeof(*d_tree),MPI_BYTE,recvPeer,0,
                          mpi.comm,&requests[0]));
      CUKD_MPI_CALL(Isend(d_tree,sendCount*sizeof(*d_tree),MPI_BYTE,sendPeer,0,
                          mpi.comm,&requests[1]));
      CUKD_MPI_CALL(Waitall(2,requests,MPI_STATUSES_IGNORE));
      
      N = recvCount;
      std::swap(d_tree,d_tree_recv);
    }
    // -----------------------------------------------------------------------------
    runQuery<<<divRoundUp(numQueries,1024),1024>>>
      (/* tree */d_tree,N,
       /* query params */d_cand,k,maxRadius,
       /* query points */d_queries,numQueries,
       round);
    CUKD_CUDA_CALL(DeviceSynchronize());
  }
  std::cout << "done all queries..." << std::endl;
  float *d_finalResults = 0;
  CUKD_CUDA_CALL(MallocManaged((void **)&d_finalResults,myPoints.size()*sizeof(float)));
  extractFinalResult<<<divRoundUp(numQueries,1024),1024>>>
    (d_finalResults,numQueries,k,d_cand);
  CUKD_CUDA_CALL(DeviceSynchronize());

  MPI_Barrier(mpi.comm);

  for (int i=0;i<mpi.size;i++) {
    MPI_Barrier(mpi.comm);
    if (i == mpi.rank) {
      FILE *file = fopen(outFileName.c_str(),i==0?"wb":"ab");
      fwrite(d_finalResults,sizeof(float),numQueries,file);
      fclose(file);
    }
    MPI_Barrier(mpi.comm);
  }
  MPI_Finalize();
}
