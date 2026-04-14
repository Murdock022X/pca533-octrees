#include "cstone/cuda/device_vector.h"
#include "cstone/cuda/errorcheck.cuh"
#include "cstone/domain/domain.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/sfc/common.hpp"
#include "cstone/tree/definitions.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <highfive/H5File.hpp>
#include <string>
#include <algorithm>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <vector>
#include "cstone/sfc/sfc.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/tree/octree_gpu.h"
#include "cstone/tree/update_gpu.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/gather.hpp"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/focus/source_center_gpu.h"
#include "cstone/traversal/collisions_gpu.h"
#include "cstone/tree/update_gpu.cuh"
#include "cstone/tree/octree_gpu.h"
#include "cstone/findneighbors.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/primitives/gather.hpp"
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <nvtx3/nvToolsExt.h>
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/sfc/sfc_gpu.h"
#include "cstone/tree/octree_gpu.h"
#include "cstone/tree/update_gpu.cuh"
#include "pcah5.hpp"
#include "runner.hpp"
#include "save_octree.cuh"
#include "utils.hpp"
#include "cstone/sfc/box.hpp"

template<class KeyType>
__global__ void
computeSfcKeysKernel(KeyType* keys, const float4* vals, size_t numKeys, const cstone::Box<float> box)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numKeys)
    {
      float4 val = vals[tid];
        KeyType rmKey = cstone::removeKey<KeyType>{};
        if (keys[tid] != rmKey) { keys[tid] = cstone::sfc3D<KeyType>(val.x, val.y, val.z, box); }
    }
}

template<class KeyType>
void computeSfcKeysGpu(const float4* vals, KeyType* keys, size_t numKeys, const cstone::Box<float>& box)
{
    if (numKeys == 0) { return; }

    constexpr int threadsPerBlock = 256;
    computeSfcKeysKernel<<<(numKeys + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(keys, vals, numKeys, box);
    checkGpuErrors(cudaGetLastError());
}

void processGpuVec(cstone::Box<Real> &box, 
  cstone::DeviceVector<KeyType> &d_keys, 
  cstone::DeviceVector<KeyType> &d_keys_tmp, 
  cstone::DeviceVector<int> &d_ordering, float4* d_values_tmp, 
  cstone::DeviceVector<Real> &tmp,
  cstone::DeviceVector<char> &cubTmpStorage, uint64_t tempStorageEle,
  cstone::DeviceVector<unsigned> &d_counts, 
  cstone::DeviceVector<cstone::TreeNodeIndex> &workArray,
  cstone::DeviceVector<cstone::LocalIndex> &d_layout,
  cstone::DeviceVector<KeyType> &d_tree, cstone::DeviceVector<KeyType> &tmpTree, 
  cstone::OctreeData<KeyType, cstone::GpuTag> &octreeGpuData, 
  float4* d_vals,
  int bucketSize, size_t np) {

  nvtxRangePushA("ComputeKeys");
  computeSfcKeysGpu(d_vals, cstone::sfcKindPointer(rawPtr(d_keys)), np, box);
  nvtxRangePop();

  nvtxRangePushA("SortKeys");
  // cstone::sequenceGpu(rawPtr(d_ordering), np, 0);
  cstone::sortByKeyGpu(rawPtr(d_keys), rawPtr(d_keys) + np, d_vals, rawPtr(d_keys_tmp), d_values_tmp, rawPtr(cubTmpStorage), tempStorageEle);
  nvtxRangePop();

  // nvtxRangePushA("ReorderXYZK");
  // thrust::gather(thrust::device, rawPtr(d_ordering), rawPtr(d_ordering) + np, rawPtr(d_x), tmp.data());
  // thrust::copy(thrust::device, rawPtr(tmp), rawPtr(tmp) + np, rawPtr(d_x));
  // thrust::gather(thrust::device, rawPtr(d_ordering), rawPtr(d_ordering) + np, rawPtr(d_y), tmp.data());
  // thrust::copy(thrust::device, rawPtr(tmp), rawPtr(tmp) + np, rawPtr(d_y));
  // thrust::gather(thrust::device, rawPtr(d_ordering), rawPtr(d_ordering) + np, rawPtr(d_z), tmp.data());
  // thrust::copy(thrust::device, rawPtr(tmp), rawPtr(tmp) + np, rawPtr(d_z));
  // nvtxRangePop();

  if (d_tree.size() == 0)
  {
      // initial guess on first call. use previous tree as guess on subsequent calls
      d_tree = std::vector<KeyType>{0, cstone::nodeRange<KeyType>(0)};
      d_counts = std::vector<unsigned>{unsigned(np)};
  }

  nvtxRangePushA("UpdateLeaves");
  while (!cstone::updateOctreeGpu<KeyType>({rawPtr(d_keys), d_keys.size()}, bucketSize, d_tree, d_counts,
                                                 tmpTree, workArray));
  nvtxRangePop();

  nvtxRangePushA("UpdateInternal");
  octreeGpuData.resize(cstone::nNodes(d_tree));
  cstone::buildOctreeGpu(rawPtr(d_tree), octreeGpuData.data());

  d_layout.resize(d_counts.size() + 1);
  cstone::fillGpu(rawPtr(d_layout), rawPtr(d_layout) + 1, cstone::LocalIndex(0));
  cstone::inclusiveScanGpu(rawPtr(d_counts), rawPtr(d_counts) + d_counts.size(), rawPtr(d_layout) + 1);
  nvtxRangePop();
}

void processGpuVecGather(cstone::Box<float> &box, 
  cstone::DeviceVector<KeyType> &d_keys, 
  cstone::DeviceVector<KeyType> &d_keys_tmp, 
  cstone::DeviceVector<int> &d_ordering, cstone::DeviceVector<int> &d_values_tmp, 
  float4* tmp, 
  cstone::DeviceVector<char> &cubTmpStorage, uint64_t tempStorageEle,
  cstone::DeviceVector<unsigned> &d_counts, 
  cstone::DeviceVector<cstone::TreeNodeIndex> &workArray,
  cstone::DeviceVector<cstone::LocalIndex> &d_layout,
  cstone::DeviceVector<KeyType> &d_tree, cstone::DeviceVector<KeyType> &tmpTree, 
  cstone::OctreeData<KeyType, cstone::GpuTag> &octreeGpuData, 
  float4* d_vals,
  int bucketSize, size_t np) {

  nvtxRangePushA("ComputeKeys");
  computeSfcKeysGpu(d_vals, cstone::sfcKindPointer(rawPtr(d_keys)), np, box);
  cudaDeviceSynchronize();
  nvtxRangePop();

  nvtxRangePushA("SortKeys");
  cstone::sequenceGpu(rawPtr(d_ordering), np, 0);
  cstone::sortByKeyGpu(rawPtr(d_keys), rawPtr(d_keys) + np, rawPtr(d_ordering), rawPtr(d_keys_tmp), rawPtr(d_values_tmp), rawPtr(cubTmpStorage), tempStorageEle);
  nvtxRangePop();

  nvtxRangePushA("ReorderXYZK");
  thrust::gather(thrust::device, rawPtr(d_ordering), rawPtr(d_ordering) + np, d_vals, tmp);
  thrust::copy(thrust::device, tmp, tmp + np, d_vals);
  nvtxRangePop();

  if (d_tree.size() == 0)
  {
      // initial guess on first call. use previous tree as guess on subsequent calls
      d_tree = std::vector<KeyType>{0, cstone::nodeRange<KeyType>(0)};
      d_counts = std::vector<unsigned>{unsigned(np)};
  }

  nvtxRangePushA("UpdateLeaves");
  while (!cstone::updateOctreeGpu<KeyType>({rawPtr(d_keys), d_keys.size()}, bucketSize, d_tree, d_counts,
                                                 tmpTree, workArray));
  nvtxRangePop();

  nvtxRangePushA("UpdateInternal");
  octreeGpuData.resize(cstone::nNodes(d_tree));
  cstone::buildOctreeGpu(rawPtr(d_tree), octreeGpuData.data());

  d_layout.resize(d_counts.size() + 1);
  cstone::fillGpu(rawPtr(d_layout), rawPtr(d_layout) + 1, cstone::LocalIndex(0));
  cstone::inclusiveScanGpu(rawPtr(d_counts), rawPtr(d_counts) + d_counts.size(), rawPtr(d_layout) + 1);
  nvtxRangePop();
}

void processGpu(cstone::Box<Real> &box, 
  cstone::DeviceVector<KeyType> &d_keys, 
  cstone::DeviceVector<KeyType> &d_keys_tmp, 
  cstone::DeviceVector<int> &d_ordering, cstone::DeviceVector<int> &d_values_tmp, 
  cstone::DeviceVector<Real> &tmp, 
  cstone::DeviceVector<char> &cubTmpStorage, uint64_t tempStorageEle,
  cstone::DeviceVector<unsigned> &d_counts, 
  cstone::DeviceVector<cstone::TreeNodeIndex> &workArray,
  cstone::DeviceVector<cstone::LocalIndex> &d_layout,
  cstone::DeviceVector<KeyType> &d_tree, cstone::DeviceVector<KeyType> &tmpTree, 
  cstone::OctreeData<KeyType, cstone::GpuTag> &octreeGpuData, 
  cstone::DeviceVector<Real> &d_x, cstone::DeviceVector<Real> &d_y, cstone::DeviceVector<Real> &d_z, 
  int bucketSize, size_t np) {

  nvtxRangePushA("ComputeKeys");
  cstone::computeSfcKeysGpu(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), cstone::sfcKindPointer(rawPtr(d_keys)), np, box);
  nvtxRangePop();

  nvtxRangePushA("SortKeys");
  cstone::sequenceGpu(rawPtr(d_ordering), np, 0);
  cstone::sortByKeyGpu(rawPtr(d_keys), rawPtr(d_keys) + np, rawPtr(d_ordering), rawPtr(d_keys_tmp), rawPtr(d_values_tmp), rawPtr(cubTmpStorage), tempStorageEle);
  nvtxRangePop();

  nvtxRangePushA("ReorderXYZK");
  thrust::gather(thrust::device, rawPtr(d_ordering), rawPtr(d_ordering) + np, rawPtr(d_x), tmp.data());
  thrust::copy(thrust::device, rawPtr(tmp), rawPtr(tmp) + np, rawPtr(d_x));
  thrust::gather(thrust::device, rawPtr(d_ordering), rawPtr(d_ordering) + np, rawPtr(d_y), tmp.data());
  thrust::copy(thrust::device, rawPtr(tmp), rawPtr(tmp) + np, rawPtr(d_y));
  thrust::gather(thrust::device, rawPtr(d_ordering), rawPtr(d_ordering) + np, rawPtr(d_z), tmp.data());
  thrust::copy(thrust::device, rawPtr(tmp), rawPtr(tmp) + np, rawPtr(d_z));
  nvtxRangePop();

  if (d_tree.size() == 0)
  {
      // initial guess on first call. use previous tree as guess on subsequent calls
      d_tree = std::vector<KeyType>{0, cstone::nodeRange<KeyType>(0)};
      d_counts = std::vector<unsigned>{unsigned(np)};
  }

  nvtxRangePushA("UpdateLeaves");
  while (!cstone::updateOctreeGpu<KeyType>({rawPtr(d_keys), d_keys.size()}, bucketSize, d_tree, d_counts,
                                                 tmpTree, workArray));
  nvtxRangePop();

  nvtxRangePushA("UpdateInternal");
  octreeGpuData.resize(cstone::nNodes(d_tree));
  cstone::buildOctreeGpu(rawPtr(d_tree), octreeGpuData.data());

  d_layout.resize(d_counts.size() + 1);
  cstone::fillGpu(rawPtr(d_layout), rawPtr(d_layout) + 1, cstone::LocalIndex(0));
  cstone::inclusiveScanGpu(rawPtr(d_counts), rawPtr(d_counts) + d_counts.size(), rawPtr(d_layout) + 1);
  nvtxRangePop();
}

// std::pair<double, double> runnerGpu(const std::vector<KeyType> &keys, const std::vector<Real> &ix,
//                const std::vector<Real> &iy, const std::vector<Real> &iz,
//                const std::vector<Real> &h, const std::vector<Real> &px,
//                const std::vector<Real> &py, const std::vector<Real> &pz, int rank,
//                int numRanks, int bucketSize, int bucketSizeFocus, float theta,
//                std::string group_name, bool save) {
//   cstone::Box<Real> box{-1.15, 1.15};

//   std::vector<Real> x(ix), y(iy), z(iz);

//   std::pair<double, double> t;

//   size_t np = keys.size();
//   int call_count = 1;

//   std::vector<float4> vals(keys.size());
//   for (size_t i = 0; i < keys.size(); ++i) {
//     vals[i] = make_float4(x[i], y[i], z[i], h[i]);
//   }

//   std::cout << "Running GPU Octree Build and Sync Benchmark with " << np
//             << " particles, bucket size: " << bucketSize
//             << ", theta: " << theta << std::endl;
  
//   cstone::DeviceVector<KeyType> d_keys(keys.size());
//   cstone::DeviceVector<KeyType> d_tree, tmpTree;
//   cstone::OctreeData<KeyType, cstone::GpuTag> octreeGpuData;
//   cstone::DeviceVector<KeyType> d_keys_tmp(keys.size());
//   cstone::DeviceVector<unsigned> d_counts;
//   cstone::DeviceVector<cstone::TreeNodeIndex> workArray;

//   cstone::DeviceVector<int> d_ordering(keys.size());
//   cstone::DeviceVector<Real> tmp(keys.size());
//   float4 *d_vals, *d_values_tmp;
//   cudaMalloc<float4>(&d_vals, vals.size());
//   cudaMemcpy(d_vals, vals.data(), vals.size() * sizeof(float4), cudaMemcpyHostToDevice);
//   cudaMalloc<float4>(&d_values_tmp, vals.size() * sizeof(float4));

//   cstone::DeviceVector<cstone::LocalIndex> d_layout;

//   uint64_t tempStorageEle = cstone::sortByKeyTempStorage<KeyType, float4>(np);
//   cstone::DeviceVector<char> cubTmpStorage(tempStorageEle);

//   std::cout << "Temp storage bytes: " << tempStorageEle << std::endl;

//   auto f = [&]() {
//     processGpuVec(box, d_keys, d_keys_tmp, d_ordering, d_values_tmp, tmp, cubTmpStorage, tempStorageEle, 
//       d_counts, workArray, d_layout, d_tree, tmpTree, octreeGpuData, d_vals, bucketSize, np);
//   };
  
//   nvtxRangePushA("Initial");
//   float sync_ms = timeGpu(f);
//   nvtxRangePop();

//   t.first = sync_ms;

//   if (rank == 0)
//     std::cout << "\tUpdate Octree Initial: " << sync_ms << "us, call count: " << call_count
//               << std::endl;

//   // if (save) {
//   //   std::vector<KeyType> keys_host(d_keys.size());
//   //   cudaMemcpy(x.data(), d_ix.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
//   //   cudaMemcpy(y.data(), d_iy.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
//   //   cudaMemcpy(z.data(), d_iz.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
//   //   cudaMemcpy(keys_host.data(), d_keys.data(), np * sizeof(KeyType), cudaMemcpyDeviceToHost);
//   //   saveOctreeH5Gpu(box, octreeGpuData, d_tree, group_name + "_initial", rank, numRanks, x, y, z, keys_host);
//   // }

//   #pragma omp parallel for
//   for (auto i = 0; i < vals.size(); ++i) {
//     vals[i].x += px[i];
//     vals[i].y += py[i];
//     vals[i].z += pz[i];
//   }

//   cudaMemcpy(d_vals, vals.data(), vals.size() * sizeof(float4), cudaMemcpyHostToDevice);

//   nvtxRangePushA("Perturb");
//   sync_ms = timeGpu(f);
//   nvtxRangePop();

//   t.second = sync_ms;

//   call_count = 1;

//   if (rank == 0)
//     std::cout << "\tPerturbation update time: " << sync_ms << " us, call count: " << call_count
//               << std::endl;
  

//   cudaFree(d_vals);
//   cudaFree(d_values_tmp);
//   // if (save) {
//   //   std::vector<KeyType> keys_host(d_keys.size());
//   //   cudaMemcpy(x.data(), d_ix.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
//   //   cudaMemcpy(y.data(), d_iy.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
//   //   cudaMemcpy(z.data(), d_iz.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
//   //   cudaMemcpy(keys_host.data(), d_keys.data(), np * sizeof(KeyType), cudaMemcpyDeviceToHost);
//   //   saveOctreeH5Gpu(box, octreeGpuData, d_tree, group_name + "_perturbed", rank, numRanks, x, y, z, keys_host);
//   // }

//   return t;
// }

std::pair<double, double> runnerGpu(const std::vector<KeyType> &keys, const std::vector<Real> &ix,
               const std::vector<Real> &iy, const std::vector<Real> &iz,
               const std::vector<Real> &h, const std::vector<Real> &px,
               const std::vector<Real> &py, const std::vector<Real> &pz, int rank,
               int numRanks, int bucketSize, int bucketSizeFocus, float theta,
               std::string group_name, bool save) {
  cstone::Box<float> box{-1.15, 1.15};

  std::pair<double, double> t;

  size_t np = keys.size();
  int call_count = 1;

  std::cout << "Running GPU Octree Build and Sync Benchmark with " << np
            << " particles, bucket size: " << bucketSize
            << ", theta: " << theta << std::endl;
  
  cstone::DeviceVector<KeyType> d_keys(keys.size());
  cstone::DeviceVector<KeyType> d_tree, tmpTree;
  cstone::OctreeData<KeyType, cstone::GpuTag> octreeGpuData;
  cstone::DeviceVector<KeyType> d_keys_tmp(keys.size());
  cstone::DeviceVector<unsigned> d_counts;
  cstone::DeviceVector<cstone::TreeNodeIndex> workArray;

  cstone::DeviceVector<int> d_ordering(keys.size()), d_values_tmp(keys.size());
  float4* tmp;
  cudaMalloc<float4>(&tmp, keys.size() * sizeof(float4));
  // cstone::DeviceVector<Real> d_ix(x), d_iy(y), d_iz(z);
  cstone::DeviceVector<cstone::LocalIndex> d_layout;

  std::vector<float4> vals(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    vals[i] = make_float4(ix[i], iy[i], iz[i], h[i]);
  }

  float4 *d_vals;
  cudaMalloc<float4>(&d_vals, vals.size()*sizeof(float4));
  cudaMemcpy(d_vals, vals.data(), vals.size() * sizeof(float4), cudaMemcpyHostToDevice);

  uint64_t tempStorageEle = cstone::sortByKeyTempStorage<KeyType, cstone::LocalIndex>(np);
  cstone::DeviceVector<char> cubTmpStorage(tempStorageEle);

  auto f = [&]() {
    processGpuVecGather(box, d_keys, d_keys_tmp, d_ordering, d_values_tmp, tmp, cubTmpStorage, tempStorageEle, 
      d_counts, workArray, d_layout, d_tree, tmpTree, octreeGpuData, d_vals, bucketSize, np);
  };
  
  nvtxRangePushA("Initial");
  float sync_ms = timeGpu(f);
  nvtxRangePop();

  t.first = sync_ms;

  if (rank == 0)
    std::cout << "\tUpdate Octree Initial: " << sync_ms << "us, call count: " << call_count
              << std::endl;

  // if (save) {
  //   std::vector<KeyType> keys_host(d_keys.size());
  //   cudaMemcpy(x.data(), d_ix.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
  //   cudaMemcpy(y.data(), d_iy.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
  //   cudaMemcpy(z.data(), d_iz.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
  //   cudaMemcpy(keys_host.data(), d_keys.data(), np * sizeof(KeyType), cudaMemcpyDeviceToHost);
  //   saveOctreeH5Gpu(box, octreeGpuData, d_tree, group_name + "_initial", rank, numRanks, x, y, z, keys_host);
  // }

  #pragma omp parallel for
  for (auto i = 0; i < vals.size(); ++i) {
    vals[i].x += px[i];
    vals[i].y += py[i];
    vals[i].z += pz[i];
  }

  cudaMemcpy(d_vals, vals.data(), vals.size() * sizeof(float4), cudaMemcpyHostToDevice);

  nvtxRangePushA("Perturb");
  sync_ms = timeGpu(f);
  nvtxRangePop();

  t.second = sync_ms;

  call_count = 1;

  if (rank == 0)
    std::cout << "\tPerturbation update time: " << sync_ms << " us, call count: " << call_count
              << std::endl;

  cudaFree(d_vals);
  
  // if (save) {
  //   std::vector<KeyType> keys_host(d_keys.size());
  //   cudaMemcpy(x.data(), d_ix.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
  //   cudaMemcpy(y.data(), d_iy.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
  //   cudaMemcpy(z.data(), d_iz.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
  //   cudaMemcpy(keys_host.data(), d_keys.data(), np * sizeof(KeyType), cudaMemcpyDeviceToHost);
  //   saveOctreeH5Gpu(box, octreeGpuData, d_tree, group_name + "_perturbed", rank, numRanks, x, y, z, keys_host);
  // }

  return t;
}

// std::pair<double, double> runnerGpu(const std::vector<KeyType> &keys, const std::vector<Real> &ix,
//                const std::vector<Real> &iy, const std::vector<Real> &iz,
//                const std::vector<Real> &h, const std::vector<Real> &px,
//                const std::vector<Real> &py, const std::vector<Real> &pz, int rank,
//                int numRanks, int bucketSize, int bucketSizeFocus, float theta,
//                std::string group_name, bool save) {
//   cstone::Box<Real> box{-1.15, 1.15};

//   std::vector<Real> x(ix), y(iy), z(iz);

//   std::pair<double, double> t;

//   size_t np = keys.size();
//   int call_count = 1;

//   std::cout << "Running GPU Octree Build and Sync Benchmark with " << np
//             << " particles, bucket size: " << bucketSize
//             << ", theta: " << theta << std::endl;
  
//   cstone::DeviceVector<KeyType> d_keys(keys.size());
//   cstone::DeviceVector<KeyType> d_tree, tmpTree;
//   cstone::OctreeData<KeyType, cstone::GpuTag> octreeGpuData;
//   cstone::DeviceVector<KeyType> d_keys_tmp(keys.size());
//   cstone::DeviceVector<unsigned> d_counts;
//   cstone::DeviceVector<cstone::TreeNodeIndex> workArray;

//   cstone::DeviceVector<int> d_ordering(keys.size()), d_values_tmp(keys.size());
//   cstone::DeviceVector<Real> tmp(keys.size());
//   cstone::DeviceVector<Real> d_ix(x), d_iy(y), d_iz(z);
//   cstone::DeviceVector<cstone::LocalIndex> d_layout;

//   uint64_t tempStorageEle = cstone::sortByKeyTempStorage<KeyType, cstone::LocalIndex>(np);
//   cstone::DeviceVector<char> cubTmpStorage(tempStorageEle);

//   auto f = [&]() {
//     processGpu(box, d_keys, d_keys_tmp, d_ordering, d_values_tmp, tmp, cubTmpStorage, tempStorageEle, 
//       d_counts, workArray, d_layout, d_tree, tmpTree, octreeGpuData, d_ix, d_iy, d_iz, bucketSize, np);
//   };
  
//   nvtxRangePushA("Initial");
//   float sync_ms = timeGpu(f);
//   nvtxRangePop();

//   t.first = sync_ms;

//   if (rank == 0)
//     std::cout << "\tUpdate Octree Initial: " << sync_ms << "us, call count: " << call_count
//               << std::endl;

//   // if (save) {
//   //   std::vector<KeyType> keys_host(d_keys.size());
//   //   cudaMemcpy(x.data(), d_ix.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
//   //   cudaMemcpy(y.data(), d_iy.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
//   //   cudaMemcpy(z.data(), d_iz.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
//   //   cudaMemcpy(keys_host.data(), d_keys.data(), np * sizeof(KeyType), cudaMemcpyDeviceToHost);
//   //   saveOctreeH5Gpu(box, octreeGpuData, d_tree, group_name + "_initial", rank, numRanks, x, y, z, keys_host);
//   // }

//   #pragma omp parallel for
//   for (auto i = 0; i < ix.size(); ++i) {
//     x[i] += px[i];
//     y[i] += py[i];
//     z[i] += pz[i];
//   }

//   d_ix = x;
//   d_iy = y;
//   d_iz = z;

//   nvtxRangePushA("Perturb");
//   sync_ms = timeGpu(f);
//   nvtxRangePop();

//   t.second = sync_ms;

//   call_count = 1;

//   if (rank == 0)
//     std::cout << "\tPerturbation update time: " << sync_ms << " us, call count: " << call_count
//               << std::endl;
  
//   // if (save) {
//   //   std::vector<KeyType> keys_host(d_keys.size());
//   //   cudaMemcpy(x.data(), d_ix.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
//   //   cudaMemcpy(y.data(), d_iy.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
//   //   cudaMemcpy(z.data(), d_iz.data(), np * sizeof(Real), cudaMemcpyDeviceToHost);
//   //   cudaMemcpy(keys_host.data(), d_keys.data(), np * sizeof(KeyType), cudaMemcpyDeviceToHost);
//   //   saveOctreeH5Gpu(box, octreeGpuData, d_tree, group_name + "_perturbed", rank, numRanks, x, y, z, keys_host);
//   // }

//   return t;
// }

std::pair<double, double> runnerGpuMulti(const std::vector<KeyType> &keys, const std::vector<Real> &ix,
               const std::vector<Real> &iy, const std::vector<Real> &iz,
               const std::vector<Real> &h, const std::vector<Real> &px,
               const std::vector<Real> &py, const std::vector<Real> &pz, int rank,
               int numRanks, int bucketSize, int bucketSizeFocus, float theta,
               std::string group_name, bool save) {
  size_t free_initial_byte, free_byte;
  size_t total_initial_byte, total_byte;

  std::pair <double, double> t;

  cudaDeviceSynchronize();

  // Get memory info
  checkGpuErrors(cudaMemGetInfo(&free_initial_byte, &total_initial_byte));

  size_t used_initial_byte = total_initial_byte - free_initial_byte;

  cstone::Domain<KeyType, Real, cstone::GpuTag> domain(
      rank, numRanks, bucketSize, bucketSizeFocus, theta);

  cstone::DeviceVector<KeyType> d_keys(keys.size());
  cstone::DeviceVector<Real> d_ix(ix), d_iy(iy), d_iz(iz), d_h(h);
  cstone::DeviceVector<Real> d_px(px), d_py(py), d_pz(pz);
  std::vector<Real> s1, s2, s3;
  cstone::DeviceVector<Real> d_s1, d_s2, d_s3;

  std::vector<Real> x(keys.size()), y(keys.size()), z(keys.size());

  // Convert to a lambda to measure the time taken by the sync function
  auto sync_f = [&]() {
    domain.sync(d_keys, d_ix, d_iy, d_iz, d_h, std::tie(d_px, d_py, d_pz),
                std::tie(d_s1, d_s2, d_s3));
  };

  nvtxRangePush("DomainSyncInitial");
  float sync_ms = timeGpu(sync_f);
  nvtxRangePop();

  t.first = sync_ms;

  checkGpuErrors(cudaMemGetInfo(&free_byte, &total_byte));
  size_t consumed = total_byte - free_byte - used_initial_byte;

  if (rank == 0)
    std::cout << "\tDomain Sync Initial: " << sync_ms
              << "us, Memory Usage: " << consumed / (1024 * 1024) << "Mb"
              << std::endl;

  // x.resize(d_ix.size());
  // y.resize(d_iy.size());
  // z.resize(d_iz.size());
  // keys.resize(d_keys.size());
  // cudaMemcpy(x.data(), d_ix.data(), d_ix.size() * sizeof(Real), cudaMemcpyDeviceToHost);
  // cudaMemcpy(y.data(), d_iy.data(), d_iy.size() * sizeof(Real), cudaMemcpyDeviceToHost);
  // cudaMemcpy(z.data(), d_iz.data(), d_iz.size() * sizeof(Real), cudaMemcpyDeviceToHost);
  // cudaMemcpy(keys.data(), d_keys.data(), d_keys.size() * sizeof(KeyType), cudaMemcpyDeviceToHost);

  // if (save)
  //   saveDomainOctreeH5Gpu(domain, group_name + "_initial", rank, numRanks, x, y, z, keys);

  thrust::transform(thrust::device, d_ix.data() + domain.startIndex(),
                    d_ix.data() + domain.endIndex(),
                    d_px.data() + domain.startIndex(),
                    d_ix.data() + domain.startIndex(), thrust::plus<Real>());
  thrust::transform(thrust::device, d_iy.data() + domain.startIndex(),
                    d_iy.data() + domain.endIndex(),
                    d_py.data() + domain.startIndex(),
                    d_iy.data() + domain.startIndex(), thrust::plus<Real>());
  thrust::transform(thrust::device, d_iz.data() + domain.startIndex(),
                    d_iz.data() + domain.endIndex(),
                    d_pz.data() + domain.startIndex(),
                    d_iz.data() + domain.startIndex(), thrust::plus<Real>());

  nvtxRangePush("DomainSyncPerturb");
  sync_ms = timeGpu(sync_f);
  nvtxRangePop();

  t.second = sync_ms;

  checkGpuErrors(cudaMemGetInfo(&free_byte, &total_byte));
  consumed = total_byte - free_byte - used_initial_byte;

  if (rank == 0)
    std::cout << "\tDomain Sync with Perturbations: " << sync_ms
              << "us, Memory Usage: " << consumed / (1024 * 1024) << "Mb"
              << std::endl;

  // x.resize(d_ix.size());
  // y.resize(d_iy.size());
  // z.resize(d_iz.size());
  // keys.resize(d_keys.size());
  // cudaMemcpy(x.data(), d_ix.data(), d_ix.size() * sizeof(Real), cudaMemcpyDeviceToHost);
  // cudaMemcpy(y.data(), d_iy.data(), d_iy.size() * sizeof(Real), cudaMemcpyDeviceToHost);
  // cudaMemcpy(z.data(), d_iz.data(), d_iz.size() * sizeof(Real), cudaMemcpyDeviceToHost);
  // cudaMemcpy(keys.data(), d_keys.data(), d_keys.size() * sizeof(KeyType), cudaMemcpyDeviceToHost);

  // if (save)
  //   saveDomainOctreeH5Gpu(domain, group_name + "_perturbed", rank, numRanks, x, y, z, keys);
  return t;
}
