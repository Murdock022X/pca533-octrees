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

#include "pcah5.hpp"
#include "runner.hpp"
#include "save_octree.cuh"
#include "utils.hpp"

void runnerGpu(std::vector<KeyType> &keys, std::vector<Real> &ix,
               std::vector<Real> &iy, std::vector<Real> &iz,
               std::vector<Real> &h, std::vector<Real> &px,
               std::vector<Real> &py, std::vector<Real> &pz, int rank,
               int numRanks, int bucketSize, int bucketSizeFocus, float theta,
               std::string group_name) {
  size_t free_initial_byte, free_byte;
  size_t total_initial_byte, total_byte;

  cstone::Box<Real> box{-1.5, 1.5};

  unsigned int np = keys.size();

  std::cout << "Running GPU Octree Build and Sync Benchmark with " << np
            << " particles, bucket size: " << bucketSize
            << ", theta: " << theta << std::endl;

  // Get memory info
  checkGpuErrors(cudaMemGetInfo(&free_initial_byte, &total_initial_byte));

  size_t used_initial_byte = total_initial_byte - free_initial_byte;
  
  std::vector<cstone::LocalIndex> sfcOrder(np);
  std::iota(begin(sfcOrder), end(sfcOrder), cstone::LocalIndex(0));

  cstone::computeSfcKeys(ix.data(), iy.data(), iz.data(), cstone::sfcKindPointer(keys.data()), np, box);
  cstone::sort_by_key(keys.data(), keys.data() + np, sfcOrder.data());

  std::vector<Real> temp(ix.size());
  cstone::gather<cstone::LocalIndex>(sfcOrder, ix.data(), temp.data());
  std::swap(ix, temp);
  cstone::gather<cstone::LocalIndex>(sfcOrder, iy.data(), temp.data());
  std::swap(iy, temp);
  cstone::gather<cstone::LocalIndex>(sfcOrder, iz.data(), temp.data());
  std::swap(iz, temp);

  cstone::DeviceVector<KeyType> d_keys(keys.data(), keys.data() + np);

  cstone::DeviceVector<KeyType> tree    = std::vector<KeyType>{0, cstone::nodeRange<KeyType>(0)};
  cstone::DeviceVector<unsigned> counts = std::vector<unsigned>{np};

  cstone::DeviceVector<KeyType> tmpTree;
  cstone::DeviceVector<cstone::TreeNodeIndex> workArray;

  std::vector<Real> x(keys.size()), y(keys.size()), z(keys.size());

  int call_count = 0;

  // Convert to a lambda to measure the time taken by the sync function
  auto fullBuild = [&]()
  {
    while (!cstone::updateOctreeGpu<KeyType>({rawPtr(d_keys), np}, bucketSize, tree, counts, tmpTree,
                                      workArray)) { call_count++; };
  };

  float sync_ms = timeGpu(fullBuild);

  checkGpuErrors(cudaMemGetInfo(&free_byte, &total_byte));
  size_t consumed = total_byte - free_byte - used_initial_byte;

  if (rank == 0)
    std::cout << "\tUpdate Octree Initial: " << sync_ms
              << "us, Memory Usage: " << consumed / (1024 * 1024) << "Mb"
              << std::endl;

  // thrust::copy(thrust::host, d_keys.data(), d_keys.data() + d_keys.size(), keys.begin());

  // saveOctreeH5Gpu(, group_name + "_initial", rank, numRanks, x, y, z, keys);

  #pragma omp parallel for
  for (auto i = 0; i < ix.size(); ++i) {
    ix[i] += px[i];
    iy[i] += py[i];
    iz[i] += pz[i];
  }

  std::iota(begin(sfcOrder), end(sfcOrder), cstone::LocalIndex(0));

  cstone::computeSfcKeys(ix.data(), iy.data(), iz.data(), cstone::sfcKindPointer(keys.data()), np, box);
  cstone::sort_by_key(keys.data(), keys.data() + np, sfcOrder.data());

  cstone::gather<cstone::LocalIndex>(sfcOrder, ix.data(), temp.data());
  std::swap(ix, temp);
  cstone::gather<cstone::LocalIndex>(sfcOrder, iy.data(), temp.data());
  std::swap(iy, temp);
  cstone::gather<cstone::LocalIndex>(sfcOrder, iz.data(), temp.data());
  std::swap(iz, temp);
  d_keys = keys;

  call_count = 0;

  sync_ms = timeGpu(fullBuild);

  checkGpuErrors(cudaMemGetInfo(&free_byte, &total_byte));
  consumed = total_byte - free_byte - used_initial_byte;

  if (rank == 0)
    std::cout << "\tDomain Sync with Perturbations: " << sync_ms
              << "us, Memory Usage: " << consumed / (1024 * 1024) << "Mb"
              << std::endl;
  
  // thrust::copy(thrust::host, d_keys.data(), d_keys.data() + d_keys.size(), keys.begin());

  // saveOctreeH5Gpu(domain, group_name + "_perturbed", x, y, z, keys);

  sync_ms = timeGpu(fullBuild);

  checkGpuErrors(cudaMemGetInfo(&free_byte, &total_byte));
  consumed = total_byte - free_byte - used_initial_byte;

  if (rank == 0)
    std::cout << "\tDomain Sync without Perturbations: " << sync_ms
              << "us, Memory Usage: " << consumed / (1024 * 1024) << "Mb\n"
              << std::endl;
}

void runnerGpuMulti(std::vector<KeyType> &keys, std::vector<Real> &ix,
               std::vector<Real> &iy, std::vector<Real> &iz,
               std::vector<Real> &h, std::vector<Real> &px,
               std::vector<Real> &py, std::vector<Real> &pz, int rank,
               int numRanks, int bucketSize, int bucketSizeFocus, float theta,
               std::string group_name) {
  size_t free_initial_byte, free_byte;
  size_t total_initial_byte, total_byte;

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

  float sync_ms = timeGpu(sync_f);

  checkGpuErrors(cudaMemGetInfo(&free_byte, &total_byte));
  size_t consumed = total_byte - free_byte - used_initial_byte;

  if (rank == 0)
    std::cout << "\tDomain Sync Initial: " << sync_ms
              << "us, Memory Usage: " << consumed / (1024 * 1024) << "Mb"
              << std::endl;

  thrust::copy(thrust::host, d_ix.data(), d_ix.data() + d_ix.size(), x.begin());
  thrust::copy(thrust::host, d_iy.data(), d_iy.data() + d_iy.size(), y.begin());
  thrust::copy(thrust::host, d_iz.data(), d_iz.data() + d_iz.size(), z.begin());
  thrust::copy(thrust::host, d_keys.data(), d_keys.data() + d_keys.size(), keys.begin());

  saveDomainOctreeH5Gpu(domain, group_name + "_initial", rank, numRanks, x, y, z, keys);

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

  sync_ms = timeGpu(sync_f);

  checkGpuErrors(cudaMemGetInfo(&free_byte, &total_byte));
  consumed = total_byte - free_byte - used_initial_byte;

  if (rank == 0)
    std::cout << "\tDomain Sync with Perturbations: " << sync_ms
              << "us, Memory Usage: " << consumed / (1024 * 1024) << "Mb"
              << std::endl;

  thrust::copy(thrust::host, d_ix.data(), d_ix.data() + d_ix.size(), x.begin());
  thrust::copy(thrust::host, d_iy.data(), d_iy.data() + d_iy.size(), y.begin());
  thrust::copy(thrust::host, d_iz.data(), d_iz.data() + d_iz.size(), z.begin());
  thrust::copy(thrust::host, d_keys.data(), d_keys.data() + d_keys.size(), keys.begin());

  saveDomainOctreeH5Gpu(domain, group_name + "_perturbed", rank, numRanks, x, y, z, keys);

  sync_ms = timeGpu(sync_f);

  checkGpuErrors(cudaMemGetInfo(&free_byte, &total_byte));
  consumed = total_byte - free_byte - used_initial_byte;

  if (rank == 0)
    std::cout << "\tDomain Sync without Perturbations: " << sync_ms
              << "us, Memory Usage: " << consumed / (1024 * 1024) << "Mb\n"
              << std::endl;
}
