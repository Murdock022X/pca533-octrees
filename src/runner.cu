#include "cstone/cuda/device_vector.h"
#include "cstone/cuda/errorcheck.cuh"
#include "cstone/domain/domain.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/sfc/common.hpp"
#include "cstone/tree/definitions.h"
#include "pcah5.hpp"
#include "runner.hpp"
#include "save_octree.cuh"
#include "utils.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <highfive/H5File.hpp>
#include <string>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <vector>

void runnerGpu(std::vector<KeyType> &keys, std::vector<Real> &ix,
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

  saveDomainOctreeH5Gpu(domain, group_name + "_initial", rank, numRanks);

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

  saveDomainOctreeH5Gpu(domain, group_name + "_perturbed", rank, numRanks);

  sync_ms = timeGpu(sync_f);

  checkGpuErrors(cudaMemGetInfo(&free_byte, &total_byte));
  consumed = total_byte - free_byte - used_initial_byte;

  if (rank == 0)
    std::cout << "\tDomain Sync without Perturbations: " << sync_ms
              << "us, Memory Usage: " << consumed / (1024 * 1024) << "Mb\n"
              << std::endl;
}
