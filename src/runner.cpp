#include "runner.hpp"
#include "cstone/domain/domain.hpp"
#include "save_octree.hpp"
#include "utils.hpp"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

void runner(HighFive::File &file, std::string group_name, int rank,
            int numRanks, bool gpu) {
  if (!file.exist(group_name))
    throw std::runtime_error("Group does not exist in the dataset file: " +
                             group_name);

  auto [ix, iy, iz, px, py, pz] = read_dataset<Real>(file, group_name);

  size_t start = rank * ix.size() / numRanks;
  size_t end = (rank + 1) * ix.size() / numRanks;

  std::cout << "Dataset loaded [" << group_name << "] -> n = " << ix.size()
            << "rank = " << rank << " with range [" << start << ", " << end
            << ")" << std::endl;

  std::vector<Real> ix_local(ix.begin() + start, ix.begin() + end);
  std::vector<Real> iy_local(iy.begin() + start, iy.begin() + end);
  std::vector<Real> iz_local(iz.begin() + start, iz.begin() + end);
  std::vector<Real> px_local(px.begin() + start, px.begin() + end);
  std::vector<Real> py_local(py.begin() + start, py.begin() + end);
  std::vector<Real> pz_local(pz.begin() + start, pz.begin() + end);

  std::vector<Real> h(end - start, 0.1);

  int bucketSize = 1024;
  int bucketSizeFocus = 8;
  float theta = 0.6f;

  std::vector<KeyType> keys(end - start);
  if (!gpu) {
    runnerCpu(keys, ix_local, iy_local, iz_local, h, px_local, py_local,
              pz_local, rank, numRanks, bucketSize, bucketSizeFocus, theta,
              group_name);
  } else {
    runnerGpu(keys, ix_local, iy_local, iz_local, h, px_local, py_local,
              pz_local, rank, numRanks, bucketSize, bucketSizeFocus, theta,
              group_name);
  }
}

void runnerCpu(std::vector<KeyType> &keys, std::vector<Real> &ix,
               std::vector<Real> &iy, std::vector<Real> &iz,
               std::vector<Real> &h, std::vector<Real> &px,
               std::vector<Real> &py, std::vector<Real> &pz, int rank,
               int numRanks, int bucketSize, int bucketSizeFocus, float theta,
               std::string group_name) {
  cstone::Domain<KeyType, Real, cstone::CpuTag> domain(
      rank, numRanks, bucketSize, bucketSizeFocus, theta);

  std::vector<Real> s1, s2, s3;
  auto sync_f = [&]() {
    domain.sync(keys, ix, iy, iz, h, std::tie(px, py, pz),
                std::tie(s1, s2, s3));
  };

  float sync_ms = timeCpu(sync_f);

  if (rank == 0) {
    std::cout << "\tDomain Sync Initial: " << sync_ms << "us" << std::endl;
  }

  saveDomainOctreeH5Cpu(domain, group_name + "_initial", rank, numRanks);

#pragma omp parallel for
  for (auto i = domain.startIndex(); i < domain.endIndex(); ++i) {
    ix[i] += px[i];
    iy[i] += py[i];
    iz[i] += pz[i];
  }

  sync_ms = timeCpu(sync_f);

  if (rank == 0) {
    std::cout << "\tDomain Sync with Perturbations: " << sync_ms << "us"
              << std::endl;
  }

  saveDomainOctreeH5Cpu(domain, group_name + "_perturbed", rank, numRanks);

  sync_ms = timeCpu(sync_f);

  if (rank == 0) {
    std::cout << "\tDomain Sync without Perturbations: " << sync_ms << "us"
              << std::endl;
  }
}
