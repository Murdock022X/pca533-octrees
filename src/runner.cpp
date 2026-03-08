#include "runner.hpp"
#include "cstone/domain/domain.hpp"
#include "save_octree.hpp"
#include "utils.hpp"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>
#include <numeric>
#include <span>

void runner(HighFive::File &file, std::string group_name, int rank,
            int numRanks, bool gpu, bool lets, int bucketSize, int bucketSizeFocus,
            float theta) {
  if (!file.exist(group_name))
    throw std::runtime_error("Group does not exist in the dataset file: " +
                             group_name);

  auto [ix, iy, iz, px, py, pz] = read_dataset<Real>(file, group_name);

  size_t start = rank * ix.size() / numRanks;
  size_t end = (rank + 1) * ix.size() / numRanks;

  std::cout << "Dataset loaded [" << group_name << "] -> n = " << ix.size()
            << ", rank = " << rank << ", subdomain [" << start << ", " << end
            << ")" << std::endl;

  std::vector<Real> ix_local(ix.begin() + start, ix.begin() + end);
  std::vector<Real> iy_local(iy.begin() + start, iy.begin() + end);
  std::vector<Real> iz_local(iz.begin() + start, iz.begin() + end);
  std::vector<Real> px_local(px.begin() + start, px.begin() + end);
  std::vector<Real> py_local(py.begin() + start, py.begin() + end);
  std::vector<Real> pz_local(pz.begin() + start, pz.begin() + end);

  std::vector<Real> h(end - start, 0.1);
  std::vector<KeyType> keys(end - start);

  for (int i = 0; i < 10; i++) {
  if (!gpu && !lets) {
    runnerCpu(keys, ix_local, iy_local, iz_local, h, px_local, py_local,
              pz_local, rank, numRanks, bucketSize, bucketSizeFocus, theta,
              group_name);
  } else if (!gpu && lets) {
    runnerCpuMulti(keys, ix_local, iy_local, iz_local, h, px_local, py_local,
              pz_local, rank, numRanks, bucketSize, bucketSizeFocus, theta,
              group_name);
  } else if (gpu && !lets) {
    runnerGpu(keys, ix_local, iy_local, iz_local, h, px_local, py_local,
              pz_local, rank, numRanks, bucketSize, bucketSizeFocus, theta,
              group_name);
  } else if (gpu && lets) {
    runnerGpuMulti(keys, ix_local, iy_local, iz_local, h, px_local, py_local,
              pz_local, rank, numRanks, bucketSize, bucketSizeFocus, theta,
              group_name);
  } else {
    throw std::runtime_error("Invalid combination of gpu and lets flags");
  }
  }
}

void processCpu(cstone::Box<Real> &box, 
  std::vector<KeyType> &d_keys, 
  std::vector<KeyType> &d_keys_tmp, 
  std::vector<unsigned> &d_ordering, std::vector<unsigned> &d_values_tmp, 
  std::vector<Real> &tmp, 
  std::vector<char> &cubTmpStorage, uint64_t tempStorageEle,
  std::vector<unsigned> &d_counts, 
  std::vector<cstone::TreeNodeIndex> &workArray,
  std::vector<cstone::LocalIndex> &d_layout,
  std::vector<KeyType> &d_tree, std::vector<KeyType> &tmpTree, 
  cstone::OctreeData<KeyType, cstone::CpuTag> &octreeData, 
  std::vector<Real> &d_x, std::vector<Real> &d_y, std::vector<Real> &d_z, 
  int bucketSize, size_t np) {

  cstone::computeSfcKeys(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), cstone::sfcKindPointer(rawPtr(d_keys)), np, box);
  std::iota(d_ordering.begin(), d_ordering.end(), 0);
  cstone::sort_by_key(d_keys.begin(), d_keys.end(), d_ordering.begin());

  cstone::gatherCpu(std::span(d_ordering.data(), np), d_x.data(), tmp.data());
  std::swap(d_x, tmp);
  cstone::gatherCpu(std::span(d_ordering.data(), np), d_y.data(), tmp.data());
  std::swap(d_x, tmp);
  cstone::gatherCpu(std::span(d_ordering.data(), np), d_z.data(), tmp.data());
  std::swap(d_x, tmp);

  if (d_tree.size() == 0)
  {
      // initial guess on first call. use previous tree as guess on subsequent calls
      d_tree = std::vector<KeyType>{0, cstone::nodeRange<KeyType>(0)};
      d_counts = std::vector<unsigned>{unsigned(np)};
  }

  while (!cstone::updateOctree<KeyType>({rawPtr(d_keys), d_keys.size()}, bucketSize, d_tree, d_counts));

  octreeData.resize(cstone::nNodes(d_tree));
  cstone::updateInternalTree({d_tree.data(), d_tree.size()}, octreeData.data());

  d_layout.resize(d_counts.size() + 1);
  d_layout[0] = cstone::LocalIndex(0);
  std::inclusive_scan(d_counts.begin(), d_counts.end(), d_layout.begin() + 1);
}

void runnerCpu(std::vector<KeyType> &keys, std::vector<Real> &ix,
               std::vector<Real> &iy, std::vector<Real> &iz,
               std::vector<Real> &h, std::vector<Real> &px,
               std::vector<Real> &py, std::vector<Real> &pz, int rank,
               int numRanks, int bucketSize, int bucketSizeFocus, float theta,
               std::string group_name) {
  cstone::Box<Real> box{-1.5, 1.5};

  size_t np = keys.size();
  int call_count = 1;

  std::cout << "Running GPU Octree Build and Sync Benchmark with " << np
            << " particles, bucket size: " << bucketSize
            << ", theta: " << theta << std::endl;
  
  std::vector<KeyType> d_keys(keys.size());
  std::vector<KeyType> d_tree, tmpTree;
  cstone::OctreeData<KeyType, cstone::CpuTag> octreeData;
  std::vector<KeyType> d_keys_tmp(keys.size());
  std::vector<unsigned> d_counts;
  std::vector<cstone::TreeNodeIndex> workArray;

  std::vector<unsigned> d_ordering(keys.size()), d_values_tmp(keys.size());
  std::vector<Real> tmp(keys.size());
  std::vector<cstone::LocalIndex> d_layout;

  uint64_t tempStorageEle = cstone::sortByKeyTempStorage<KeyType, cstone::LocalIndex>(np);
  std::vector<char> cubTmpStorage(tempStorageEle);

  auto f = [&]() {
    processCpu(box, d_keys, d_keys_tmp, d_ordering, d_values_tmp, tmp, cubTmpStorage, tempStorageEle, 
      d_counts, workArray, d_layout, d_tree, tmpTree, octreeData, ix, iy, iz, bucketSize, np);
  };

  float sync_ms = timeCpu(f);

  if (rank == 0)
    std::cout << "\tUpdate Octree Initial: " << sync_ms << "us, call count: " << call_count
              << std::endl;

  // thrust::copy(thrust::host, d_keys.data(), d_keys.data() + d_keys.size(), keys.begin());

  // saveOctreeH5Gpu(, group_name + "_initial", rank, numRanks, x, y, z, keys);

  #pragma omp parallel for
  for (auto i = 0; i < ix.size(); ++i) {
    ix[i] += px[i];
    iy[i] += py[i];
    iz[i] += pz[i];
  }

  sync_ms = timeCpu(f);

  call_count = 1;

  if (rank == 0)
    std::cout << "\tPerturbation update time: " << sync_ms << " us, call count: " << call_count
              << std::endl;
  
  // thrust::copy(thrust::host, d_keys.data(), d_keys.data() + d_keys.size(), keys.begin());

  // saveOctreeH5Gpu(domain, group_name + "_perturbed", x, y, z, keys);
}

void runnerCpuMulti(std::vector<KeyType> &keys, std::vector<Real> &ix,
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

  saveDomainOctreeH5Cpu(domain, group_name + "_initial", rank, numRanks, ix, iy, iz, keys);

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

  saveDomainOctreeH5Cpu(domain, group_name + "_perturbed", rank, numRanks, ix, iy, iz, keys);

  sync_ms = timeCpu(sync_f);

  if (rank == 0) {
    std::cout << "\tDomain Sync without Perturbations: " << sync_ms << "us"
              << std::endl;
  }
}
