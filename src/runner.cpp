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
#include <fstream>

void runner(HighFive::File &file, std::string group_name, int rank,
            int numRanks, bool gpu, bool lets, int bucketSize, int bucketSizeFocus,
            float theta, bool save) {
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

  int trials = 10;

  std::vector<double> t_no_pt (9);
  std::vector<double> t_pt (9);

  for (int i = 0; i < trials; i++) {
    std::pair<double, double> t;
    if (!gpu && !lets) {
      t = runnerCpu(keys, ix_local, iy_local, iz_local, h, px_local, py_local,
                pz_local, rank, numRanks, bucketSize, bucketSizeFocus, theta,
                group_name, false);
    } else if (!gpu && lets) {
      t = runnerCpuMulti(keys, ix_local, iy_local, iz_local, h, px_local, py_local,
                pz_local, rank, numRanks, bucketSize, bucketSizeFocus, theta,
                group_name, false);
    } else if (gpu && !lets) {
      t = runnerGpu(keys, ix_local, iy_local, iz_local, h, px_local, py_local,
                pz_local, rank, numRanks, bucketSize, bucketSizeFocus, theta,
                group_name, false);
    } else if (gpu && lets) {
      t = runnerGpuMulti(keys, ix_local, iy_local, iz_local, h, px_local, py_local,
                pz_local, rank, numRanks, bucketSize, bucketSizeFocus, theta,
                group_name, false);
    } else {
      throw std::runtime_error("Invalid combination of gpu and lets flags");
    }

    if (i != 0) { // skip first trial for warmup
      t_no_pt[i-1] = t.first;
      t_pt[i-1] = t.second;
    }
  }

  double avg_no_pt = std::accumulate(t_no_pt.begin(), t_no_pt.end(), 0.0) / t_no_pt.size();
  double avg_pt = std::accumulate(t_pt.begin(), t_pt.end(), 0.0) / t_pt.size();
  double min_no_pt = *std::min_element(t_no_pt.begin(), t_no_pt.end());
  double min_pt = *std::min_element(t_pt.begin(), t_pt.end());
  double max_no_pt = *std::max_element(t_no_pt.begin(), t_no_pt.end());
  double max_pt = *std::max_element(t_pt.begin(), t_pt.end());
  double stddev_no_pt = std::sqrt(std::accumulate(t_no_pt.begin(), t_no_pt.end(), 0.0, 
    [avg_no_pt](double acc, double x) { return acc + (x - avg_no_pt) * (x - avg_no_pt); }) / t_no_pt.size());
  double stddev_pt = std::sqrt(std::accumulate(t_pt.begin(), t_pt.end(), 0.0, 
    [avg_pt](double acc, double x) { return acc + (x - avg_pt) * (x - avg_pt); }) / t_pt.size());

  if (rank == 0) {
    std::cout << "No Perturbations: Average time: " << avg_no_pt << "us, Min: " << min_no_pt << "us, Max: " << max_no_pt << "us, StdDev: " << stddev_no_pt << "us" << std::endl;
    std::cout << "With Perturbations: Average time: " << avg_pt << " us, Min: " << min_pt << "us, Max: " << max_pt << "us, StdDev: " << stddev_pt << "us" << std::endl;
  }

  if (save) {
    std::ofstream out(group_name + "_timings.csv");
    out << "trial,no_pt_us,pt_us\n";
    for (size_t i = 0; i < t_no_pt.size(); i++)      out << (i+1) << "," << t_no_pt[i] << "," << t_pt[i] << "\n";
    out.close();
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

std::pair<double, double> runnerCpu(const std::vector<KeyType> &keys, const std::vector<Real> &ix,
               const std::vector<Real> &iy, const std::vector<Real> &iz,
               const std::vector<Real> &h, const std::vector<Real> &px,
               const std::vector<Real> &py, const std::vector<Real> &pz, int rank,
               int numRanks, int bucketSize, int bucketSizeFocus, float theta,
               std::string group_name, bool save) {
  cstone::Box<Real> box{-1.5, 1.5};

  size_t np = keys.size();
  int call_count = 1;

  std::pair<double, double> t;

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

  std::vector<Real> x(ix), y(iy), z(iz);

  auto f = [&]() {
    processCpu(box, d_keys, d_keys_tmp, d_ordering, d_values_tmp, tmp, cubTmpStorage, tempStorageEle, 
      d_counts, workArray, d_layout, d_tree, tmpTree, octreeData, x, y, z, bucketSize, np);
  };

  float sync_ms = timeCpu(f);
  t.first = sync_ms;

  if (rank == 0)
    std::cout << "\tUpdate Octree Initial: " << sync_ms << "us, call count: " << call_count
              << std::endl;

  // thrust::copy(thrust::host, d_keys.data(), d_keys.data() + d_keys.size(), keys.begin());

  // saveOctreeH5Gpu(, group_name + "_initial", rank, numRanks, x, y, z, keys);

  #pragma omp parallel for
  for (auto i = 0; i < ix.size(); ++i) {
    x[i] += px[i];
    y[i] += py[i];
    z[i] += pz[i];
  }

  sync_ms = timeCpu(f);
  t.second = sync_ms;

  call_count = 1;

  if (rank == 0)
    std::cout << "\tPerturbation update time: " << sync_ms << " us, call count: " << call_count
              << std::endl;

  return t;
  
  // thrust::copy(thrust::host, d_keys.data(), d_keys.data() + d_keys.size(), keys.begin());

  // saveOctreeH5Gpu(domain, group_name + "_perturbed", x, y, z, keys);
}

std::pair<double, double> runnerCpuMulti(const std::vector<KeyType> &keys, const std::vector<Real> &ix,
               const std::vector<Real> &iy, const std::vector<Real> &iz,
               const std::vector<Real> &h, const std::vector<Real> &px,
               const std::vector<Real> &py, const std::vector<Real> &pz, int rank,
               int numRanks, int bucketSize, int bucketSizeFocus, float theta,
               std::string group_name, bool save) {
  cstone::Domain<KeyType, Real, cstone::CpuTag> domain(
      rank, numRanks, bucketSize, bucketSizeFocus, theta);

  std::pair<double, double> t;

  std::vector<KeyType> k(keys);
  std::vector<Real> x(ix), y(iy), z(iz);
  std::vector<Real> hh(h);
  std::vector<Real> s1, s2, s3;
  auto sync_f = [&]() {
    domain.sync(k, x, y, z, hh, std::tuple{},
                std::tie(s1, s2, s3));
  };

  float sync_ms = timeCpu(sync_f);
  t.first = sync_ms;

  if (rank == 0) {
    std::cout << "\tDomain Sync Initial: " << sync_ms << "us" << std::endl;
  }

  if (save)
    saveDomainOctreeH5Cpu(domain, group_name + "_initial", rank, numRanks, x, y, z, k);

#pragma omp parallel for
  for (auto i = domain.startIndex(); i < domain.endIndex(); ++i) {
    x[i] += px[i];
    y[i] += py[i];
    z[i] += pz[i];
  }

  sync_ms = timeCpu(sync_f);

  t.second = sync_ms;

  if (rank == 0) {
    std::cout << "\tDomain Sync with Perturbations: " << sync_ms << "us"
              << std::endl;
  }

  if (save)
    saveDomainOctreeH5Cpu(domain, group_name + "_perturbed", rank, numRanks, x, y, z, k);

  if (rank == 0) {
    std::cout << "\tDomain Sync without Perturbations: " << sync_ms << "us"
              << std::endl;
  }

  return t;
}
