#pragma once

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

#include "save_octree.hpp"

inline std::vector<KeyType> copyDeviceKeys(const KeyType *ptr, size_t n) {
  thrust::host_vector<KeyType> tmp(n);
  thrust::copy(thrust::device_ptr<const KeyType>(ptr),
               thrust::device_ptr<const KeyType>(ptr + n), tmp.begin());
  return std::vector<KeyType>(tmp.begin(), tmp.end());
}

inline std::vector<cstone::TreeNodeIndex>
copyDeviceNodeIdx(const cstone::TreeNodeIndex *ptr, size_t n) {
  thrust::host_vector<cstone::TreeNodeIndex> tmp(n);
  thrust::copy(thrust::device_ptr<const cstone::TreeNodeIndex>(ptr),
               thrust::device_ptr<const cstone::TreeNodeIndex>(ptr + n),
               tmp.begin());
  return std::vector<cstone::TreeNodeIndex>(tmp.begin(), tmp.end());
}

inline OctreeHostData
collectOctreeFromViewGpu(const cstone::OctreeView<const KeyType> &view,
                         const KeyType *leavesPtr, size_t leavesSize) {
  constexpr size_t levelRangeSize = cstone::maxTreeLevel<KeyType>{} + 2;
  return {copyDeviceKeys(leavesPtr, leavesSize),
          copyDeviceKeys(view.prefixes, view.numNodes),
          copyDeviceNodeIdx(view.childOffsets, view.numNodes),
          copyDeviceNodeIdx(view.internalToLeaf, view.numNodes),
          copyDeviceNodeIdx(view.d_levelRange, levelRangeSize)};
}

inline OctreeHostData collectGlobalOctreeGpu(
    const cstone::Domain<KeyType, Real, cstone::GpuTag> &domain) {
  auto view = domain.globalTree();
  return collectOctreeFromViewGpu(view, view.leaves, view.numLeafNodes + 1);
}

inline OctreeHostData collectFocusOctreeGpu(
    const cstone::Domain<KeyType, Real, cstone::GpuTag> &domain) {
  auto view = domain.focusTree().octreeViewAcc();
  auto leaves = domain.focusTree().treeLeavesAcc();
  return collectOctreeFromViewGpu(view, leaves.data(), leaves.size());
}

inline void saveDomainOctreeH5Gpu(
    const cstone::Domain<KeyType, Real, cstone::GpuTag> &domain,
    const std::string &spec, int rank, int numRanks) {
  auto globalTree = domain.globalTree();
  if (globalTree.numLeafNodes == 0) {
    return;
  }

  std::string safeSpec = sanitizeSpec(spec);
  std::filesystem::create_directories("outputs");
  std::filesystem::path outputPath =
      std::filesystem::path("outputs") /
      ("domain_" + safeSpec + "_rank" + std::to_string(rank) + ".h5");

  HighFive::File out(outputPath.string(), HighFive::File::Overwrite);
  auto box = domain.box();

  std::vector<Real> boxVec{box.xmin(), box.xmax(), box.ymin(),
                           box.ymax(), box.zmin(), box.zmax()};
  out.createDataSet("domain_box", boxVec);
  out.createAttribute("rank", rank);
  out.createAttribute("num_ranks", numRanks);
  out.createAttribute("focus_start_cell", domain.startCell());
  out.createAttribute("focus_end_cell", domain.endCell());

  writeOctreeGroup(out, "global_octree", collectGlobalOctreeGpu(domain), box);
  writeOctreeGroup(out, "focus_octree", collectFocusOctreeGpu(domain), box);

  if (rank == 0) {
    std::cout << "\tSaved octree HDF5: " << outputPath << std::endl;
  }
}
