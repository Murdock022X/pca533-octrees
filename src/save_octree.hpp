#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "cstone/domain/domain.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/sfc/common.hpp"
#include "cstone/tree/octree.hpp"

struct OctreeHostData
{
    std::vector<KeyType> leaves;
    std::vector<KeyType> prefixes;
    std::vector<cstone::TreeNodeIndex> childOffset;
    std::vector<cstone::TreeNodeIndex> internalToLeaf;
    std::vector<cstone::TreeNodeIndex> levelRange;
};

inline std::string sanitizeSpec(std::string spec)
{
    std::replace_if(spec.begin(), spec.end(),
                    [](char c) { return !(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.'); },
                    '_');
    return spec;
}

inline void writeOctreeGroup(HighFive::File& out,
                             const std::string& groupName,
                             const OctreeHostData& oct,
                             const cstone::Box<Real>& box)
{
    int numNodes = int(oct.prefixes.size());
    int numLeafNodes = int(oct.leaves.size()) - 1;
    if (numNodes <= 0 || numLeafNodes < 0) { return; }

    std::vector<Real> cx(numNodes), cy(numNodes), cz(numNodes);
    std::vector<Real> sx(numNodes), sy(numNodes), sz(numNodes);
    std::vector<unsigned> level(numNodes), isLeaf(numNodes);
    std::vector<KeyType> startKey(numNodes);

    std::vector<cstone::Vec3<Real>> centers(numNodes), sizes(numNodes);
    cstone::nodeFpCenters<KeyType>(std::span<const KeyType>(oct.prefixes.data(), oct.prefixes.size()),
                                   centers.data(), sizes.data(), box);

    for (int i = 0; i < numNodes; ++i)
    {
        cx[i] = centers[i][0];
        cy[i] = centers[i][1];
        cz[i] = centers[i][2];
        sx[i] = sizes[i][0];
        sy[i] = sizes[i][1];
        sz[i] = sizes[i][2];
        level[i] = cstone::decodePrefixLength(oct.prefixes[i]) / 3;
        isLeaf[i] = (oct.childOffset[i] == 0) ? 1u : 0u;
        startKey[i] = cstone::decodePlaceholderBit(oct.prefixes[i]);
    }

    auto group = out.createGroup(groupName);
    group.createAttribute("num_nodes", numNodes);
    group.createAttribute("num_leaf_nodes", numLeafNodes);
    group.createDataSet("leaves", oct.leaves);
    group.createDataSet("prefixes", oct.prefixes);
    group.createDataSet("child_offset", oct.childOffset);
    group.createDataSet("internal_to_leaf", oct.internalToLeaf);
    group.createDataSet("level_range", oct.levelRange);
    group.createDataSet("level", level);
    group.createDataSet("is_leaf", isLeaf);
    group.createDataSet("start_key", startKey);
    group.createDataSet("cx", cx);
    group.createDataSet("cy", cy);
    group.createDataSet("cz", cz);
    group.createDataSet("sx", sx);
    group.createDataSet("sy", sy);
    group.createDataSet("sz", sz);
}

inline OctreeHostData collectOctreeFromViewCpu(const cstone::OctreeView<const KeyType>& view,
                                               std::span<const KeyType> leaves)
{
    constexpr size_t levelRangeSize = cstone::maxTreeLevel<KeyType>{} + 2;
    return {
        std::vector<KeyType>(leaves.begin(), leaves.end()),
        std::vector<KeyType>(view.prefixes, view.prefixes + view.numNodes),
        std::vector<cstone::TreeNodeIndex>(view.childOffsets, view.childOffsets + view.numNodes),
        std::vector<cstone::TreeNodeIndex>(view.internalToLeaf, view.internalToLeaf + view.numNodes),
        std::vector<cstone::TreeNodeIndex>(view.levelRange, view.levelRange + levelRangeSize)};
}

inline OctreeHostData collectFocusOctreeCpu(const cstone::Domain<KeyType, Real, cstone::CpuTag>& domain)
{
    return collectOctreeFromViewCpu(domain.focusTree().octreeViewAcc(), domain.focusTree().treeLeaves());
}

inline OctreeHostData collectGlobalOctreeCpu(const cstone::Domain<KeyType, Real, cstone::CpuTag>& domain)
{
    auto view = domain.globalTree();
    return collectOctreeFromViewCpu(view, std::span<const KeyType>(view.leaves, view.numLeafNodes + 1));
}

inline void saveDomainOctreeH5Cpu(const cstone::Domain<KeyType, Real, cstone::CpuTag>& domain,
                                  const std::string& spec,
                                  int rank,
                                  int numRanks)
{
    auto globalTree = domain.globalTree();
    if (globalTree.numLeafNodes == 0) { return; }

    std::string safeSpec = sanitizeSpec(spec);
    std::filesystem::create_directories("outputs");
    std::filesystem::path outputPath =
        std::filesystem::path("outputs") / ("domain_" + safeSpec + "_rank" + std::to_string(rank) + ".h5");

    HighFive::File out(outputPath.string(), HighFive::File::Overwrite);
    auto box = domain.box();

    std::vector<Real> boxVec{box.xmin(), box.xmax(), box.ymin(), box.ymax(), box.zmin(), box.zmax()};
    out.createDataSet("domain_box", boxVec);
    out.createAttribute("rank", rank);
    out.createAttribute("num_ranks", numRanks);
    out.createAttribute("focus_start_cell", domain.startCell());
    out.createAttribute("focus_end_cell", domain.endCell());

    writeOctreeGroup(out, "global_octree", collectGlobalOctreeCpu(domain), box);
    writeOctreeGroup(out, "focus_octree", collectFocusOctreeCpu(domain), box);

    if (rank == 0) { std::cout << "\tSaved octree HDF5: " << outputPath << std::endl; }
}
