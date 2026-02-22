#pragma once

#include "cstone/domain/domain.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/sfc/common.hpp"
#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cctype>
#include "cstone/tree/definitions.h"
#include "pcah5.hpp"
#include <highfive/H5File.hpp>
#include <chrono>
#include "cstone/cuda/device_vector.h"
#include "cstone/domain/domain.hpp"

void saveDomainOctreeCsvCpu(const cstone::Domain<KeyType, Real, cstone::CpuTag>& domain, const std::string& spec, int rank)
{
    auto tree = domain.globalTree();
    if (tree.numNodes == 0) { return; }

    std::vector<KeyType> prefixes(tree.prefixes, tree.prefixes + tree.numNodes);
    std::vector<cstone::Vec3<Real>> centers(tree.numNodes), sizes(tree.numNodes);
    cstone::nodeFpCenters<KeyType>(prefixes, centers.data(), sizes.data(), domain.box());

    std::string safe_spec = spec;
    std::replace_if(safe_spec.begin(), safe_spec.end(),
                    [](char c) { return !(std::isalnum(c) || c == '-' || c == '_' || c == '.'); }, '_');

    fs::create_directories("outputs");
    fs::path output_path = fs::path("outputs") / ("domain_octree_" + safe_spec + "_rank" + std::to_string(rank) + ".csv");
    std::ofstream out(output_path);
    if (!out) { throw std::runtime_error("Failed to open octree output file: " + output_path.string()); }

    out << "node,level,is_leaf,child_offset,prefix,start_key,cx,cy,cz,sx,sy,sz\n";
    for (int i = 0; i < tree.numNodes; ++i)
    {
        KeyType prefix = tree.prefixes[i];
        unsigned level = cstone::decodePrefixLength(prefix) / 3;
        auto childOffset = tree.childOffsets[i];
        bool isLeaf = (childOffset == 0);

        out << i << "," << level << "," << (isLeaf ? 1 : 0) << "," << childOffset << "," << prefix << ","
            << cstone::decodePlaceholderBit(prefix) << ","
            << centers[i][0] << "," << centers[i][1] << "," << centers[i][2] << ","
            << sizes[i][0] << "," << sizes[i][1] << "," << sizes[i][2] << "\n";
    }

    if (rank == 0)
    {
        std::cout << "\tSaved domain octree CSV: " << output_path << " (" << tree.numNodes << " nodes)" << std::endl;
    }
}