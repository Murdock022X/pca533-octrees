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
#include <thrust/transform.h>
#include <thrust/functional.h>
#include "cstone/tree/definitions.h"
#include "pcah5.hpp"
#include <highfive/H5File.hpp>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include "cstone/cuda/device_vector.h"
#include "cstone/cuda/errorcheck.cuh"
#include "cstone/domain/domain.hpp"

void saveDomainOctreeCsvGpu(const cstone::Domain<KeyType, Real, cstone::GpuTag>& domain, const std::string& spec, int rank)
{
    auto tree = domain.globalTree();
    if (tree.numNodes == 0) { return; }

    std::vector<cstone::Vec3<Real>> centers(tree.numNodes), sizes(tree.numNodes);
    thrust::host_vector<KeyType> prefixes(tree.numNodes);
    thrust::copy(thrust::device_ptr<const KeyType>(tree.prefixes), // Start device iterator
                 thrust::device_ptr<const KeyType>(tree.prefixes + tree.numNodes), // End device iterator
                 prefixes.begin());
    cstone::nodeFpCenters<KeyType>(std::span(prefixes.data(), prefixes.size()), centers.data(), sizes.data(), domain.box());

    std::string safe_spec = spec;
    std::replace_if(safe_spec.begin(), safe_spec.end(),
                    [](char c) { return !(std::isalnum(c) || c == '-' || c == '_' || c == '.'); }, '_');

    fs::create_directories("outputs");
    fs::path output_path = fs::path("outputs") / ("domain_octree_" + safe_spec + "_rank" + std::to_string(rank) + ".csv");
    std::ofstream out(output_path);
    if (!out) { throw std::runtime_error("Failed to open octree output file: " + output_path.string()); }

    thrust::host_vector<cstone::TreeNodeIndex> childOffsets(tree.numNodes);
    thrust::copy(thrust::device_ptr<const cstone::TreeNodeIndex>(tree.childOffsets), // Start device iterator
                 thrust::device_ptr<const cstone::TreeNodeIndex>(tree.childOffsets + tree.numNodes), // End device iterator
                 childOffsets.begin());

    out << "node,level,is_leaf,child_offset,prefix,start_key,cx,cy,cz,sx,sy,sz\n";
    for (int i = 0; i < tree.numNodes; ++i)
    {
        KeyType prefix = prefixes[i];
        unsigned level = cstone::decodePrefixLength(prefix) / 3;
        auto childOffset = childOffsets[i];
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
