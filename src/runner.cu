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

using Real    = double;
using KeyType = uint64_t;
namespace fs = std::filesystem;

template<class F>
float timeCpu(F&& f)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

template<class Accelerator>
void saveDomainOctreeCsv(const cstone::Domain<KeyType, Real, Accelerator>& domain, const std::string& group_name, int rank)
{
    auto tree = domain.globalTree();
    if (tree.numNodes == 0) { return; }

    std::vector<cstone::Vec3<Real>> centers(tree.numNodes), sizes(tree.numNodes);
    thrust::host_vector<KeyType> prefixes(tree.numNodes);
    thrust::copy(thrust::device_ptr<const KeyType>(tree.prefixes), // Start device iterator
                 thrust::device_ptr<const KeyType>(tree.prefixes + tree.numNodes), // End device iterator
                 prefixes.begin());
    cstone::nodeFpCenters<KeyType>(std::span(prefixes.data(), prefixes.size()), centers.data(), sizes.data(), domain.box());

    std::string safe_group = group_name;
    std::replace_if(safe_group.begin(), safe_group.end(),
                    [](char c) { return !(std::isalnum(c) || c == '-' || c == '_' || c == '.'); }, '_');

    fs::create_directories("outputs");
    fs::path output_path = fs::path("outputs") / ("domain_octree_" + safe_group + "_rank" + std::to_string(rank) + ".csv");
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

void runnerCpu(std::vector<KeyType>& keys,
               std::vector<Real>& ix,
               std::vector<Real>& iy,
               std::vector<Real>& iz,
               std::vector<Real>& h,
               std::vector<Real>& px,
               std::vector<Real>& py,
               std::vector<Real>& pz,
               int rank,
               int numRanks,
               int bucketSize,
               int bucketSizeFocus,
               float theta);

void runner(HighFive::File& file, std::string group_name, int rank, int numRanks, bool gpu) {
    if (!file.exist(group_name))
        throw std::runtime_error("Group does not exist in the dataset file: " + group_name);

    auto [ix, iy, iz, px, py, pz] = read_dataset<Real>(file, group_name);

    std::cout << "Dataset loaded [" << group_name << "] -> n = " << ix.size() << std::endl;

    // fill x,y,z,h with some initial values on each rank
    std::vector<Real> h(ix.size(), 0.1);
    
    int bucketSize = 1024;
    int bucketSizeFocus = 8;
    float theta = 0.6f;

    std::vector<KeyType> keys(ix.size());
    if (!gpu)
    {
        runnerCpu(keys, ix, iy, iz, h, px, py, pz, rank, numRanks, bucketSize, bucketSizeFocus, theta);
        return;
    }
    else
    {
        size_t free_initial_byte, free_byte;
        size_t total_initial_byte, total_byte;

        // Get memory info
        checkGpuErrors(cudaMemGetInfo(&free_initial_byte, &total_initial_byte));

        size_t used_initial_byte = total_initial_byte - free_initial_byte;

        cstone::Domain<KeyType, Real, cstone::GpuTag> domain(rank, numRanks, bucketSize, bucketSizeFocus, theta);

        cstone::DeviceVector<KeyType> d_keys(keys.size());
        cstone::DeviceVector<Real> d_ix(ix), d_iy(iy), d_iz(iz), d_h(h);
        cstone::DeviceVector<Real> d_px(px), d_py(py), d_pz(pz);
        std::vector<Real> s1, s2, s3;
        cstone::DeviceVector<Real> d_s1, d_s2, d_s3;

        // Convert to a lambda to measure the time taken by the sync function
        auto sync_f = [&]()
        {
            domain.sync(d_keys, d_ix, d_iy, d_iz, d_h, std::tie(d_px, d_py, d_pz), std::tie(d_s1, d_s2, d_s3));
        };

        float sync_ms = timeCpu(sync_f);

        checkGpuErrors(cudaMemGetInfo(&free_byte, &total_byte));
        size_t consumed = total_byte - free_byte - used_initial_byte;

            
        if (rank == 0)
            std::cout << "\tDomain Sync Initial: " << sync_ms << "us, Memory Usage: " << consumed / (1024 * 1024) << "Mb" << std::endl; 

        // thrust::transform(d_ix.begin(), d_ix.end(), d_px.begin(), d_ix.begin(), thrust::plus<Real>());
        // thrust::transform(d_iy.begin(), d_iy.end(), d_py.begin(), d_iy.begin(), thrust::plus<Real>());
        // thrust::transform(d_iz.begin(), d_iz.end(), d_pz.begin(), d_iz.begin(), thrust::plus<Real>());
        
        sync_ms = timeCpu(sync_f);

        checkGpuErrors(cudaMemGetInfo(&free_byte, &total_byte));
        consumed = total_byte - free_byte - used_initial_byte;
            
        if (rank == 0)
            std::cout << "\tDomain Sync with Perturbations: " << sync_ms << "us, Memory Usage: " << consumed / (1024 * 1024) << "Mb" << std::endl;

        sync_ms = timeCpu(sync_f);

        checkGpuErrors(cudaMemGetInfo(&free_byte, &total_byte));
        consumed = total_byte - free_byte - used_initial_byte;
            
        if (rank == 0)
            std::cout << "\tDomain Sync without Perturbations: " << sync_ms << "us, Memory Usage: " << consumed / (1024 * 1024) << "Mb\n" << std::endl; 

        saveDomainOctreeCsv(domain, group_name, rank);
    }
}
