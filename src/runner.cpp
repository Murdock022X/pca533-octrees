#include "cstone/domain/domain.hpp"
#include <vector>
#include <chrono>
#include <iostream>
#include <tuple>
#include <cstdint>
#include "runner.hpp"
#include "save_octree.hpp"
#include "utils.hpp"

void runner(HighFive::File& file, std::string group_name, int rank, int numRanks, bool gpu)
{
    if (!file.exist(group_name))
        throw std::runtime_error("Group does not exist in the dataset file: " + group_name);

    auto [ix, iy, iz, px, py, pz] = read_dataset<Real>(file, group_name);

    std::cout << "Dataset loaded [" << group_name << "] -> n = " << ix.size() << std::endl;

    std::vector<Real> h(ix.size(), 0.1);

    int bucketSize = 1024;
    int bucketSizeFocus = 8;
    float theta = 0.6f;

    std::vector<KeyType> keys(ix.size());
    if (!gpu)
    {
        runnerCpu(keys, ix, iy, iz, h, px, py, pz, rank, numRanks, bucketSize, bucketSizeFocus, theta, group_name);
    }
    else
    {
        runnerGpu(keys, ix, iy, iz, h, px, py, pz, rank, numRanks, bucketSize, bucketSizeFocus, theta, group_name);
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
               float theta,
               std::string group_name)
{
    cstone::Domain<KeyType, Real, cstone::CpuTag> domain(rank, numRanks, bucketSize, bucketSizeFocus, theta);

    std::vector<Real> s1, s2, s3;
    auto sync_f = [&]() { domain.sync(keys, ix, iy, iz, h, std::tuple{}, std::tie(s1, s2, s3)); };

    float sync_ms = timeCpu(sync_f);

    if (rank == 0) { std::cout << "\tDomain Sync Initial: " << sync_ms << "us" << std::endl; }

    saveDomainOctreeCsvCpu(domain, group_name + "_initial", rank);

#pragma omp parallel for
    for (auto i = domain.startIndex(); i < domain.endIndex(); ++i)
    {
        ix[i] += px[i];
        iy[i] += py[i];
        iz[i] += pz[i];
    }

    sync_ms = timeCpu(sync_f);

    if (rank == 0) { std::cout << "\tDomain Sync with Perturbations: " << sync_ms << "us" << std::endl; }

    saveDomainOctreeCsvCpu(domain, group_name + "_perturbed", rank);

    sync_ms = timeCpu(sync_f);

    if (rank == 0) { std::cout << "\tDomain Sync without Perturbations: " << sync_ms << "us" << std::endl; }
}
