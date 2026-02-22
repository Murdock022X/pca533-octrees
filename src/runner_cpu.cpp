#include "cstone/domain/domain.hpp"
#include <vector>
#include <chrono>
#include <iostream>
#include <tuple>
#include <cstdint>

using Real    = double;
using KeyType = uint64_t;

template<class F>
float timeCpu(F&& f)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
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
               float theta)
{
    cstone::Domain<KeyType, Real> domain(rank, numRanks, bucketSize, bucketSizeFocus, theta);

    std::vector<Real> s1, s2, s3;
    auto sync_f = [&]() { domain.sync(keys, ix, iy, iz, h, std::tie(px, py, pz), std::tie(s1, s2, s3)); };

    float sync_ms = timeCpu(sync_f);

    if (rank == 0) { std::cout << "\tDomain Sync Initial: " << sync_ms << "us" << std::endl; }

#pragma omp parallel for
    for (auto i = domain.startIndex(); i < domain.endIndex(); ++i)
    {
        ix[i] += px[i];
        iy[i] += py[i];
        iz[i] += pz[i];
    }

    sync_ms = timeCpu(sync_f);

    if (rank == 0) { std::cout << "\tDomain Sync with Perturbations: " << sync_ms << "us" << std::endl; }

    sync_ms = timeCpu(sync_f);

    if (rank == 0) { std::cout << "\tDomain Sync without Perturbations: " << sync_ms << "us" << std::endl; }
}
