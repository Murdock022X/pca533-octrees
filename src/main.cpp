#include <mpi.h>
#include <vector>
#include <chrono>
#include <filesystem>

#include "cstone/domain/domain.hpp"
#include "pcah5.hpp"

namespace fs = std::filesystem;

using Real    = double;
using KeyType = unsigned;

template<class F>
float timeCpu(F&& f)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <dataset filepath> <group names ...>" << std::endl;
        return 1;
    }

    MPI_Init(&argc, &argv);
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    fs::path dataset_path(argv[1]);
    if (!fs::exists(dataset_path))
        throw std::runtime_error("Dataset file does not exist: " + dataset_path.string());
    if (!fs::is_regular_file(dataset_path))
        throw std::runtime_error("Dataset path is not a regular file: " + dataset_path.string());

    HighFive::File file(dataset_path.string(), HighFive::File::ReadOnly);

    for (int i = 2; i < argc; ++i)
    {
        std::string group_name(argv[i]);
        if (!file.exist(group_name))
            throw std::runtime_error("Group does not exist in the dataset file: " + group_name);

        auto [ix, iy, iz, px, py, pz] = read_dataset<Real>(file, group_name);

        std::cout << "Dataset loaded -> n = " << ix.size() << std::endl;

        // fill x,y,z,h with some initial values on each rank
        std::vector<Real> h(ix.size(), 0.1);
        
        int bucketSize = 50;
        int bucketSizeFocus = 10;
        float theta = 0.6f;

        // Construct the domain
        cstone::Domain<KeyType, Real> domain(rank, numRanks, bucketSize, bucketSizeFocus, theta);
        
        std::vector<KeyType> keys(ix.size());
        std::vector<Real> s1, s2, s3;

        // std::vector<double> timing;
        
        // Convert to a lambda to measure the time taken by the sync function
        auto sync_f = [&]()
        {
            domain.sync(keys,ix,iy,iz,h,std::tie(px,py,pz),std::tie(s1,s2,s3));
        };

        float sync_ms = timeCpu(sync_f);
            
        if (rank == 0)
            std::cout << "\tDomain Sync Initial: " << sync_ms << "us" << std::endl; 
        
        // x,y,z,h now contain all particles of a part of the global octree,
        // including their halos.
        // std::vector<Real> density(domain.nParticlesWithHalos());

        // compute physical quantities, e.g. densities for particles in the assigned ranges:
        // computeDensity(density,x,y,z,h,domain.startIndex(),domain.endIndex());
        // auto exchange_f = [&]()
        // {
        // 	domain.exchangeHalos(std::tie(density), s1, s2);
        // };
        // float exchange_ms = timeCpu(exchange_f);
        
        // if (rank == 0)
        // 	std::cout << "\tExchange Halos: " << exchange_ms << "ms" << std::endl;

        #pragma omp parallel for
        for (auto i = domain.startIndex(); i < domain.endIndex(); ++i)
        {
            ix[i] += px[i];
            iy[i] += py[i];
            iz[i] += pz[i];
        }

        sync_ms = timeCpu(sync_f);
            
        if (rank == 0)
            std::cout << "\tDomain Sync with Perturbations: " << sync_ms << "us" << std::endl;

        sync_ms = timeCpu(sync_f);
            
        if (rank == 0)
            std::cout << "\tDomain Sync without Perturbations: " << sync_ms << "us" << std::endl; 
        
        // x,y,z,h now contain all particles of a part of the global octree,
        // including their halos.
        // density.resize(domain.nParticlesWithHalos());

        // compute physical quantities, e.g. densities for particles in the assigned ranges:
        // computeDensity(density,x,y,z,h,domain.startIndex(),domain.endIndex());
        // exchange_ms = timeCpu(exchange_f);
        
        // if (rank == 0)
            // std::cout << "\tExchange Halos: " << exchange_ms << "ms" << std::endl;
    }
	
    MPI_Finalize();

    return 0;
}
