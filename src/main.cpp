#include <mpi.h>
#include <vector>
#include <chrono>
#include <filesystem>

#include "runner.hpp"
#include "pcah5.hpp"

namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << "[--gpu] <dataset filepath> <group names ...>" << std::endl;
        return 1;
    }

    MPI_Init(&argc, &argv);

    bool gpu = false;
    if (std::string(argv[1]) == "--gpu")
    {        
        gpu = true;
        argc -= 2;
        argv += 2;
    } else {
        argc -= 1;
        argv += 1;
    }

    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    fs::path dataset_path(argv[0]);
    if (!fs::exists(dataset_path))
        throw std::runtime_error("Dataset file does not exist: " + dataset_path.string());
    if (!fs::is_regular_file(dataset_path))
        throw std::runtime_error("Dataset path is not a regular file: " + dataset_path.string());

    HighFive::File file(dataset_path.string(), HighFive::File::ReadOnly);

    for (int i = 1; i < argc; ++i)
    {
        std::string group_name(argv[i]);
        runner(file, group_name, rank, numRanks, gpu);
    }
	
    MPI_Finalize();

    return 0;
}
