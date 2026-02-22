#include <chrono>
#include <filesystem>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

#include "pcah5.hpp"
#include "runner.hpp"

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
  auto printUsage = [&]() {
    std::cerr << "Usage: " << argv[0]
              << " [--gpu] [--theta <value>] [--bucket-size-global <value>] "
                 "[--bucket-size-focus <value>] <dataset filepath> <group "
                 "names ...>"
              << std::endl;
  };

  bool gpu = false;
  double theta = 0.6;
  int bucketSizeGlobal = 1024;
  int bucketSizeFocus = 64;

  int positionalStart = 1;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      printUsage();
      return 0;
    }
    if (arg == "--gpu") {
      gpu = true;
    } else if (arg == "--theta") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --theta" << std::endl;
        printUsage();
        return 1;
      }
      try {
        theta = std::stod(argv[++i]);
      } catch (const std::exception &) {
        std::cerr << "Invalid value for --theta: " << argv[i] << std::endl;
        return 1;
      }
    } else if (arg == "--bucket-size-global") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --bucket-size-global" << std::endl;
        printUsage();
        return 1;
      }
      try {
        bucketSizeGlobal = std::stoi(argv[++i]);
      } catch (const std::exception &) {
        std::cerr << "Invalid value for --bucket-size-global: " << argv[i]
                  << std::endl;
        return 1;
      }
    } else if (arg == "--bucket-size-focus") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --bucket-size-focus" << std::endl;
        printUsage();
        return 1;
      }
      try {
        bucketSizeFocus = std::stoi(argv[++i]);
      } catch (const std::exception &) {
        std::cerr << "Invalid value for --bucket-size-focus: " << argv[i]
                  << std::endl;
        return 1;
      }
    } else if (!arg.empty() && arg[0] == '-') {
      std::cerr << "Unknown option: " << arg << std::endl;
      printUsage();
      return 1;
    } else {
      positionalStart = i;
      break;
    }
  }

  if (positionalStart >= argc || positionalStart + 1 >= argc) {
    printUsage();
    return 1;
  }

  MPI_Init(&argc, &argv);

  int rank = 0, numRanks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  fs::path dataset_path(argv[positionalStart]);
  if (!fs::exists(dataset_path))
    throw std::runtime_error("Dataset file does not exist: " +
                             dataset_path.string());
  if (!fs::is_regular_file(dataset_path))
    throw std::runtime_error("Dataset path is not a regular file: " +
                             dataset_path.string());

  HighFive::File file(dataset_path.string(), HighFive::File::ReadOnly);

  for (int i = positionalStart + 1; i < argc; ++i) {
    std::string group_name(argv[i]);
    runner(file, group_name, rank, numRanks, gpu, bucketSizeGlobal,
           bucketSizeFocus, static_cast<float>(theta));
  }

  MPI_Finalize();

  return 0;
}
