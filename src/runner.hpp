#pragma once

#include "pcah5.hpp"
#include <highfive/H5File.hpp>
#include <string>
#include <filesystem>

using Real = double;
using KeyType = uint64_t;
namespace fs = std::filesystem;

void runnerCpu(std::vector<KeyType> &keys, std::vector<Real> &ix,
               std::vector<Real> &iy, std::vector<Real> &iz,
               std::vector<Real> &h, std::vector<Real> &px,
               std::vector<Real> &py, std::vector<Real> &pz, int rank,
               int numRanks, int bucketSize, int bucketSizeFocus, float theta,
               std::string group_name);

void runnerGpu(std::vector<KeyType> &keys, std::vector<Real> &ix,
               std::vector<Real> &iy, std::vector<Real> &iz,
               std::vector<Real> &h, std::vector<Real> &px,
               std::vector<Real> &py, std::vector<Real> &pz, int rank,
               int numRanks, int bucketSize, int bucketSizeFocus, float theta,
               std::string group_name);

void runner(HighFive::File &file, std::string group_name, int rank,
            int numRanks, bool gpu, int bucketSize, int bucketSizeFocus,
            float theta);
