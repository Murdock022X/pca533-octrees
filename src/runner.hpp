#pragma once

#include "pcah5.hpp"
#include <highfive/H5File.hpp>
#include <string>
#include <filesystem>

using Real = double;
using KeyType = uint64_t;
namespace fs = std::filesystem;

void runnerCpu(const std::vector<KeyType> &keys, const std::vector<Real> &ix,
               const std::vector<Real> &iy, const std::vector<Real> &iz,
               const std::vector<Real> &h, const std::vector<Real> &px,
               const std::vector<Real> &py, const std::vector<Real> &pz, int rank,
               int numRanks, int bucketSize, int bucketSizeFocus, float theta,
               std::string group_name, bool save);

void runnerGpu(const std::vector<KeyType> &keys, const std::vector<Real> &ix,
               const std::vector<Real> &iy, const std::vector<Real> &iz,
               const std::vector<Real> &h, const std::vector<Real> &px,
               const std::vector<Real> &py, const std::vector<Real> &pz, int rank,
               int numRanks, int bucketSize, int bucketSizeFocus, float theta,
               std::string group_name, bool save);

void runnerCpuMulti(const std::vector<KeyType> &keys, const std::vector<Real> &ix,
               const std::vector<Real> &iy, const std::vector<Real> &iz,
               const std::vector<Real> &h, const std::vector<Real> &px,
               const std::vector<Real> &py, const std::vector<Real> &pz, int rank,
               int numRanks, int bucketSize, int bucketSizeFocus, float theta,
               std::string group_name, bool save);

void runnerGpuMulti(const std::vector<KeyType> &keys, const std::vector<Real> &ix,
               const std::vector<Real> &iy, const std::vector<Real> &iz,
               const std::vector<Real> &h, const std::vector<Real> &px,
               const std::vector<Real> &py, const std::vector<Real> &pz, int rank,
               int numRanks, int bucketSize, int bucketSizeFocus, float theta,
               std::string group_name, bool save);

void runner(HighFive::File &file, std::string group_name, int rank,
            int numRanks, bool gpu, bool lets, int bucketSize, int bucketSizeFocus,
            float theta, bool save);
