#pragma once

#include <highfive/H5File.hpp>
#include <string>

void runner(HighFive::File& file, std::string group_name, int rank, int numRanks, bool gpu);
