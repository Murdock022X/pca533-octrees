#define CATCH_CONFIG_CPP11_TO_STRING
#define CATCH_CONFIG_COLOUR_ANSI
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_NO_POSIX_SIGNALS

#include "catch.hpp"
#include "pcah5.hpp"
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

TEST_CASE("HDF5Read", "[unit]") {
	const char* path_value = std::getenv("TEST_HDF5_PATH");
	REQUIRE(path_value != nullptr);
	fs::path hdf5_path(path_value);
	REQUIRE(fs::exists(hdf5_path));
	REQUIRE(fs::is_regular_file(hdf5_path));
	HighFive::File file(hdf5_path.string(), HighFive::File::ReadOnly);
	auto [ix, iy, iz, px, py, pz] = read_dataset<double>(file, "test");
	REQUIRE(ix.size() == 1000);
	REQUIRE(iy.size() == 1000);
	REQUIRE(iz.size() == 1000);
	REQUIRE(px.size() == 1000);
	REQUIRE(py.size() == 1000);
	REQUIRE(pz.size() == 1000);
}
