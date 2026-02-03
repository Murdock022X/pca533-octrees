#pragma once

#include <random>

template <typename Distribution, typename T>
std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> dist_3d(UniformRandomBitGenerator& gen, Distribution& d, size_t n) {
	std::vector<T> x(n), y(n), z(n);

	for (size_t i = 0; i < n; i++) {
		x[i] = d(gen);
		y[i] = d(gen);
		z[i] = d(gen);
	}

	return std::make_tuple(x, y, z);
}

