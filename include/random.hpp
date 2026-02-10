#pragma once

#include <random>
#include <concepts>
#include <vector>
#include <tuple>

template <typename Distribution, typename Generator>
requires std::uniform_random_bit_generator<Generator> 
      && std::floating_point<typename Distribution::result_type>
auto dist_3d(Generator &gen, Distribution &d, size_t n) {
    using T = typename Distribution::result_type;
    std::vector<T> x(n), y(n), z(n);

    for (size_t i = 0; i < n; i++) {
        x[i] = d(gen);
        y[i] = d(gen);
        z[i] = d(gen);
    }

    // Using move to ensure we don't accidentally trigger expensive copies
    return std::make_tuple(std::move(x), std::move(y), std::move(z));
}

