#pragma once

#include <highfive/H5File.hpp>
#include <concepts>

template <std::floating_point T>
std::tuple<std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>, std::vector<T>> read_dataset(const std::string& filename, const std::string& group_name) {
    HighFive::File file(filename, HighFive::File::ReadOnly);

    auto grp = file.getGroup(group_name);

    auto ix = grp.getDataSet("ix");
    auto iy = grp.getDataSet("iy");
    auto iz = grp.getDataSet("iz");
    auto px = grp.getDataSet("px");
    auto py = grp.getDataSet("py");
    auto pz = grp.getDataSet("pz");

    std::vector<T> ix_vec, iy_vec, iz_vec, px_vec, py_vec, pz_vec;
    ix.read(ix_vec);
    iy.read(iy_vec);
    iz.read(iz_vec);
    px.read(px_vec);
    py.read(py_vec);
    pz.read(pz_vec);

    return std::make_tuple(ix_vec, iy_vec, iz_vec, px_vec, py_vec, pz_vec);
}
