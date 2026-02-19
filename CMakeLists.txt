cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(octrees VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_CXX_STANDARD 20)
set(DEFAULT_BUILD_TYPE "Release")

find_package(HDF5 REQUIRED COMPONENTS C CXX)

add_subdirectory(src)
add_subdirectory(test)
