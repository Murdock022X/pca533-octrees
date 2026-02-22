add_subdirectory(cornerstone)

add_executable(pca main.cpp runner.hpp runner.cu runner_cpu.cpp pcah5.hpp)

target_include_directories(pca PUBLIC ${HDF5_INCLUDE_DIRS} ../HighFive/include cornerstone/include)
target_link_libraries(pca PRIVATE ${HDF5_LIBRARIES} cstone_gpu)

set_source_files_properties(runner.cu PROPERTIES COMPILE_DEFINITIONS USE_CUDA)
