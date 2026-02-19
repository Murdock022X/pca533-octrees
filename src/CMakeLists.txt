add_subdirectory(cornerstone)

add_executable(pca main.cpp)

target_include_directories(pca PUBLIC ${HDF5_INCLUDE_DIRS} ../HighFive/include cornerstone/include)
target_link_libraries(pca PRIVATE ${HDF5_LIBRARIES} cstone_gpu)
