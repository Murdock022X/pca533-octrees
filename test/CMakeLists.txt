add_executable(octree_tests main.cpp)

target_include_directories(octree_tests PRIVATE ../include ../src ../HighFive/include ${HDF5_INCLUDE_DIRS})
target_link_libraries(octree_tests PRIVATE ${HDF5_LIBRARIES})
