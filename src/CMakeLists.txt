# add_subdirectory(cornerstone)
add_library(pca STATIC 
    pcah5.hpp
    # h5.cpp
)

target_include_directories(pca PUBLIC ${HDF5_INCLUDE_DIRS} ../HighFive/include)
