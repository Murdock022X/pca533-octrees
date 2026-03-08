
.PHONY: build clean format tidy

build:
	cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCSTONE_WITH_GPU_AWARE_MPI=ON

clean:
	rm -rf ./build	

format:
	find . -regex '{src,test}*\.\(cpp\|hpp\|c\|h\|cu\|cuh\)' -exec clang-format -style=file -i {} \;

tidy:
	run-clang-tidy-19 -j4 -p build/

