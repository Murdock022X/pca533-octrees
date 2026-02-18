
.PHONY: format tidy
format:
	find . -regex '{src,test}*\.\(cpp\|hpp\|c\|h\|cu\|cuh\)' -exec clang-format -style=file -i {} \;

tidy:
	run-clang-tidy-19 -j4 -p build/

