
.PHONY: format
format:
	find . -regex '{src,test}*\.\(cpp\|hpp\|c\|h\|cu\|cuh\)' -exec clang-format -style=file -i {} \;

