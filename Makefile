.PHONY: all build test test-python bench clean format install

CUDA_ARCH ?= "80;86;89;90;100"
BUILD_DIR = build

all: build

build:
	@mkdir -p $(BUILD_DIR)
	cmake -B $(BUILD_DIR) -S src/csrc -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)
	cmake --build $(BUILD_DIR) -j$$(nproc)

test: build
	@for t in $(BUILD_DIR)/test_*; do echo "Running $$t..."; $$t || exit 1; done
	@echo "All C++ tests passed."

test-python: install
	LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 pytest src/python/tests -v

bench: build
	@for b in $(BUILD_DIR)/bench_*; do echo "Running $$b..."; $$b; done

clean:
	rm -rf $(BUILD_DIR)

format:
	find src/csrc -name "*.cu" -o -name "*.cuh" -o -name "*.h" -o -name "*.cpp" | xargs clang-format -i
	black src/python

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"