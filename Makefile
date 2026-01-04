.PHONY: all build test test-python bench bench-python clean format lint install install-dev force-install

CUDA_ARCH ?= "80;86;89;90;100"
BUILD_DIR = build
STAMP_DIR = .stamps

CSRC_FILES := $(shell find src/csrc -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.cpp' 2>/dev/null)
BINDING_FILES := src/python/bindings.cu setup.py pyproject.toml

all: build

$(STAMP_DIR):
	@mkdir -p $(STAMP_DIR)

$(STAMP_DIR)/build: $(CSRC_FILES) | $(STAMP_DIR)
	@mkdir -p $(BUILD_DIR)
	cmake -B $(BUILD_DIR) -S src/csrc -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)
	cmake --build $(BUILD_DIR) -j$$(nproc)
	@touch $@

build: $(STAMP_DIR)/build

$(STAMP_DIR)/install: $(CSRC_FILES) $(BINDING_FILES) | $(STAMP_DIR)
	@echo "csrc code was updated, rebuilding full mhc extension..."
	pip install -e . -q
	@touch $@

install: $(STAMP_DIR)/install

force-install:
	pip install -e .
	@mkdir -p $(STAMP_DIR)
	@touch $(STAMP_DIR)/install

install-dev:
	pip install -e ".[dev]"
	@mkdir -p $(STAMP_DIR)
	@touch $(STAMP_DIR)/install

test: build
	@failed=0; \
	for t in $(BUILD_DIR)/test_*; do \
		echo "Running $$t..."; \
		if ! $$t; then \
			failed=1; \
		fi; \
	done; \
	if [ $$failed -eq 0 ]; then \
		echo "All C++ tests passed."; \
	else \
		echo "Some C++ tests FAILED."; \
		exit 1; \
	fi

test-python: install
	LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 pytest src/python/tests -v

bench: build
	@for b in $(BUILD_DIR)/bench_*; do echo "Running $$b..."; $$b; done

bench-python: install
	LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python src/python/benchmarks/bench_layer.py --all-configs --backward

clean:
	rm -rf $(BUILD_DIR) $(STAMP_DIR)

format:
	find src/csrc -name "*.cu" -o -name "*.cuh" -o -name "*.h" -o -name "*.cpp" | xargs clang-format -i
	black src/python

lint:
	@echo "python linting with ruff..."
	ruff check src/python/
	@echo ""
	@echo "c++ / cuda linting with cppcheck..."
	find src/csrc \( -name "*.cu" -o -name "*.cuh" -o -name "*.h" -o -name "*.cpp" \) -not -path "src/csrc/build/*" | \
		xargs cppcheck --enable=warning,performance,portability \
		--suppress=missingIncludeSystem \
		--suppress=unmatchedSuppression \
		--suppress=syntaxError \
		--suppress=shiftTooManyBits \
		--suppress=uninitMemberVar \
		--suppress=duplicateAssignExpression \
		--inline-suppr \
		--language=c++ \
		-I src/csrc/include \
		-I src/csrc/kernels
