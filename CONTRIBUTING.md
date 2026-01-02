# Contributing

## Run tests

The pre-commit hook as described below doesn't currently run the tests. Verify all tests pass before committing.

### Run Python Tests

```bash
make test-python
```

### Run C++ / CUDA Tests

```bash
make test
```

## Pre-commit Hook

This project uses a pre-commit hook to automatically format code using the rules in `.clang-format` and run tests before each commit.

**Setup:**
```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

**Process:**
1. Runs `clang-format` on all staged `.cu`, `.cuh`, and `.h` files
2. Runs `black` formatter on all `.py` files

If any step fails, the commit will be aborted.

**Format Manually:**
```bash
find csrc -name "*.cu" -o -name "*.cuh" | xargs clang-format -i
```

## Code Style

- 4 space indents
- 100 column limit
- K & R braces
- Left pointer alignment
