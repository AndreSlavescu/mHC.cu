# mHC.cu
CUDA implementation of Manifold-Constrained Hyper-Connections (mHC)

## Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90 -DMHC_ENABLE_PDL=ON # test platform is H100 SXM5
cmake --build build -j4
```

For multi-architecture builds:
```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;100"
```

## Run Tests

```bash
./build/test_rmsnorm
./build/test_sinkhorn_knopp
./build/test_stream_mix_tc
./build/test_mhc_layer
```

## Run Benchmarks

```bash
./build/bench_rmsnorm
./build/bench_sinkhorn_knopp
./build/bench_mhc_layer
./build/bench_fused_rmsnorm_matmul
./build/bench_stream_ops_backward
./build/bench_rmsnorm_backward
```

## Contributing

### Pre-commit Hook

This project uses a pre-commit hook to automatically format code using the rules in `.clang-format` and run tests before each commit.

**Setup:**
```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

**Process:**
1. Runs `clang-format` on all staged `.cu` and `.cuh` files
2. Builds the project
3. Runs all tests

If any step fails, the commit will be aborted.

**Format Manually:**
```bash
find csrc -name "*.cu" -o -name "*.cuh" | xargs clang-format -i
```

## Paper

**mHC: Manifold-Constrained Hyper-Connections**  
https://arxiv.org/abs/2512.24880

Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, Wenfeng Liang

DeepSeek-AI

## Citation

```bibtex
@article{xie2025mhc,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={Xie, Zhenda and Wei, Yixuan and Cao, Huanqi and Zhao, Chenggang and Deng, Chengqi and Li, Jiashi and Dai, Damai and Gao, Huazuo and Chang, Jiang and Zhao, Liang and Zhou, Shangyan and Xu, Zhean and Zhang, Zhengyan and Zeng, Wangding and Hu, Shengding and Wang, Yuqing and Yuan, Jingyang and Wang, Lean and Liang, Wenfeng},
  journal={arXiv preprint arXiv:2512.24880},
  year={2025}
}
```
