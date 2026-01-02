# mHC.cu

unofficial CUDA implementation of mHC: Manifold-Constrained Hyper-Connections by DeepSeek-AI

## Installation

```bash
make install      # install PyTorch extension
make install-dev  # install with dev dependencies
```

## Build

```bash
make              # build C++ / CUDA source for all architectures
make CUDA_ARCH=90 # build for specific arch (H100)
make clean        # clean build
```

## Test

```bash
make test         # C++ tests
make test-python  # Python tests
```

## Benchmark

```bash
make bench        # run all benchmarks
```

## Format

```bash
make format       # clang-format + python black formatting
```

## Usage

```python
import torch
from mhc import MHCLayer

layer = MHCLayer(hidden_dim=4096, expansion_rate=4).cuda()
x = torch.randn(8, 4, 4096, device="cuda")  # [B, n, C]
y = layer(x)  # [B, n, C]
```

## Paper

**mHC: Manifold-Constrained Hyper-Connections**  
https://arxiv.org/abs/2512.24880

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
