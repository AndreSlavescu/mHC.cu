import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_cuda_arch_flags():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build mhc")
    major, minor = torch.cuda.get_device_capability()
    return [f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]


setup(
    name="mhc",
    version="0.1.0",
    description="CUDA implementation of Manifold-Constrained Hyper-Connections",
    author="DeepSeek-AI (paper), Andre Slavescu (implementation)",
    url="https://github.com/AndreSlavescu/mHC.cu",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    ext_modules=[
        CUDAExtension(
            name="mhc_cuda",
            sources=["src/python/bindings.cu"],
            include_dirs=[
                os.path.join(os.path.dirname(__file__), "src/csrc/include"),
                os.path.join(os.path.dirname(__file__), "src/csrc/kernels"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--use_fast_math",
                ]
                + get_cuda_arch_flags(),
            },
            libraries=["cublas", "cublasLt"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=["torch>=2.0.0"],
    extras_require={
        "dev": ["black", "pytest"],
    },
)
