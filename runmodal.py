import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import modal

APP_NAME = "mhc-cu"
HF_CACHE_VOLUME = "mhc-hf-cache"
RUNS_VOLUME = "mhc-runs"

app = modal.App(APP_NAME)


base_image = modal.Image.from_registry(
    "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.10"
).apt_install(
    "build-essential",
    "cmake",
    "ninja-build",
    "python3-dev",
    "python3-pip",
)
native_image = base_image.run_commands("ln -sf /usr/bin/python3 /usr/bin/python")
# Define all images unconditionally so Modal's container import always sees every function.
python_image = native_image.run_commands(
    "python3 -m pip install numpy ninja pytest",
    "python3 -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128",
)
train_image = python_image.run_commands(
    "python3 -m pip install datasets sentencepiece tiktoken tqdm transformers accelerate>=0.26.0",
)
project_root = Path(__file__).parent
ignore_patterns = [
    ".git",
    ".stamps",
    ".ruff_cache",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "runs",
    "*.egg-info",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.o",
]
native_image = native_image.add_local_dir(
    project_root, remote_path="/workspace", copy=False, ignore=ignore_patterns
)
python_image = python_image.add_local_dir(
    project_root, remote_path="/workspace", copy=False, ignore=ignore_patterns
)
train_image = train_image.add_local_dir(
    project_root, remote_path="/workspace", copy=False, ignore=ignore_patterns
)
runs_volume = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


def _print_env() -> None:
    try:
        import torch
    except ImportError:
        print("PyTorch: not installed")
        return
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def _run_make(target: str, cuda_arch: Optional[str]) -> None:
    env = os.environ.copy()
    env.setdefault("CC", "gcc")
    env.setdefault("CXX", "g++")
    env.setdefault("CUDACXX", "/usr/local/cuda/bin/nvcc")
    env.setdefault("PIP_NO_BUILD_ISOLATION", "1")
    if cuda_arch:
        env["CUDA_ARCH"] = cuda_arch
    print(f"Running: make {target}")
    subprocess.run(["make", target], check=True, cwd="/workspace", env=env)


def _run_tests(scope: str, cuda_arch: Optional[str]) -> None:
    if scope in ("native", "all"):
        _run_make("test", cuda_arch)
    if scope in ("python", "all"):
        _run_make("test-python", cuda_arch)


def _run_benchmarks(scope: str, cuda_arch: Optional[str]) -> None:
    if scope in ("native", "all"):
        _run_make("bench", cuda_arch)
    if scope in ("python", "all"):
        _run_make("bench-python", cuda_arch)


def _run_job(mode: str, scope: str, cuda_arch: Optional[str]) -> None:
    sys.path.insert(0, "/workspace")
    _print_env()
    if mode == "test":
        _run_tests(scope, cuda_arch)
        return
    if mode == "bench":
        _run_benchmarks(scope, cuda_arch)
        return
    if mode == "all":
        _run_tests(scope, cuda_arch)
        _run_benchmarks(scope, cuda_arch)
        return
    raise ValueError("mode must be 'test', 'bench', or 'all'")


def _run_train(train_args: Optional[str], cuda_arch: Optional[str]) -> Tuple[str, str]:
    sys.path.insert(0, "/workspace/src/python")
    _print_env()
    _run_make("install", cuda_arch)
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("MHC_RUNS_ROOT", "/runs")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if cuda_arch:
        os.environ["CUDA_ARCH"] = cuda_arch
    import mhc.trainer as trainer

    args = shlex.split(train_args) if train_args else []
    if "--runs-root" not in args:
        args += ["--runs-root", "/runs"]
    run_name, out_dir = trainer.run_training(args)
    return run_name, out_dir


if python_image is not None:

    @app.function(gpu="H100", image=python_image, timeout=3600)
    def run_h100(mode: str, scope: str, cuda_arch: Optional[str]) -> None:
        _run_job(mode, scope, cuda_arch)

    @app.function(gpu="B200", image=python_image, timeout=3600)
    def run_b200(mode: str, scope: str, cuda_arch: Optional[str]) -> None:
        _run_job(mode, scope, cuda_arch)


if train_image is not None:

    @app.function(
        gpu="H100",
        image=train_image,
        volumes={"/runs": runs_volume, "/root/.cache/huggingface": hf_cache_volume},
        timeout=12 * 3600,
    )
    def train_h100(
        train_args: Optional[str], cuda_arch: Optional[str]
    ) -> Tuple[str, str]:
        return _run_train(train_args, cuda_arch)

    @app.function(
        gpu="B200",
        image=train_image,
        volumes={"/runs": runs_volume, "/root/.cache/huggingface": hf_cache_volume},
        timeout=12 * 3600,
    )
    def train_b200(
        train_args: Optional[str], cuda_arch: Optional[str]
    ) -> Tuple[str, str]:
        return _run_train(train_args, cuda_arch)


@app.function(gpu="H100", image=native_image, timeout=3600)
def run_h100_native(mode: str, scope: str, cuda_arch: Optional[str]) -> None:
    _run_job(mode, scope, cuda_arch)


@app.function(gpu="B200", image=native_image, timeout=3600)
def run_b200_native(mode: str, scope: str, cuda_arch: Optional[str]) -> None:
    _run_job(mode, scope, cuda_arch)


def _default_arch(gpu: str) -> Optional[str]:
    gpu_norm = gpu.strip().lower()
    if gpu_norm == "h100":
        return "90"
    if gpu_norm == "b200":
        return "100"
    return None


@app.local_entrypoint()
def main(
    gpu: str = "h100",
    mode: str = "test",
    scope: str = "all",
    cuda_arch: Optional[str] = None,
    train_args: Optional[str] = None,
    download: bool = False,
) -> None:
    gpu_norm = gpu.strip().lower()
    mode_norm = mode.strip().lower()
    scope_norm = scope.strip().lower()
    resolved_arch = cuda_arch or _default_arch(gpu_norm)
    if gpu_norm == "h100":
        if mode_norm == "train":
            if "train_h100" not in globals():
                raise RuntimeError(
                    "Training image not built for this scope. Rerun with --mode train."
                )
            run_name, _ = train_h100.remote(train_args, resolved_arch)
            if download:
                _download_run(run_name)
            return
        if scope_norm not in ("native", "python", "all"):
            raise ValueError("scope must be 'native', 'python', or 'all'")
        if scope_norm == "native":
            run_h100_native.remote(mode_norm, scope_norm, resolved_arch)
        else:
            if "run_h100" not in globals():
                raise RuntimeError(
                    "Python image not built for this scope. Rerun without --scope native."
                )
            run_h100.remote(mode_norm, scope_norm, resolved_arch)
        return
    if gpu_norm == "b200":
        if mode_norm == "train":
            if "train_b200" not in globals():
                raise RuntimeError(
                    "Training image not built for this scope. Rerun with --mode train."
                )
            run_name, _ = train_b200.remote(train_args, resolved_arch)
            if download:
                _download_run(run_name)
            return
        if scope_norm not in ("native", "python", "all"):
            raise ValueError("scope must be 'native', 'python', or 'all'")
        if scope_norm == "native":
            run_b200_native.remote(mode_norm, scope_norm, resolved_arch)
        else:
            if "run_b200" not in globals():
                raise RuntimeError(
                    "Python image not built for this scope. Rerun without --scope native."
                )
            run_b200.remote(mode_norm, scope_norm, resolved_arch)
        return
    raise ValueError("gpu must be 'h100' or 'b200'")


def _download_run(run_name: str) -> None:
    if not run_name:
        return
    run_name = run_name.strip()
    local_root = Path("runs")
    local_root.mkdir(exist_ok=True)
    local_dir = local_root / run_name
    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            RUNS_VOLUME,
            f"/runs/{run_name}",
            str(local_dir),
        ],
        check=True,
    )
