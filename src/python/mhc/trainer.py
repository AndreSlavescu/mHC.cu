from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from itertools import chain
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_callback import TrainerCallback
from transformers.utils import SAFE_WEIGHTS_NAME

try:
    from transformers.trainer_utils import TRAINING_ARGS_NAME
except ImportError:  # pragma: no cover - compatibility with older transformers
    try:
        from transformers.utils import TRAINING_ARGS_NAME
    except ImportError:
        TRAINING_ARGS_NAME = "training_args.bin"

from .model import ModelConfig, MHCTransformer, mhc_cuda

MODEL_PRESETS = {
    "3b": {
        "n_layers": 12,
        "hidden_dim": 1280,
        "ffn_dim": 896,
        "n_heads": 16,
        "rope_dim": 64,
        "vocab_size": 129280,
        "max_seq_len": 4096,
    },
    "9b": {
        "n_layers": 18,
        "hidden_dim": 1920,
        "ffn_dim": 1280,
        "n_heads": 24,
        "rope_dim": 64,
        "vocab_size": 129280,
        "max_seq_len": 4096,
    },
    "27b": {
        "n_layers": 30,
        "hidden_dim": 2560,
        "ffn_dim": 1536,
        "n_heads": 32,
        "rope_dim": 64,
        "vocab_size": 129280,
        "max_seq_len": 4096,
    },
    "3b-1t": {
        "n_layers": 12,
        "hidden_dim": 1280,
        "ffn_dim": 896,
        "n_heads": 16,
        "rope_dim": 64,
        "vocab_size": 129280,
        "max_seq_len": 4096,
    },
}

TRAIN_PRESETS = {
    "3b": {
        "batch_size": 320,
        "training_steps": 30000,
        "base_lr": 8.6e-4,
    },
    "9b": {
        "batch_size": 512,
        "training_steps": 50000,
        "base_lr": 5.9e-4,
    },
    "27b": {
        "batch_size": 1280,
        "training_steps": 50000,
        "base_lr": 4.0e-4,
    },
    "3b-1t": {
        "batch_size": 2560,
        "training_steps": 100000,
        "base_lr": 9.0e-4,
    },
}


def _bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def _get_cuda_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda_mem_allocated_gb": _bytes_to_gib(torch.cuda.memory_allocated()),
        "cuda_mem_reserved_gb": _bytes_to_gib(torch.cuda.memory_reserved()),
        "cuda_mem_peak_gb": _bytes_to_gib(torch.cuda.max_memory_allocated()),
        "cuda_mem_peak_reserved_gb": _bytes_to_gib(torch.cuda.max_memory_reserved()),
        "cuda_mem_free_gb": _bytes_to_gib(free),
        "cuda_mem_total_gb": _bytes_to_gib(total),
    }


def _optimizer_state_gb(optimizer: torch.optim.Optimizer) -> float:
    total_bytes = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total_bytes += value.numel() * value.element_size()
    return _bytes_to_gib(total_bytes)


def _param_grad_gb(model: torch.nn.Module) -> tuple[float, float]:
    param_bytes = 0
    grad_bytes = 0
    for param in model.parameters():
        param_bytes += param.numel() * param.element_size()
        if param.grad is not None:
            grad_bytes += param.grad.numel() * param.grad.element_size()
    return _bytes_to_gib(param_bytes), _bytes_to_gib(grad_bytes)


def _resolve_text_field(example: dict) -> str:
    for key in ("text", "content", "document", "doc", "data"):
        if key in example:
            return key
    for key, value in example.items():
        if isinstance(value, str):
            return key
    raise ValueError("Unable to infer text field from dataset samples.")


def _prepare_hf_dataset(dataset, tokenizer, text_field: str, seq_len: int, stride: int):
    eos_token_id = tokenizer.eos_token_id

    def tokenize(batch: dict) -> dict:
        encoded = tokenizer(
            batch[text_field],
            add_special_tokens=False,
            return_attention_mask=False,
        )
        if eos_token_id is not None:
            encoded["input_ids"] = [
                ids + [eos_token_id] for ids in encoded["input_ids"]
            ]
        return encoded

    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )

    block_size = seq_len + 1
    step = stride or seq_len
    if step <= 0:
        step = seq_len

    def group_texts(batch: dict) -> dict:
        concatenated = list(chain.from_iterable(batch["input_ids"]))
        total_length = len(concatenated)
        if total_length < block_size:
            return {"input_ids": [], "labels": []}
        input_ids = []
        labels = []
        for i in range(0, total_length - block_size + 1, step):
            window = concatenated[i : i + block_size]
            input_ids.append(window[:-1])
            labels.append(window[1:])
        return {"input_ids": input_ids, "labels": labels}

    dataset = dataset.map(group_texts, batched=True)
    return dataset


def _nonempty_example(example: dict) -> bool:
    return len(example["input_ids"]) > 0


def _apply_scale(config: dict, scale: float) -> dict:
    if scale == 1.0:
        return config
    hidden_dim = max(16, int(config["hidden_dim"] * scale))
    n_heads = max(1, int(config["n_heads"] * scale))
    hidden_dim = max(n_heads, (hidden_dim // n_heads) * n_heads)
    ffn_dim = max(16, int(config["ffn_dim"] * scale))
    config = dict(config)
    config.update(hidden_dim=hidden_dim, n_heads=n_heads, ffn_dim=ffn_dim)
    return config


def _build_model_config(args, vocab_size: int) -> ModelConfig:
    preset = _apply_scale(MODEL_PRESETS[args.preset], args.scale)
    hidden_dim = args.hidden_dim or preset["hidden_dim"]
    n_heads = args.n_heads or preset["n_heads"]
    if hidden_dim % n_heads != 0:
        hidden_dim = (hidden_dim // n_heads) * n_heads
    return ModelConfig(
        vocab_size=args.vocab_size or vocab_size or preset["vocab_size"],
        n_layers=args.n_layers or preset["n_layers"],
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        ffn_dim=args.ffn_dim or preset["ffn_dim"],
        max_seq_len=args.seq_len or preset["max_seq_len"],
        expansion_rate=args.expansion_rate,
        sinkhorn_iters=args.sinkhorn_iters,
        alpha_init=args.alpha_init,
        rmsnorm_eps=args.rmsnorm_eps,
        sinkhorn_eps=args.sinkhorn_eps,
        rope_dim=args.rope_dim or preset["rope_dim"],
        rope_theta=args.rope_theta,
        dropout=args.dropout,
        sdp_kernel=args.sdp_kernel,
        recompute_ratio=args.recompute_ratio,
        recompute_mhc=args.recompute_mhc,
        use_dynamic_h=True,
    )


class _MetricsCallback(TrainerCallback):
    def __init__(
        self,
        metrics_path: Path,
        log_memory: bool,
        tokens_per_step: int,
    ) -> None:
        self.metrics_path = metrics_path
        self.log_memory = log_memory
        self.tokens_per_step = tokens_per_step
        self.start_time: Optional[float] = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        log = dict(logs)
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            tokens_seen = state.global_step * self.tokens_per_step
            log["tokens_seen"] = tokens_seen
            log["tokens_per_s"] = tokens_seen / max(elapsed, 1e-6)
        log_cuda = self.log_memory and torch.cuda.is_available()
        if log_cuda:
            log.update(_get_cuda_memory())
            model = kwargs.get("model")
            optimizer = kwargs.get("optimizer")
            if model is not None:
                param_gb, grad_gb = _param_grad_gb(model)
                log["param_gb"] = param_gb
                log["grad_gb"] = grad_gb
            if optimizer is not None:
                log["optim_state_gb"] = _optimizer_state_gb(optimizer)
        lr = log.get("learning_rate")
        loss = log.get("loss")
        tokens_per_s = log.get("tokens_per_s")
        if loss is not None and lr is not None and tokens_per_s is not None:
            print(
                f"step={state.global_step} loss={loss:.4f} lr={lr:.3e} "
                f"tokens/s={tokens_per_s:.1f}"
            )
        if log_cuda and "cuda_mem_allocated_gb" in log:
            print(
                "cuda_mem="
                f"{log['cuda_mem_allocated_gb']:.1f}G/"
                f"{log['cuda_mem_reserved_gb']:.1f}G "
                f"peak={log['cuda_mem_peak_gb']:.1f}G"
            )
            if "param_gb" in log and "optim_state_gb" in log:
                print(
                    "mem_breakdown="
                    f"params={log.get('param_gb', 0.0):.1f}G "
                    f"grads={log.get('grad_gb', 0.0):.1f}G "
                    f"optim={log.get('optim_state_gb', 0.0):.1f}G"
                )
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(log) + "\n")


class MHCTrainer(Trainer):
    def __init__(
        self,
        *args,
        logits_chunk_size: int,
        loss_dtype: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.logits_chunk_size = logits_chunk_size
        self.loss_dtype = loss_dtype

    def _safe_state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        state = model.state_dict()
        token_key = "token_emb.weight"
        head_key = "lm_head.weight"
        if token_key in state and head_key in state:
            if state[token_key].data_ptr() == state[head_key].data_ptr():
                state[head_key] = state[head_key].clone()
        safe_state: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                tensor = value.detach().cpu()
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                safe_state[key] = tensor
        return safe_state

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ) -> None:
        if not self.is_world_process_zero():
            return
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        state_dict = self._safe_state_dict(self.model)
        from safetensors.torch import save_file

        save_file(
            state_dict,
            os.path.join(output_dir, SAFE_WEIGHTS_NAME),
            metadata={"format": "pt"},
        )
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _resolve_inputs(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        if labels is None:
            labels = inputs.get("label")
        if labels is None and input_ids is not None:
            labels = input_ids.clone()
            if labels.dim() == 2 and labels.size(1) > 1:
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = -100
        if input_ids is None or labels is None:
            raise ValueError("inputs must contain input_ids and labels")
        return input_ids, labels

    def _chunked_ce_loss(
        self, hidden: torch.Tensor, labels: torch.Tensor, lm_head: torch.nn.Module
    ) -> float:
        hidden_flat = hidden.reshape(-1, hidden.size(-1))
        labels_flat = labels.reshape(-1)
        total_tokens = labels_flat.numel()
        if total_tokens == 0:
            return 0.0
        total_loss_val = 0.0
        head_dtype = lm_head.weight.dtype
        grad_accum = self.args.gradient_accumulation_steps
        for start in range(0, total_tokens, self.logits_chunk_size):
            end = min(start + self.logits_chunk_size, total_tokens)
            chunk = hidden_flat[start:end]
            if chunk.dtype != head_dtype:
                chunk = chunk.to(head_dtype)
            logits = lm_head(chunk)
            logits_for_loss = logits.float() if self.loss_dtype == "fp32" else logits
            chunk_loss = F.cross_entropy(
                logits_for_loss,
                labels_flat[start:end],
                reduction="sum",
                ignore_index=-100,
            )
            total_loss_val += chunk_loss.item()
            chunk_loss = chunk_loss / float(total_tokens * grad_accum)
            self.accelerator.backward(chunk_loss, retain_graph=end < total_tokens)
            del logits, chunk_loss
        return total_loss_val / max(total_tokens, 1)

    def _training_step_chunked(
        self, model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        hidden = model(input_ids, return_hidden=True)
        loss_val = self._chunked_ce_loss(hidden, labels, model.lm_head)
        loss_val = loss_val / float(self.args.gradient_accumulation_steps)
        return hidden.new_tensor(loss_val, dtype=torch.float32)

    def _training_step_full(
        self, model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        logits = model(input_ids)
        logits_for_loss = logits.float() if self.loss_dtype == "fp32" else logits
        loss = F.cross_entropy(
            logits_for_loss.view(-1, logits_for_loss.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        loss = loss / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss)
        return loss

    def training_step(self, model, inputs, _num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        input_ids, labels = self._resolve_inputs(inputs)

        with self.compute_loss_context_manager():
            if self.logits_chunk_size > 0:
                loss = self._training_step_chunked(model, input_ids, labels)
            else:
                loss = self._training_step_full(model, input_ids, labels)

        return loss.detach()


def run_training(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="mHC trainer")
    parser.add_argument("--preset", default="3b", choices=sorted(MODEL_PRESETS))
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--dataset", default="Trelis/tiny-shakespeare")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--lr-scale",
        choices=("none", "linear", "sqrt"),
        default="linear",
        help="Scale preset LR by effective_batch / preset_batch when --lr is unset.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--expansion-rate", type=int, default=4)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--alpha-init", type=float, default=0.01)
    parser.add_argument("--rmsnorm-eps", type=float, default=1e-6)
    parser.add_argument("--sinkhorn-eps", type=float, default=1e-6)
    parser.add_argument("--rope-dim", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--sdp-kernel",
        choices=("auto", "flash", "math"),
        default="flash",
        help="Scaled dot-product attention kernel selection.",
    )
    parser.add_argument(
        "--recompute-ratio",
        type=float,
        default=0.0,
        help="Fraction of transformer blocks to checkpoint (0 disables, 1 checkpoints all).",
    )
    parser.add_argument(
        "--recompute-mhc",
        action="store_true",
        help="Recompute residual-stream kernels in backward without rerunning attention/MLP.",
    )
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--ffn-dim", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--loss-dtype",
        choices=("bf16", "fp32"),
        default="fp32",
        help="Compute LM head + loss in this dtype for stability.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument(
        "--log-memory",
        action="store_true",
        help="Log CUDA memory stats alongside loss.",
    )
    parser.add_argument(
        "--logits-chunk-size",
        type=int,
        default=0,
        help="Compute LM head + loss in token chunks to reduce peak memory.",
    )
    parser.add_argument(
        "--fused",
        action="store_true",
        help="Enable fused mHC kernels (experimental).",
    )
    parser.add_argument(
        "--no-fused",
        action="store_true",
        help="Disable fused mHC kernels.",
    )
    parser.add_argument("--runs-root", default=os.getenv("MHC_RUNS_ROOT", "runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream dataset samples instead of fully caching to disk.",
    )
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    args = parser.parse_args(argv)
    if args.logits_chunk_size < 0:
        raise ValueError("logits_chunk_size must be >= 0")

    preset_train = TRAIN_PRESETS[args.preset]
    if args.batch_size is None:
        args.batch_size = preset_train["batch_size"]
    if args.max_steps is None:
        args.max_steps = preset_train["training_steps"]
    effective_batch = args.batch_size * args.grad_accum

    base_lr = args.lr or preset_train["base_lr"]
    lr_scale_factor = 1.0
    if args.lr is None and args.lr_scale != "none":
        lr_scale_factor = effective_batch / preset_train["batch_size"]
        if args.lr_scale == "sqrt":
            lr_scale_factor = math.sqrt(lr_scale_factor)
        base_lr *= lr_scale_factor

    warmup_steps = args.warmup_steps or min(2000, args.max_steps)
    if args.fused and args.no_fused:
        raise ValueError("Use at most one of --fused or --no-fused.")
    use_fused_mhc = args.fused and not args.no_fused
    if use_fused_mhc and mhc_cuda is None:
        print("mhc_cuda not found; falling back to python mHC path.")
        use_fused_mhc = False
    if use_fused_mhc and args.sinkhorn_eps != args.rmsnorm_eps:
        fused_eps = max(args.sinkhorn_eps, args.rmsnorm_eps)
        print(f"Fused mHC uses a single eps; using eps={fused_eps:g}.")
        args.sinkhorn_eps = fused_eps
        args.rmsnorm_eps = fused_eps

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, use_fast=True, token=args.hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        args.dataset,
        split=args.dataset_split,
        streaming=args.streaming,
        token=args.hf_token,
    )
    sample = next(iter(dataset)) if args.streaming else dataset[0]
    text_field = args.text_field or _resolve_text_field(sample)

    model_config = _build_model_config(args, tokenizer.vocab_size)
    model_config.use_fused_mhc = use_fused_mhc
    model = MHCTransformer(model_config)
    if args.dtype == "bf16":
        model = model.to(dtype=torch.bfloat16)

    seq_len = model_config.max_seq_len
    stride = args.stride or seq_len
    dataset = _prepare_hf_dataset(dataset, tokenizer, text_field, seq_len, stride)
    dataset = dataset.filter(_nonempty_example)
    if args.shuffle_buffer:
        if args.streaming:
            dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
        else:
            dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.with_format("torch")

    run_name = args.run_name or f"mhc-{args.preset}-{int(time.time())}"
    out_dir = Path(args.runs_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.json").open("w") as f:
        json.dump(
            {
                "preset": args.preset,
                "preset_batch_size": preset_train["batch_size"],
                "preset_training_steps": preset_train["training_steps"],
                "base_lr": base_lr,
                "lr_scale": args.lr_scale,
                "lr_scale_factor": lr_scale_factor,
                "warmup_steps": warmup_steps,
                "dataset": args.dataset,
                "tokenizer": args.tokenizer,
                "text_field": text_field,
                "model_config": asdict(model_config),
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "effective_batch_size": effective_batch,
                "max_steps": args.max_steps,
                "grad_clip": args.grad_clip,
                "sdp_kernel": args.sdp_kernel,
                "recompute_ratio": args.recompute_ratio,
                "recompute_mhc": args.recompute_mhc,
                "logits_chunk_size": args.logits_chunk_size,
                "use_fused_mhc": use_fused_mhc,
            },
            f,
            indent=2,
        )

    if args.log_memory and torch.cuda.is_available():
        param_gb, _ = _param_grad_gb(model)
        print(f"model_params={param_gb:.2f}G (bf16 storage)")

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        warmup_steps=warmup_steps,
        learning_rate=base_lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.beta1,
        adam_beta2=args.beta2,
        adam_epsilon=args.adam_eps,
        max_grad_norm=args.grad_clip,
        bf16=args.dtype == "bf16",
        logging_steps=args.log_every,
        logging_first_step=True,
        save_steps=args.save_every,
        save_strategy="steps" if args.save_every > 0 else "no",
        logging_strategy="steps",
        report_to=[],
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        run_name=run_name,
        remove_unused_columns=False,
        save_safetensors=True,
    )

    tokens_per_step = args.batch_size * args.grad_accum * seq_len
    metrics_path = out_dir / "metrics.jsonl"

    trainer = MHCTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
        logits_chunk_size=args.logits_chunk_size,
        loss_dtype=args.loss_dtype,
    )
    trainer.add_callback(
        _MetricsCallback(
            metrics_path=metrics_path,
            log_memory=args.log_memory,
            tokens_per_step=tokens_per_step,
        )
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    return run_name, str(out_dir)


if __name__ == "__main__":
    run_training()
