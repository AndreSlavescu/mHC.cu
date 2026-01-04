#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn


class GPUContextCleaner:
    def __init__(self, device: torch.device = None):
        if device is None:
            device = torch.device("cuda")
        self.device = device

        props = torch.cuda.get_device_properties(device)
        l2_cache_bytes = props.L2_cache_size
        self.flush_size = max(l2_cache_bytes * 2, 64 * 1024 * 1024)

        self.flush_tensor = torch.empty(
            self.flush_size // 4, dtype=torch.float32, device=device
        )

    def clear(self):
        torch.cuda.synchronize(self.device)
        self.flush_tensor.fill_(1.0)
        torch.cuda.synchronize(self.device)


class NaiveMHCLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-5,
        alpha_init: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

        self.rmsnorm_weight = nn.Parameter(torch.ones(hidden_dim, dtype=torch.float32))
        self.H_pre = nn.Parameter(torch.zeros(expansion_rate, dtype=torch.float32))
        self.H_post = nn.Parameter(torch.zeros(expansion_rate, dtype=torch.float32))
        H_res_init = alpha_init * torch.randn(expansion_rate, expansion_rate)
        self.H_res = nn.Parameter(H_res_init.float())

    def sinkhorn_knopp(self, M: torch.Tensor) -> torch.Tensor:
        M = torch.exp(M)
        for _ in range(self.sinkhorn_iters):
            M = M / (M.sum(dim=1, keepdim=True) + self.eps)
            M = M / (M.sum(dim=0, keepdim=True) + self.eps)
        return M

    def rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.rmsnorm_weight

    def forward(self, x_expanded: torch.Tensor) -> torch.Tensor:
        B, n, C = x_expanded.shape

        H_pre_activated = torch.sigmoid(self.H_pre)
        H_post_activated = 2.0 * torch.sigmoid(self.H_post)
        M = self.sinkhorn_knopp(self.H_res)

        x_agg = torch.einsum("i,bic->bc", H_pre_activated, x_expanded)
        x_normed = self.rmsnorm(x_agg)
        y_dist = H_post_activated.view(1, n, 1) * x_normed.view(B, 1, C)
        x_mixed = torch.einsum("ij,bjc->bic", M, x_expanded)

        return x_mixed + y_dist


class NaiveMHCLayerDynamic(nn.Module):
    """
    Dynamic H computation as per the paper (Equations: 5-10):
    1. Flatten x -> RMSNorm -> Linear projections to get tilde_H values
    2. H_pre = sigmoid(tilde_H_pre), H_post = 2*sigmoid(tilde_H_post)
    3. M = Sinkhorn-Knopp(tilde_H_res)
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-5,
        alpha_init: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        n = expansion_rate
        C = hidden_dim

        self.rmsnorm_weight = nn.Parameter(torch.ones(hidden_dim, dtype=torch.float32))

        self.phi_pre = nn.Parameter(torch.randn(n * C, n) * 0.02)
        self.phi_post = nn.Parameter(torch.randn(n * C, n) * 0.02)
        self.phi_res = nn.Parameter(torch.randn(n * C, n * n) * 0.02)

        self.b_pre = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.zeros(n, n))

        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init))

    def sinkhorn_knopp(self, M: torch.Tensor) -> torch.Tensor:
        M = torch.exp(M)
        for _ in range(self.sinkhorn_iters):
            M = M / (M.sum(dim=-1, keepdim=True) + self.eps)
            M = M / (M.sum(dim=-2, keepdim=True) + self.eps)
        return M

    def rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.rmsnorm_weight

    def forward(self, x_expanded: torch.Tensor) -> torch.Tensor:
        B, n, C = x_expanded.shape

        x_flat = x_expanded.reshape(B, n * C)
        rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + self.eps)
        x_norm = x_flat / rms

        tilde_H_pre = self.alpha_pre * (x_norm @ self.phi_pre) + self.b_pre
        tilde_H_post = self.alpha_post * (x_norm @ self.phi_post) + self.b_post
        tilde_H_res = (
            self.alpha_res * (x_norm @ self.phi_res).reshape(B, n, n) + self.b_res
        )

        H_pre = torch.sigmoid(tilde_H_pre)
        H_post = 2.0 * torch.sigmoid(tilde_H_post)
        M = self.sinkhorn_knopp(tilde_H_res)

        x_agg = torch.einsum("bi,bic->bc", H_pre, x_expanded)
        x_normed = self.rmsnorm(x_agg)
        y_dist = H_post.unsqueeze(-1) * x_normed.unsqueeze(1)
        x_mixed = torch.einsum("bij,bjc->bic", M, x_expanded)

        return x_mixed + y_dist


def benchmark_forward(layer, B, n, C, device, cleaner, warmup=10, runs=100):
    for _ in range(warmup):
        x = torch.randn(B, n, C, device=device, dtype=torch.float32)
        _ = layer(x)
        del x
    torch.cuda.synchronize()

    torch.cuda.empty_cache()

    times = []
    for _ in range(runs):
        x = torch.randn(B, n, C, device=device, dtype=torch.float32)

        cleaner.clear()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = layer(x)
        end_event.record()

        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

        del x

    times = sorted(times)
    trim = runs // 10
    if trim > 0:
        times = times[trim:-trim]

    avg_ms = sum(times) / len(times)
    return avg_ms


def benchmark_backward(layer, B, n, C, device, cleaner, warmup=10, runs=100):
    for _ in range(warmup):
        x = torch.randn(B, n, C, device=device, dtype=torch.float32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        del x, out, loss
    torch.cuda.synchronize()

    for p in layer.parameters():
        if p.grad is not None:
            p.grad.zero_()

    torch.cuda.empty_cache()

    times = []
    for _ in range(runs):
        x = torch.randn(B, n, C, device=device, dtype=torch.float32, requires_grad=True)

        cleaner.clear()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        out = layer(x)
        loss = out.sum()
        loss.backward()
        end_event.record()

        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

        for p in layer.parameters():
            if p.grad is not None:
                p.grad.zero_()

        del x, out, loss

    times = sorted(times)
    trim = runs // 10
    if trim > 0:
        times = times[trim:-trim]

    avg_ms = sum(times) / len(times)
    return avg_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark mHC Layer")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden", type=int, default=1280, help="Hidden dimension")
    parser.add_argument("--expansion", type=int, default=4, help="Expansion rate (n)")
    parser.add_argument(
        "--sinkhorn-iters", type=int, default=20, help="Sinkhorn iterations"
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Benchmark runs")
    parser.add_argument(
        "--backward", action="store_true", help="Benchmark backward pass too"
    )
    parser.add_argument(
        "--all-configs", action="store_true", help="Run all standard configs"
    )
    args = parser.parse_args()

    device = torch.device("cuda")

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cleaner = GPUContextCleaner(device)

    from mhc import MHCLayer

    if args.all_configs:
        configs = [
            (320, 1280, 4),
            (512, 1920, 4),
            (1280, 2560, 4),
            (2560, 1280, 4),
        ]
    else:
        configs = [(args.batch, args.hidden, args.expansion)]

    print("mHC Layer Python Benchmark")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Sinkhorn iterations: {args.sinkhorn_iters}")
    print(f"Warmup: {args.warmup}, Runs: {args.runs}")
    print()

    print("Static H Path (shared H across batch)")
    print("-" * 80)
    print(
        f"{'Batch':>6} {'Hidden':>6} {'n':>4} {'Fused (ms)':>12} {'Naive (ms)':>12} "
        f"{'Speedup':>10}"
    )
    print("-" * 80)

    for B, C, n in configs:
        fused_layer = MHCLayer(
            hidden_dim=C,
            expansion_rate=n,
            sinkhorn_iters=args.sinkhorn_iters,
            use_dynamic_h=False,
        ).to(device)

        naive_layer = NaiveMHCLayer(
            hidden_dim=C,
            expansion_rate=n,
            sinkhorn_iters=args.sinkhorn_iters,
        ).to(device)

        fused_time = benchmark_forward(
            fused_layer, B, n, C, device, cleaner, args.warmup, args.runs
        )
        naive_time = benchmark_forward(
            naive_layer, B, n, C, device, cleaner, args.warmup, args.runs
        )

        speedup = naive_time / fused_time

        print(
            f"{B:>6} {C:>6} {n:>4} {fused_time:>12.3f} {naive_time:>12.3f} "
            f"{speedup:>9.2f}x"
        )

        del fused_layer, naive_layer
        torch.cuda.empty_cache()

    print()
    print(
        "Dynamic H Path (per-batch H values which is how the paper is implementing mHC layer)"
    )
    print("-" * 80)
    print(
        f"{'Batch':>6} {'Hidden':>6} {'n':>4} {'Fused (ms)':>12} {'Naive (ms)':>12} "
        f"{'Speedup':>10}"
    )
    print("-" * 80)

    for B, C, n in configs:
        fused_layer = MHCLayer(
            hidden_dim=C,
            expansion_rate=n,
            sinkhorn_iters=args.sinkhorn_iters,
            use_dynamic_h=True,
        ).to(device)

        naive_layer = NaiveMHCLayerDynamic(
            hidden_dim=C,
            expansion_rate=n,
            sinkhorn_iters=args.sinkhorn_iters,
        ).to(device)

        fused_time = benchmark_forward(
            fused_layer, B, n, C, device, cleaner, args.warmup, args.runs
        )

        naive_time = benchmark_forward(
            naive_layer, B, n, C, device, cleaner, args.warmup, args.runs
        )

        speedup = naive_time / fused_time

        print(
            f"{B:>6} {C:>6} {n:>4} {fused_time:>12.3f} {naive_time:>12.3f} "
            f"{speedup:>9.2f}x"
        )

        del fused_layer, naive_layer
        torch.cuda.empty_cache()

    if args.backward:
        print()
        print("Backward Pass Static H (forward + backward)")
        print("-" * 80)
        print(
            f"{'Batch':>6} {'Hidden':>6} {'n':>4} {'Fused (ms)':>12} {'Naive (ms)':>12} "
            f"{'Speedup':>10}"
        )
        print("-" * 80)

        for B, C, n in configs:
            fused_layer = MHCLayer(
                hidden_dim=C,
                expansion_rate=n,
                sinkhorn_iters=args.sinkhorn_iters,
                use_dynamic_h=False,
            ).to(device)

            naive_layer = NaiveMHCLayer(
                hidden_dim=C,
                expansion_rate=n,
                sinkhorn_iters=args.sinkhorn_iters,
            ).to(device)

            naive_layer.H_pre.data = fused_layer.H_pre.data.clone()
            naive_layer.H_post.data = fused_layer.H_post.data.clone()
            naive_layer.H_res.data = fused_layer.H_res.data.clone()
            naive_layer.rmsnorm_weight.data = (
                fused_layer.rmsnorm_weight.data.float().clone()
            )

            fused_time = benchmark_backward(
                fused_layer, B, n, C, device, cleaner, args.warmup, args.runs
            )
            naive_time = benchmark_backward(
                naive_layer, B, n, C, device, cleaner, args.warmup, args.runs
            )

            speedup = naive_time / fused_time

            print(
                f"{B:>6} {C:>6} {n:>4} {fused_time:>12.3f} {naive_time:>12.3f} "
                f"{speedup:>9.2f}x"
            )

            del fused_layer, naive_layer
            torch.cuda.empty_cache()

        print()
        print(
            "Backward Pass Dynamic H Path (forward + backward) (this is how the paper is implementing mHC layer)"
        )
        print("-" * 80)
        print(
            f"{'Batch':>6} {'Hidden':>6} {'n':>4} {'Fused (ms)':>12} {'Naive (ms)':>12} "
            f"{'Speedup':>10}"
        )
        print("-" * 80)

        for B, C, n in configs:
            fused_layer = MHCLayer(
                hidden_dim=C,
                expansion_rate=n,
                sinkhorn_iters=args.sinkhorn_iters,
                use_dynamic_h=True,
            ).to(device)

            naive_layer = NaiveMHCLayerDynamic(
                hidden_dim=C,
                expansion_rate=n,
                sinkhorn_iters=args.sinkhorn_iters,
            ).to(device)

            fused_time = benchmark_backward(
                fused_layer, B, n, C, device, cleaner, args.warmup, args.runs
            )

            naive_time = benchmark_backward(
                naive_layer, B, n, C, device, cleaner, args.warmup, args.runs
            )

            speedup = naive_time / fused_time

            print(
                f"{B:>6} {C:>6} {n:>4} {fused_time:>12.3f} {naive_time:>12.3f} "
                f"{speedup:>9.2f}x"
            )

            del fused_layer, naive_layer
            torch.cuda.empty_cache()

    print()


if __name__ == "__main__":
    main()
