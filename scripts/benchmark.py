"""
Throughput benchmark for DSTT-T models.

Measures tokens/second for forward and forward+backward passes.

Usage:
    python scripts/benchmark.py --config tiny --seq_len 256 --warmup 10 --iters 50
"""

import argparse
import time
import torch
from dstt import DSTTConfig, DSTTTransformer
from dstt.utils import count_parameters, format_params


def benchmark(config_name: str, seq_len: int, batch_size: int, warmup: int, iters: int):
    configs = {
        "tiny": DSTTConfig.tiny,
        "base": DSTTConfig.base,
        "large": DSTTConfig.large,
    }
    config = configs[config_name]()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DSTTTransformer(config).to(device)
    model.train()

    print(f"Config: {config_name} | Device: {device}")
    print(f"Params: {format_params(count_parameters(model))}")
    print(f"Batch: {batch_size} | Seq: {seq_len}")
    print("-" * 50)

    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(warmup):
        out = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Forward benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            out = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    fwd_time = (time.perf_counter() - t0) / iters
    fwd_tps = batch_size * seq_len / fwd_time

    # Forward+backward benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = model(x)
        loss = out.sum()
        loss.backward()
        model.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()
    fwdbwd_time = (time.perf_counter() - t0) / iters
    fwdbwd_tps = batch_size * seq_len / fwdbwd_time

    print(f"Forward:          {fwd_time*1000:.1f} ms | {fwd_tps:.0f} tok/s")
    print(f"Forward+Backward: {fwdbwd_time*1000:.1f} ms | {fwdbwd_tps:.0f} tok/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tiny")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    benchmark(args.config, args.seq_len, args.batch_size, args.warmup, args.iters)
