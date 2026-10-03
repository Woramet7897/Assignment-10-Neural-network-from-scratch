"""Benchmark inference speed, latency, FPS, and memory footprint of custom UNet.

Measures:
- Parameter count
- Checkpoint file size on disk (MB)
- Peak inference memory footprint (GPU VRAM / CPU RSS)
- Inference latency per single frame (batch size 1, 64x36)
- Throughput in Frames Per Second (FPS)
Saves results to assets/memory_footprint.json.
"""
import argparse
import json
import os
import time
from pathlib import Path

import psutil
import torch

from model import UNet, count_parameters


def benchmark_model(
    checkpoint_path: Path,
    device_name: str = "auto",
    warmup_runs: int = 100,
    bench_runs: int = 500
) -> dict:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"Benchmarking on device: {device}")

    # File size
    file_size_bytes = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0
    file_size_mb = file_size_bytes / (1024 * 1024)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {"img_w": 64, "img_h": 36, "base_ch": 16})
    base_ch = config.get("base_ch", 16)
    img_w = config.get("img_w", 64)
    img_h = config.get("img_h", 36)

    model = UNet(in_channels=3, num_classes=1, base_ch=base_ch)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_params = count_parameters(model)

    # Prepare dummy input (batch size 1, 64x36)
    dummy_input = torch.randn(1, 3, img_h, img_w, device=device)

    # Memory Tracking Setup
    process = psutil.Process()
    rss_before_mb = process.memory_info().rss / (1024 * 1024)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Latency Timing
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(bench_runs):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time_sec = end_time - start_time
    avg_latency_ms = (total_time_sec / bench_runs) * 1000.0
    fps = bench_runs / total_time_sec

    # Memory Measurements
    rss_after_mb = process.memory_info().rss / (1024 * 1024)
    cpu_rss_delta_mb = max(0.0, rss_after_mb - rss_before_mb)

    if device.type == "cuda":
        peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_gpu_mem_mb = 0.0

    results = {
        "device": str(device),
        "input_resolution": f"{img_w}x{img_h}",
        "batch_size": 1,
        "total_parameters": total_params,
        "parameters_million": round(total_params / 1e6, 4),
        "checkpoint_file_size_mb": round(file_size_mb, 3),
        "peak_gpu_memory_allocated_mb": round(peak_gpu_mem_mb, 2),
        "process_rss_delta_mb": round(cpu_rss_delta_mb, 2),
        "warmup_runs": warmup_runs,
        "benchmark_runs": bench_runs,
        "average_latency_ms": round(avg_latency_ms, 3),
        "throughput_fps": round(fps, 1)
    }

    # Print summary
    print("\n" + "=" * 60)
    print("      INFERENCE & MEMORY FOOTPRINT BENCHMARK")
    print("=" * 60)
    print(f" Device                       : {results['device']}")
    print(f" Input Tensor Resolution      : (1, 3, {img_h}, {img_w}) [W={img_w}, H={img_h}]")
    print(f" Model Parameter Count        : {results['total_parameters']:,} ({results['parameters_million']}M)")
    print(f" Checkpoint Size on Disk      : {results['checkpoint_file_size_mb']} MB")
    if device.type == "cuda":
        print(f" Peak GPU VRAM Allocated     : {results['peak_gpu_memory_allocated_mb']} MB")
    print(f" Process RSS Memory Delta     : {results['process_rss_delta_mb']} MB")
    print(f" Average Latency per Frame    : {results['average_latency_ms']} ms")
    print(f" Inference Throughput         : {results['throughput_fps']} FPS")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference performance and memory footprint.")
    parser.add_argument("--checkpoint", default="checkpoints/unet_lane/best.pt", help="Path to checkpoint")
    parser.add_argument("--device", default="auto", help="Device to benchmark on ('auto', 'cuda', 'cpu')")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=500, help="Number of benchmark iterations")
    parser.add_argument("--output-json", default="assets/memory_footprint.json", help="Output JSON path")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    results = benchmark_model(ckpt_path, device_name=args.device, warmup_runs=args.warmup, bench_runs=args.runs)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results saved to: {out_json}")


if __name__ == "__main__":
    main()
