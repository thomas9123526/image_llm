"""
Benchmark Z-Image-Turbo following the plan in image_generation_production_todo.md:
warm-up runs, N measured runs, fixed prompts + seeds, CUDA-synchronized timing,
p50/p95/p99 latency, images/min and peak VRAM. Writes a JSON report.

    python benchmark.py                       # 5 warm-up + 100 runs, batch 1
    python benchmark.py --batch-size 2 --runs 50
    python benchmark.py --attention flash     # needs flash-attn installed
    python benchmark.py --compile
    python benchmark.py --steps 5             # 4-step quality/speed experiment
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from zimage_engine import EngineConfig, ZImageEngine

PROMPTS = [
    "Photorealistic portrait of a young woman with freckles, natural window light, 85mm, shallow depth of field",
    "A busy Tokyo crosswalk at night in the rain, neon reflections, cinematic, photorealistic",
    "Close-up of a ripe strawberry with water droplets on a dark background, macro photography",
    "An old stone farmhouse in Tuscany at golden hour, cypress trees, rolling hills",
    "A golden retriever puppy running through autumn leaves in a park, motion blur background",
    "Modern minimalist living room interior, large windows, oak floor, soft morning light",
    "Elderly man reading a newspaper in a Paris cafe, film photography look, 35mm",
    "Aerial drone shot of a turquoise lagoon with a small sailboat, high detail",
]


def pct(values, p):
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def env_info() -> dict:
    import diffusers, transformers

    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--attention", default=None, help="flash | _flash_3 (default SDPA)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--save-every", type=int, default=10, help="save every Nth image (0 = none; saves disk)")
    ap.add_argument("--out-dir", default="bench_results")
    args = ap.parse_args()

    cfg = EngineConfig.from_env()
    cfg.attention_backend = args.attention
    cfg.compile_transformer = args.compile
    cfg.max_batch_size = max(cfg.max_batch_size, args.batch_size)
    engine = ZImageEngine(cfg)

    t0 = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - t0

    gen_kw = dict(width=args.width, height=args.height, steps=args.steps)
    for i in range(args.warmup):
        r = engine.generate(PROMPTS[i % len(PROMPTS)], seed=i, num_images=args.batch_size, **gen_kw)
        print(f"warm-up {i + 1}/{args.warmup}: {r.elapsed_s:.2f}s")

    torch.cuda.reset_peak_memory_stats()
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    latencies, failures = [], 0
    wall0 = time.perf_counter()
    for i in range(args.runs):
        prompts = [PROMPTS[(i * args.batch_size + j) % len(PROMPTS)] for j in range(args.batch_size)]
        seeds = [1000 + i * args.batch_size + j for j in range(args.batch_size)]
        try:
            r = engine.generate(prompts, seed=seeds, **gen_kw)
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"run {i}: FAILED {e}")
            continue
        latencies.append(r.elapsed_s)
        if args.save_every and i % args.save_every == 0:
            r.images[0].save(img_dir / f"run{i:04d}_seed{seeds[0]}.png")
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{args.runs}  last={r.elapsed_s:.2f}s  p50={statistics.median(latencies):.2f}s")
    wall = time.perf_counter() - wall0

    images_done = len(latencies) * args.batch_size
    report = {
        "env": env_info(),
        "settings": {**vars(args)},
        "load_seconds": round(load_s, 1),
        "runs_ok": len(latencies),
        "failures": failures,
        "latency_per_call_s": {
            "mean": round(statistics.mean(latencies), 3),
            "p50": round(pct(latencies, 50), 3),
            "p95": round(pct(latencies, 95), 3),
            "p99": round(pct(latencies, 99), 3),
            "min": round(min(latencies), 3),
            "max": round(max(latencies), 3),
        },
        "seconds_per_image": round(sum(latencies) / images_done, 3),
        "images_per_minute": round(images_done / wall * 60, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
    }
    name = f"bench_bs{args.batch_size}_steps{args.steps}_{args.attention or 'sdpa'}{'_compile' if args.compile else ''}.json"
    (out_dir / name).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Report written to {out_dir / name}")


if __name__ == "__main__":
    main()
