# 07 – Benchmark 🖥️

Follows the benchmark plan in the project todo: 5 warm-up runs, 100 measured runs, fixed prompts and
seeds, CUDA-synchronized timing.

```bash
cd zimage_service
python benchmark.py                                   # baseline: batch 1, 8 steps, SDPA
python benchmark.py --batch-size 2 --runs 50
python benchmark.py --batch-size 4 --runs 25
python benchmark.py --compile                         # torch.compile (Ubuntu; on Windows see troubleshooting)
python benchmark.py --steps 5 --out-dir bench_4step   # 4-step quality experiment
```
On the H200, prefix with `CUDA_VISIBLE_DEVICES=0` if the machine has several GPUs.

Results are written to `bench_results/bench_*.json` and include environment versions, p50/p95/p99
latency, seconds per image, **images/minute** and peak VRAM. Every 10th image is saved to
`bench_results/images/` so you can check quality.

Time needed: about 6–10 min per 100-run benchmark at batch 1.

Bring the JSON files back so the 5090 and H200 numbers can be compared against the 50 images/min
target (≥ 62.5 images/min demonstrated for 20% headroom).
