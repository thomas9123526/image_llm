# Z-Image-Turbo service (RTX 5090 / H200)

> **Offline machines:** follow the step-by-step guides in [../cmds/README.md](../cmds/README.md) instead of
> section 1 below (wheels are collected in `../pkgs/`).

| File | Purpose |
|---|---|
| `quick_test.py` | Smoke test: load the model, generate one image, print timing + VRAM |
| `zimage_engine.py` | `ZImageEngine` class: load once, thread-safe `generate()`, batching, seeds, stats |
| `server.py` | FastAPI server for other users: sync + async (job) endpoints, API keys, bounded queue |
| `client.py` | Python client (`ZImageClient`) + CLI example |
| `benchmark.py` | Benchmark per the todo plan: warm-up, 100 runs, p50/p95/p99, img/min, peak VRAM → JSON |
| `convert_to_bf16.py` | Optional: re-save the FP32 transformer (23 GB) as BF16 (12 GB) to free disk |

## 1. Set up the 5090 machine

```powershell
python -m venv .venv
.venv\Scripts\activate
# RTX 5090 (Blackwell) needs a CUDA 12.8+ PyTorch build
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"   # should list sm_120
```

The model path defaults to `..\models\Z-Image-Turbo`. Override with `set ZIMAGE_MODEL_PATH=D:\path\to\Z-Image-Turbo`.

## 2. Quick test

```powershell
python quick_test.py
python quick_test.py --prompt "a red fox in fresh snow, photorealistic" --seed 7 --out fox.png
```

## 3. Use as a class

```python
from zimage_engine import ZImageEngine, EngineConfig

engine = ZImageEngine(EngineConfig())   # or EngineConfig.from_env()
engine.load()
engine.warmup()

r = engine.generate("a photorealistic cat on a windowsill", seed=1)
r.images[0].save("cat.png")

r = engine.generate(["a red car", "a blue boat"], seed=[10, 11])   # batch of 2
print(r.elapsed_s, r.seeds)
```

## 4. Run the API server

```powershell
set ZIMAGE_API_KEYS=key-alice,key-bob
python server.py            # listens on 0.0.0.0:8000, ONE process only
```

Open Windows Firewall for port 8000 if other machines need access. Interactive docs: `http://<server-ip>:8000/docs`.

**Sync** (wait for the image):
```bash
curl -X POST http://SERVER:8000/v1/images/generate -H "X-API-Key: key-alice" \
     -H "Content-Type: application/json" \
     -d '{"prompt":"a lighthouse at dusk, photorealistic","response_format":"png","seed":42}' \
     --output out.png
```

**Async** (get a job id, poll later):
```bash
curl -X POST http://SERVER:8000/v1/jobs -H "X-API-Key: key-alice" -H "Content-Type: application/json" \
     -d '{"prompt":"a lighthouse at dusk","num_images":2}'
curl http://SERVER:8000/v1/jobs/<job_id> -H "X-API-Key: key-alice"
curl http://SERVER:8000/v1/jobs/<job_id>/images/0 -H "X-API-Key: key-alice" --output 0.png
```

**Python:** `python client.py --url http://SERVER:8000 --key key-alice --mode async -n 2`

Request fields: `prompt` (required), `width`/`height` (256–2048, multiple of 16, default 1024), `steps` (default 9 = 8 DiT forwards), `seed`, `num_images` (1–4), `response_format` (`b64_json` | `png`, sync only).

Error codes: `401` bad key · `422` bad input · `429` too many active jobs for your key · `503` queue full · `504` sync wait timed out.

## 5. Benchmark

```powershell
python benchmark.py                          # baseline: bs1, 8 steps, SDPA
python benchmark.py --batch-size 2 --runs 50
python benchmark.py --compile
python benchmark.py --steps 5                # 4-step experiment (check quality!)
```
Reports go to `bench_results\*.json`. Only every 10th image is saved to spare disk.

## 6. Free ~11 GB of disk (optional)

```powershell
python convert_to_bf16.py --swap   # needs ~24 GB RAM, ~13 GB free disk temporarily
python quick_test.py               # verify
rmdir /s ..\models\Z-Image-Turbo\transformer_fp32_backup
```

## Design notes / what is intentionally left for later

- One model process per GPU, one generation at a time (single queue consumer) — matches the todo's initial operating settings.
- Jobs are kept **in memory**; async results are deleted after `ZIMAGE_JOB_TTL` (1 h). A server restart loses queued jobs.
- Next steps from the todo (section D/F): Redis Streams queue + separate GPU worker processes (needed for the H200 worker), S3 storage, Prometheus metrics, per-key rate limiting, HTTPS via Nginx/Traefik.
