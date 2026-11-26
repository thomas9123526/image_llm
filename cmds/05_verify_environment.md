# 05 – Verify the environment 🖥️

With the venv activated:
```bat
python zimage_service\check_env.py
```
```bash
python zimage_service/check_env.py
```
If the model is somewhere else, pass `--model D:\models\Z-Image-Turbo` or set `ZIMAGE_MODEL_PATH`.

It checks:
- Python version, torch + CUDA, GPU name and memory
- **the torch build supports this GPU** (5090 = `sm_120`, H200 = `sm_90`)
- BF16 support, plus a real BF16 matmul and SDPA attention kernel on the GPU
- transformers has `Qwen3Model`, and diffusers has `ZImagePipeline`
- model config and weight files, total size and free disk

Then test the API server logic without loading the model (about 15 s, uses a CPU stub instead of the GPU):
```bash
python zimage_service/tests/test_api_stub.py      # expect: 16/16 passed
```

`check_env.py` must end with `ALL REQUIRED CHECKS PASSED`. Each `FAIL` line names what's wrong; see [10_troubleshooting.md](10_troubleshooting.md).
