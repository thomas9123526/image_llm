# 10 – Troubleshooting

### torch does not support the RTX 5090
`check_env.py` shows `FAIL torch build supports sm_120`, or you get `CUDA error: no kernel image is available`.
The 5090 needs **torch ≥ 2.7 built for CUDA 12.8+** (`cu128`) and an NVIDIA driver ≥ 570. You'll need
offline torch wheels (about 3 GB), downloaded on a **Windows** internet machine:
```powershell
py -3.12 -m pip download torch --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple --only-binary=:all: -d pkgs\5090\torch_py312
```
On the 5090: `python -m pip install --no-index --find-links pkgs\5090\torch_py312 torch`
For Ubuntu, download on a Linux machine or in WSL, because pip only fetches the `nvidia-*` CUDA wheels
when the download runs on Linux.

### `No matching distribution found for X` during install_offline
A dependency is missing from the folder, usually an OS-specific one. Add `X` to
`pkgs/requirements/linux_extra.txt` (H200) or `windows_extra.txt` (5090), rerun step 02 for that
target, and copy the folder again.

### `pip check`: `transformers X requires huggingface-hub<..., but you have ...`
The new diffusers needed a newer huggingface_hub than your transformers allows. Fix:
`python cmds/scripts/install_offline.py --target <t> --with-transformers`.
If transformers wheels weren't shipped, rerun step 02 with `--relax huggingface_hub transformers tokenizers`.

### `ImportError: cannot import name 'ZImagePipeline'`
diffusers < 0.36 is being imported. Check
`python -c "import diffusers; print(diffusers.__version__, diffusers.__file__)"` and make sure the venv is activated.

### Loading tries to connect to huggingface.co or hangs
Set `HF_HUB_OFFLINE=1` and use a **local folder path** for the model. The code and start scripts already do this.

### Out of memory (CUDA OOM)
The 5090 (32 GB) runs 1024×1024 at batch 1–2 comfortably. Lower `--batch-size`, close other GPU
programs (check `nvidia-smi`), or set `ZIMAGE_CPU_OFFLOAD=1` (slow).

### `--compile` fails on Windows
torch.compile needs Triton, which isn't officially available on Windows. Skip `--compile` on the 5090
(or install the `triton-windows` wheel offline). It works on the H200 / Ubuntu.

### Model load is very slow, freezes the PC, or runs out of RAM
It's reading 23 GB of FP32 weights into RAM first (model-card setting). On a PC with ≤ 32 GB RAM,
use low-memory loading. It works for quick_test, benchmark and the server:
- Windows: `set ZIMAGE_LOW_CPU_MEM=1`, then rerun
- Ubuntu: `export ZIMAGE_LOW_CPU_MEM=1`, then rerun

Otherwise, Do step 09 (BF16 copy), and keep the model on an SSD.

### Server returns 429 / 503 / 504
- 429: that API key already has `ZIMAGE_MAX_PER_KEY` active jobs
- 503: the queue is full (`ZIMAGE_QUEUE_MAX`)
- 504: a sync request waited longer than `ZIMAGE_SYNC_TIMEOUT`. Use async `/v1/jobs` when the queue is long.

### Roll back
Delete the `.venv` folder. The system Python was never modified. The `pip list` from before every
install is saved in `cmds/logs/`.
