# Offline deployment guide: Z-Image-Turbo on RTX 5090 (Windows) and H200 (Ubuntu)

The GPU machines have **no internet**. Steps marked 🌐 run on the internet machine; steps marked 🖥️ run on the GPU machine.
Every step has its own guide in this folder.

| # | Where | Guide | What happens |
|---|---|---|---|
| 01 | 🖥️ 5090 + H200 | [01_inventory_on_targets.md](01_inventory_on_targets.md) | **Optional, skipped by default.** Records installed versions so downloads can be pinned to match |
| 02 | 🌐 | [02_download_wheels.md](02_download_wheels.md) | Download wheels into `pkgs/5090/` and `pkgs/h200/`, each with `py310`, `py311`, `py312`, `py313` |
| 03 | 🌐 → 🖥️ | [03_transfer.md](03_transfer.md) | What to copy to each machine |
| 04 | 🖥️ | [04_install_offline.md](04_install_offline.md) | Create a venv that reuses the installed torch, then install the wheels offline |
| 05 | 🖥️ | [05_verify_environment.md](05_verify_environment.md) | `check_env.py`: checks the GPU, torch build, packages and model files |
| 06 | 🖥️ | [06_quick_test.md](06_quick_test.md) | Generate the first image |
| 07 | 🖥️ | [07_benchmark.md](07_benchmark.md) | Baseline benchmark (100 runs) |
| 08 | 🖥️ | [08_api_server.md](08_api_server.md) | Run the API for other users, and test it from a client |
| 09 | 🖥️ | [09_free_disk_bf16.md](09_free_disk_bf16.md) | Optional, **not planned**: shrink the model from 31 GB to about 20 GB |
| 10 | - | [10_troubleshooting.md](10_troubleshooting.md) | Common errors and fixes |
| 11 | 🌐 → 🖥️ | [11_snapshot_updates.md](11_snapshot_updates.md) | Send only changed files later (`tools/snapshot_changes.py`) |

## Default path: skip step 01
Assumption: torch (with CUDA), transformers and CUDA are already installed on both GPU machines.
Go straight to **02 → 03 → 04 → 05**. Without an inventory, step 02 runs in *blind mode*:
- downloads the newest compatible wheels for our extra packages, **plus transformers + tokenizers wheels as a fallback**;
- never downloads torch or CUDA.

In step 04, pip keeps everything already installed (torch, transformers, numpy, ...) if it is compatible,
and installs only what is missing into the venv. If the installed transformers conflicts with the new
diffusers/huggingface_hub, pip automatically uses the shipped transformers wheel instead.
`check_env.py` (step 05) then confirms the torch build actually supports the GPU.

Use step 01 only if step 04 fails with a version conflict you can't resolve.

## Folder layout
```
image_llm/
├── zimage_service/      code: quick_test, engine class, API server, client, benchmark, check_env
├── models/Z-Image-Turbo model weights (31 GB, or about 20 GB after step 09)
├── pkgs/
│   ├── requirements/    lists of what gets downloaded (edit these to add packages)
│   ├── targets.json     5090 = windows, h200 = linux
│   ├── inventory/       filled by step 01
│   ├── 5090/py310..py313 filled by step 02
│   └── h200/py310..py313 filled by step 02
└── cmds/                these guides + scripts/
```

## What gets downloaded (torch and CUDA are never downloaded)
diffusers ≥ 0.36, accelerate (no-deps), safetensors, huggingface_hub, pillow, numpy, fastapi, pydantic,
uvicorn, httptools, uvloop (Ubuntu only), requests, psutil, pyyaml, and all their dependencies.
transformers + tokenizers are included only if the target's version is missing or older than 4.51.
Python 3.10, 3.11, 3.12 and 3.13 are all prepared (about 55–65 MB per folder, about 470 MB in total).
