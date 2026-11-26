# 04 – Install offline 🖥️

Use a **virtual environment with `--system-site-packages`**. It reuses the torch, CUDA and transformers
already installed, but new packages go only into the venv, so the system Python is never modified.
To undo everything, delete `.venv`.

## RTX 5090 – Windows
```bat
cd D:\image_llm
py -3.12 -m venv --system-site-packages .venv
.venv\Scripts\activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python cmds\scripts\install_offline.py --target 5090 --dry-run
python cmds\scripts\install_offline.py --target 5090
```
- Any of Python 3.10 / 3.11 / 3.12 / 3.13 works (e.g. `py -3.10`). The script automatically picks the matching `pkgs\5090\py3XX` folder.
- If torch is in a conda env: `conda activate <env>` first and skip the venv, or create the venv from that env's python.

## H200 – Ubuntu
```bash
cd /opt/image_llm
python3.12 -m venv --system-site-packages .venv
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python cmds/scripts/install_offline.py --target h200 --dry-run
python cmds/scripts/install_offline.py --target h200
```
If `venv` fails with `ensurepip is not available` (Ubuntu without `python3.12-venv`, which can't be installed offline):
```bash
python3.12 -m venv --system-site-packages --without-pip .venv
source .venv/bin/activate
python -m pip --version      # uses the system pip, installs into .venv
```

## What the script does
1. Picks `pkgs/<target>/py3XX` based on the Python version running it.
2. Saves `pip list` to `cmds/logs/*_freeze_before.txt` (for rollback).
3. `pip install --no-index --find-links pkgs/<target>/py3XX -r requirements.txt -c constraints.txt`
4. Includes the shipped transformers wheels in the resolution. The **installed** transformers is kept
   if it's ≥ 4.51 and compatible; otherwise pip switches to the shipped one (inside the venv only).
   `--with-transformers` forces the upgrade.
5. `pip install --no-deps accelerate`
6. `pip check` reports any dependency conflicts.

It never contacts the internet (`--no-index`) and never installs or upgrades torch.
