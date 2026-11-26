# 02 – Download the wheels 🌐

Run on the internet machine (this Windows PC). It needs local Python **3.10 and 3.12**; both are already
installed here (`py -3.10`, `py -3.12`). The script uses the matching interpreter for each target
version, so dependencies that vary by Python version resolve correctly.

```powershell
cd D:\hkc\image_llm
py -3.10 -m pip install -U pip
py -3.12 -m pip install -U pip

python cmds\scripts\download_wheels.py --clean
```
This builds all 4 folders. To build just some:
```powershell
python cmds\scripts\download_wheels.py --targets 5090 --pythons 3.12
python cmds\scripts\download_wheels.py --targets h200
```

With no `pkgs/inventory/` (step 01 skipped), every folder is built in **blind mode**: you'll see
`mode=blind` in the output and `"mode": "blind"` in `manifest.json`. That's expected.

## What it does, per target and Python version
1. Reads `pkgs/inventory/<target>/py3XX/` if it exists (blind mode: no pins, and transformers wheels are added as a fallback). Installed versions become **constraints**
   (pins), and torch, nvidia-* and triton are always excluded.
2. `pip download --only-binary=:all:` with `--platform win_amd64` (5090) or `--platform manylinux_2_xx_x86_64`
   (H200) and `--python-version 3.X`, for `pkgs/requirements/common.txt` (+ `linux_extra.txt` for the H200).
3. Downloads `accelerate` with `--no-deps` so pip doesn't fetch torch.
4. **Offline check:** a `pip install --dry-run --no-index` from the folder alone. It must print
   `offline resolution check: OK`.
5. Writes `requirements*.txt`, `constraints.txt`, `manifest.json` and `download.log` into the folder.
   The install script needs these files, so keep them.

Expected output at the end:
```
==== SUMMARY ====
  5090/py3.10      OK
  5090/py3.12      OK
  h200/py3.10      OK
  h200/py3.12      OK
```

## If something fails
- `ResolutionImpossible ... huggingface-hub==0.xx` → the newest diffusers needs a newer version of a
  package pinned on the target. Let it upgrade:
  `python cmds\scripts\download_wheels.py --targets h200 --relax huggingface_hub`
  If transformers then complains on the target, also ship transformers: add
  `--relax huggingface_hub transformers tokenizers` and install with `--with-transformers` in step 04.
- `No matching distribution found for X` → X has no wheel for that platform or Python version. Check `download.log`.
- Adding a package: add a line to `pkgs/requirements/common.txt` and rerun.

## Note on H200 wheels downloaded from Windows
pip evaluates OS-specific dependency markers for the **download** machine. Linux-only dependencies are
therefore listed by hand in `pkgs/requirements/linux_extra.txt` (currently `uvloop`). If step 04 on
Ubuntu reports a missing package, add it there and rerun with `--targets h200`.
