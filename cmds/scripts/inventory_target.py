#!/usr/bin/env python3
"""
STEP 01 - run on each OFFLINE GPU machine (5090 / H200). Standard library only.

Records what is already installed so download_wheels.py fetches compatible versions
and does not re-download / upgrade torch, CUDA or transformers.

    python cmds/scripts/inventory_target.py --target 5090
    python3 cmds/scripts/inventory_target.py --target h200

Run it with EVERY Python you plan to use (e.g. both 3.10 and 3.12).
Output: pkgs/inventory/<target>/py3XX/{env.json, pip_freeze.txt, pip_debug.txt, nvidia_smi.txt}
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TORCH_PROBE = r"""
import json
d = {}
try:
    import torch
    d["torch"] = torch.__version__
    d["torch_cuda"] = torch.version.cuda
    d["cuda_available"] = torch.cuda.is_available()
    d["arch_list"] = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    d["gpus"] = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        d["gpus"].append({"name": p.name, "capability": f"{p.major}.{p.minor}",
                          "mem_gb": round(p.total_memory / 1024**3, 1)})
    d["bf16_supported"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
except Exception as e:
    d["torch_error"] = repr(e)
for mod in ("transformers", "diffusers", "accelerate", "huggingface_hub", "tokenizers", "safetensors"):
    try:
        m = __import__(mod)
        d[mod] = getattr(m, "__version__", "?")
    except Exception as e:
        d[mod] = None
print(json.dumps(d))
"""


def run(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:  # noqa: BLE001
        return -1, "", repr(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="name used for the pkgs folder, e.g. 5090 or h200")
    ap.add_argument("--out", default=str(ROOT / "pkgs" / "inventory"))
    args = ap.parse_args()

    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    out = Path(args.out) / args.target / f"py{pyver.replace('.', '')}"
    out.mkdir(parents=True, exist_ok=True)

    is_linux = sys.platform.startswith("linux")
    glibc = platform.libc_ver()[1] if is_linux else None

    env = {
        "target": args.target,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "python": pyver,
        "python_full": platform.python_version(),
        "python_executable": sys.executable,
        "in_virtualenv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "os": "linux" if is_linux else ("windows" if sys.platform == "win32" else sys.platform),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "glibc": glibc,
        "disk_free_gb": round(shutil.disk_usage(ROOT).free / 1024**3, 1),
    }

    code, pipv, _ = run([sys.executable, "-m", "pip", "--version"])
    env["pip"] = pipv.strip() if code == 0 else None

    code, freeze, err = run([sys.executable, "-m", "pip", "list", "--format=freeze"])
    (out / "pip_freeze.txt").write_text(freeze if code == 0 else f"# pip list failed: {err}\n", encoding="utf-8")

    code, dbg, _ = run([sys.executable, "-m", "pip", "debug", "--verbose"])
    (out / "pip_debug.txt").write_text(dbg, encoding="utf-8")

    code, probe, err = run([sys.executable, "-c", TORCH_PROBE], timeout=300)
    try:
        env["packages"] = json.loads(probe.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        env["packages"] = {"probe_error": err[-2000:]}

    smi = shutil.which("nvidia-smi")
    if smi:
        _, s, _ = run([smi])
        (out / "nvidia_smi.txt").write_text(s, encoding="utf-8")
        _, q, _ = run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        env["nvidia_smi"] = q.strip()

    (out / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    pk = env["packages"]
    print(json.dumps(env, indent=2))
    print(f"\nSaved inventory to: {out}")
    # Quick human-readable warnings
    if pk.get("torch_error") or not pk.get("torch"):
        print("WARNING: torch could not be imported with this Python.")
    elif not pk.get("cuda_available"):
        print("WARNING: torch is installed but CUDA is NOT available.")
    else:
        caps = [g["capability"] for g in pk.get("gpus", [])]
        if "12.0" in caps and "sm_120" not in pk.get("arch_list", []):
            print("WARNING: RTX 5090 (sm_120) is not in this torch build's arch list -> need a cu128+ torch build.")
    if not pk.get("transformers"):
        print("NOTE: transformers not found - it will be added to the download list.")


if __name__ == "__main__":
    main()
