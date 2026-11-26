"""
STEP 05 - verify the GPU machine is ready BEFORE loading the 6B model.

    python zimage_service/check_env.py
    python zimage_service/check_env.py --model D:\\models\\Z-Image-Turbo

Exit code 0 = all required checks passed.
"""
import argparse
import importlib
import os
import platform
import shutil
import sys
from pathlib import Path

DEFAULT_MODEL = os.environ.get(
    "ZIMAGE_MODEL_PATH", str(Path(__file__).resolve().parent.parent / "models" / "Z-Image-Turbo")
)
REQUIRED_MODEL_FILES = ["model_index.json", "transformer/config.json", "text_encoder/config.json",
                        "vae/config.json", "tokenizer/tokenizer.json", "scheduler/scheduler_config.json"]

failures = []


def check(ok: bool, label: str, detail: str = "", required: bool = True) -> None:
    mark = "OK  " if ok else ("FAIL" if required else "WARN")
    print(f"[{mark}] {label}{(' - ' + detail) if detail else ''}")
    if not ok and required:
        failures.append(label)


def version_of(mod: str):
    try:
        return getattr(importlib.import_module(mod), "__version__", "?")
    except Exception as e:  # noqa: BLE001
        return None if isinstance(e, ImportError) else f"import error: {e!r}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    print(f"Python {platform.python_version()} on {platform.platform()}\n")
    check(sys.version_info[:2] in ((3, 10), (3, 11), (3, 12)), "Python version 3.10-3.12", platform.python_version())

    # --- torch / GPU
    try:
        import torch
        check(True, "torch import", f"{torch.__version__} (CUDA {torch.version.cuda})")
        cuda = torch.cuda.is_available()
        check(cuda, "CUDA available")
        if cuda:
            p = torch.cuda.get_device_properties(0)
            cap = f"sm_{p.major}{p.minor}"
            check(True, "GPU", f"{p.name}, {p.total_memory / 1024**3:.0f} GB, {cap}")
            arch = torch.cuda.get_arch_list()
            check(cap in arch or f"compute_{p.major}{p.minor}" in arch, f"torch build supports {cap}",
                  "RTX 5090 needs a cu128+ torch build" if cap == "sm_120" else ", ".join(arch))
            check(torch.cuda.is_bf16_supported(), "BF16 supported")
            check(p.total_memory / 1024**3 >= 24, "GPU memory >= 24 GB", required=False)
            try:
                x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
                torch.cuda.synchronize()
                check(bool(torch.isfinite(x @ x).all()), "CUDA BF16 matmul test")
                import torch.nn.functional as F
                q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.bfloat16)
                F.scaled_dot_product_attention(q, q, q)
                torch.cuda.synchronize()
                check(True, "SDPA attention test")
            except Exception as e:  # noqa: BLE001
                check(False, "CUDA kernel test", repr(e))
    except Exception as e:  # noqa: BLE001
        check(False, "torch import", repr(e))

    # --- python packages
    print()
    for mod, req in [("transformers", True), ("diffusers", True), ("accelerate", False), ("safetensors", True),
                     ("huggingface_hub", True), ("PIL", True), ("fastapi", True), ("uvicorn", True),
                     ("pydantic", True), ("requests", False)]:
        v = version_of(mod)
        check(v is not None and not str(v).startswith("import error"), f"{mod}", str(v), required=req)

    try:
        from transformers import Qwen3Model  # noqa: F401
        check(True, "transformers has Qwen3Model")
    except Exception as e:  # noqa: BLE001
        check(False, "transformers has Qwen3Model", f"need transformers>=4.51 ({e!r})")
    try:
        from diffusers import ZImagePipeline  # noqa: F401
        check(True, "diffusers has ZImagePipeline")
    except Exception as e:  # noqa: BLE001
        check(False, "diffusers has ZImagePipeline", f"need diffusers>=0.36 ({e!r})")

    # --- model files
    print()
    root = Path(args.model)
    check(root.exists(), "model folder", str(root))
    if root.exists():
        missing = [f for f in REQUIRED_MODEL_FILES if not (root / f).exists()]
        check(not missing, "model config files", "missing: " + ", ".join(missing) if missing else "")
        for comp in ("transformer", "text_encoder", "vae"):
            n = len(list((root / comp).glob("*.safetensors")))
            check(n > 0, f"{comp} weights", f"{n} safetensors file(s)")
        size = sum(f.stat().st_size for f in root.rglob("*.safetensors")) / 1024**3
        check(True, "model size on disk", f"{size:.1f} GB")
        free = shutil.disk_usage(root).free / 1024**3
        check(free >= 5, "free disk >= 5 GB", f"{free:.1f} GB free", required=False)

    print("\n" + ("ALL REQUIRED CHECKS PASSED - run quick_test.py next" if not failures
                  else f"{len(failures)} REQUIRED CHECK(S) FAILED: {failures}"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
