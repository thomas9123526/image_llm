"""
OPTIONAL disk saver: the downloaded transformer is stored in FP32 (~23 GB).
The pipeline casts it to BF16 at load time anyway, so a BF16 copy (~12 GB) gives
identical inference results and saves ~11 GB of disk (and faster loading).

    python convert_to_bf16.py                    # writes transformer_bf16/ next to transformer/
    python convert_to_bf16.py --swap             # then also replaces transformer/ (keeps a backup until you delete it)

After verifying quick_test.py works with the new weights, delete the backup folder
`transformer_fp32_backup` to reclaim the space.

Needs ~24 GB free system RAM (the FP32 model is loaded on CPU). No GPU needed.
"""
import argparse
import os
import shutil
from pathlib import Path

import torch
from diffusers import ZImageTransformer2DModel

DEFAULT_MODEL = os.environ.get(
    "ZIMAGE_MODEL_PATH", str(Path(__file__).resolve().parent.parent / "models" / "Z-Image-Turbo")
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--swap", action="store_true", help="move bf16 copy into place of transformer/")
    args = ap.parse_args()

    root = Path(args.model)
    src, dst = root / "transformer", root / "transformer_bf16"
    need_gb = 13
    free_gb = shutil.disk_usage(root).free / 1024**3
    if free_gb < need_gb:
        raise SystemExit(f"Need ~{need_gb} GB free on disk, only {free_gb:.1f} GB available")

    print("Loading FP32 transformer on CPU (this takes a few minutes)...")
    model = ZImageTransformer2DModel.from_pretrained(src, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
    print(f"Saving BF16 copy to {dst} ...")
    model.save_pretrained(dst, safe_serialization=True, max_shard_size="5GB")
    del model

    if args.swap:
        backup = root / "transformer_fp32_backup"
        src.rename(backup)
        dst.rename(src)
        print(f"Swapped. Test with quick_test.py, then delete {backup} to free ~11 GB.")
    else:
        print(f"Done. Point to it by re-running with --swap, or inspect {dst}.")


if __name__ == "__main__":
    main()
