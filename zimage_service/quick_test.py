"""
Quick smoke test for Z-Image-Turbo on a single GPU.

Usage:
    python quick_test.py
    python quick_test.py --prompt "a red fox in snow, photorealistic" --seed 7
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # offline GPU machine
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from diffusers import ZImagePipeline

DEFAULT_MODEL = os.environ.get(
    "ZIMAGE_MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "models" / "Z-Image-Turbo"),
)

DEFAULT_PROMPT = (
    "Photorealistic portrait of an elderly fisherman on a wooden pier at sunrise, "
    "weathered skin, knitted wool cap, soft golden light, shallow depth of field, "
    "85mm lens, highly detailed"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=9, help="9 => 8 DiT forwards (model-card default)")
    ap.add_argument("--out", default="quick_test.png")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU not found - check your PyTorch CUDA build."
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda}")

    t0 = time.perf_counter()
    pipe = ZImagePipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        # Model card uses False (loads the full FP32 transformer, ~24 GB, into system RAM first).
        # On a PC with <= 32 GB RAM set ZIMAGE_LOW_CPU_MEM=1 to load tensor-by-tensor instead.
        low_cpu_mem_usage=os.environ.get("ZIMAGE_LOW_CPU_MEM", "0").lower() in ("1", "true", "yes"),
    )
    pipe.to("cuda")
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # First call includes CUDA kernel warm-up, so time a second call too.
    for label in ("cold", "warm"):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        image = pipe(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=0.0,  # must be 0 for Turbo
            generator=torch.Generator("cuda").manual_seed(args.seed),
        ).images[0]
        torch.cuda.synchronize()
        print(f"{label} generation: {time.perf_counter() - t0:.2f}s")

    image.save(args.out)
    print(f"Saved {args.out} | peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")


if __name__ == "__main__":
    main()
