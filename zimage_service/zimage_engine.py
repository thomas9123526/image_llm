"""
Reusable Z-Image-Turbo engine.

Loads the pipeline once and keeps it resident on the GPU. Thread-safe: calls are
serialized with a lock so exactly one generation runs on the GPU at a time.

Example:
    from zimage_engine import ZImageEngine, EngineConfig

    engine = ZImageEngine(EngineConfig(model_path="../models/Z-Image-Turbo"))
    engine.load()
    engine.warmup()
    result = engine.generate("a photorealistic cat on a windowsill", seed=1)
    result.images[0].save("cat.png")
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from PIL import Image

log = logging.getLogger("zimage.engine")

# GPU machines are offline: never try to reach huggingface.co (set to 0 to allow).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MIN_SIDE, MAX_SIDE, SIDE_MULTIPLE = 256, 2048, 16  # VAE /8 x patch 2 => multiples of 16
MAX_SEED = 2**32 - 1


def _default_model_path() -> str:
    return os.environ.get(
        "ZIMAGE_MODEL_PATH",
        str(Path(__file__).resolve().parent.parent / "models" / "Z-Image-Turbo"),
    )


@dataclass
class EngineConfig:
    model_path: str = field(default_factory=_default_model_path)
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    attention_backend: Optional[str] = None  # None=SDPA, "flash", "_flash_3" (needs flash-attn installed)
    compile_transformer: bool = False        # torch.compile: faster steady state, slow first call
    cpu_offload: bool = False                # only for small GPUs; not needed on a 5090
    low_cpu_mem_usage: bool = False          # model card uses False
    default_steps: int = 9                   # 9 => 8 DiT forwards
    max_batch_size: int = 4

    @classmethod
    def from_env(cls) -> "EngineConfig":
        def b(name: str, default: bool) -> bool:
            return os.environ.get(name, str(default)).lower() in ("1", "true", "yes")

        return cls(
            model_path=_default_model_path(),
            device=os.environ.get("ZIMAGE_DEVICE", "cuda"),
            attention_backend=os.environ.get("ZIMAGE_ATTENTION") or None,
            compile_transformer=b("ZIMAGE_COMPILE", False),
            cpu_offload=b("ZIMAGE_CPU_OFFLOAD", False),
            low_cpu_mem_usage=b("ZIMAGE_LOW_CPU_MEM", False),
            default_steps=int(os.environ.get("ZIMAGE_STEPS", 9)),
            max_batch_size=int(os.environ.get("ZIMAGE_MAX_BATCH", 4)),
        )


@dataclass
class GenerationResult:
    images: List[Image.Image]
    seeds: List[int]
    width: int
    height: int
    steps: int
    elapsed_s: float


class ZImageEngine:
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.pipe = None
        self._lock = threading.Lock()
        self.loaded_at: Optional[float] = None
        self.total_images = 0

    # ------------------------------------------------------------------ lifecycle
    @property
    def is_loaded(self) -> bool:
        return self.pipe is not None

    def load(self) -> None:
        from diffusers import ZImagePipeline  # imported lazily so the module imports fast

        cfg = self.config
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available - install a CUDA build of PyTorch (cu128+ for RTX 5090).")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        t0 = time.perf_counter()
        log.info("Loading Z-Image-Turbo from %s", cfg.model_path)
        pipe = ZImagePipeline.from_pretrained(
            cfg.model_path, torch_dtype=cfg.dtype, low_cpu_mem_usage=cfg.low_cpu_mem_usage
        )
        if cfg.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(cfg.device)

        if cfg.attention_backend:
            pipe.transformer.set_attention_backend(cfg.attention_backend)
            log.info("Attention backend: %s", cfg.attention_backend)
        if cfg.compile_transformer:
            pipe.transformer.compile()
            log.info("torch.compile enabled on transformer (first call will be slow)")

        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.loaded_at = time.time()
        log.info("Model loaded in %.1fs", time.perf_counter() - t0)

    def warmup(self, runs: int = 2, width: int = 1024, height: int = 1024) -> None:
        for i in range(runs):
            r = self.generate("warm-up: a photo of a mountain lake", width=width, height=height, seed=i)
            log.info("Warm-up %d/%d: %.2fs", i + 1, runs, r.elapsed_s)
        self.total_images -= runs

    def unload(self) -> None:
        with self._lock:
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------ inference
    @staticmethod
    def validate_size(width: int, height: int) -> None:
        for name, v in (("width", width), ("height", height)):
            if not (MIN_SIDE <= v <= MAX_SIDE) or v % SIDE_MULTIPLE:
                raise ValueError(f"{name}={v} must be in [{MIN_SIDE}, {MAX_SIDE}] and a multiple of {SIDE_MULTIPLE}")

    def generate(
        self,
        prompt: Union[str, Sequence[str]],
        *,
        width: int = 1024,
        height: int = 1024,
        steps: Optional[int] = None,
        seed: Optional[Union[int, Sequence[int]]] = None,
        num_images: int = 1,
    ) -> GenerationResult:
        """Generate one or more images.

        prompt: a string (repeated num_images times) or a list of prompts (one image each).
        seed:   None => random; int => base seed (seed, seed+1, ...); list => one per image.
        """
        if not self.is_loaded:
            raise RuntimeError("Engine not loaded - call load() first")

        prompts = [prompt] * num_images if isinstance(prompt, str) else list(prompt)
        n = len(prompts)
        if n == 0 or n > self.config.max_batch_size:
            raise ValueError(f"batch size must be 1..{self.config.max_batch_size}, got {n}")
        if any(not p or not p.strip() for p in prompts):
            raise ValueError("prompt must not be empty")
        self.validate_size(width, height)
        steps = steps or self.config.default_steps

        if seed is None:
            seeds = [random.randint(0, MAX_SEED) for _ in range(n)]
        elif isinstance(seed, int):
            seeds = [(seed + i) % (MAX_SEED + 1) for i in range(n)]
        else:
            seeds = list(seed)
            if len(seeds) != n:
                raise ValueError("number of seeds must match number of images")

        gen_device = "cpu" if self.config.cpu_offload else self.config.device
        generators = [torch.Generator(gen_device).manual_seed(s) for s in seeds]

        with self._lock, torch.inference_mode():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = self.pipe(
                prompt=prompts,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=0.0,  # Turbo models must use 0
                generator=generators,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

        self.total_images += n
        return GenerationResult(out.images, seeds, width, height, steps, elapsed)

    # ------------------------------------------------------------------ info
    def info(self) -> dict:
        d = {
            "loaded": self.is_loaded,
            "model_path": self.config.model_path,
            "total_images": self.total_images,
            "uptime_s": round(time.time() - self.loaded_at, 1) if self.loaded_at else None,
        }
        if torch.cuda.is_available():
            d.update(
                gpu=torch.cuda.get_device_name(0),
                vram_allocated_gb=round(torch.cuda.memory_allocated() / 1024**3, 2),
                vram_peak_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                vram_total_gb=round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            )
        return d
