"""
Z-Image-Turbo HTTP API server (single GPU, single process).

Run (ONE uvicorn worker only - each worker would load its own copy of the model):
    set ZIMAGE_API_KEYS=key-alice,key-bob
    python server.py
  or
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1

Endpoints:
    GET  /health                      liveness
    GET  /ready                       200 once the model is loaded
    GET  /v1/stats                    queue + GPU stats
    POST /v1/images/generate          synchronous: waits and returns the image(s)
    POST /v1/jobs                     asynchronous: returns 202 + job_id
    GET  /v1/jobs/{job_id}            job status
    GET  /v1/jobs/{job_id}/images/{i} PNG result of an async job

Environment variables (all optional):
    ZIMAGE_MODEL_PATH     path to models/Z-Image-Turbo
    ZIMAGE_API_KEYS       comma-separated API keys. Empty => auth DISABLED (dev only)
    ZIMAGE_HOST / ZIMAGE_PORT                 default 0.0.0.0 / 8000
    ZIMAGE_QUEUE_MAX      max queued jobs before returning 503 (default 32)
    ZIMAGE_MAX_PER_KEY    max queued+running jobs per API key (default 4)
    ZIMAGE_SYNC_TIMEOUT   seconds a sync request waits (default 120)
    ZIMAGE_JOB_TTL        seconds finished jobs + images are kept (default 3600)
    ZIMAGE_OUTPUT_DIR     where async results are stored (default ./outputs)
    ZIMAGE_WARMUP         warm-up generations at startup (default 2)
    ZIMAGE_ATTENTION, ZIMAGE_COMPILE, ZIMAGE_STEPS, ZIMAGE_MAX_BATCH  -> see zimage_engine.EngineConfig
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

from zimage_engine import MAX_SEED, ZImageEngine, EngineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("zimage.server")

# ----------------------------------------------------------------------------- settings
API_KEYS = {k.strip() for k in os.environ.get("ZIMAGE_API_KEYS", "").split(",") if k.strip()}
QUEUE_MAX = int(os.environ.get("ZIMAGE_QUEUE_MAX", 32))
MAX_PER_KEY = int(os.environ.get("ZIMAGE_MAX_PER_KEY", 4))
SYNC_TIMEOUT = float(os.environ.get("ZIMAGE_SYNC_TIMEOUT", 120))
JOB_TTL = float(os.environ.get("ZIMAGE_JOB_TTL", 3600))
OUTPUT_DIR = Path(os.environ.get("ZIMAGE_OUTPUT_DIR", Path(__file__).resolve().parent / "outputs"))
WARMUP_RUNS = int(os.environ.get("ZIMAGE_WARMUP", 2))
MAX_PROMPT_CHARS = 2000


# ----------------------------------------------------------------------------- schemas
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)
    width: int = Field(1024, ge=256, le=2048, multiple_of=16)
    height: int = Field(1024, ge=256, le=2048, multiple_of=16)
    steps: int = Field(9, ge=1, le=20, description="9 => 8 DiT forwards (recommended)")
    seed: Optional[int] = Field(None, ge=0, le=MAX_SEED)
    num_images: int = Field(1, ge=1, le=4)
    response_format: Literal["b64_json", "png"] = "b64_json"  # sync endpoint only

    @field_validator("prompt")
    @classmethod
    def strip_prompt(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("prompt must not be blank")
        return v


@dataclass
class Job:
    id: str
    req: GenerateRequest
    owner: str
    status: str = "queued"  # queued | running | completed | failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    seeds: List[int] = field(default_factory=list)
    gen_seconds: Optional[float] = None
    keep_in_memory: bool = False          # sync jobs: images returned directly, not stored
    images_png: List[bytes] = field(default_factory=list)
    image_files: List[Path] = field(default_factory=list)
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def public(self, request: Request) -> dict:
        d = {
            "job_id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "queue_wait_s": round(self.started_at - self.created_at, 3) if self.started_at else None,
            "generation_s": round(self.gen_seconds, 3) if self.gen_seconds else None,
            "params": self.req.model_dump(exclude={"response_format"}),
            "seeds": self.seeds or None,
            "error": self.error,
        }
        if self.status == "completed" and self.image_files:
            d["images"] = [
                str(request.url_for("get_job_image", job_id=self.id, index=i)) for i in range(len(self.image_files))
            ]
        return d


# ----------------------------------------------------------------------------- state
engine = ZImageEngine(EngineConfig.from_env())
jobs: Dict[str, Job] = {}
queue: "asyncio.Queue[Job]" = None  # created inside the running loop
stats = {"completed": 0, "failed": 0, "rejected": 0, "gen_seconds_total": 0.0}


def _png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _run_job_blocking(job: Job) -> None:
    """Runs in a worker thread: GPU generation + PNG encoding/saving."""
    r = job.req
    result = engine.generate(
        r.prompt, width=r.width, height=r.height, steps=r.steps, seed=r.seed, num_images=r.num_images
    )
    job.seeds, job.gen_seconds = result.seeds, result.elapsed_s
    pngs = [_png_bytes(im) for im in result.images]
    if job.keep_in_memory:
        job.images_png = pngs
    else:
        job_dir = OUTPUT_DIR / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        for i, data in enumerate(pngs):
            p = job_dir / f"{i}.png"
            p.write_bytes(data)
            job.image_files.append(p)


async def gpu_worker() -> None:
    """Single consumer => exactly one generation on the GPU at a time."""
    while True:
        job = await queue.get()
        job.status, job.started_at = "running", time.time()
        try:
            await asyncio.to_thread(_run_job_blocking, job)
            job.status = "completed"
            stats["completed"] += 1
            stats["gen_seconds_total"] += job.gen_seconds or 0
            log.info("job %s done in %.2fs (waited %.2fs)", job.id, job.gen_seconds, job.started_at - job.created_at)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            job.status, job.error = "failed", "GPU out of memory - try a smaller size or fewer images"
            stats["failed"] += 1
            log.error("job %s OOM", job.id)
        except ValueError as e:
            job.status, job.error = "failed", str(e)
            stats["failed"] += 1
        except Exception as e:  # noqa: BLE001 - keep the worker alive no matter what
            job.status, job.error = "failed", f"internal error: {type(e).__name__}"
            stats["failed"] += 1
            log.exception("job %s failed", job.id)
        finally:
            job.finished_at = time.time()
            job.done.set()
            queue.task_done()


async def janitor() -> None:
    """Deletes finished jobs and their files after JOB_TTL (protects your disk)."""
    while True:
        await asyncio.sleep(60)
        cutoff = time.time() - JOB_TTL
        for jid in [j.id for j in jobs.values() if j.finished_at and j.finished_at < cutoff]:
            jobs.pop(jid, None)
            shutil.rmtree(OUTPUT_DIR / jid, ignore_errors=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global queue
    queue = asyncio.Queue(maxsize=QUEUE_MAX)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not API_KEYS:
        log.warning("ZIMAGE_API_KEYS is empty - authentication is DISABLED. Do not expose this publicly.")
    await asyncio.to_thread(engine.load)
    if WARMUP_RUNS:
        await asyncio.to_thread(engine.warmup, WARMUP_RUNS)
    tasks = [asyncio.create_task(gpu_worker()), asyncio.create_task(janitor())]
    log.info("Server ready")
    yield
    for t in tasks:
        t.cancel()
    engine.unload()


app = FastAPI(title="Z-Image-Turbo API", version="1.0", lifespan=lifespan)


# ----------------------------------------------------------------------------- auth / admission
async def require_api_key(
    x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)
) -> str:
    if not API_KEYS:
        return "anonymous"
    key = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None)
    if key not in API_KEYS:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key")
    return key


def _enqueue(req: GenerateRequest, owner: str, keep_in_memory: bool) -> Job:
    if req.num_images > engine.config.max_batch_size:
        raise HTTPException(422, f"num_images must be <= {engine.config.max_batch_size}")
    active = sum(1 for j in jobs.values() if j.owner == owner and j.status in ("queued", "running"))
    if active >= MAX_PER_KEY:
        stats["rejected"] += 1
        raise HTTPException(429, f"too many active jobs for this key (max {MAX_PER_KEY})", headers={"Retry-After": "5"})
    job = Job(id=uuid.uuid4().hex, req=req, owner=owner, keep_in_memory=keep_in_memory)
    try:
        queue.put_nowait(job)
    except asyncio.QueueFull:
        stats["rejected"] += 1
        raise HTTPException(503, "server busy, queue full", headers={"Retry-After": "10"})
    jobs[job.id] = job
    return job


def _get_owned_job(job_id: str, owner: str) -> Job:
    job = jobs.get(job_id)
    if not job or job.owner != owner:
        raise HTTPException(404, "job not found")
    return job


# ----------------------------------------------------------------------------- routes
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    if not engine.is_loaded:
        return JSONResponse({"ready": False}, status_code=503)
    return {"ready": True}


@app.get("/v1/stats")
async def get_stats(_: str = Depends(require_api_key)):
    done = stats["completed"]
    return {
        **stats,
        "avg_generation_s": round(stats["gen_seconds_total"] / done, 3) if done else None,
        "queue_depth": queue.qsize() if queue else 0,
        "queue_max": QUEUE_MAX,
        "engine": engine.info(),
    }


@app.post("/v1/images/generate")
async def generate_sync(req: GenerateRequest, owner: str = Depends(require_api_key)):
    job = _enqueue(req, owner, keep_in_memory=True)
    try:
        await asyncio.wait_for(job.done.wait(), timeout=SYNC_TIMEOUT)
    except asyncio.TimeoutError:
        # The job keeps running on the GPU; its in-memory result is simply discarded.
        jobs.pop(job.id, None)
        raise HTTPException(504, "generation timed out; use POST /v1/jobs for long queues")
    try:
        if job.status != "completed":
            raise HTTPException(500 if job.error and job.error.startswith("internal") else 422, job.error)
        meta = {"X-Job-Id": job.id, "X-Seeds": ",".join(map(str, job.seeds)),
                "X-Generation-Seconds": f"{job.gen_seconds:.3f}"}
        if req.response_format == "png":
            if len(job.images_png) != 1:
                raise HTTPException(422, "response_format=png supports num_images=1 only")
            return Response(job.images_png[0], media_type="image/png", headers=meta)
        return {
            "job_id": job.id,
            "seeds": job.seeds,
            "generation_s": round(job.gen_seconds, 3),
            "images": [base64.b64encode(b).decode() for b in job.images_png],
        }
    finally:
        jobs.pop(job.id, None)  # sync results are not kept


@app.post("/v1/jobs", status_code=202)
async def create_job(req: GenerateRequest, request: Request, owner: str = Depends(require_api_key)):
    job = _enqueue(req, owner, keep_in_memory=False)
    return {
        "job_id": job.id,
        "status": job.status,
        "queue_position": queue.qsize(),
        "status_url": str(request.url_for("get_job", job_id=job.id)),
    }


@app.get("/v1/jobs/{job_id}", name="get_job")
async def get_job(job_id: str, request: Request, owner: str = Depends(require_api_key)):
    return _get_owned_job(job_id, owner).public(request)


@app.get("/v1/jobs/{job_id}/images/{index}", name="get_job_image")
async def get_job_image(job_id: str, index: int, owner: str = Depends(require_api_key)):
    job = _get_owned_job(job_id, owner)
    if job.status != "completed":
        raise HTTPException(409, f"job is {job.status}")
    if not 0 <= index < len(job.image_files):
        raise HTTPException(404, "image not found")
    return FileResponse(job.image_files[index], media_type="image/png", filename=f"{job_id}_{index}.png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("ZIMAGE_HOST", "0.0.0.0"),
        port=int(os.environ.get("ZIMAGE_PORT", 8000)),
        workers=1,
    )
