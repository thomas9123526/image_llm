# Image Generation Production Project --- Partner Handoff

## Goal

Build a Python-based production text-to-image service using Hugging Face
open-weight models, targeting:

-   **Workload:** 10--50 generated images per minute
-   **Output:** 1024 × 1024
-   **Style:** Photorealistic images
-   **Deployment:** Python API / production inference
-   **Licensing:** Non-commercial use is acceptable
-   **Hardware:** One NVIDIA RTX 5090 (32 GB VRAM) and one NVIDIA H200
    (141 GB HBM)

The aim is to establish a reliable baseline first, then optimize
throughput and add a higher-quality model route.

------------------------------------------------------------------------

## Proposed model choices

### 1. Z-Image-Turbo --- baseline / standard generation

-   Hugging Face: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
-   Approximate model size: 6B parameters
-   Intended role: Fast standard photorealistic generation
-   Initial precision: BF16
-   Initial resolution: 1024 × 1024
-   Initial inference steps: 8
-   Proposed deployment: Run on the RTX 5090 first; also benchmark on
    the H200.

### 2. Qwen-Image-2512 --- premium quality route

-   Hugging Face: https://huggingface.co/Qwen/Qwen-Image-2512
-   Approximate model size: 20B parameters
-   Intended role: Higher-detail / premium generation, subject to
    quality evaluation
-   Initial precision: BF16 on H200
-   Initial resolution: 1024 × 1024
-   Start with the model card's recommended inference settings (the
    earlier discussion used 50 steps; verify against the current model
    card and installed Diffusers version).
-   Do not assume it can meet the same throughput as Z-Image-Turbo.

### 3. FLUX.2 \[dev\] --- optional evaluation

-   Hugging Face: https://huggingface.co/black-forest-labs/FLUX.2-dev
-   Approximate model size: 32B parameters
-   Intended role: Optional premium comparison on H200.
-   BF16 weights alone are roughly 64 GB; full inference requires
    additional memory.
-   Check and accept the Hugging Face model terms before use. Its
    license is not equivalent to Apache 2.0 and may restrict commercial
    use.
-   Only add it if a controlled quality comparison shows a meaningful
    benefit.

### Model-selection decision

Start with **Z-Image-Turbo on both GPUs** to get a clean, comparable
throughput baseline. Add Qwen-Image-2512 as a separate H200 premium
worker after the baseline is stable. Evaluate FLUX.2 \[dev\] later, not
as a launch dependency.

Model names, repository availability, model-card instructions, licenses,
and actual memory requirements must be rechecked before deployment. VRAM
numbers discussed so far are planning estimates, not guarantees.

------------------------------------------------------------------------

## Proposed architecture

1.  **API gateway:** Nginx or Traefik.
2.  **API:** FastAPI accepts generation requests and returns a job ID
    with HTTP `202 Accepted`.
3.  **Queue:** Redis Streams or another durable job queue.
4.  **Scheduler:** Routes jobs by model / quality tier and available
    worker capacity.
5.  **GPU workers:** One persistent model process per GPU; do not reload
    a model for each request.
6.  **Storage:** S3-compatible object storage for generated images.
7.  **Status API:** `GET /jobs/{job_id}` returns queued, running,
    completed, or failed status and the result URL.
8.  **Monitoring:** Prometheus + Grafana; NVIDIA DCGM Exporter for GPU
    metrics.

### Initial routing

-   Standard requests: Z-Image-Turbo worker.
-   Premium requests: Qwen-Image-2512 worker on H200, after validation.
-   Overflow: Route standard requests to whichever Z-Image worker has
    available capacity.
-   Keep model-specific queues so slow premium jobs do not block
    standard jobs.

### Important implementation requirements

-   Use bounded queues and backpressure.
-   Add authentication, rate limiting, request validation, timeouts and
    idempotency.
-   Use atomic job claiming, retry limits and visibility timeouts.
-   Persist job metadata; do not rely only on expiring Redis keys.
-   Store outputs in object storage and return authenticated or
    time-limited URLs.
-   Limit each GPU to one active generation initially; increase only
    after benchmarking.
-   Do not run multiple processes on the same GPU unless the memory and
    performance impact have been tested.

------------------------------------------------------------------------

## Performance planning --- estimates versus measurements

### Public benchmark mentioned in the discussion

A third-party RTX 5090 Z-Image-Turbo benchmark was cited at
approximately:

-   3.18 seconds per 1024 × 1024 image
-   About 19 images per minute
-   About 21.7 GB peak VRAM

This is a reported external benchmark, **not a measurement on our own
system**. Reproduce it with our exact model revision, software versions,
prompt set and settings before using it for capacity commitments.

### Planning estimates previously discussed

These are hypotheses for capacity planning only, not verified H200
results:

-   RTX 5090 + Z-Image-Turbo: approximately 19 images/minute based on
    the cited third-party result.
-   H200 + Z-Image-Turbo: approximately 20--30 images/minute as an
    unverified estimate.
-   Combined Z-Image capacity: approximately 39--49 images/minute as an
    unverified estimate.
-   H200 + Qwen-Image-2512: approximately 6--12 seconds/image was
    discussed as a rough estimate, not a benchmark.

**Do not promise 50 images/minute from these estimates.** To sustain 50
images/minute with 20% capacity headroom, demonstrate at least 62.5
images/minute under representative load. The two-GPU setup has not yet
been shown to meet this.

### Benchmark plan

For each GPU/model configuration:

1.  Record exact GPU model, driver, CUDA, PyTorch, Diffusers and model
    revision.
2.  Run at least 5 warm-up generations.
3.  Run at least 100 measured generations.
4.  Use a fixed set of representative prompts and seeds.
5.  Synchronize CUDA around timing measurements.
6.  Record generation latency (p50, p95, p99), images/minute, peak VRAM,
    GPU utilization, power, temperature and failures.
7.  Repeat at batch sizes 1, 2 and 4, if memory allows.
8.  Compare quality at the same output resolution using human review and
    a fixed evaluation set.
9.  Test mixed standard/premium traffic and sustained load, not only
    isolated generations.

### Initial operating settings to test

  Setting                              Initial value
  ------------------------------------ ----------------------------------------------
  Resolution                           1024 × 1024
  Precision                            BF16
  Z-Image steps                        8
  Initial batch size                   1
  Initial active generations per GPU   1
  Batch tests                          1, 2, 4
  Batching window                      50--150 ms, only if batching helps
  Target utilization                   Roughly 70--90%, measured rather than forced

Test four-step Z-Image only as a separate quality/performance
experiment. Do not reduce steps without checking photorealism and prompt
adherence.

------------------------------------------------------------------------

## TODO checklist

### A. Verify models and environment

-   [ ] Confirm the Hugging Face repositories are accessible.
-   [ ] Review each model's current license and usage restrictions.
-   [ ] Pin model revisions / commit hashes.
-   [ ] Confirm current official Diffusers pipeline classes and
    model-card inference examples.
-   [ ] Verify BF16 support and actual VRAM usage on both GPUs.
-   [ ] Record OS, NVIDIA driver, CUDA, PyTorch, Transformers and
    Diffusers versions.
-   [ ] Build a reproducible environment (Docker image or locked Python
    dependencies).

### B. Establish single-GPU baselines

-   [ ] Run Z-Image-Turbo BF16 on RTX 5090 at 1024 × 1024, 8 steps.
-   [ ] Run Z-Image-Turbo BF16 on H200 at identical settings.
-   [ ] Measure warm and steady-state latency and peak VRAM.
-   [ ] Run at least 100 generations per configuration.
-   [ ] Save outputs and benchmark logs.
-   [ ] Evaluate photorealism and prompt adherence on a shared prompt
    set.

### C. Optimize standard generation

-   [ ] Test batch sizes 1, 2 and 4 on each GPU.
-   [ ] Test SDPA / supported attention backends.
-   [ ] Benchmark `torch.compile` separately after a working baseline.
-   [ ] Test four-step Z-Image against eight-step quality.
-   [ ] Measure throughput and p95 latency under concurrent API load.
-   [ ] Select settings based on quality, throughput and memory---not
    utilization alone.

### D. Build the service

-   [ ] Implement FastAPI `POST /generate` returning a job ID.
-   [ ] Implement `GET /jobs/{job_id}`.
-   [ ] Add Redis-backed queue and atomic job claiming.
-   [ ] Build one persistent Z-Image worker for RTX 5090.
-   [ ] Build one persistent Z-Image worker for H200.
-   [ ] Add scheduler and worker health checks.
-   [ ] Add failure handling, retries, timeouts and dead-letter
    handling.
-   [ ] Add authentication, rate limiting and input validation.
-   [ ] Store images in S3-compatible object storage.
-   [ ] Return result URLs and generation metadata.

### E. Add premium model

-   [ ] Load Qwen-Image-2512 BF16 on H200 in a separate worker.
-   [ ] Follow the current official model-card inference settings.
-   [ ] Benchmark latency, peak VRAM and quality.
-   [ ] Add a premium queue and explicit routing policy.
-   [ ] Evaluate FLUX.2 \[dev\] only if there is a clear quality
    requirement and its license is acceptable.

### F. Production readiness and capacity

-   [ ] Add Prometheus, Grafana and NVIDIA DCGM Exporter.
-   [ ] Track queue depth, throughput, p50/p95/p99 latency, GPU memory,
    utilization, temperature and failures.
-   [ ] Run a sustained load test at 10 images/minute.
-   [ ] Run a sustained load test at 25 images/minute.
-   [ ] Run a sustained load test at 50 images/minute.
-   [ ] Test burst traffic and queue recovery.
-   [ ] Confirm at least 62.5 images/minute demonstrated capacity before
    committing to 50/minute with 20% headroom.
-   [ ] If capacity is insufficient, optimize batching/model settings or
    add GPU capacity.

------------------------------------------------------------------------

## Decisions requested from the team

1.  Confirm that Z-Image-Turbo is the standard model for the first
    benchmark.
2.  Confirm that Qwen-Image-2512 is the candidate premium model on H200.
3.  Choose Redis Streams versus Celery / another queue implementation.
4.  Choose S3-compatible storage provider and retention policy.
5.  Agree on the photorealism evaluation prompts and acceptance
    criteria.
6.  Agree on the required latency SLA and whether 50 images/minute is
    sustained or peak demand.
7.  Decide whether another GPU is available if the two-GPU system cannot
    demonstrate the required headroom.

## Useful references

-   Z-Image-Turbo: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
-   Qwen-Image-2512: https://huggingface.co/Qwen/Qwen-Image-2512
-   FLUX.2 \[dev\]: https://huggingface.co/black-forest-labs/FLUX.2-dev
-   Diffusers optimization:
    https://huggingface.co/docs/diffusers/optimization/overview
-   NVIDIA DCGM Exporter: https://github.com/NVIDIA/dcgm-exporter

------------------------------------------------------------------------

**Current recommendation:** Build the queue/API/storage skeleton and
benchmark Z-Image-Turbo on both GPUs first. Treat all H200 throughput
figures as unverified until measured on the actual deployment. Add Qwen
as a separate premium path only after the standard service is stable.
