"""
Example client for the Z-Image-Turbo API.

    python client.py --url http://GPU-SERVER-IP:8000 --key key-alice --prompt "a lighthouse at dusk"
    python client.py --url http://GPU-SERVER-IP:8000 --key key-alice --mode async -n 2
"""
import argparse
import base64
import time

import requests


class ZImageClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 180):
        self.base = base_url.rstrip("/")
        self.s = requests.Session()
        if api_key:
            self.s.headers["X-API-Key"] = api_key
        self.timeout = timeout

    def generate(self, prompt: str, **params) -> list[bytes]:
        """Synchronous: blocks until images are ready. Returns PNG bytes."""
        r = self.s.post(f"{self.base}/v1/images/generate", json={"prompt": prompt, **params}, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        print(f"job {data['job_id']} seeds={data['seeds']} gen={data['generation_s']}s")
        return [base64.b64decode(b) for b in data["images"]]

    def submit(self, prompt: str, **params) -> str:
        """Asynchronous: returns a job_id immediately."""
        r = self.s.post(f"{self.base}/v1/jobs", json={"prompt": prompt, **params}, timeout=30)
        r.raise_for_status()
        return r.json()["job_id"]

    def wait(self, job_id: str, poll_s: float = 1.0) -> list[bytes]:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            r = self.s.get(f"{self.base}/v1/jobs/{job_id}", timeout=30)
            r.raise_for_status()
            job = r.json()
            if job["status"] == "completed":
                return [self.s.get(u, timeout=60).content for u in job["images"]]
            if job["status"] == "failed":
                raise RuntimeError(job["error"])
            time.sleep(poll_s)
        raise TimeoutError(job_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default="")
    ap.add_argument("--prompt", default="Photorealistic street market in Hanoi at night, neon signs, rain reflections")
    ap.add_argument("--mode", choices=["sync", "async"], default="sync")
    ap.add_argument("-n", "--num-images", type=int, default=1)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    args = ap.parse_args()

    c = ZImageClient(args.url, args.key)
    params = {"num_images": args.num_images, "width": args.width, "height": args.height}
    if args.seed is not None:
        params["seed"] = args.seed

    t0 = time.time()
    if args.mode == "sync":
        images = c.generate(args.prompt, **params)
    else:
        job_id = c.submit(args.prompt, **params)
        print("submitted", job_id)
        images = c.wait(job_id)
    for i, png in enumerate(images):
        name = f"client_out_{i}.png"
        open(name, "wb").write(png)
        print("saved", name)
    print(f"round trip {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
