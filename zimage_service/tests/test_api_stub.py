"""
End-to-end API test of server.py with the GPU replaced by a CPU stub (no model load, ~15 s).
Run on any machine after install_offline.py:   python zimage_service/tests/test_api_stub.py
"""
import base64
import io
import os
import sys
import tempfile
import threading
import time

os.environ["ZIMAGE_API_KEYS"] = "k1,k2"
os.environ["ZIMAGE_WARMUP"] = "0"
os.environ["ZIMAGE_MAX_PER_KEY"] = "2"
os.environ["ZIMAGE_OUTPUT_DIR"] = tempfile.mkdtemp()
SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVICE_DIR)

import requests
import uvicorn
from PIL import Image

import server
from zimage_engine import GenerationResult, ZImageEngine


class StubEngine(ZImageEngine):
    def load(self):
        self.pipe = object()
        self.loaded_at = time.time()

    def generate(self, prompt, *, width=1024, height=1024, steps=None, seed=None, num_images=1):
        self.validate_size(width, height)
        time.sleep(0.5)  # pretend GPU work
        imgs = [Image.new("RGB", (width, height), (i * 60, 100, 200)) for i in range(num_images)]
        seeds = [(seed or 0) + i for i in range(num_images)]
        self.total_images += num_images
        return GenerationResult(imgs, seeds, width, height, steps or 9, 0.5)

    def info(self):
        return {"loaded": True, "total_images": self.total_images}


server.engine = StubEngine()
cfg = uvicorn.Config(server.app, host="127.0.0.1", port=8765, log_level="warning")
srv = uvicorn.Server(cfg)
threading.Thread(target=srv.run, daemon=True).start()
B = "http://127.0.0.1:8765"
for _ in range(100):
    try:
        if requests.get(B + "/ready", timeout=1).status_code == 200:
            break
    except Exception:
        pass
    time.sleep(0.1)

H = {"X-API-Key": "k1"}
results = []


def check(name, cond, info=""):
    results.append(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")


check("health", requests.get(B + "/health").json() == {"status": "ok"})
check("ready", requests.get(B + "/ready").status_code == 200)
check("no key -> 401", requests.post(B + "/v1/images/generate", json={"prompt": "x"}).status_code == 401)
check("bearer auth", requests.get(B + "/v1/stats", headers={"Authorization": "Bearer k2"}).status_code == 200)
r = requests.post(B + "/v1/images/generate", json={"prompt": "x", "width": 1000}, headers=H)
check("width not /16 -> 422", r.status_code == 422)
check("blank prompt -> 422", requests.post(B + "/v1/images/generate", json={"prompt": "   "}, headers=H).status_code == 422)

r = requests.post(B + "/v1/images/generate", json={"prompt": "fox", "seed": 5, "num_images": 2,
                                                   "width": 512, "height": 768}, headers=H)
ok = r.status_code == 200 and len(r.json()["images"]) == 2 and r.json()["seeds"] == [5, 6]
im = Image.open(io.BytesIO(base64.b64decode(r.json()["images"][0]))) if ok else None
check("sync b64, 2 images, seeds", ok and im.size == (512, 768), str(r.status_code))

r = requests.post(B + "/v1/images/generate", json={"prompt": "fox", "response_format": "png"}, headers=H)
check("sync png", r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG",
      r.headers.get("X-Seeds", ""))

r = requests.post(B + "/v1/jobs", json={"prompt": "async fox", "num_images": 2}, headers=H)
check("async -> 202", r.status_code == 202, r.text[:80])
jid = r.json()["job_id"]
check("other key cannot see job -> 404",
      requests.get(f"{B}/v1/jobs/{jid}", headers={"X-API-Key": "k2"}).status_code == 404)
for _ in range(50):
    j = requests.get(f"{B}/v1/jobs/{jid}", headers=H).json()
    if j["status"] in ("completed", "failed"):
        break
    time.sleep(0.2)
check("async completed", j["status"] == "completed" and len(j["images"]) == 2, j["status"])
img = requests.get(j["images"][1], headers=H)
check("async image download", img.status_code == 200 and img.content[:4] == b"\x89PNG")

# per-key limit: MAX_PER_KEY=2 -> third concurrent job rejected
codes = [requests.post(B + "/v1/jobs", json={"prompt": f"p{i}"}, headers=H).status_code for i in range(3)]
check("per-key limit -> 429", codes == [202, 202, 429], str(codes))

st = requests.get(B + "/v1/stats", headers=H).json()
check("stats", st["completed"] >= 3 and "queue_depth" in st, f"completed={st['completed']}")

# client.py class against live server
from client import ZImageClient  # noqa: E402

time.sleep(2)
c = ZImageClient(B, "k2")
check("client.generate", len(c.generate("cat", width=256, height=256)) == 1)
check("client.submit/wait", len(c.wait(c.submit("cat", num_images=1, width=256, height=256))) == 1)

srv.should_exit = True
print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
