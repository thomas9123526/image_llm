# 06 – Quick test: the first image 🖥️

**Windows (5090)**
```bat
cd D:\image_llm
.venv\Scripts\activate
cd zimage_service
python quick_test.py
python quick_test.py --prompt "a red fox in fresh snow, photorealistic" --seed 7 --out fox.png
```
**Ubuntu (H200)**
```bash
cd /opt/image_llm && source .venv/bin/activate && cd zimage_service
CUDA_VISIBLE_DEVICES=0 python quick_test.py
```

Expected (rough numbers, to be confirmed):
- Model load: 30–120 s (the FP32 transformer is read from disk and cast to BF16; faster after step 09)
- First (cold) generation: noticeably slower
- Warm generation at 1024×1024: about 3 s on the 5090 (third-party figure), probably faster on the H200
- Peak VRAM: about 20–24 GB

Open `quick_test.png`. It should be a sharp, photorealistic fisherman portrait. Same seed + same prompt = same image.

## Try the engine class
```python
from zimage_engine import ZImageEngine, EngineConfig

e = ZImageEngine(EngineConfig())
e.load(); e.warmup()
r = e.generate(["a red car on a coastal road", "a blue boat in a harbor"], seed=[1, 2])
for i, im in enumerate(r.images):
    im.save(f"batch_{i}.png")
print(r.elapsed_s, e.info())
```
