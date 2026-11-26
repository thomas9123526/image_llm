# 09 – Optional: shrink the model 31 GB → about 20 GB 🖥️

The downloaded `transformer/` is stored in **FP32 (23 GB)**. The pipeline casts it to BF16 on load
anyway, so a BF16 copy (about 12 GB) produces the same images and loads faster.

Requirements: about 24 GB free RAM, about 13 GB free disk temporarily, no GPU needed.
Do it after step 04 (it needs diffusers) and after step 06 has passed once, so you have a reference image.

```bash
cd zimage_service
python quick_test.py --out before.png          # reference image
python convert_to_bf16.py --swap               # transformer/ -> transformer_fp32_backup/, new BF16 -> transformer/
python quick_test.py --out after.png           # should look the same as before.png
```
If it looks the same, delete the backup:
- Windows: `rmdir /s /q ..\models\Z-Image-Turbo\transformer_fp32_backup`
- Ubuntu: `rm -rf ../models/Z-Image-Turbo/transformer_fp32_backup`

Tip: do this once on the 5090, then copy the smaller model folder to the H200 to save transfer time.
