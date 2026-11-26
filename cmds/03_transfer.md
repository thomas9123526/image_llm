# 03 – Transfer to the GPU machines 🌐 → 🖥️

Keep the same folder structure on each machine (for example `D:\image_llm` on the 5090, `/opt/image_llm` on the H200).

| Copy | 5090 (Windows) | H200 (Ubuntu) | Size |
|---|---|---|---|
| `zimage_service/` | ✅ | ✅ | < 1 MB |
| `cmds/` | ✅ | ✅ | < 1 MB |
| `SHA256SUMS.txt` | ✅ | ✅ | < 1 MB |
| `pkgs/5090/` | ✅ | – | ~0.2 GB |
| `pkgs/h200/` | – | ✅ | ~0.25 GB |
| `models/Z-Image-Turbo/` | ✅ (if not already there) | ✅ (if not already there) | 31 GB (or ~20 GB after step 09) |

`pkgs/inventory/`, `__pycache__` and the model's `.cache` folder don't need to be copied.

Example copy commands:
```bat
robocopy D:\hkc\image_llm E:\image_llm /E /XD __pycache__ inventory .cache
```
```bash
rsync -a --exclude __pycache__ --exclude inventory --exclude .cache /media/usb/image_llm/ /opt/image_llm/
```

**Verify the copy** of the code and wheels (the checksums in `SHA256SUMS.txt` were created on the internet PC; copy that file too):
```bash
python cmds/scripts/bundle_checksums.py verify --target 5090     # or --target h200
```
Expect `N OK, 0 missing, 0 corrupt`. If you edit code or re-download wheels, recreate the checksums first with
`python cmds/scripts/bundle_checksums.py create`.

Check the model copied fully with `python zimage_service/check_env.py` (step 05). It lists the weight
files and the total size; the original is about 31 GB.

**USB tip:** FAT32 sticks can't hold files over 4 GB, and the model shards are about 5–10 GB. Format the stick as exFAT or NTFS.
