# 11 – Move code changes with snapshots 🌐 → 🖥️

`tools/snapshot_changes.py` remembers the state of the project. Later it lists what changed and
copies **only the new and changed files** into a transfer folder, keeping the folder tree. You don't
need to copy the whole project again after every edit.

It runs only on this PC (`D:\hkc\image_llm`). The GPU machines just receive the files.

## How it works
- `snapshot` records every file's size, modification time and SHA-256 in `.snapshots/<name>.json`.
- `copy` compares the project with the latest snapshot and copies new and changed files to the target
  folder. A file that was only touched (same content) is **not** copied.
- `--update` saves a new snapshot after copying, so the next update starts from that point.
- The excludes (for example `models`) are stored **inside** the snapshot and reused automatically.
- Files are never copied into the project itself. The target must be outside `D:\hkc\image_llm`.

## Already done: first full move (2026-11-26)
```bat
python tools\snapshot_changes.py snapshot --name empty --exclude "*"
python tools\snapshot_changes.py copy D:\hkc\transfer\hkc_1126_code\igcode --snapshot empty --exclude models --exclude pkgs/inventory --exclude outputs --exclude bench_results --exclude cmds/logs --update
```
`empty` is a baseline with 0 files, so everything counted as new: 480 files, 487 MB, without the model.
`--update` then saved the real baseline. You don't need to repeat this, except for a full fresh copy (see below).

---

## Routine: send an update

### On this PC
```bat
cd D:\hkc\image_llm

REM 1. refresh checksums (always, after any edit)
python cmds\scripts\bundle_checksums.py create

REM 2. see what changed since the last copy
python tools\snapshot_changes.py changes

REM 3. dry run, then real copy into a NEW dated folder
python tools\snapshot_changes.py copy D:\hkc\transfer\hkc_MMDD_update\igcode --dry-run
python tools\snapshot_changes.py copy D:\hkc\transfer\hkc_MMDD_update\igcode --update
```
Replace `MMDD` with today's date, e.g. `hkc_1203_update`. Use a new folder for each update so nothing gets mixed up.

The transfer folder will contain only the changed files plus:
```
_changes\CHANGES.txt        new / changed / deleted files and the baseline snapshot used
_changes\deleted.txt        files deleted on this PC
_changes\delete_files.ps1   deletes those files on a Windows copy
_changes\delete_files.sh    deletes those files on an Ubuntu copy
```
`SHA256SUMS.txt` is always in the list, because step 1 rewrites it. That's expected: the GPU machine
needs the new copy to verify.

### On the GPU machine
Stop the API server first if it's running.

**RTX 5090 (Windows)**, with the project at `D:\image_llm` as an example:
```bat
robocopy E:\hkc_MMDD_update\igcode D:\image_llm /E /XD _changes
powershell -ExecutionPolicy Bypass -File E:\hkc_MMDD_update\igcode\_changes\delete_files.ps1 D:\image_llm
cd D:\image_llm
python cmds\scripts\bundle_checksums.py verify --target 5090
```

**H200 (Ubuntu)**, with the project at `/opt/image_llm` as an example:
```bash
rsync -a --exclude _changes /media/usb/hkc_MMDD_update/igcode/ /opt/image_llm/
bash /media/usb/hkc_MMDD_update/igcode/_changes/delete_files.sh /opt/image_llm
cd /opt/image_llm
python3 cmds/scripts/bundle_checksums.py verify --target h200
```

Read `_changes/CHANGES.txt` before running `delete_files`. If nothing was deleted, the script does nothing.

`verify` must print `... OK, 0 missing, 0 corrupt`. Then:
- Code changed → run `python zimage_service/tests/test_api_stub.py`, then restart the server.
- Wheels changed → rerun `install_offline.py --target <5090|h200>` (guide 04).

---

## Other commands
| Task | Command |
|---|---|
| List snapshots | `python tools\snapshot_changes.py list` |
| Changes since a specific snapshot | `python tools\snapshot_changes.py changes --snapshot 20261126-101500` |
| Re-send changes since an older snapshot (e.g. a USB stick got lost) | `python tools\snapshot_changes.py copy D:\hkc\transfer\resend\igcode --snapshot <older name>` (no `--update`) |
| Full fresh copy of everything (new machine) | the "first full move" command above, into a new folder |
| Only one machine's wheels | add `--exclude pkgs/h200` (going to the 5090) or `--exclude pkgs/5090` (going to the H200), but see the warning below |

## Things to know
- **Don't pass `--exclude` on routine updates.** Passing any `--exclude` **replaces** the excludes stored
  in the snapshot (only the built-in defaults remain). Then `models` would be scanned and could get
  copied. If you do need an extra exclude, repeat all of them:
  `--exclude models --exclude pkgs/inventory --exclude outputs --exclude bench_results --exclude cmds/logs --exclude <new>`
- **Forgot `--update`?** Nothing breaks. The next copy simply includes those changes again.
- **Re-downloaded wheels** (`download_wheels.py --clean`): unchanged wheels have the same content, so they
  aren't copied. Only new versions are copied, and replaced old versions appear in `delete_files`.
- **Don't delete `.snapshots/`.** It's the memory of what the GPU machines have. It's never copied.
- **The model is excluded.** It was moved separately. To update model files, copy them by hand.
- `tools/` is copied too but isn't needed on the GPU machines. It's harmless there.
