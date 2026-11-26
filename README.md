# image_llm – Z-Image-Turbo text-to-image service

Python service for [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) on an RTX 5090 (Windows)
and an H200 (Ubuntu), built for **offline** GPU machines.

| Folder | Contents |
|---|---|
| [zimage_service/](zimage_service/) | quick test, `ZImageEngine` class, FastAPI server, client, benchmark, environment check, API test |
| [cmds/](cmds/) | step-by-step guides 01–11 and helper scripts (download / install wheels offline, start server, checksums) |
| [pkgs/](pkgs/) | requirement lists + `targets.json`. Wheels are **not** in git; build them with `download_wheels.py` |
| [tools/](tools/) | `snapshot_changes.py`: copy only changed files to the offline machines |
| [image_generation_production_todo.md](image_generation_production_todo.md) | project plan and TODO checklist |

Not in this repo: model weights (`models/Z-Image-Turbo`, 31 GB) and downloaded wheels (`pkgs/5090`, `pkgs/h200`).

## Start here
Follow [cmds/README.md](cmds/README.md). In short:
1. On a machine with internet: `python cmds/scripts/download_wheels.py --clean` → `pkgs/5090/py3XX`, `pkgs/h200/py3XX`
2. Copy the project and the model to the GPU machine
3. `python cmds/scripts/install_offline.py --target 5090` (or `h200`) inside a `--system-site-packages` venv
4. `python zimage_service/check_env.py`, then `python zimage_service/quick_test.py`
5. Serve: `cmds/scripts/start_server.bat` / `start_server.sh` (see [cmds/08_api_server.md](cmds/08_api_server.md))
