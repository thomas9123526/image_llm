# 01 – Inventory the GPU machines 🖥️

**Why:** so step 02 downloads versions that match what's already installed (torch, transformers,
huggingface_hub, numpy and so on), and nothing gets upgraded by accident.

1. Copy the `cmds/` folder to each GPU machine with a USB stick. The script uses only the standard library, so no install is needed.
2. Run it with **every Python version you might use**. It writes into `pkgs/inventory/` next to `cmds/`.

**RTX 5090 (Windows, cmd or PowerShell)**
```bat
cd E:\image_llm
py -3.10 cmds\scripts\inventory_target.py --target 5090
py -3.12 cmds\scripts\inventory_target.py --target 5090
```
- If the `py` launcher isn't available, use the full path, e.g.
  `C:\Python312\python.exe cmds\scripts\inventory_target.py --target 5090`
- If torch lives in a conda env or venv, activate it first, then run
  `python cmds\scripts\inventory_target.py --target 5090`

**H200 (Ubuntu)**
```bash
cd /media/<usb>/image_llm
python3.10 cmds/scripts/inventory_target.py --target h200
python3.12 cmds/scripts/inventory_target.py --target h200
```

3. Read the warnings at the end of the output:
   - `RTX 5090 (sm_120) is not in this torch build's arch list` → the installed torch is too old for the 5090.
     See [10_troubleshooting.md](10_troubleshooting.md).
   - `torch could not be imported` → this Python isn't the one with torch. Try a different interpreter or env.
4. Bring `pkgs/inventory/` back to the internet machine, at `D:\hkc\image_llm\pkgs\inventory\`.

Result:
```
pkgs/inventory/5090/py310/  env.json  pip_freeze.txt  pip_debug.txt  nvidia_smi.txt
pkgs/inventory/5090/py312/  ...
pkgs/inventory/h200/py310/  ...
pkgs/inventory/h200/py312/  ...
```
