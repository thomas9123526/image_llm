#!/usr/bin/env python3
"""
STEP 04 - run on the OFFLINE GPU machine. Standard library only.

Installs the wheels from pkgs/<target>/py3XX into the CURRENT Python (use a venv, see guide 04).
Never touches torch/CUDA. Uses --no-index, so nothing is fetched from the internet.

    python  cmds\\scripts\\install_offline.py --target 5090
    python3 cmds/scripts/install_offline.py --target h200
    ... --dry-run                 # show what would be installed
    ... --with-transformers       # force-upgrade to the shipped transformers wheels

Already-installed packages (torch, CUDA, transformers, numpy, ...) are kept whenever they satisfy
the requirements; only missing or incompatible packages are installed into the venv.
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIN_TRANSFORMERS = (4, 51)


def run(cmd) -> int:
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd])


def installed_version(dist: str):
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:  # noqa: BLE001
        return None


def vtuple(v: str):
    import re
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def nonempty(p: Path) -> bool:
    return p.exists() and any(l.strip() and not l.startswith("#") for l in p.read_text().splitlines())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["5090", "h200"])
    ap.add_argument("--pkgs", default=str(ROOT / "pkgs"))
    ap.add_argument("--with-transformers", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ver = f"{sys.version_info.major}{sys.version_info.minor}"
    wheel_dir = Path(args.pkgs) / args.target / f"py{ver}"
    if not wheel_dir.exists():
        sys.exit(f"ERROR: {wheel_dir} not found. Packages were collected for: "
                 f"{[p.name for p in (Path(args.pkgs) / args.target).glob('py*')]}. "
                 f"Run this script with a matching Python version.")

    print(f"Python      : {sys.version.split()[0]} ({sys.executable})")
    print(f"In venv     : {sys.prefix != getattr(sys, 'base_prefix', sys.prefix)}")
    print(f"Wheel folder: {wheel_dir}")

    torch_v = installed_version("torch")
    print(f"torch       : {torch_v or 'NOT INSTALLED'}")
    if not torch_v:
        print("WARNING: torch not visible from this Python. If you use a venv, create it with "
              "--system-site-packages so the pre-installed torch is reused (see guide 04).")

    log_dir = ROOT / "cmds" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    before = log_dir / f"{args.target}_py{ver}_{stamp}_freeze_before.txt"
    before.write_text(subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"],
                                     capture_output=True, text=True).stdout)
    print(f"Saved current package list to {before} (for rollback)")

    base = [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", wheel_dir]
    if args.dry_run:
        base.append("--dry-run")
    constraints = wheel_dir / "constraints.txt"
    cflags = ["-c", constraints] if nonempty(constraints) else []

    reqs = ["-r", wheel_dir / "requirements.txt"]
    tf_v = installed_version("transformers")
    tf_file = wheel_dir / "requirements-transformers.txt"
    if nonempty(tf_file):
        # transformers is part of the resolution. With pip's default "only-if-needed" strategy the
        # INSTALLED transformers is kept if it is >= 4.51 and compatible with the new diffusers /
        # huggingface_hub; otherwise pip uses the shipped (compatible) transformers wheel instead.
        print(f"transformers: installed {tf_v} -> kept unless incompatible (shipped wheels as fallback)")
        reqs += ["-r", tf_file]
        if args.with_transformers:
            reqs += ["--upgrade", "transformers", "tokenizers"]
    elif tf_v is None or vtuple(tf_v) < MIN_TRANSFORMERS:
        sys.exit(f"ERROR: transformers {tf_v} is missing/too old (need >= 4.51) and no transformers wheels "
                 f"were shipped. Re-run step 02.")
    else:
        print(f"transformers: {tf_v} (OK, kept)")

    rc = run(base + ["--upgrade-strategy", "only-if-needed"] + reqs + cflags)
    if rc != 0:
        sys.exit("ERROR: install failed. The message above names the missing/conflicting package. "
                 "See cmds/10_troubleshooting.md.")

    nodeps = wheel_dir / "requirements-nodeps.txt"
    if nonempty(nodeps):
        rc = run(base + ["--no-deps", "-r", nodeps])
        if rc != 0:
            sys.exit("ERROR: installing no-deps packages (accelerate) failed.")

    if args.dry_run:
        print("\nDry run finished - nothing was installed.")
        return

    print("\n--- pip check (dependency consistency) ---")
    rc = run([sys.executable, "-m", "pip", "check"])
    after = log_dir / f"{args.target}_py{ver}_{stamp}_freeze_after.txt"
    after.write_text(subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"],
                                    capture_output=True, text=True).stdout)
    print(f"\nInstall finished. pip check {'OK' if rc == 0 else 'reported issues (see above)'}.")
    print("Next: python zimage_service/check_env.py   (guide 05)")


if __name__ == "__main__":
    main()
