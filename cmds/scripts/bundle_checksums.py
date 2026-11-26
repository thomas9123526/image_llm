#!/usr/bin/env python3
"""
SHA-256 checksums for the transfer bundle (code + wheels; NOT the model). Standard library only.

On the internet PC (before copying):   python cmds/scripts/bundle_checksums.py create
On the GPU machine (after copying):    python cmds/scripts/bundle_checksums.py verify --target 5090
                                        python3 cmds/scripts/bundle_checksums.py verify --target h200
"""
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ["zimage_service", "cmds", "pkgs"]
SKIP_PARTS = {"__pycache__", "logs", "outputs", "bench_results", "inventory", ".venv"}
SUMS = ROOT / "SHA256SUMS.txt"


def files():
    for top in INCLUDE:
        for p in sorted((ROOT / top).rglob("*")):
            rel = p.relative_to(ROOT)
            if p.is_file() and not SKIP_PARTS.intersection(rel.parts) and p.suffix != ".pyc":
                yield rel


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["create", "verify"])
    ap.add_argument("--target", help="verify only files for this target (other pkgs/<target> folders are skipped)")
    args = ap.parse_args()

    if args.mode == "create":
        lines = [f"{sha256(ROOT / r)}  {r.as_posix()}" for r in files()]
        SUMS.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {len(lines)} checksums to {SUMS}")
        return

    bad = missing = ok = 0
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        parts = rel.split("/")
        if args.target and len(parts) > 2 and parts[0] == "pkgs" and parts[1] not in ("requirements", args.target):
            continue
        p = ROOT / rel
        if not p.exists():
            missing += 1
            print("MISSING ", rel)
        elif sha256(p) != digest:
            bad += 1
            print("CORRUPT ", rel)
        else:
            ok += 1
    print(f"\n{ok} OK, {missing} missing, {bad} corrupt")
    sys.exit(1 if (bad or missing) else 0)


if __name__ == "__main__":
    main()
