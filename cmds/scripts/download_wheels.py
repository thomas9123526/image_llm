#!/usr/bin/env python3
"""
STEP 02 - run on the machine WITH internet. Standard library only.

Downloads every wheel needed to run zimage_service offline into:
    pkgs/<target>/py310/   pkgs/<target>/py312/

    python cmds/scripts/download_wheels.py                     # all targets, all pythons
    python cmds/scripts/download_wheels.py --targets 5090 --pythons 3.12
    python cmds/scripts/download_wheels.py --relax huggingface_hub   # drop a version pin from inventory
    python cmds/scripts/download_wheels.py --py310 "C:\\Python310\\python.exe" --py312 "C:\\...\\python.exe"

If pkgs/inventory/<target>/py3XX/ exists (from step 01), versions of already-installed packages
are pinned to what the target has, so nothing already installed gets upgraded.
Without inventory it runs in "blind" mode: newest compatible versions + transformers wheels as a fallback.

Never downloads torch / CUDA (they are expected to be on the targets already).
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKGS = ROOT / "pkgs"
REQ_DIR = PKGS / "requirements"
INV_DIR = PKGS / "inventory"

# Never pinned / never downloaded: the GPU stack already on the targets.
GPU_STACK_PREFIXES = ("torch", "torchvision", "torchaudio", "triton", "pytorch-triton", "nvidia-",
                      "xformers", "flash-attn", "cuda-", "pip", "setuptools", "wheel")
MIN_TRANSFORMERS = (4, 51)  # Qwen3Model support


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def req_name(line: str) -> str:
    return norm(re.split(r"[\s\[<>=!~;@]", line.strip(), 1)[0])


def vtuple(v: str):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def read_reqs(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split(" #")[0].split("\t#")[0].strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def find_interpreter(ver: str, explicit: str = None):
    """Prefer an interpreter of the SAME Python version so environment markers resolve correctly."""
    if explicit:
        return [explicit]
    if f"{sys.version_info.major}.{sys.version_info.minor}" == ver:
        return [sys.executable]
    for cand in (["py", f"-{ver}"], [f"python{ver}"]):
        try:
            r = subprocess.run(cand + ["-c", "import sys;print('%d.%d'%sys.version_info[:2])"],
                               capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and r.stdout.strip() == ver:
                return cand
        except Exception:  # noqa: BLE001
            pass
    return None


def interp_version(interp) -> str:
    r = subprocess.run(interp + ["-c", "import sys;print('%d.%d'%sys.version_info[:2])"],
                       capture_output=True, text=True, timeout=60)
    return r.stdout.strip()


def _packaging():
    try:
        from packaging.markers import Marker  # noqa: F401
        from packaging.requirements import Requirement
    except ImportError:  # fall back to pip's vendored copy
        from pip._vendor.packaging.requirements import Requirement
    return Requirement


def marker_env(ver: str, os_name: str) -> dict:
    win = os_name == "windows"
    return {
        "python_version": ver, "python_full_version": ver + ".0",
        "implementation_name": "cpython", "implementation_version": ver + ".0",
        "platform_python_implementation": "CPython",
        "sys_platform": "win32" if win else "linux", "platform_system": "Windows" if win else "Linux",
        "os_name": "nt" if win else "posix", "platform_machine": "AMD64" if win else "x86_64",
        "platform_release": "", "platform_version": "", "extra": "",
    }


def missing_by_markers(out: Path, ver: str, os_name: str) -> list:
    """pip evaluates environment markers (python_version, sys_platform, ...) for the HOST interpreter,
    not for --python-version/--platform. Re-evaluate every wheel's Requires-Dist for the TARGET and
    return requirements whose package is not in the folder."""
    import zipfile

    Requirement = _packaging()
    env = marker_env(ver, os_name)
    have = {norm(p.name.split("-")[0]) for p in out.glob("*.whl")}
    missing = {}
    for whl in out.glob("*.whl"):
        with zipfile.ZipFile(whl) as z:
            meta = next((n for n in z.namelist() if n.endswith(".dist-info/METADATA")), None)
            if not meta:
                continue
            for line in z.read(meta).decode("utf-8", "replace").splitlines():
                if not line.startswith("Requires-Dist:"):
                    continue
                try:
                    req = Requirement(line.split(":", 1)[1].strip())
                except Exception:  # noqa: BLE001
                    continue
                n = norm(req.name)
                if n in have or n.startswith(GPU_STACK_PREFIXES):
                    continue
                if req.marker is None or req.marker.evaluate(env):
                    spec = str(req.specifier)
                    missing.setdefault(n, set()).add(spec)
    return [n + ",".join(s for s in sorted(specs) if s) for n, specs in sorted(missing.items())]


def platform_tags(os_name: str, glibc: str) -> list:
    if os_name == "windows":
        return ["win_amd64"]
    minor = vtuple(glibc or "2.28")[1] if glibc else 28
    tags = [f"manylinux_2_{m}_x86_64" for m in range(minor, 4, -1)]
    tags += [t for k, t in ((17, "manylinux2014_x86_64"), (12, "manylinux2010_x86_64"), (5, "manylinux1_x86_64"))
             if k <= minor]
    return tags


def load_inventory(target: str, ver: str):
    d = INV_DIR / target / f"py{ver.replace('.', '')}"
    env = json.loads((d / "env.json").read_text(encoding="utf-8")) if (d / "env.json").exists() else None
    installed = None
    if (d / "pip_freeze.txt").exists():
        installed = {}
        for line in (d / "pip_freeze.txt").read_text(encoding="utf-8").splitlines():
            if "==" in line and not line.startswith("#"):
                n, v = line.split("==", 1)
                installed[norm(n)] = v.strip()
    return env, installed


def pip(interp, args, log):
    cmd = interp + ["-m", "pip"] + args
    log.write("\n$ " + " ".join(cmd) + "\n")
    print("  $", " ".join(cmd[:6]), "...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    log.write(r.stdout + r.stderr)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
    return r.returncode


def build_one(target: str, ver: str, cfg: dict, args) -> bool:
    print(f"\n=== {target} / Python {ver} ===")
    interp = find_interpreter(ver, getattr(args, f"py{ver.replace('.', '')}", None))
    if interp is None:
        interp = [sys.executable]
        print(f"  NOTE: no local Python {ver}; using Python {interp_version(interp)} with --python-version {ver}. "
              f"Version-specific dependencies are fixed up by the marker check below.")

    env, installed = load_inventory(target, ver)
    os_name = (env or {}).get("os") or cfg.get("os", "linux")
    glibc = (env or {}).get("glibc") or cfg.get("glibc")
    mode = "inventory" if installed is not None else "blind"
    print(f"  mode={mode} os={os_name} glibc={glibc} interpreter={' '.join(interp)}")

    host_os = "windows" if sys.platform == "win32" else "linux"
    if host_os != os_name:
        print(f"  NOTE: downloading {os_name} wheels on a {host_os} host - OS-specific deps are fixed up "
              f"by the marker check (and pkgs/requirements/{os_name}_extra.txt)")

    reqs = read_reqs(REQ_DIR / "common.txt") + read_reqs(REQ_DIR / f"{os_name}_extra.txt")
    nodeps = read_reqs(REQ_DIR / "nodeps.txt")
    tf_stack = read_reqs(REQ_DIR / "transformers_stack.txt")
    relax = {norm(x) for x in args.relax}
    constraints = []
    install_tf = False

    if installed is not None:
        tf = installed.get("transformers")
        if tf is None or vtuple(tf) < MIN_TRANSFORMERS:
            print(f"  transformers on target = {tf} -> will ship transformers wheels")
            install_tf = True
            relax |= {"transformers", "tokenizers", "huggingface-hub", "safetensors"}
        for n, v in sorted(installed.items()):
            if n.startswith(GPU_STACK_PREFIXES) or n in relax:
                continue
            constraints.append(f"{n}=={v}")
        nodeps = [r for r in nodeps if req_name(r) not in installed]
    # Blind mode: also ship transformers wheels so the install step can fall back to them if needed.
    download_reqs = reqs + (tf_stack if (install_tf or installed is None) else [])

    out = Path(args.out_root) / target / f"py{ver.replace('.', '')}"
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "requirements.txt").write_text("\n".join(reqs) + "\n", encoding="utf-8")
    (out / "requirements-nodeps.txt").write_text("\n".join(nodeps) + "\n", encoding="utf-8")
    if install_tf or installed is None:
        (out / "requirements-transformers.txt").write_text("\n".join(tf_stack) + "\n", encoding="utf-8")
    (out / "constraints.txt").write_text("\n".join(constraints) + "\n", encoding="utf-8")
    dl_file = out / "_download_list.txt"
    dl_file.write_text("\n".join(download_reqs) + "\n", encoding="utf-8")

    abi = f"cp{ver.replace('.', '')}"
    target_flags = ["--only-binary=:all:", "--python-version", ver, "--implementation", "cp", "--abi", abi]
    for t in platform_tags(os_name, glibc):
        target_flags += ["--platform", t]

    ok = True
    with open(out / "download.log", "w", encoding="utf-8") as log:
        pip(interp, ["--version"], log)
        rc = pip(interp, ["download", "--dest", str(out), *target_flags, "-r", str(dl_file),
                          "-c", str(out / "constraints.txt")], log)
        if rc != 0:
            ok = False
            print("  ERROR resolving main packages. If a pinned (inventory) version conflicts, "
                  "rerun with --relax <package>. See download.log.")
        if nodeps and ok:
            rc = pip(interp, ["download", "--dest", str(out), *target_flags, "--no-deps",
                              "-r", str(out / "requirements-nodeps.txt")], log)
            ok = ok and rc == 0

        # Marker fix-up: pip evaluated python_version / sys_platform markers for the HOST. Re-check every
        # wheel's dependencies for the TARGET and fetch whatever is missing, until the set is closed.
        for _ in range(8):
            if not ok:
                break
            missing = missing_by_markers(out, ver, os_name)
            if not missing:
                print("  marker closure check: OK")
                break
            print(f"  marker fix-up: adding {missing}")
            fix_file = out / "_marker_fixups.txt"
            fix_file.write_text("\n".join(missing) + "\n", encoding="utf-8")
            rc = pip(interp, ["download", "--dest", str(out), *target_flags, "-r", str(fix_file),
                              "-c", str(out / "constraints.txt")], log)
            ok = ok and rc == 0
        else:
            ok = False
            print("  marker closure check: FAILED (did not converge)")

        # Offline resolution check: can everything be installed from this folder alone?
        if ok and not args.skip_verify:
            with tempfile.TemporaryDirectory() as tmp:
                rc = pip(interp, ["install", "--dry-run", "--ignore-installed", "--no-index",
                                  "--find-links", str(out), "--target", tmp, *target_flags,
                                  "-r", str(dl_file), "-c", str(out / "constraints.txt")], log)
            print("  offline resolution check:", "OK" if rc == 0 else "FAILED (see download.log)")
            ok = ok and rc == 0

    wheels = sorted(p.name for p in out.glob("*.whl"))
    size_mb = sum(p.stat().st_size for p in out.glob("*.whl")) / 1024**2
    manifest = {
        "target": target, "python": ver, "os": os_name, "glibc": glibc, "mode": mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "platform_tags": platform_tags(os_name, glibc), "ok": ok,
        "ships_transformers": install_tf or installed is None,
        "wheel_count": len(wheels), "size_mb": round(size_mb, 1), "wheels": wheels,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  {'OK' if ok else 'FAILED'}: {len(wheels)} wheels, {size_mb:.0f} MB -> {out}")
    return ok


def main() -> None:
    cfg_all = json.loads((PKGS / "targets.json").read_text(encoding="utf-8"))
    targets_default = [k for k in cfg_all if not k.startswith("_")]

    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=targets_default)
    ap.add_argument("--pythons", nargs="+", default=None, help="default: from targets.json (3.10 3.12)")
    ap.add_argument("--py310", help="path to a local Python 3.10 interpreter")
    ap.add_argument("--py312", help="path to a local Python 3.12 interpreter")
    ap.add_argument("--relax", nargs="*", default=[], help="packages whose inventory pin should be dropped")
    ap.add_argument("--py311", help="path to a local Python 3.11 interpreter")
    ap.add_argument("--py313", help="path to a local Python 3.13 interpreter")
    ap.add_argument("--out-root", default=str(PKGS), help="where <target>/py3XX folders are written")
    ap.add_argument("--clean", action="store_true", help="delete existing pkgs/<target>/py3XX first")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    results = {}
    for t in args.targets:
        cfg = cfg_all.get(t, {})
        for ver in args.pythons or cfg.get("python", ["3.10", "3.12"]):
            results[f"{t}/py{ver}"] = build_one(t, ver, cfg, args)

    print("\n==== SUMMARY ====")
    for k, ok in results.items():
        print(f"  {k:16s} {'OK' if ok else 'FAILED'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
