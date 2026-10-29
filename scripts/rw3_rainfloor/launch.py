"""Bounded, unbooked-by-default rain attribution launcher; parent imports no JAX."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
REPO = Path("<USER_HOME>/src/wrf_gpu2")
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/v45-rainfloor")
CASE = Path("<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227")


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args])


def verify_sources(arm):
    patch = HERE / Path(arm["patch"]).name
    if hashlib.sha256(patch.read_bytes()).hexdigest() != arm["patch_sha256"]:
        raise ValueError("reviewed patch changed")
    if git("rev-parse", arm["pin"] + ":src").decode().strip() != arm["src_tree"]:
        raise ValueError("candidate source tree mismatch")
    changed = git("diff", "--name-only", arm["base"], arm["pin"]).decode().splitlines()
    if changed != arm["changed_files"] or any(not p.startswith("src/") for p in changed):
        raise ValueError("candidate changes outside registered source delta")


def argv(source, case, out, hours, smoke):
    result = [sys.executable, str(HERE / "model_driver.py"), "run", "--input-dir", str(case),
              "--output-dir", str(out / "wrfout"), "--proof-dir", str(out / "proof"),
              "--scratch-dir", str(out / "scratch"), "--domains-from-namelist",
              "--emit-initial-history", "--hours", str(hours)]
    return result + (["--dry-run"] if smoke else [])


def window(cap):
    start, end = (datetime.fromisoformat(os.environ[k].replace("Z", "+00:00"))
                  for k in ("GA_START_UTC", "GA_END_UTC"))
    now = datetime.now(timezone.utc)
    if not start <= now or (end - now).total_seconds() < cap + 30:
        raise ValueError("outside booked sequencer window or insufficient cap remaining")


def check_cold_gate(path, pin):
    gate = json.loads(path.read_text())
    if gate.get("source_pin") != pin or gate.get("E41") != "PASS" or gate.get("platform") != "gpu":
        raise ValueError("pin-bound manager-reviewed cold GPU E41 gate required before warm arm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=["NO_TOPO", "NO_QSQ", "NO_E84"])
    parser.add_argument("phase", choices=["cold", "warm"])
    args = parser.parse_args()
    arm = next(a for a in json.loads((HERE / "SOURCE_VARIANTS.json").read_text())["arms"] if a["arm"] == args.arm)
    verify_sources(arm)
    smoke = os.environ.get("WRF_GPU2_SEQ_SMOKE") == "1"
    hours, cap = (3, 1800) if args.phase == "cold" else (72, 720)
    window(cap)
    snapshot = None
    if smoke:
        if os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1":
            raise ValueError("CPU smoke must never inherit GPU lock")
        out = LANE / f"smoke_{args.arm}_{args.phase}"
        if out.exists():
            raise ValueError(f"fresh smoke output required: {out}")
        out.mkdir(parents=True)
        snapshot = out / "source"
        snapshot.mkdir()
        # One small immutable source archive per smoke, immediately removed.
        with tarfile.open(fileobj=io.BytesIO(git("archive", arm["pin"], "src"))) as tar:
            tar.extractall(snapshot, filter="data")
        (snapshot / "data").mkdir()
        (snapshot / "data/wrf_pristine").symlink_to("<USER_HOME>/src/wrf_pristine")
        source, case = snapshot, CASE
        cache = out / "unused_cache"
    else:
        # A lease alone is not a booking. Manager supplies paths, resource
        # approval and private immutable snapshots for the actual booked arm.
        booking_path = Path(os.environ["V45_RAINFLOOR_BOOKING"])
        booking = json.loads(booking_path.read_text())
        if booking["approved_by"] != "manager" or booking["disk_approved"] is not True:
            raise ValueError("manager booking and DISK-PLAN required")
        bound = booking["arms"][args.arm]
        if bound["pin"] != arm["pin"]:
            raise ValueError("booking source pin mismatch")
        source, case, root = map(Path, (bound["snapshot"], booking["case"], bound["run_root"]))
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
        if head != arm["pin"] or dirty:
            raise ValueError("measured snapshot is not the clean immutable registered pin")
        if shutil.disk_usage(root).free < (365 + 30) * 2**30:
            raise ValueError("365 GiB ALISIOS floor plus 30 GiB arm reserve required")
        out, cache = root / args.phase, root / "private_cache"
        if out.exists():
            raise ValueError("output already exists; no overwrite/retry best-of-N")
        if args.phase == "warm":
            check_cold_gate(root / "COLD_GATE_PASS.json", arm["pin"])
        out.mkdir(parents=True)
    env = dict(PATH="<USER_HOME>/miniconda3/bin:/usr/local/cuda/bin:/usr/bin:/bin", HOME=os.environ["HOME"],
               USER=os.environ.get("USER", "user"), LANG="C.UTF-8", PYTHONPATH=str(source / "src"),
               PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu" if smoke else "cuda",
               GPUWRF_JAX_CACHE_DIR=str(cache / "jax"), GPUWRF_XLA_AUTOTUNE_CACHE_DIR=str(cache / "autotune"),
               TRITON_CACHE_DIR=str(cache / "triton"))
    command = argv(source, case, out, hours, smoke)
    if not smoke:
        command = [str(REPO / "scripts/with_gpu_lock.sh"), "--timeout", "30", "--label", "v45-rainfloor", "--",
                   "timeout", str(cap), *command]
    started = time.monotonic()
    rc = 2
    try:
        with (out / "run.log").open("w") as log:
            rc = subprocess.run(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=180 if smoke else cap + 45).returncode
    finally:
        receipt = dict(arm=args.arm, phase=args.phase, source_pin=arm["pin"], source_tree=arm["src_tree"],
                       argv=argv(source, case, out, hours, False), smoke_added_argv=["--dry-run"] if smoke else [],
                       source_archive=bool(smoke), hours=hours, cap_s=cap, rc=rc,
                       wall_s=time.monotonic() - started, cache=str(cache),
                       scope="PLAN_ONLY; GPU science/performance/E41 NOT_APPLICABLE" if smoke else "GPU diagnostic; score vs original CPU-WRF pending")
        (out / "RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if snapshot is not None:
            shutil.rmtree(snapshot)
            if cache.exists():
                shutil.rmtree(cache)
    print(json.dumps(receipt))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
