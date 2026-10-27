"""CPU-only lifecycle controls; never acquire the official GPU lock or import JAX."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def stamp(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    tree = here.parents[1]
    spec = importlib.util.spec_from_file_location("guard", here / "night_guard.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    now = 1_800_000_000
    admitted = guard.admit(stamp(now), stamp(now + 2390), now, 2390)
    assert admitted["need_seconds"] == 2390
    negatives = [(None, None), (stamp(now + 1), stamp(now + 2500)),
                 (stamp(now - 2500), stamp(now - 1)), (stamp(now), stamp(now + 2389)),
                 (stamp(now).replace("Z", "+00:00"), stamp(now + 2500))]
    for start, end in negatives:
        try:
            guard.admit(start, end, now, 2390)
        except ValueError:
            continue
        raise AssertionError("inadmissible window accepted")
    result = {"guard": {"positive": 1, "negative": len(negatives)}, "runs": []}
    base_env = dict(os.environ, JAX_PLATFORMS="cpu")
    base_env.pop("GA_START_UTC", None)
    base_env.pop("GA_END_UTC", None)
    for label, dates in (("missing", {}), ("expired", {"GA_START_UTC": "2020-01-01T00:00:00Z",
                                                     "GA_END_UTC": "2020-01-01T00:40:00Z"})):
        run = subprocess.run(["bash", str(here / "night40.sh")], env=base_env | dates,
                             capture_output=True, text=True, timeout=15)
        (out / (label + ".log")).write_text(run.stdout + run.stderr)
        assert run.returncode == 4 and "GPU NOT STARTED" in run.stdout, run
    result["actual_entry_refusals"] = 2
    # Fixture tree changes only service boundaries: git status and the GPU
    # wrapper are backend-dark shims. Launcher/controller bytes are unchanged.
    fixture = out / "fixture"
    scripts = fixture / "scripts"
    proof = fixture / "proofs/w44_lsmparams"
    proof.mkdir(parents=True)
    (scripts / "release").mkdir(parents=True)
    for name in ("night40.sh", "night_guard.py", "supervise.py"):
        shutil.copyfile(here / name, proof / name)
        assert (proof / name).read_bytes() == (here / name).read_bytes()
    shutil.copyfile(tree / "scripts/release/bundle_consumer.py", scripts / "release/bundle_consumer.py")
    shutil.copyfile(tree / "scripts/with_ram_slot.sh", scripts / "with_ram_slot.sh")
    (scripts / "with_ram_slot.sh").chmod(0o700)
    worker = out / "worker.py"
    worker.write_text('''import json,os,signal,subprocess,sys,time
from pathlib import Path
signal.signal(signal.SIGTERM,signal.SIG_IGN)
signal.signal(signal.SIGINT,signal.SIG_IGN)
record=Path(os.environ["W2_TEST_RECORD"])
if len(sys.argv)==1:
    child=subprocess.Popen([sys.executable,__file__,"sampler"],pass_fds=(9,))
    role="worker"
else:
    role="sampler"
with record.open("a") as f:
    f.write(json.dumps({"role":role,"pid":os.getpid(),"pgid":os.getpgrp(),"fd9":os.fstat(9).st_ino})+"\\n")
while True: time.sleep(.05)
''')
    shim = out / "bin"
    shim.mkdir()
    (shim / "git").write_text("#!/bin/sh\nexit 0\n")
    (shim / "python").write_text("#!/usr/bin/env python3\nimport os,sys\n"
        "a=sys.argv[1:]\n"
        "if a and a[0].endswith('/root_gpu.py'):\n"
        "    if os.environ.get('W2_TEST_STAGE')=='success': sys.exit(0)\n"
        "    if os.environ.get('W2_TEST_STAGE')=='e41': sys.exit(0)\n"
        "    a=[os.environ['W2_TEST_WORKER']]\n"
        "if a and a[0].endswith('/wave_cubin_gate.py'):\n"
        "    if os.environ.get('W2_TEST_STAGE')=='success': sys.exit(0)\n"
        "    import fcntl\n"
        "    fd=os.open(os.environ['W2_TEST_LOCK'],os.O_WRONLY|os.O_CREAT,0o600)\n"
        "    os.dup2(fd,9,inheritable=True); fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        "    a=[os.environ['W2_TEST_WORKER']]\n"
        "os.execv(sys.executable,[sys.executable]+a)\n")
    (scripts / "with_gpu_lock.sh").write_text('''#!/usr/bin/env bash
set -euo pipefail
[[ $1 == --label && $2 == w44-lsmparams && $3 == --timeout && $4 == 1 && $5 == -- ]]
shift 5
exec 9>"$W2_TEST_LOCK"
flock -n 9
export GPUWRF_GPU_LOCK_HELD=1 GPUWRF_GPU_LOCK_FD=9 GPUWRF_GPU_LOCK_FILE="$W2_TEST_LOCK"
printf 'dummylock\\n' >> "$W2_TEST_CALLS"
exec "$@"
''')
    (shim / "nsys").write_text("#!/bin/sh\nexit 0\n")
    for path in (shim / "git", shim / "python", shim / "nsys", scripts / "with_gpu_lock.sh"):
        path.chmod(0o700)
    real_now = int(subprocess.check_output(["bash", "<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh"]))
    env = base_env | {"PATH": str(shim) + ":" + os.environ["PATH"],
        "GA_START_UTC": stamp(real_now - 1), "GA_END_UTC": stamp(real_now + 2500),
        "W2_TEST_WORKER": str(worker)}
    # The real launcher uses fixed run paths. Redirect those only in the
    # boundary environment via a symlink-free shell path substitution.
    # Source transformation is explicit and limited to the artifact root.
    original = (proof / "night40.sh").read_text()
    old = "<USER_HOME>/wrf_gpu2_lanes/w44-lsmparams"
    assert original.count("LANE=" + old) == 1
    (proof / "night40.sh").write_text(original.replace(old, str(out / "arms")))
    result["fixture_launcher_delta"] = "artifact root only; controller and timing/topology unchanged"
    for label, ram, expire in (("top_group_term", False, False),
                               ("actual_e41_ram_scope_term", True, False),
                               ("whole_timeout", False, True)):
        record, lock = out / (label + ".pids"), out / (label + ".lock")
        local = env | {"W2_TEST_RECORD": str(record), "W2_TEST_LOCK": str(lock),
                       "W2_TEST_CALLS": str(out / (label + ".calls"))}
        if ram:
            local["W2_TEST_STAGE"] = "e41"
        # Distinct root prevents the launcher's no-reuse guard firing.
        (proof / "night40.sh").write_text(original.replace(old, str(out / label)))
        if expire:
            command = [sys.executable, str(here / "supervise.py"), "--whole-step", "--seconds", "1", "--",
                       "bash", str(proof / "night40.sh")]
        else:
            command = ["timeout", "--signal=TERM", "30", "bash", str(proof / "night40.sh")]
        t0 = time.monotonic()
        with (out / (label + ".log")).open("w") as log:
            process = subprocess.Popen(command, env=local, stdout=log, stderr=log, start_new_session=True)
            rows = []
            emergency = []
            try:
                deadline = t0 + 15
                while time.monotonic() < deadline:
                    rows = [json.loads(s) for s in record.read_text().splitlines()] if record.exists() else []
                    if len(rows) == 2 or process.poll() is not None:
                        break
                    time.sleep(.02)
                assert len(rows) == 2, (label, rows, process.poll())
                assert len({row["pgid"] for row in rows}) == 1, rows
                if not expire:
                    os.killpg(process.pid, signal.SIGTERM)
                rc = process.wait(timeout=15)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                # Only an unsuccessful fixture needs emergency exact-PID
                # cleanup. Healthy controls prove absence before this point.
                for row in rows:
                    if Path('/proc', str(row["pid"])).exists():
                        emergency.append(row["pid"])
                        print("FIXTURE EMERGENCY CLEANUP", row, file=sys.stderr)
                        os.kill(row["pid"], signal.SIGKILL)
        assert not emergency, (label, "survived before fixture cleanup", emergency)
        survivors = [row["pid"] for row in rows if Path('/proc', str(row["pid"])).exists()]
        assert not survivors, (label, survivors)
        with lock.open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert rc in ((124,) if expire else (143, -signal.SIGTERM)), (label, rc)
        result["runs"].append({"case": label, "rc": rc, "seconds": round(time.monotonic()-t0, 3),
            "processes": rows, "survivors": survivors, "dummy_lock_reacquired": True})
    success = out / "both_families_success"
    calls = out / "success.calls"
    (proof / "night40.sh").write_text(original.replace(old, str(success)))
    run = subprocess.run(["bash", str(proof / "night40.sh")], env=env | {
        "W2_TEST_STAGE": "success", "W2_TEST_LOCK": str(out / "success.lock"),
        "W2_TEST_CALLS": str(calls)}, capture_output=True, text=True, timeout=30)
    (out / "success.log").write_text(run.stdout + run.stderr)
    assert run.returncode == 0, run
    assert len(calls.read_text().splitlines()) == 6
    assert not (success / "gpu/night40/cache/lsm").exists()
    assert not (success / "gpu/night40/cache/jarvis").exists()
    result["both_families_success"] = {"rc": 0, "private_lock_calls": 6,
        "both_caches_removed": True, "gpu_or_profiler_started": False}
    (out / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
