"""v7 launch smoke: the exact entry script and REAL root_gpu.py model path on CPU.

Substitutions (only): artifact root inside the shell text, a private flock instead of
the official GPU lock, JAX_PLATFORMS cuda->cpu right before JAX import (product env
recorded and checked first), an nsys shim that runs the profiled command unprofiled,
and E41 runs the REAL wave_cubin_gate.py on the CPU cache: its verdict is recorded
but a CPU cache can never PASS native E41, so any finished verdict (rc 0/2 + receipt)
counts as 'phase reached'. Run the whole smoke inside ONE 12G RAM slot.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--entry", choices=("cuda_mini_v7.sh", "gpu_v7.sh"), required=True)
    parser.add_argument("--timeout", type=int, default=15000)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    source = here.parents[1]
    fixture, bin_dir = out / "fixture", out / "bin"
    proof = fixture / "proofs/w44_lsmparams"
    proof.mkdir(parents=True)
    (fixture / "scripts/release").mkdir(parents=True)
    bin_dir.mkdir()
    tools = {}
    for name in (args.entry, "night_guard.py", "supervise.py", "root_gpu.py", "field_table.py", "launch_smoke_model.py"):
        data = (here / name).read_bytes()
        (proof / name).write_bytes(data)
        tools[name] = hashlib.sha256(data).hexdigest()
    (fixture / "src").symlink_to(source / "src", target_is_directory=True)
    (fixture / "scripts/e41").symlink_to(source / "scripts/e41", target_is_directory=True)
    for name in ("with_ram_slot.sh", "release/bundle_consumer.py", "wave_cubin_gate.py"):
        shutil.copyfile(source / "scripts" / name, fixture / "scripts" / name)
    (fixture / "scripts/with_ram_slot.sh").chmod(0o700)
    original = (here / args.entry).read_text()
    shell = original.replace("<USER_HOME>/wrf_gpu2_lanes/w44-lsmparams", str(out / "arms"))
    # CPU compile of the real program takes 6-20 min per executable (LS62): scale ONLY the timer values
    # (GPU caps stay as sealed); every command, env assignment and phase order is unchanged.
    timers = [("timeout --foreground --signal=TERM 735", "timeout --foreground --signal=TERM 7200", 1),
              ("--seconds 720", "--seconds 7180", 1),
              ("timeout --foreground --signal=TERM 115", "timeout --foreground --signal=TERM 2400", 2),
              ("--seconds 100", "--seconds 2380", 2),
              ("WHOLE=1200\n", "WHOLE=14400\n", 1 if args.entry == "cuda_mini_v7.sh" else 0),
              ("WHOLE=2370\n", "WHOLE=20000\n", 1 if args.entry == "gpu_v7.sh" else 0)]
    for old, new, count in timers:
        assert shell.count(old) == count, (old, shell.count(old))
        shell = shell.replace(old, new)
    (proof / args.entry).write_text(shell)
    wrapper = fixture / "scripts/with_gpu_lock.sh"
    wrapper.write_text('''#!/usr/bin/env bash
set -euo pipefail
[[ $1 == --label && $2 == w44-lsmparams && $3 == --timeout && $4 == 1 && $5 == -- ]]
shift 5
exec 9>"$W2_LAUNCH_SMOKE_ROOT/private.lock"
flock -n 9
export GPUWRF_GPU_LOCK_HELD=1 GPUWRF_GPU_LOCK_FD=9 GPUWRF_GPU_LOCK_FILE="$W2_LAUNCH_SMOKE_ROOT/private.lock"
printf 'private-lock\\n' >> "$W2_LAUNCH_SMOKE_ROOT/lock_calls.txt"
exec "$@"
''')
    wrapper.chmod(0o700)
    git = bin_dir / "git"
    git.write_text("#!/usr/bin/env python3\nimport os,sys\na=sys.argv[1:]\n"
        "if a[:1]==['-C']: a=a[2:]\n"
        "os.execv('/usr/bin/git',['git','-C',os.environ['W2_SMOKE_SOURCE']]+a)\n")
    git.chmod(0o700)
    python = bin_dir / "python"
    python.write_text("#!/usr/bin/env python3\nimport json,os,subprocess,sys\nfrom pathlib import Path\n"
        "a=sys.argv[1:]\n"
        "if a and a[0].endswith('/wave_cubin_gate.py'):\n"
        "    r=subprocess.run([sys.executable]+a,capture_output=True,text=True)\n"
        "    sys.stdout.write(r.stdout);sys.stderr.write(r.stderr)\n"
        "    p=Path(a[2]);p.mkdir(parents=True,exist_ok=True)\n"
        "    files=sorted(str(x.relative_to(p)) for x in p.rglob('*') if x.is_file() and x.name!='cpu_smoke_e41.json')\n"
        "    cpu_refusal='no native CUDA ELF in executable: '+a[1] in r.stderr\n"
        "    (p/'cpu_smoke_e41.json').write_text(json.dumps({'CPU_smoke_only':True,'native_E41_pass':False,'real_tool_rc':r.returncode,'receipt_files':files,'expected_cpu_cache_refusal':cpu_refusal}))\n"
        "    sys.exit(0 if (r.returncode in (0,2) and files) or (r.returncode==1 and cpu_refusal) else 9)\n"
        "os.execv(sys.executable,[sys.executable]+a)\n")
    python.chmod(0o700)
    nsys = bin_dir / "nsys"
    nsys.write_text("#!/usr/bin/env python3\nimport os,sys\n"
        "i=sys.argv.index('python');os.execvp('python',sys.argv[i:])\n")
    nsys.chmod(0o700)
    now = int(subprocess.check_output(["bash", "<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh"]))
    stamp = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    env = dict(os.environ, JAX_PLATFORMS="cpu", W2_LAUNCH_SMOKE_ROOT=str(out), W2_SMOKE_CAPPED="1",
               W2_SMOKE_SOURCE=str(source), GA_START_UTC=stamp(now - 1), GA_END_UTC=stamp(now + args.timeout + 60),
               PATH=str(bin_dir) + ":" + os.environ["PATH"])
    started = time.monotonic()
    with (out / "launcher.log").open("w") as log:
        run = subprocess.Popen(["bash", str(proof / args.entry)], env=env, stdout=log, stderr=log,
                               start_new_session=True)
        try:
            run.wait(timeout=args.timeout)
        finally:
            if run.poll() is None:
                os.killpg(run.pid, signal.SIGTERM)
                run.wait(timeout=30)
    elapsed = time.monotonic() - started
    arm = "v7mini" if args.entry == "cuda_mini_v7.sh" else "v7"
    families = ["lsm"] if arm == "v7mini" else ["lsm", "jarvis"]
    children = []
    for family in families:
        run_dir = out / "arms/gpu" / arm / family
        for phase in ("prime", "measure", "measure_profile"):
            path = run_dir / (phase + ".json")
            if not path.exists():
                continue
            value = json.loads(path.read_text())
            assert value["product_child_env"]["JAX_PLATFORMS"] == "cuda"
            assert value["platform"] == "cpu" and value["launch_smoke"] and not value["gpu_executed"]
            assert value["same_aot_keys"] and value["changed_loaded"] and not value["changed_new_native_blobs"]
            assert value["changed_difference"]
            # Record-only GPU ON-default table (manager 05:16Z): must exist for every domain of the run.
            table = value["default_field_table"]
            assert sorted(table) == sorted(value["alignment_dt_s"]) and all(t["fields_compared"] for t in table.values())
            children.append({"family": family, "phase": phase, "path": str(path),
                             "default_byte_exact": value["default_byte_exact"],
                             "default_field_table_max_ratio": {d: t["max_ratio"] for d, t in table.items()},
                             "default_difference_leaves": len(value["default_difference"]),
                             "changed_difference_leaves": len(value["changed_difference"]),
                             "median_s": value.get("median_s"), "alignment_dt_s": value["alignment_dt_s"]})
        e41 = run_dir / "e41/cpu_smoke_e41.json"
        children.append({"family": family, "phase": "e41", "path": str(e41),
                         "value": json.loads(e41.read_text()) if e41.exists() else None})
    expected = 4 * len(families)
    result = {"verdict": "LAUNCH_SMOKE_PASS" if run.returncode == 0 and len(children) == expected else "FAIL",
              "entry": args.entry, "source_pin": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
              "tool_sha256": tools, "launcher_rc": run.returncode, "elapsed_s": elapsed, "children": children,
              "official_gpu_lock_called": False, "gpu_executed": False, "native_qualification": False,
              "substitutions": ["artifact root only in exact shell", "private device flock", "JAX_PLATFORMS cuda->cpu before JAX import",
                                "nsys shim runs the command unprofiled", "E41 = real tool on CPU cache, verdict recorded, never a native PASS",
                                "timer values scaled for CPU compile (sealed GPU caps unchanged)"]}
    if run.returncode == 0:
        with (out / "private.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for family in families:
            assert not (out / "arms/gpu" / arm / "cache" / family).exists()
    (out / "report.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "children"}, indent=2))
    assert result["verdict"] == "LAUNCH_SMOKE_PASS", result


if __name__ == "__main__":
    main()
