"""w44-wrfdebug original-CPU-WRF startup arm (cycling heap-abort root cause).

Same E219 CPU-only env, physical cores 8,9,12, 3x1 decomposition and stop rule
(all 3 initial history files + 10 root steps, 240 s) as w44-warmstart's
cpu_startup_bisect.py. Inputs are read-only symlinks to the warmstart sets;
only the namelist keys named on the command line change. Startup diagnostic only.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, "<USER_HOME>/wrf_gpu2_lanes/w44-warmstart/CPU_BISECT_TOOL_V2")
import cpu_twin as twin  # constants/ENV/teardown only; never its post()/experiment()

WS = Path("<DATA_ROOT>/wrf_gpu2/scratch/w44-warmstart")
OUT = Path("<DATA_ROOT>/wrf_gpu2/scratch/w44-wrfdebug/runs")
EXES = {"orig": twin.BUILD / "run/wrf.exe",
        "patched": Path("<DATA_ROOT>/wrf_gpu2/scratch/w44-wrfdebug/build/patched/wrf.exe"),
        "orig_bounds": Path("<DATA_ROOT>/wrf_gpu2/scratch/w44-wrfdebug/build/orig_bounds/wrf.exe"),
        "patched_bounds": Path("<DATA_ROOT>/wrf_gpu2/scratch/w44-wrfdebug/build/patched_bounds/wrf.exe")}
SETS = {"A": WS / "may_pair/20260503_18z_a1", "C": WS / "native_may04_00_newforcing",
        "D": WS / "native_may04_00_real_schema", "B": None}
NML = {"fresh": WS / "may_pair/20260503_18z_a1/namelist.input",
       "native": WS / "native_may04_00_newforcing/namelist.input"}
CPUS = [8, 9, 12]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ranks(run):
    found = []
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                if (p / "cwd").resolve() == run and (p / "exe").resolve() == (run / "wrf.exe").resolve():
                    found.append(int(p.name))
            except (OSError, RuntimeError):
                pass
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--inputs", choices=list(SETS), required=True)
    ap.add_argument("--nml", choices=list(NML), required=True)
    ap.add_argument("--exe", choices=list(EXES), required=True)
    ap.add_argument("--cycling", choices=["keep", "true", "false"], default="keep")
    ap.add_argument("--prec-acc-dt", type=float, default=None)
    ap.add_argument("--hour1", action="store_true", help="stop after the +1 h history of all 3 domains (6 files, d01 step>=70)")
    a = ap.parse_args()
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert not Path("/tmp/wrf_gpu2_quiet").exists(), "QUIET"
    assert sorted(os.sched_getaffinity(0)) == CPUS
    meta = OUT / a.name
    run = meta / "cpu"
    assert not meta.exists(), "fresh attempt required"
    run.mkdir(parents=True)
    files = {}
    for name in [*(f"wrfinput_d{i:02}" for i in (1, 2, 3)), "wrfbdy_d01", *(f"wrflowinp_d{i:02}" for i in (1, 2, 3))]:
        if a.inputs == "B":
            src = (WS / "may_pair/previous_at_new_time" / (name.replace("wrfinput_", "wrfout_") + "_2026-05-04_00:00:00")
                   if name.startswith("wrfinput") else SETS["A"] / name)
        else:
            src = SETS[a.inputs] / name
        src = twin.safe(src)
        (run / name).symlink_to(src)
        files[name] = dict(source=str(src), sha256=sha(src))
    original = NML[a.nml].read_text()
    text = original
    for key, value in [("nproc_x", 3), ("nproc_y", 1)]:
        text, n = re.subn(r"(?m)^(\s*" + key + r"\s*=\s*)2,", lambda m: m[1] + str(value) + ",", text)
        assert n == 1
    if a.cycling != "keep":
        val = ".true." if a.cycling == "true" else ".false."
        text, n = re.subn(r"(?m)^\s*cycling\s*=.*$", " cycling = " + val + ",", text)
        if n == 0:
            text, n = re.subn(r"(?m)^&time_control\s*$", "&time_control\n cycling = " + val + ",", text)
        assert n == 1
    if a.prec_acc_dt is not None:
        assert "prec_acc_dt" not in original
        v = f"{a.prec_acc_dt:g}"
        text, n = re.subn(r"(?m)^&physics\s*$", f"&physics\n prec_acc_dt = {v}, {v}, {v},", text)
        assert n == 1
    (run / "namelist.input").write_text(text)
    (meta / "namelist.diff").write_text("".join(difflib.unified_diff(original.splitlines(True), text.splitlines(True))))
    prep = json.loads((WS / "cpu_native_may04_00_3r/PREPARE.json").read_text())
    for name, row in prep["tables"].items():
        assert sha(row["path"]) == row["sha256"], name
        (run / name).symlink_to(row["path"])
    # Thompson tables measured by warmstart arm A with the SAME executable source (sealed SHA); avoids ~65 s rebuild.
    for name in ("qr_acr_qg_V4.dat", "qr_acr_qsV2.dat", "freezeH2O.dat"):
        (run / name).symlink_to(WS / "startup_A_3r/cpu" / name)
    exe = EXES[a.exe]
    (run / "wrf.exe").symlink_to(exe)
    (run / "rank.sh").write_text("#!/bin/bash\nset -eu\nCPUS=(8 9 12)\nexec taskset -c \"${CPUS[$OMPI_COMM_WORLD_RANK]}\" nice -n 19 ./wrf.exe\n")
    (run / "rank.sh").chmod(0o755)
    prepare = dict(arm=a.name, args=vars(a), inputs=files, exe=str(exe), exe_sha256=sha(exe),
                   namelist_sha256=sha(run / "namelist.input"), launcher_sha256=sha(__file__), rank_cpus=CPUS)
    (meta / "PREPARE.json").write_text(json.dumps(prepare, indent=2) + "\n")
    env = os.environ.copy()
    env.update(twin.ENV)
    env["LD_LIBRARY_PATH"] = str(twin.WENV / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = str(twin.WENV / "bin") + ":" + env["PATH"]
    cmd = ["timeout", "--signal=TERM", "--kill-after=20", "900" if a.hour1 else "240", str(twin.WENV / "bin/mpirun"),
           "--use-hwthread-cpus", "--bind-to", "none", "-np", "3", "./rank.sh"]
    t0 = time.monotonic()
    status, detail, checked = "FAIL", "", False
    proc = None
    try:
        with (meta / "cpu.log").open("xb") as log:
            proc = subprocess.Popen(cmd, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                if not checked and len(ranks(run)) == 3:
                    rows = []
                    for pid in ranks(run):
                        p = Path("/proc") / str(pid)
                        renv = dict(x.split("=", 1) for x in (p / "environ").read_text().split("\0") if "=" in x)
                        r = int(renv["OMPI_COMM_WORLD_RANK"])
                        nv = [os.readlink(fd) for fd in (p / "fd").iterdir() if "/dev/nvidia" in os.readlink(fd)]
                        row = dict(pid=pid, rank=r, affinity=sorted(os.sched_getaffinity(pid)),
                                   libcuda=(p / "maps").read_text().count("libcuda"), nvidia_fds=nv)
                        assert row["affinity"] == [CPUS[r]] and row["libcuda"] == 0 and not nv, row
                        assert {k: renv.get(k) for k in twin.ENV} == twin.ENV
                        rows.append(row)
                    (meta / "CPU_ONLY_RANKS.json").write_text(json.dumps(rows, indent=2) + "\n")
                    checked = True
                rsl = (run / "rsl.error.0000").read_text(errors="replace") if (run / "rsl.error.0000").exists() else ""
                steps = len(re.findall(r"Timing for main:.*domain\s+1:", rsl))
                need_files, need_steps = (6, 70) if a.hour1 else (3, 10)
                if len(list(run.glob("wrfout_d0*"))) >= need_files and steps >= need_steps:
                    status, detail = "STARTUP_PASS", f"initial3history+rootsteps{steps}"
                    break
                time.sleep(1)
            if proc.poll() is not None:
                detail = "MPI exited rc=" + str(proc.returncode)
    except BaseException as exc:
        detail = repr(exc)
    finally:
        if proc is not None:
            twin.teardown(proc)
    outputs = []
    from netCDF4 import Dataset, chartostring
    for p in sorted(run.glob("wrfout_*")):
        row = dict(file=p.name, bytes=p.stat().st_size, sha256=sha(p))
        try:
            with Dataset(p) as ds:
                row.update(times=chartostring(ds["Times"][:]).tolist(), variables=len(ds.variables))
        except OSError as exc:
            row["read_error"] = str(exc)
            status = "FAIL"
        outputs.append(row)
    assert all(sha(run / n) == r["sha256"] for n, r in files.items())
    res = dict(status=status, detail=detail, process_exit=proc.returncode if proc else None, arm=a.name,
               args=vars(a), cpu_only_checked=checked, wall_s=round(time.monotonic() - t0, 2), outputs=outputs,
               inputs_byte_exact=True, scope="startup diagnostic, no forecast/fidelity", cache="NONE")
    (meta / "RESULT.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res), flush=True)
    return 0 if status == "STARTUP_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
