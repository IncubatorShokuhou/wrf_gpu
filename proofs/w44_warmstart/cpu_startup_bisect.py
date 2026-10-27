"""Manager-requested A/B/C ORIGINAL-WRF startup controls; inputs never edited.

Run in a 12 GiB scope with E219 env, physical cores 8,9,12 (or approved
four-rank fallback 8,9,12,13). Stop after all initial history writes + 10 root
steps; this is a startup diagnostic, never a forecast fidelity result.
"""
import argparse
import difflib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import cpu_twin as twin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["A", "B", "C", "D"])
    parser.add_argument("--ranks", type=int, choices=[3, 4], default=3)
    args = parser.parse_args()
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert not Path("/tmp/wrf_gpu2_quiet").exists(), "QUIET"
    cpus = [8, 9, 12, 13][:args.ranks]
    assert sorted(os.sched_getaffinity(0)) == cpus
    meta = twin.ROOT / ("startup_" + args.arm + "_" + str(args.ranks) + "r")
    run = meta / "cpu"
    assert not meta.exists(), "fresh attempt required"
    assert os.statvfs(twin.ROOT).f_bavail * os.statvfs(twin.ROOT).f_frsize >= 200 * 2**30
    run.mkdir(parents=True)
    twin.RUN, twin.META = run, meta
    fresh = twin.ROOT / "may_pair/20260503_18z_a1"
    native = twin.ROOT / "native_may04_00_newforcing"
    real_schema = twin.ROOT / "native_may04_00_real_schema"
    previous = twin.ROOT / "may_pair/previous_at_new_time"
    files = {}
    for name in [*(f"wrfinput_d{i:02}" for i in (1, 2, 3)), "wrfbdy_d01",
                 *(f"wrflowinp_d{i:02}" for i in (1, 2, 3))]:
        if args.arm in ("C", "D"):
            source = (real_schema if args.arm == "D" else native) / name
        elif args.arm == "B" and name.startswith("wrfinput"):
            source = previous / name.replace("wrfinput_", "wrfout_")
            source = source.with_name(source.name + "_2026-05-04_00:00:00")
        else:
            source = fresh / name
        source = twin.safe(source)
        (run / name).symlink_to(source)
        files[name] = {"source": str(source), "sha256": twin.sha(source), "bytes": source.stat().st_size}
    original = ((fresh if args.arm == "A" else native) / "namelist.input").read_text()
    text = original
    nx, ny = (3, 1) if args.ranks == 3 else (2, 2)
    for key, value in [("nproc_x", nx), ("nproc_y", ny)]:
        text, count = re.subn(r"(?m)^(\s*" + key + r"\s*=\s*)2,", lambda m: m[1] + str(value) + ",", text)
        assert count == 1
    (run / "namelist.input").write_text(text)
    (meta / "namelist.diff").write_text("".join(difflib.unified_diff(original.splitlines(True), text.splitlines(True))))
    original_prep = json.loads((twin.ROOT / "cpu_native_may04_00_3r/PREPARE.json").read_text())
    for name, row in original_prep["tables"].items():
        assert twin.sha(row["path"]) == row["sha256"], name
        (run / name).symlink_to(row["path"])
    # Same executable's measured A table cache avoids repeating its 65-second
    # Thompson setup. Values/hash are sealed; this never changes model inputs.
    reused_tables = {}
    if args.arm == "D":
        for name in ("qr_acr_qg_V4.dat", "qr_acr_qsV2.dat", "freezeH2O.dat"):
            source = twin.ROOT / "startup_A_3r/cpu" / name
            (run / name).symlink_to(source)
            reused_tables[name] = {"source": str(source), "sha256": twin.sha(source)}
    assert twin.sha(twin.BUILD / "run/wrf.exe") == twin.ORIGINAL_EXE_SHA
    (run / "wrf.exe").symlink_to(twin.BUILD / "run/wrf.exe")
    (run / "rank.sh").write_text("#!/bin/bash\nset -eu\nCPUS=(" + " ".join(map(str, cpus)) + ")\nexec taskset -c \"${CPUS[$OMPI_COMM_WORLD_RANK]}\" nice -n 19 ./wrf.exe\n")
    (run / "rank.sh").chmod(0o755)
    twin.write("PREPARE.json", dict(arm=args.arm, rank_cpus=cpus, inputs=files,
        executable_sha256=twin.ORIGINAL_EXE_SHA, namelist_sha256=twin.sha(run / "namelist.input"),
        namelist_source="original May0503" if args.arm == "A" else "GPU native inputset3h",
        namelist_diff_keys=["nproc_x", "nproc_y"], input_edits="NONE; readonly symlinks",
        stop="all3initialhistory+10rootsteps", cache="NONE", launcher_sha256=twin.sha(__file__),
        common_tool_sha256=twin.sha(Path(twin.__file__)), reused_tables=reused_tables))
    env = os.environ.copy()
    env.update(twin.ENV)
    env["LD_LIBRARY_PATH"] = str(twin.WENV / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = str(twin.WENV / "bin") + ":" + env["PATH"]
    cmd = ["timeout", "--signal=TERM", "--kill-after=20", "240", str(twin.WENV / "bin/mpirun"),
           "--use-hwthread-cpus", "--bind-to", "none", "-np", str(args.ranks), "./rank.sh"]
    t0 = time.monotonic()
    started = twin.utc()
    status, detail, rank_checks = "FAIL", "", 0
    def interrupt(signum, frame):
        raise KeyboardInterrupt(signum)
    signal.signal(signal.SIGTERM, interrupt)
    proc = None
    try:
        with (meta / "cpu.log").open("xb") as log:
            proc = subprocess.Popen(cmd, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            twin.write("PROCESS.json", dict(pid=proc.pid, sid=proc.pid, command=cmd, started_utc=started))
            while proc.poll() is None:
                if not rank_checks and len(twin.ranks()) == args.ranks:
                    rows = []
                    for pid in twin.ranks():
                        p = Path("/proc") / str(pid)
                        rank_env = dict(x.split("=", 1) for x in (p / "environ").read_text().split("\0") if "=" in x)
                        rank = int(rank_env["OMPI_COMM_WORLD_RANK"])
                        maps = (p / "maps").read_text()
                        nvidia = [str(fd.resolve()) for fd in (p / "fd").iterdir() if "/dev/nvidia" in str(fd.resolve())]
                        row = dict(pid=pid, rank=rank, affinity=sorted(os.sched_getaffinity(pid)),
                                   libcuda_mappings=maps.count("libcuda"), nvidia_fds=nvidia,
                                   env={k: rank_env.get(k) for k in twin.ENV})
                        assert row["affinity"] == [cpus[rank]] and row["libcuda_mappings"] == 0 and not nvidia, row
                        assert row["env"] == twin.ENV
                        rows.append(row)
                    twin.write("CPU_ONLY_RANKS.json", dict(utc=twin.utc(), ranks=rows))
                    rank_checks += 1
                rsl = (run / "rsl.error.0000").read_text(errors="replace") if (run / "rsl.error.0000").exists() else ""
                steps = len(re.findall(r"Timing for main:.*domain\s+1:", rsl))
                if len(list(run.glob("wrfout_d0*"))) >= 3 and steps >= 10:
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
    rc = proc.returncode if proc is not None else None
    outputs = []
    from netCDF4 import Dataset, chartostring
    for p in sorted(run.glob("wrfout_*")):
        row = dict(file=p.name, bytes=p.stat().st_size, sha256=twin.sha(p))
        try:
            with Dataset(p) as ds:
                row.update(times=chartostring(ds["Times"][:]).tolist(), variables=len(ds.variables))
        except OSError as exc:
            row["read_error"] = str(exc)
            status = "FAIL"
        outputs.append(row)
    identity = all(twin.sha(run / n) == r["sha256"] for n, r in files.items())
    assert identity
    result = dict(status=status, detail=detail, process_exit=rc, arm=args.arm, rank_cpus=cpus,
        cpu_only_rank_checks=rank_checks, started_utc=started, ended_utc=twin.utc(), wall_s=time.monotonic()-t0,
        inputs_byte_exact=identity, outputs=outputs, scope="startup diagnostic, no fullforecast/fidelity", cache="NONE")
    twin.write("RESULT.json", result)
    message = f"CPU-BISECT {args.arm}/{args.ranks}r {status}: {detail}; wall={result['wall_s']:.2f}s; inputsSHAexact; {meta}/RESULT.json; no forecastclaim"
    twin.post(message)
    with twin.EXP.open("a") as f:
        f.write("\t".join([twin.utc(), "w44-warmstart", "gpt-6.1-sol", f"CPU-BISECT-{args.arm}-{args.ranks}r",
            "managerA/B/CoriginalWRFstartup", json.dumps(result, separators=(",", ":")),
            "artifact=" + str(meta / "RESULT.json") + ";cache=NONE;CPUphysical" + str(cpus) + ";nice19 E21912G timeout240;source/dataUNEDITED"]) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if status == "STARTUP_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
