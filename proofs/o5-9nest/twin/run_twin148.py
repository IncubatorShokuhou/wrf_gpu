"""#148 original CPU-WRF twin (lane o5-9nest), adapted from w43-twin0227 C2_4x1/run_twin.py and w41-swddirscreen cpu_twin.sh.
No JAX, no GPU.

prepare: ONE GuardedRead session (admission A-O5-9NEST-148, proofs/o5-9nest/gread148.py) copies the reference run files
         (wrfinput/wrflowinp of the kept domains renamed to the twin numbering, wrfbdy_d01, the 3 Thompson tables the reference
         linked, reference namelist) into ROOT/cpu, verifying each sha256 against the admission; other tables + wrf.exe are
         symlinks into the build (exe sha 1d0f5779 asserted). Twin namelist = make_twin_namelist.py (n18k5: d01,d02,d04,d05,d06
         renumbered + nproc); --run-hours N (trial only) additionally sets run_hours. Writes PREPARE.json + namelist.diff.
run:     inside `scripts/with_ram_slot.sh --max <G>` (memcg asserted). One MPI rank per listed CPU (taskset per rank, nice 19),
         E219 CPU-only OpenMPI env, live per-rank proof (affinity, libcuda 0, no /dev/nvidia fds). ALISIOS registry
         `register.py wrf start/end` (--registry real | dry [lane-private ATLAS dir] | none). Hard deadline (--deadline-utc):
         budget = deadline - now - 120 s. Stops early (clean teardown of the whole MPI session) on ROOT/STOP (stop_twin.sh, for
         the ALISIOS 5-min return), on <DATA_ROOT> free < FLOOR_GIB, or at the budget. Partial runs are valid (every complete
         hourly frame counts). finish(): CPU_OUTPUTS.json (sha per file), OUTPUT_ADMISSION_LIST.txt (realpath sha256 bytes, for
         the ALISIOS output admission), cpu_run.json, registry end, MAILBOX END line.

usage: python run_twin148.py prepare --root R --mode n18k5 --nproc 3 2 [--run-hours 1]
       scripts/with_ram_slot.sh --label o5-9nest-twin --max 16G -- python run_twin148.py run --root R \
           --cpus 5,6,7,21,22,23 --deadline-utc 2026-10-25T06:00:00Z --registry real
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import gread148  # noqa: E402

BUILD = Path("<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/install_gen2_dmpar")
WENV = Path("<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build")
EXE_SHA = "1d0f5779fff143361e470e37f89c734497ed5a7bb55d0c53b7e3fccc569dd49b"
SRC_CPU = "<DATA_ROOT>/alisios/scratch/n148-cpuref-20261018_12z/cpu"
REG = "<USER_HOME>/src/alisios/ops/northstar/data_map/register.py"
LANE_DIR = Path("<USER_HOME>/wrf_gpu2_lanes/o5-9nest")
MB = Path("<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md")
FLOOR_GIB = 365  # manager disk rule 2026-10-24 (ALISIOS target >= 360)
FORBIDDEN_CPUS = {0, 2, 3, 4, 10, 11, 19, 20, 26, 27, 14, 15, 30, 31}  # ALISIOS nightly/wind300 + OS/agents
ENV = dict(JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OMPI_MCA_opal_cuda_support="0", OMPI_MCA_accelerator="null",
           OMPI_MCA_rcache="grdma", OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,sm,tcp", OMPI_MCA_coll="^ucc,hcoll",
           UCX_TLS="self,sm,tcp", HWLOC_COMPONENTS="-cuda,-nvml")
DOMS = {"n18": [f"d0{i}" for i in range(1, 10)], "n18k5": ["d01", "d02", "d04", "d05", "d06"]}
T0 = datetime(2026, 10, 18, 18, tzinfo=timezone.utc)


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def free_gib(p="<DATA_ROOT>"):
    s = os.statvfs(p)
    return s.f_bavail * s.f_frsize / 2**30


def post(msg):
    line = f"{utc()} o5-9nest: {msg}"
    print(line, flush=True)
    with MB.open("a") as f:
        f.write(line + "\n")


def source_files(mode):
    """admitted realpath -> name in the twin run dir."""
    m = {}
    for i, d in enumerate(DOMS[mode], 1):
        for kind in ("wrfinput", "wrflowinp"):
            m[os.path.realpath(f"{SRC_CPU}/{kind}_{d}")] = f"{kind}_d{i:02d}"
    m[os.path.realpath(f"{SRC_CPU}/wrfbdy_d01")] = "wrfbdy_d01"
    for t in ("freezeH2O.dat", "qr_acr_qg_V4.dat", "qr_acr_qsV2.dat"):
        m[os.path.realpath(f"{SRC_CPU}/{t}")] = t
    m[os.path.realpath(f"{SRC_CPU}/namelist.input")] = "reference_namelist.input"
    return m


def prepare(a):
    root = Path(a.root)
    run_dir = root / "cpu"
    assert not run_dir.exists(), "fresh attempt required"
    assert not Path("/tmp/wrf_gpu2_quiet").exists(), "QUIET"
    assert free_gib() >= FLOOR_GIB + 1, f"<DATA_ROOT> free {free_gib():.0f} GiB < {FLOOR_GIB}+1"
    assert sha(BUILD / "run/wrf.exe") == EXE_SHA
    files = source_files(a.mode)
    sess, rdir, listed = gread148.session(sorted(files), f"STAGE_{root.name}")
    run_dir.mkdir(parents=True)
    staged = {}
    with sess:
        for src, name in files.items():
            h = hashlib.sha256()
            with open(src, "rb") as fi, open(run_dir / name, "xb") as fo:
                for b in iter(lambda: fi.read(4 << 20), b""):
                    h.update(b)
                    fo.write(b)
            if h.hexdigest() != listed[src]["sha256"]:
                raise ValueError(f"sha mismatch vs admission: {src}")
            staged[name] = {"source": src, "sha256": h.hexdigest()}
    gread148.close_receipt(sess, rdir, {"n_staged": len(staged)})
    tables = {}
    for p in sorted((BUILD / "run").iterdir()):
        if p.name in ("namelist.input", "wrf.exe", "real.exe", "ndown.exe", "tc.exe") or p.name in staged \
                or p.name.startswith(("wrfinput", "wrfbdy", "wrflowinp", "wrfout", "wrfrst", "rsl.")) or not p.is_file():
            continue
        (run_dir / p.name).symlink_to(p.resolve())
        tables[p.name] = sha(p)
    (run_dir / "wrf.exe").symlink_to((BUILD / "run/wrf.exe").resolve())
    ref_nl = run_dir / "reference_namelist.input"
    out = subprocess.run([sys.executable, str(HERE / "make_twin_namelist.py"), str(ref_nl), str(run_dir / "namelist.input"),
                          a.mode, str(a.nproc[0]), str(a.nproc[1])], check=True, capture_output=True, text=True).stdout
    if a.run_hours is not None or a.run_minutes is not None:  # trial only
        nl = (run_dir / "namelist.input").read_text()
        for key, val in (("run_hours", a.run_hours or 0), ("run_minutes", a.run_minutes or 0)):
            nl, n = re.subn(r"(?m)^(\s*" + key + r"\s*=\s*)\d+\s*,", lambda m: f"{m.group(1)}{val},", nl)
            assert n == 1, key
        (run_dir / "namelist.input").write_text(nl)
    (root / "namelist.diff").write_text("".join(difflib.unified_diff(
        ref_nl.read_text().splitlines(True), (run_dir / "namelist.input").read_text().splitlines(True), "reference", "twin")))
    trial_min = None if a.run_hours is None and a.run_minutes is None else 60 * (a.run_hours or 0) + (a.run_minutes or 0)
    rec = dict(prepared_utc=utc(), mode=a.mode, nproc=a.nproc, run_hours_override=a.run_hours, trial_minutes=trial_min, domains_ref=DOMS[a.mode],
               staged=staged, tables=tables, executable_sha256=EXE_SHA, twin_namelist_sha256=sha(run_dir / "namelist.input"),
               namelist_receipt=json.loads(out), admission=gread148.ADMISSION, stage_receipts=str(rdir),
               launcher_sha256=sha(__file__), builder_sha256=sha(HERE / "make_twin_namelist.py"), cache="NONE")
    (root / "PREPARE.json").write_text(json.dumps(rec, indent=1))
    print("PREPARED", run_dir, "staged", len(staged), "tables", len(tables))


def ranks(run_dir):
    found = []
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                if (p / "cwd").resolve() == run_dir and (p / "exe").resolve() == (run_dir / "wrf.exe").resolve():
                    found.append(int(p.name))
            except (OSError, RuntimeError):
                pass
    return sorted(found)


def inspect(run_dir, cpus):
    rows = []
    for pid in ranks(run_dir):
        p = Path("/proc") / str(pid)
        env = dict(x.split("=", 1) for x in (p / "environ").read_text().split("\0") if "=" in x)
        devs = []
        for fd in (p / "fd").iterdir():
            try:
                t = os.readlink(fd)
            except OSError:
                continue
            if "/dev/nvidia" in t:
                devs.append(t)
        r = int(env["OMPI_COMM_WORLD_RANK"])
        row = dict(pid=pid, rank=r, affinity=sorted(os.sched_getaffinity(pid)), nice=os.getpriority(os.PRIO_PROCESS, pid),
                   libcuda=(p / "maps").read_text().count("libcuda"), nvidia_fds=devs, env={k: env.get(k) for k in ENV})
        assert row["affinity"] == [cpus[r]] and row["libcuda"] == 0 and not devs and row["env"] == ENV and row["nice"] == 19, row
        rows.append(row)
    assert len(rows) == len(cpus), rows
    return dict(utc=utc(), ranks=rows)


def session_pgroups(sid):
    groups = set()
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                if os.getsid(int(p.name)) == sid:
                    groups.add(os.getpgid(int(p.name)))
            except (ProcessLookupError, PermissionError):
                pass
    return groups


def teardown(proc):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        groups = session_pgroups(proc.pid)
        for g in groups:
            try:
                os.killpg(g, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM and groups:
            for _ in range(30):
                if not session_pgroups(proc.pid):
                    break
                time.sleep(1)
    proc.wait(timeout=120)


def registry(action, mode, rid, root, **kw):
    if mode == "none":
        return 0
    env = os.environ.copy()
    if mode == "dry":
        env["ALISIOS_ATLAS_DIR"] = str(LANE_DIR / "registry_dry")
        env["ALISIOS_DATA_MAP_NO_REGEN"] = "1"
        Path(env["ALISIOS_ATLAS_DIR"]).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-B", REG, "wrf", action, "--run-id", rid, "--by", "wrf_gpu2-o5-9nest"]
    for k, v in kw.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (root / f"registry_{action}.log").write_text(r.stdout + r.stderr)
    return r.returncode


def finish(root, run_dir, rc, started, t0, reason, prep, a, rid):
    from netCDF4 import Dataset, chartostring
    rows = []
    for p in sorted(run_dir.glob("wrfout_d0*")):
        try:
            with Dataset(p) as ds:
                times = chartostring(ds["Times"][:]).tolist()
        except OSError as exc:  # a frame being written when the run stopped
            rows.append(dict(file=p.name, incomplete=str(exc)))
            continue
        rows.append(dict(path=str(p.resolve()), file=p.name, bytes=p.stat().st_size, sha256=sha(p), times=times))
    (root / "CPU_OUTPUTS.json").write_text(json.dumps(rows, indent=1))
    (root / "OUTPUT_ADMISSION_LIST.txt").write_text("".join(f"{r['path']}\t{r['sha256']}\t{r['bytes']}\n" for r in rows if "path" in r))
    rsl = (run_dir / "rsl.out.0000").read_text() if (run_dir / "rsl.out.0000").exists() else ""
    complete = sum("path" in r for r in rows)
    tm = prep.get("trial_minutes")
    expected = len(prep["domains_ref"]) * ((72 if tm is None else tm // 60) + 1)
    success = rc == 0 and "SUCCESS COMPLETE WRF" in rsl and complete == expected
    status = "DONE_CLEAN" if success else ("ABORTED" if reason in ("budget", "deadline", "stop_request", "floor") else "FAILED")
    res = dict(rc=rc, status=status, stop_reason=reason, success_complete="SUCCESS COMPLETE WRF" in rsl, started_utc=started,
               ended_utc=utc(), wall_s=time.monotonic() - t0, wrfout_complete=complete, expected_wrfout=expected,
               outputs_sha256=sha(root / "CPU_OUTPUTS.json"), output_admission_list=str(root / "OUTPUT_ADMISSION_LIST.txt"),
               run_id=rid, registry=a.registry, cache="NONE")
    (root / "cpu_run.json").write_text(json.dumps(res, indent=1))
    reg_rc = registry("end", a.registry, rid, root, ended_utc=res["ended_utc"], rc=rc, status=status, corpus="none",
                      raw_deleted="nein", notes=f"wrf_gpu2 o5-9nest #148 CPU twin {prep['mode']} nproc {prep['nproc']}; "
                      f"{complete}/{expected} hourly frames; raw kept in wrf_gpu2 scratch until scored, outputs listed in "
                      f"{root}/OUTPUT_ADMISSION_LIST.txt")
    post(f"TWIN END {'TRIAL ' if prep.get('trial_minutes') is not None else ''}rc={rc} status={status} reason={reason} "
         f"wrfout={complete}/{expected} wall={res['wall_s']:.0f}s registry_end_rc={reg_rc} run_id={rid} root={root}")
    return 0 if status in ("DONE_CLEAN", "ABORTED") else 1


def run(a):
    root = Path(a.root)
    run_dir = (root / "cpu").resolve()
    cpus = [int(c) for c in a.cpus.split(",")]
    assert not set(cpus) & FORBIDDEN_CPUS, f"forbidden cores {set(cpus) & FORBIDDEN_CPUS}"
    prep = json.loads((root / "PREPARE.json").read_text())
    assert prep["nproc"][0] * prep["nproc"][1] == len(cpus), "ranks != nproc_x*nproc_y"
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    assert not Path("/tmp/wrf_gpu2_quiet").exists(), "QUIET"
    assert not list(run_dir.glob("wrfout_*")), "refusing restart/overwrite"
    assert not (root / "STOP").exists(), "STOP file present"
    assert sha(run_dir / "wrf.exe") == EXE_SHA and sha(run_dir / "namelist.input") == prep["twin_namelist_sha256"]
    assert free_gib() >= FLOOR_GIB + a.need_gib, f"<DATA_ROOT> free {free_gib():.0f} GiB < floor {FLOOR_GIB} + {a.need_gib}"
    cg = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().strip().split("::")[1].lstrip("/")
    mem_max = (cg / "memory.max").read_text().strip()
    assert mem_max != "max", "run inside with_ram_slot.sh (memcg cap required)"
    deadline = datetime.strptime(a.deadline_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    budget = int(deadline - time.time() - 120)
    assert budget >= 300, f"budget {budget}s too small"
    rid = f"wrf_gpu2-o5-9nest:20261018_12z:twin_{prep['mode']}_{prep['nproc'][0]}x{prep['nproc'][1]}" \
          f"{':trial' if prep.get('trial_minutes') is not None else ''}:{root.name}"
    if a.registry == "dry":
        rid += f":dry{datetime.now(timezone.utc):%H%M%S}"
    (run_dir / "rank.sh").write_text("#!/bin/bash\nset -eu\nCPUS=(" + " ".join(map(str, cpus)) + ")\n"
                                     'exec taskset -c "${CPUS[$OMPI_COMM_WORLD_RANK]}" nice -n 19 ./wrf.exe\n')
    (run_dir / "rank.sh").chmod(0o755)
    env = os.environ.copy()
    env.update(ENV)
    env["LD_LIBRARY_PATH"] = str(WENV / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = str(WENV / "bin") + ":" + env["PATH"]
    cmd = ["timeout", "--signal=TERM", "--kill-after=30", str(budget), str(WENV / "bin/mpirun"),
           "--use-hwthread-cpus", "--bind-to", "none", "-np", str(len(cpus)), "./rank.sh"]
    started, t0 = utc(), time.monotonic()
    rrc = registry("start", a.registry, rid, root, producer="wrf_gpu2-o5-9nest",
                   purpose="ALISIOS #148 noise floor: original CPU-WRF 4.7.1 decomposition twin of the 9-nest CPU reference "
                           f"({prep['mode']}: d01,d02,d04,d05,d06; nproc {prep['nproc']}); no product, no stations",
                   family="wn3_9nest_cpu_twin", tags="wrf_gpu2,o5-9nest,148,cpu_twin,cpu_wrf_4.7.1_dmpar",
                   forcing="WN3+AIFS3D", engine=f"WRF V4.7.1 dmpar original CPU (wrf.exe sha 1d0f5779), {len(cpus)} ranks",
                   nests="9/3/1 km (d01,d02 + 1 km leaves LP,LG,EH)", lead_h=72 if prep.get("trial_minutes") is None else max(1, prep["trial_minutes"] // 60),
                   issue="20261018_12z", run_dir=str(run_dir), retention="raw_delete_after_extract", started_utc=started)
    assert rrc == 0, f"registry start failed rc={rrc} (see {root}/registry_start.log)"
    (root / "LAUNCH.json").write_text(json.dumps(dict(started_utc=started, command=cmd, env=ENV, cpus=cpus, run_id=rid,
                                                     registry=a.registry, deadline_utc=a.deadline_utc, budget_s=budget,
                                                     memory_max=mem_max, launcher_sha256=sha(__file__)), indent=1))
    proc, reason, rc = None, "model", None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(signum)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with open(root / "cpu.log", "xb") as log:
            proc = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (root / "PROCESS.json").write_text(json.dumps(dict(supervisor_pid=os.getpid(), sid=proc.pid, started_utc=started)))
            deadline_start = time.monotonic() + 180
            while len(ranks(run_dir)) != len(cpus):
                assert proc.poll() is None, "MPI exited before all ranks started"
                assert time.monotonic() < deadline_start, "rank startup timed out"
                time.sleep(1)
            time.sleep(10)
            (root / "CPU_ONLY_RANKS.json").write_text(json.dumps(inspect(run_dir, cpus), indent=1))
            post(f"TWIN START {'TRIAL ' if prep.get('trial_minutes') is not None else ''}mode={prep['mode']} nproc={prep['nproc']} "
                 f"ranks={len(cpus)} cpus={a.cpus} libcuda=0x{len(cpus)} nvidia_fds=0 nice19 memcg={mem_max} budget={budget}s "
                 f"deadline={a.deadline_utc} run_id={rid} root={root} (stop: touch {root}/STOP)")
            nxt, checked = time.monotonic() + 60, False
            while proc.poll() is None:
                time.sleep(5)
                if (root / "STOP").exists():
                    reason = "stop_request"
                    break
                if not checked and len(list(run_dir.glob("wrfout_d0*"))) > len(prep["domains_ref"]):
                    (root / "CPU_ONLY_AFTER_HISTORY.json").write_text(json.dumps(inspect(run_dir, cpus), indent=1))
                    checked = True
                if time.monotonic() >= nxt:
                    n = len(list(run_dir.glob("wrfout_d01_*")))
                    (root / "PROGRESS.json").write_text(json.dumps(dict(
                        utc=utc(), wall_s=round(time.monotonic() - t0, 1), wrfout=len(list(run_dir.glob("wrfout_d0*"))),
                        d01_frames=n, memory_peak=(cg / "memory.peak").read_text().strip() if (cg / "memory.peak").exists() else None,
                        free_gib=round(free_gib(), 1)), indent=1))
                    if free_gib() < FLOOR_GIB:
                        reason = "floor"
                        break
                    nxt = time.monotonic() + 60
            rc = proc.poll()
            if rc == 124:
                reason = "budget"
    finally:
        if proc is not None:
            teardown(proc)
            rc = proc.returncode if rc is None else rc
    return finish(root, run_dir, rc, started, t0, reason, prep, a, rid)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "run"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", choices=sorted(DOMS), default="n18k5")
    ap.add_argument("--nproc", type=int, nargs=2, default=[3, 2])
    ap.add_argument("--run-hours", type=int, default=None, help="TRIAL ONLY: override run_hours")
    ap.add_argument("--run-minutes", type=int, default=None, help="TRIAL ONLY: override run_minutes (run_hours -> given or 0)")
    ap.add_argument("--cpus")
    ap.add_argument("--deadline-utc")
    ap.add_argument("--registry", choices=["real", "dry", "none"], default="dry")
    ap.add_argument("--need-gib", type=float, default=6.0, help="forecast output size + margin, checked against the floor")
    a = ap.parse_args()
    raise SystemExit(prepare(a) if a.action == "prepare" else run(a))
