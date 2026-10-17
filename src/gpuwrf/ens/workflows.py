"""Prepare explicit ensemble members, then delegate admission to parallel_cases.

The source checkout's existing scheduler owns GPU placement, memory admission,
checkpoint isolation and retries. Preparing inputs never runs the model.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess

from .members import identity

GPU_LOCK_FILE = Path("/tmp/wrf_gpu2_gpu.lock")


def _case(case, namelist, hours):
    from gpuwrf.io.namelist_parser import parse_namelist
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.verify.wrfout import _times

    case, namelist = Path(case).resolve(strict=True), Path(namelist).resolve(strict=True)
    config = parse_namelist(namelist)
    domains, clock = config.get("domains", {}), config.get("time_control", {})
    def first(group, key, default=None):
        value = group.get(key, default)
        return value[0] if isinstance(value, list) else value
    count = first(domains, "max_dom", 1)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 99:
        raise ValueError("max_dom must be an integer in [1,99]")
    dt = Fraction(str(first(domains, "time_step", 0)))
    dt += Fraction(str(first(domains, "time_step_fract_num", 0)))/Fraction(str(first(domains, "time_step_fract_den", 1)))
    if dt <= 0 or (Fraction(hours*3600)/dt).denominator != 1:
        raise ValueError("hours must contain an integral number of root time steps (dt=54s requires multiples of 3h)")
    inputs, issues = [], []
    for number in range(1, count+1):
        path = case/f"wrfinput_d{number:02d}"
        with Dataset(path) as ds:
            times = _times(ds)
        if len(times) != 1:
            raise ValueError("wrfinput must contain exactly one initialization time")
        issue = times[0].to_pydatetime()
        parts = []
        for key, default in (("start_year", None), ("start_month", None), ("start_day", None),
                             ("start_hour", 0), ("start_minute", 0), ("start_second", 0)):
            value = clock.get(key, default)
            if isinstance(value, list):
                if len(value) < number:
                    raise ValueError(f"{key} must declare every domain")
                value = value[number-1]
            if value is None:
                raise ValueError(f"namelist missing {key}")
            parts.append(value)
        if datetime(*parts, tzinfo=timezone.utc) != issue:
            raise ValueError("namelist initialization disagrees with wrfinput Times")
        issues.append(issue)
        inputs.append(path)
    if len(set(issues)) != 1:
        raise ValueError("delayed domain starts are not supported by ensemble preparation")
    boundary = case/"wrfbdy_d01"
    with Dataset(boundary) as ds:
        times = _times(ds)
    if len(times) < 2 or times != sorted(set(times)) or times[0] > issues[0] or times[-1] < issues[0]+timedelta(hours=hours):
        raise ValueError("wrfbdy Times must be ordered and cover the full requested forecast")
    inputs.append(boundary)
    # Optional forcing/input files (wrffdda, wrflowinp, etc.) stay shared too.
    # Include forcing/table inputs, never histories, restart output or logs.
    prefixes = ("wrfinput_", "wrfbdy_", "wrffdda_", "wrflowinp_", "wrfndi_", "wrfsst_", "wrfchem_")
    inputs.extend(p for p in sorted(case.iterdir()) if p.is_file() and p not in inputs
                  and (p.name.startswith(prefixes) or p.suffix == ".TBL"))
    return {"case": case, "namelist": namelist, "config": config, "issue": issues[0], "inputs": inputs}


def prepare(mode, output, *, case=None, namelists=None, cases=None, hours=24, gpus=None, max_parallel=64,
            max_per_gpu=None, pool_gib=None, checkpoint_interval_steps=0, retry_failed=0):
    import math
    from gpuwrf import __version__
    from gpuwrf.doctor.repro import create_manifest

    if mode not in ("physics", "lagged", "batch"):
        raise ValueError("mode must be physics, lagged or batch")
    if isinstance(hours, bool) or not isinstance(hours, int) or hours <= 0:
        raise ValueError("hours must be a positive integer")
    if max_parallel < 1 or (max_per_gpu is not None and max_per_gpu < 1) or checkpoint_interval_steps < 0 or retry_failed < 0:
        raise ValueError("invalid concurrency/checkpoint/retry count")
    if pool_gib is not None and (not math.isfinite(pool_gib) or pool_gib <= 0):
        raise ValueError("pool_gib must be finite and positive")
    if gpus is not None and gpus != "all" and (not gpus or any(not s.strip().isdigit() for s in gpus.split(","))
                                              or len(set(gpus.split(","))) != len(gpus.split(","))):
        raise ValueError("gpus must be all or unique comma-separated device indices")
    if mode == "physics":
        if case is None or not namelists or len(namelists) < 2 or cases:
            raise ValueError("physics requires one --case and at least two --namelists")
        if len({Path(p).resolve() for p in namelists}) != len(namelists):
            raise ValueError("duplicate physics namelist")
        specs = [_case(case, n, hours) for n in namelists]
        base = {k: v for k, v in specs[0]["config"].items() if k != "physics"}
        if any({k: v for k, v in item["config"].items() if k != "physics"} != base for item in specs[1:]):
            raise ValueError("physics members may change only &physics; geometry/time/dynamics must match")
    else:
        if case or namelists or not cases or len(cases) < 2:
            raise ValueError("lagged/batch requires at least two --cases")
        if len({Path(p).resolve() for p in cases}) != len(cases):
            raise ValueError("duplicate case directory")
        specs = [_case(p, Path(p)/"namelist.input", hours) for p in cases]
    specs.sort(key=lambda s: s["issue"])  # stable: physics retains namelist order
    common_start = max(s["issue"] for s in specs)
    common_end = min(s["issue"]+timedelta(hours=hours) for s in specs)
    if mode == "lagged":
        gaps = [b["issue"]-a["issue"] for a, b in zip(specs[:-1], specs[1:])]
        if any(gap <= timedelta(0) for gap in gaps) or len(set(gaps)) != 1:
            raise ValueError("lagged members need distinct, consecutive uniformly spaced initialization times")
        if common_end <= common_start:
            raise ValueError("lagged forecast windows have no common interval")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("prepared output exists; refusing overwrite")
    # Reuse the merged `gpuwrf manifest` API once per original case. Physics
    # members share its input/source/environment provenance; their selected
    # namelist is recorded separately in each member manifest below.
    source_cases = dict.fromkeys(s["case"] for s in specs)
    native_manifests = {path: create_manifest(path) for path in source_cases}
    known_inputs = {Path(item["path"]): {"path": str(Path(item["path"]).resolve()),
                    "bytes": item["bytes"], "sha256": item["sha256"]}
                    for native in native_manifests.values() for item in native["inputs"]}
    # Avoid hashing shared large inputs again after the native manifest.
    source_paths = {p for s in specs for p in [*s["inputs"], s["namelist"]]}
    source_ids = {p: known_inputs[p] if p in known_inputs else identity(p) for p in source_paths}
    output.mkdir(parents=True, exist_ok=False)
    (output/"source_manifests").mkdir()
    source_receipts = {}
    for index, (case_path, native) in enumerate(native_manifests.items()):
        manifest_path = output/"source_manifests"/f"source-{index:03d}.json"
        manifest_path.write_text(json.dumps(native, indent=2)+"\n")
        source_receipts[case_path] = identity(manifest_path)
    members = []
    for index, spec in enumerate(specs):
        name = f"member-{index:03d}"
        dest = output/"inputs"/name
        dest.mkdir(parents=True)
        for path in spec["inputs"]:
            (dest/path.name).symlink_to(path.resolve())
        shutil.copyfile(spec["namelist"], dest/"namelist.input")
        prepared_namelist = identity(dest/"namelist.input")
        if prepared_namelist["sha256"] != source_ids[spec["namelist"]]["sha256"]:
            raise ValueError("source namelist changed during preparation")
        manifest = {"schema": "gpuwrf.ensemble.member.v1", "id": name, "mode": mode,
                    "issue": spec["issue"].isoformat(), "hours": hours, "gpuwrf_version": __version__,
                    "source_case": str(spec["case"]), "source_namelist": source_ids[spec["namelist"]],
                    "source_manifest": source_receipts[spec["case"]],
                    "prepared_namelist": prepared_namelist, "prepared_config": spec["config"],
                    "inputs": [{"name": p.name, **source_ids[p]} for p in spec["inputs"]]}
        manifest_path = dest/"member_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        members.append({"id": name, "case": str(dest), "manifest": identity(manifest_path), "issue": manifest["issue"]})
    launcher_options = ["--max-parallel", str(max_parallel), "--checkpoint-interval-steps", str(checkpoint_interval_steps),
                        "--retry-failed", str(retry_failed)]
    for flag, value in (("--gpus", gpus), ("--max-per-gpu", max_per_gpu), ("--pool-gib", pool_gib)):
        if value is not None:
            launcher_options += [flag, str(value)]
    plan = {"schema": "gpuwrf.ensemble.plan.v1", "mode": mode, "hours": hours, "members": members,
            "out_root": str(output/"runs"), "launcher_options": launcher_options,
            "cli_options": ["--domains-from-namelist", "--hours", str(hours)],
            "common_window": [common_start.isoformat(), common_end.isoformat()] if common_end > common_start else None,
            "product_alignment": "intersection" if mode == "lagged" else "exact",
            "qualification": "CPU workflow qualified; actual GPU ensemble smoke pending"}
    (output/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    return plan


def launch(plan_path, *, dry_run=False, launcher=None):
    """Check original/prepared identities before delegating to the scheduler."""
    plan = json.loads(Path(plan_path).read_text())
    if not isinstance(plan, dict) or plan.get("schema") != "gpuwrf.ensemble.plan.v1" or len(plan.get("members", [])) < 2:
        raise ValueError("not an ensemble launch plan")
    if plan.get("cli_options") != ["--domains-from-namelist", "--hours", str(plan.get("hours"))]:
        raise ValueError("launch plan CLI options disagree with prepared forecast hours")
    allowed = {"--gpus", "--max-parallel", "--max-per-gpu", "--pool-gib", "--checkpoint-interval-steps", "--retry-failed"}
    options = plan.get("launcher_options", [])
    if (not isinstance(options, list) or len(options) % 2 or not all(isinstance(x, str) for x in options)
            or any(flag not in allowed for flag in options[::2]) or len(set(options[::2])) != len(options[::2])):
        raise ValueError("invalid ensemble scheduler options")
    if len({member["case"] for member in plan["members"]}) != len(plan["members"]):
        raise ValueError("duplicate prepared member case")
    cases, identities = [], {}
    def current_identity(path):
        path = Path(path).resolve(strict=True)
        if path not in identities:
            identities[path] = identity(path)
        return identities[path]
    for member in plan["members"]:
        case = Path(member["case"])
        if current_identity(case/"member_manifest.json") != member["manifest"]:
            raise ValueError("member manifest changed since preparation")
        manifest = json.loads((case/"member_manifest.json").read_text())
        if current_identity(manifest["source_manifest"]["path"]) != manifest["source_manifest"]:
            raise ValueError("source-case reproducibility manifest changed")
        if manifest["hours"] != plan["hours"] or manifest["mode"] != plan["mode"] or manifest["id"] != member["id"]:
            raise ValueError("launch plan disagrees with the prepared member manifest")
        if current_identity(case/"namelist.input") != manifest["prepared_namelist"]:
            raise ValueError("prepared namelist changed")
        for item in manifest["inputs"]:
            if current_identity(case/item["name"]) != {k: v for k, v in item.items() if k != "name"}:
                raise ValueError("shared member input changed since preparation")
        cases.append(str(case))
    if launcher is None:
        if GPU_LOCK_FILE.exists() and os.environ.get("GPUWRF_GPU_LOCK_HELD") != "1":
            raise ValueError("hold scripts/with_gpu_lock.sh for ensemble launch, including admission dry-run")
        launcher = Path(__file__).resolve().parents[3]/"scripts"/"run_parallel_cases.sh"
    launcher = Path(launcher)
    if not launcher.is_file():
        raise ValueError("ensemble launch requires a source checkout containing scripts/run_parallel_cases.sh")
    command = [str(launcher), "--out-root", plan["out_root"], *plan["launcher_options"], *cases]
    if dry_run:
        command.append("--dry-run")
    command += ["--", *plan["cli_options"]]
    fd = os.environ.get("GPUWRF_GPU_LOCK_FD", "")
    pass_fds = ()
    if fd:
        if not fd.isdigit() or int(fd) <= 2:
            raise ValueError("invalid GPU lock descriptor")
        descriptor = int(fd)
        stat = os.fstat(descriptor)
        lock_stat = GPU_LOCK_FILE.stat()
        if (stat.st_dev, stat.st_ino) != (lock_stat.st_dev, lock_stat.st_ino):
            raise ValueError("GPU lock descriptor does not name the official lock")
        pass_fds = (descriptor,)
    # The wrapper preserves the caller's JAX platform for children and keeps
    # its own admission process CPU-only. The official lock is inherited.
    return subprocess.run(command, pass_fds=pass_fds, check=False).returncode
