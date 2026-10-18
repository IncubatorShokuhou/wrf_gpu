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
# Track P per-member control file (gpuwrf optimize continuous): linked into the member directory like the forcing,
# identity-checked at launch, passed by the launcher as that member's --surface-controls.
SURFACE_CONTROLS_NAME = "surface_controls.npz"


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
            max_per_gpu=None, pool_gib=None, checkpoint_interval_steps=0, retry_failed=0,
            members=None, schemes=None, lengthscale_km=None, nens_start=1, physics_rows=None, single_batch=False,
            host_gb_per_case=None, surface_controls=None):
    import math
    import tempfile
    from gpuwrf import __version__
    from gpuwrf.doctor.repro import create_manifest

    if mode not in ("physics", "lagged", "batch", "stochastic", "multiphysics"):
        raise ValueError("mode must be physics, lagged, batch, stochastic or multiphysics")
    if mode not in ("stochastic", "multiphysics") and (members is not None or schemes is not None or physics_rows
                                                      or lengthscale_km is not None or nens_start != 1):
        raise ValueError("--members/--schemes/--lengthscale-km/--nens-start/--physics-rows need a preset mode")
    if isinstance(hours, bool) or not isinstance(hours, int) or hours <= 0:
        raise ValueError("hours must be a positive integer")
    if max_parallel < 1 or (max_per_gpu is not None and max_per_gpu < 1) or checkpoint_interval_steps < 0 or retry_failed < 0:
        raise ValueError("invalid concurrency/checkpoint/retry count")
    if pool_gib is not None and (not math.isfinite(pool_gib) or pool_gib <= 0):
        raise ValueError("pool_gib must be finite and positive")
    if gpus is not None and gpus != "all" and (not gpus or any(not s.strip().isdigit() for s in gpus.split(","))
                                              or len(set(gpus.split(","))) != len(gpus.split(","))):
        raise ValueError("gpus must be all or unique comma-separated device indices")
    preset_info, preset_dir = None, None
    if mode in ("stochastic", "multiphysics"):
        from gpuwrf.io.namelist_parser import parse_namelist
        from .presets import build_members, check_member_config, parse_start

        if case is None or namelists or cases:
            raise ValueError(f"{mode} preset requires one --case (members are generated from its namelist)")
        source_namelist = Path(case).resolve(strict=True)/"namelist.input"
        base_text = source_namelist.read_text(encoding="utf-8")
        base_config = parse_namelist(source_namelist)
        built = build_members(mode, base_text, base_config, members=members, schemes=schemes,
                              lengthscale_km=lengthscale_km, nens_start=nens_start, rows=physics_rows,
                              start=parse_start(base_config))
        preset_dir = tempfile.TemporaryDirectory(prefix="gpuwrf-ens-preset-")
        namelists, preset_info = [], []
        for index, (text, info) in enumerate(built):
            path = Path(preset_dir.name)/f"member-{index:03d}.namelist.input"
            path.write_text(text, encoding="utf-8")
            check_member_config(mode, base_config, parse_namelist(path),
                                () if mode == "multiphysics" and not schemes else (schemes or ("preset",)))
            namelists.append(path)
            preset_info.append(info)
    if mode in ("physics", "stochastic", "multiphysics"):
        if case is None or not namelists or len(namelists) < 2 or cases:
            raise ValueError("physics requires one --case and at least two --namelists")
        if len({Path(p).resolve() for p in namelists}) != len(namelists):
            raise ValueError("duplicate physics namelist")
        specs = [_case(case, n, hours) for n in namelists]
        free = {"physics": {"physics"}, "stochastic": {"stoch"}, "multiphysics": {"physics", "stoch"}}[mode]
        base = {k: v for k, v in specs[0]["config"].items() if k not in free}
        if any({k: v for k, v in item["config"].items() if k not in free} != base for item in specs[1:]):
            raise ValueError(f"{mode} members may change only " + "/".join("&"+g for g in sorted(free))
                             + "; geometry/time/dynamics must match")
    else:
        # single_batch: a scheduler (gpuwrf optimize) may submit one remaining batch member.
        if case or namelists or not cases or len(cases) < (1 if single_batch and mode == "batch" else 2):
            raise ValueError("lagged/batch requires at least two --cases")
        if len({Path(p).resolve() for p in cases}) != len(cases):
            raise ValueError("duplicate case directory")
        specs = [_case(p, Path(p)/"namelist.input", hours) for p in cases]
    if surface_controls is not None:
        # One control file per batch case, all or none (one program signature per batch).
        if mode != "batch" or len(surface_controls) != len(specs):
            raise ValueError("surface_controls needs batch mode and one control file per case")
        for spec, path in zip(specs, surface_controls):
            path = Path(path).resolve(strict=True)
            if path.name != SURFACE_CONTROLS_NAME or not path.is_file():
                raise ValueError(f"member control files must be regular files named {SURFACE_CONTROLS_NAME}")
            spec["inputs"].append(path)
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
    if preset_dir is not None:
        # Keep the generated namelists with the plan; the temporary copies vanish.
        (output/"preset_namelists").mkdir()
        for spec in specs:
            kept = output/"preset_namelists"/spec["namelist"].name
            shutil.copyfile(spec["namelist"], kept)
            if identity(kept)["sha256"] != source_ids[spec["namelist"]]["sha256"]:
                raise ValueError("generated preset namelist changed during preparation")
            source_ids[kept] = identity(kept)
            spec["namelist"] = kept
        preset_dir.cleanup()
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
        if preset_info is not None:
            manifest["preset"] = preset_info[index]
        manifest_path = dest/"member_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        members.append({"id": name, "case": str(dest), "manifest": identity(manifest_path), "issue": manifest["issue"]})
    launcher_options = ["--max-parallel", str(max_parallel), "--checkpoint-interval-steps", str(checkpoint_interval_steps),
                        "--retry-failed", str(retry_failed)]
    if host_gb_per_case is not None and (not math.isfinite(host_gb_per_case) or host_gb_per_case <= 0):
        raise ValueError("host_gb_per_case must be finite and positive")
    for flag, value in (("--gpus", gpus), ("--max-per-gpu", max_per_gpu), ("--pool-gib", pool_gib),
                        ("--host-gb-per-case", host_gb_per_case),
                        ("--case-surface-controls", None if surface_controls is None else SURFACE_CONTROLS_NAME)):
        if value is not None:
            launcher_options += [flag, str(value)]
    plan = {"schema": "gpuwrf.ensemble.plan.v1", "mode": mode, "hours": hours, "members": members,
            "out_root": str(output/"runs"), "launcher_options": launcher_options,
            "cli_options": ["--domains-from-namelist", "--hours", str(hours)],
            "common_window": [common_start.isoformat(), common_end.isoformat()] if common_end > common_start else None,
            "product_alignment": "intersection" if mode == "lagged" else "exact",
            "qualification": "CPU workflow qualified; actual GPU ensemble smoke pending"}
    if single_batch and mode == "batch":
        plan["min_members"] = 1
    if preset_info is not None:
        from .presets import DEFAULT_SCHEMES, MULTIPHYSICS
        plan["preset"] = {"mode": mode, "schemes": list(schemes or (() if mode == "multiphysics" else DEFAULT_SCHEMES)),
                          "lengthscale_km": lengthscale_km, "nens_start": nens_start,
                          "convention": "identical iseed_*, distinct nens per member (WRF rand_seed mixes iseed, nens, start date)",
                          "members": [{"id": m["id"], **info} for m, info in zip(members, preset_info)]}
        if mode == "multiphysics":
            plan["preset"]["table"] = [{"row": r[0], "physics": r[1], "evidence_case": r[2], "e41": r[3]} for r in MULTIPHYSICS]
        plan["qualification"] = ("CPU workflow qualified; per-scheme/per-row model qualification in docs/stochastic.md "
                                 "and docs/ensemble.md; ensemble spread/skill/calibration not qualified")
    (output/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    return plan


def launch(plan_path, *, dry_run=False, launcher=None, env=None):
    """Check original/prepared identities before delegating to the scheduler.

    ``env`` (default: this process's environment) is the members' environment, e.g. the caller's
    JAX platform while a CPU-only driver prepares them; the lock policy follows it.
    """
    environ = os.environ if env is None else env
    plan = json.loads(Path(plan_path).read_text())
    minimum = 1 if isinstance(plan, dict) and plan.get("mode") == "batch" and plan.get("min_members") == 1 else 2
    if not isinstance(plan, dict) or plan.get("schema") != "gpuwrf.ensemble.plan.v1" or len(plan.get("members", [])) < minimum:
        raise ValueError("not an ensemble launch plan")
    if plan.get("cli_options") != ["--domains-from-namelist", "--hours", str(plan.get("hours"))]:
        raise ValueError("launch plan CLI options disagree with prepared forecast hours")
    allowed = {"--gpus", "--max-parallel", "--max-per-gpu", "--pool-gib", "--checkpoint-interval-steps", "--retry-failed",
               "--host-gb-per-case", "--case-surface-controls"}
    options = plan.get("launcher_options", [])
    if (not isinstance(options, list) or len(options) % 2 or not all(isinstance(x, str) for x in options)
            or any(flag not in allowed for flag in options[::2]) or len(set(options[::2])) != len(options[::2])):
        raise ValueError("invalid ensemble scheduler options")
    if len({member["case"] for member in plan["members"]}) != len(plan["members"]):
        raise ValueError("duplicate prepared member case")
    controls = dict(zip(options[::2], options[1::2])).get("--case-surface-controls")
    if controls is not None and controls != SURFACE_CONTROLS_NAME:
        raise ValueError(f"member control files are named {SURFACE_CONTROLS_NAME}")
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
        if controls is not None and SURFACE_CONTROLS_NAME not in {item["name"] for item in manifest["inputs"]}:
            raise ValueError("member has no prepared control file")
        cases.append(str(case))
    if launcher is None:
        cpu_only = environ.get("JAX_PLATFORMS", "").strip().lower() == "cpu"  # members never touch the GPU
        if GPU_LOCK_FILE.exists() and environ.get("GPUWRF_GPU_LOCK_HELD") != "1" and not cpu_only:
            raise ValueError("hold scripts/with_gpu_lock.sh for ensemble launch, including admission dry-run")
        launcher = Path(__file__).resolve().parents[3]/"scripts"/"run_parallel_cases.sh"
    launcher = Path(launcher)
    if not launcher.is_file():
        raise ValueError("ensemble launch requires a source checkout containing scripts/run_parallel_cases.sh")
    command = [str(launcher), "--out-root", plan["out_root"], *plan["launcher_options"], *cases]
    if dry_run:
        command.append("--dry-run")
    command += ["--", *plan["cli_options"]]
    fd = environ.get("GPUWRF_GPU_LOCK_FD", "")
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
    return subprocess.run(command, pass_fds=pass_fds, check=False, env=None if env is None else dict(env)).returncode
