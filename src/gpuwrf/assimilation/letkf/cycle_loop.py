"""DA cycle loop: N d01 members per window in ONE process, analysis between windows (lane w41-da-cycle).

Layout under ``out`` (every directory is new; nothing is overwritten)::

    c00/shared/cases/<m>   first-window cases (cycle_case.write_cycle_case, land='cold'), all arms
    c00/shared/fc/<m>      member forecasts of window 0 (wrfout + proofs + receipts)
    cKK/<arm>/prep         next-window member analyses + cases (member_cycle.write_member_cycle manifest)
    cKK/<arm>/fc/<m>       member forecasts of window KK for that arm
    streams/<arm>/<m>      one continuing history stream per member (restart seam)
    checkpoints/<arm>/<m>  its restart generations (end of each window + injected analysis)
    CYCLE_RECEIPT.json

Window 0 is shared by every arm (all arms start from the same initial ensemble): it runs once per member,
checkpointed in the first restart-seam arm's stream when there is one. Arms:
``zero_update`` (the control arm: each member restarts from its own background frame through the same
file seam, i.e. analysis == background) and ``letkf`` (the a1-da/w41-da-core analysis CLI on the member
frames with the arm's observation/parameter config; it writes the next-window cases itself).

Member forecasts run in this process through :class:`member_driver.MemberDriver` (executables shared
across members, arms and windows). Analyses and case writing are host-only and run in a CPU child
process (``JAX_PLATFORMS=cpu``): they never touch the GPU, and the case admission refuses any other
backend. Every case of the loop is written through ``write_cycle_case`` with the same ``forcing_hours``,
so every window, arm and seam carries the same wrfbdy record count and shares ONE step program (the record
count is a program shape until the w41-bdyrec fix).

Seams: ``file`` (default) restarts each member from its analysis wrfinput (the port re-initialises held
physics, as WRF does from a wrfinput). ``restart`` keeps each member's model state: the analysis increment
is added to the member's end-of-window restart generation (``restart_inject``) and the same stream resumes
with ``--resume-checkpoint --extend-run``; a zero-update restart arm is the continuous forecast.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from .member_driver import (MemberDriver, members_from_manifest, refuse_protected_inputs, refuse_protected_opens,
                            safe_label, sha256)

ARM_KINDS = ("zero_update", "letkf")
SEAMS = ("file", "restart")
WRF_TIME = "%Y-%m-%d_%H:%M:%S"


def _cpu_child(argv: list[str], log: Path) -> dict[str, Any]:
    """Run a host-only step in a fresh CPU process; never inherits a GPU backend."""
    import gpuwrf

    src = str(Path(gpuwrf.__file__).resolve().parents[1])
    env = {k: v for k, v in os.environ.items() if not k.startswith(("XLA_PYTHON_CLIENT", "CUDA_VISIBLE"))}
    env.update(JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""))
    start = time.perf_counter()
    with log.open("w") as stream:
        rc = subprocess.run([sys.executable, *argv], env=env, cwd=src, stdout=stream, stderr=subprocess.STDOUT).returncode
    wall = time.perf_counter() - start
    if rc != 0:
        raise RuntimeError(f"CPU step failed rc={rc}: {' '.join(argv)} (log {log})")
    return {"argv": argv, "rc": rc, "wall_seconds": wall, "log": str(log)}


CYCLE_SCHEMA = "gpuwrf.da.cycle.v1"
_SECTION_KEYS = {"source_case", "t0", "window_hours", "cycles", "arms", "initial_members", "initial_cases",
                 "forcing_hours", "template_wrfinput", "share_executables", "require_platform"}
_ARM_KEYS = {"kind", "seam", "experiment"}


def _positive_int(where, value, *, multiple=1):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % multiple:
        raise ValueError(f"{where} must be a positive integer" + (f" multiple of {multiple}" if multiple > 1 else ""))
    return value


def _abs_path(where, value):
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{where} must be an absolute path, got {value!r}")
    return value


def validate_config_section(section) -> dict[str, Any]:
    """Validate/normalise the ``cycle`` section (delegated by ``letkf.config`` for gpuwrf.da.v1 files)."""
    from .config import iso, utc

    if not isinstance(section, dict):
        raise ValueError("cycle must be an object")
    unknown = sorted(set(section) - _SECTION_KEYS)
    if unknown:
        raise ValueError(f"cycle: unknown key(s) {unknown}; allowed {sorted(_SECTION_KEYS)}")
    for key in ("source_case", "t0", "window_hours", "cycles", "arms"):
        if key not in section:
            raise ValueError(f"cycle: missing {key!r}")
    out = {"source_case": _abs_path("cycle.source_case", section["source_case"]),
           "t0": iso(utc("cycle.t0", section["t0"])),
           "window_hours": _positive_int("cycle.window_hours", section["window_hours"], multiple=3),
           "cycles": _positive_int("cycle.cycles", section["cycles"])}
    arms = section["arms"]
    if not isinstance(arms, dict) or not arms:
        raise ValueError("cycle.arms must be a non-empty object")
    out["arms"] = {}
    for name, arm in arms.items():
        if name == "shared":
            raise ValueError("cycle.arms: arm name 'shared' is reserved")
        safe_label("cycle.arms", name)
        if not isinstance(arm, dict) or set(arm) - _ARM_KEYS or arm.get("kind") not in ARM_KINDS:
            raise ValueError(f"cycle.arms.{name}: keys {sorted(_ARM_KEYS)}, kind in {ARM_KINDS}")
        seam = arm.get("seam", "file")
        if seam not in SEAMS:
            raise ValueError(f"cycle.arms.{name}.seam must be one of {SEAMS}")
        entry = {"kind": arm["kind"], "seam": seam}
        if "experiment" in arm:
            if arm["kind"] != "letkf" or not isinstance(arm["experiment"], str):
                raise ValueError(f"cycle.arms.{name}.experiment names a letkf experiment")
            entry["experiment"] = arm["experiment"]
        out["arms"][name] = entry
    if ("initial_members" in section) == ("initial_cases" in section):
        raise ValueError("cycle: give exactly one of initial_members (label -> wrfinput) or initial_cases "
                         "(label -> case dir)")
    key = "initial_members" if "initial_members" in section else "initial_cases"
    members = section[key]
    if not isinstance(members, dict) or len(members) < 2 or "ctrl" in members:
        raise ValueError(f"cycle.{key}: >= 2 member labels other than 'ctrl'")
    out[key] = {safe_label(f"cycle.{key}", label): _abs_path(f"cycle.{key}.{label}", path)
                for label, path in members.items()}
    window, cycles = out["window_hours"], out["cycles"]
    if "forcing_hours" in section:
        out["forcing_hours"] = _positive_int("cycle.forcing_hours", section["forcing_hours"], multiple=3)
        if out["forcing_hours"] < window:
            raise ValueError("cycle.forcing_hours must be >= window_hours")
        if any(a["seam"] == "restart" for a in out["arms"].values()) and out["forcing_hours"] < window * cycles:
            raise ValueError("cycle.forcing_hours: restart-seam arms need >= cycles * window_hours (one case per "
                             "stream)")
    if "template_wrfinput" in section:
        out["template_wrfinput"] = _abs_path("cycle.template_wrfinput", section["template_wrfinput"])
    share = section.get("share_executables", True)
    if not isinstance(share, bool):
        raise ValueError("cycle.share_executables must be true or false")
    out["share_executables"] = share
    if "require_platform" in section:
        if section["require_platform"] not in ("gpu", "cpu"):
            raise ValueError("cycle.require_platform must be 'gpu' or 'cpu'")
        out["require_platform"] = section["require_platform"]
    return out


def load_config(path: Path, *, out: Path | None = None) -> dict[str, Any]:
    """Cycle config from a ``gpuwrf.da.v1`` file (its ``cycle`` section; LETKF arms use that file's analysis
    settings) or a cycle-only ``gpuwrf.da.cycle.v1`` file (zero-update arms only)."""
    path = Path(path).resolve()
    refuse_protected_inputs("cycle config", [path])
    raw = json.loads(path.read_text())
    schema = raw.get("schema") if isinstance(raw, dict) else None
    da_config = None
    if schema == "gpuwrf.da.v1":
        from . import config as dacfg

        loaded = dacfg.load(path)
        if "cycle" not in loaded.data:
            raise ValueError(f"{path}: a gpuwrf.da.v1 file needs a 'cycle' section for 'gpuwrf da cycle'")
        section = loaded.data["cycle"]
        da_config = str(path)
        names = [e["name"] for e in loaded.data["experiments"]]
        for name, arm in section["arms"].items():
            if arm["kind"] == "letkf":
                arm.setdefault("experiment", names[0])
                if arm["experiment"] not in names:
                    raise ValueError(f"cycle.arms.{name}.experiment {arm['experiment']!r} not in {names}")
        if loaded.data["ensemble"].get("fourd"):
            raise ValueError("4D (fourd) analyses are not cycled yet; drop ensemble.fourd")
    elif schema == CYCLE_SCHEMA:
        section = validate_config_section({k: v for k, v in raw.items() if k != "schema"})
        if any(arm["kind"] == "letkf" for arm in section["arms"].values()):
            raise ValueError("letkf arms need the analysis settings: use a gpuwrf.da.v1 file with a 'cycle' section")
    else:
        raise ValueError(f"{path}: schema must be 'gpuwrf.da.v1' (with a cycle section) or {CYCLE_SCHEMA!r}")
    cfg = dict(section)
    cfg["t0"] = datetime.strptime(section["t0"], "%Y-%m-%dT%H:%M:%SZ").strftime(WRF_TIME)
    cfg["out"] = None if out is None else str(out)
    cfg["da_config"] = da_config
    return cfg


def forcing_hours(cfg: Mapping[str, Any]) -> int:
    """Forcing hours every loop case keeps (``write_cycle_case``): equal for all cases, so every window,
    arm and seam has the same wrfbdy record count and shares ONE step program. Default: one window, or the
    whole horizon when a restart-seam arm exists (its stream resumes on its first case)."""
    if cfg.get("forcing_hours") is not None:
        return int(cfg["forcing_hours"])
    restart = any(arm.get("seam", "file") == "restart" for arm in cfg["arms"].values())
    return int(cfg["window_hours"]) * (int(cfg["cycles"]) if restart else 1)


def _time(cfg, cycle: int) -> str:
    t0 = datetime.strptime(cfg["t0"], WRF_TIME)
    return (t0 + timedelta(hours=cycle * int(cfg["window_hours"]))).strftime(WRF_TIME)


def first_window_cases(cfg: Mapping[str, Any], root: Path, *, hours: int | None = None
                       ) -> tuple[dict[str, Path], dict[str, Any]]:
    """Window-0 cases: given cases as-is, or member wrfinputs published via write_cycle_case(land='cold').

    ``hours`` (default one window) sets the forcing records the case keeps."""
    if cfg.get("initial_cases"):
        return {k: Path(v) for k, v in cfg["initial_cases"].items()}, {"kind": "initial_cases"}
    spec = {"source_case": cfg["source_case"], "t_a": cfg["t0"], "hours": int(hours or forcing_hours(cfg)),
            "members": cfg["initial_members"], "out": str(root / "cases")}
    spec_path = root / "first_window_spec.json"
    spec_path.write_text(json.dumps(spec, indent=1) + "\n")
    step = _cpu_child(["-m", "gpuwrf.assimilation.letkf.cycle_loop", "first-window", str(spec_path)],
                      root / "first_window.log")
    return {k: root / "cases" / k for k in cfg["initial_members"]}, step


def _member_frames(fc_root: Path, labels, t_a: str) -> dict[str, str]:
    paths = {}
    for label in labels:
        path = fc_root / label / f"wrfout_d01_{t_a}"
        if not path.is_file():
            raise ValueError(f"member {label}: no background frame {path}")
        paths[label] = str(path)
    return paths


def prepare_next(cfg: Mapping[str, Any], arm: str, fc_root: Path, labels, t_a: str, prep: Path,
                 work: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    """Analysis at ``t_a`` from the member frames under ``fc_root`` → next-window cases (CPU child)."""
    spec_arm = cfg["arms"][arm]
    frames = _member_frames(fc_root, labels, t_a)
    template = cfg.get("template_wrfinput") or str(Path(cfg["source_case"]) / "wrfinput_d01")
    work.mkdir(parents=True, exist_ok=False)
    if spec_arm["kind"] == "zero_update":
        spec = {"members": frames, "t_a": t_a, "template_wrfinput": template, "source_case": cfg["source_case"],
                "hours": forcing_hours(cfg), "out": str(prep)}
        spec_path = work / "zero_update_spec.json"
        spec_path.write_text(json.dumps(spec, indent=1) + "\n")
        step = _cpu_child(["-m", "gpuwrf.assimilation.letkf.cycle_loop", "zero-update", str(spec_path)],
                          work / "zero_update.log")
    else:
        spec = {"da_config": cfg["da_config"], "experiment": spec_arm["experiment"], "t_a": t_a,
                "member_files": [frames[label] for label in labels], "source_case": cfg["source_case"],
                "hours": forcing_hours(cfg), "template_wrfinput": template, "out": str(work / "analysis")}
        spec_path = work / "letkf_spec.json"
        spec_path.write_text(json.dumps(spec, indent=1) + "\n")
        step = _cpu_child(["-m", "gpuwrf.assimilation.letkf.cycle_loop", "letkf-step", str(spec_path)],
                          work / "letkf.log")
        result = json.loads((work / "analysis" / "cycle_step.json").read_text())
        step["analysis"] = result
        prep = Path(result["manifest"]).parent
    manifest = prep / "ENSEMBLE_MANIFEST.json"
    if not manifest.is_file():
        raise RuntimeError(f"arm {arm}: analysis at {t_a} wrote no member cases (inadmissible analysis?)")
    cases, wrfinputs = manifest_members(manifest, frames)
    step["manifest"] = {"path": str(manifest), "sha256": sha256(manifest)}
    step["analysis_wrfinputs"] = {k: str(v) for k, v in wrfinputs.items()}
    return cases, step


def manifest_members(manifest: Path, frames: Mapping[str, str]) -> tuple[dict[str, Path], dict[str, Path]]:
    """(cases, analysis wrfinputs) by loop label, matched through the background frame each analysed member
    records (``background_wrfout``), never through its label: the analysis API names members m00.. in input
    order, the zero-update writer keeps the loop labels. Exactly one analysed member per loop member."""
    members = json.loads(Path(manifest).read_text())["members"]
    by_background: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for key, member in members.items():
        if key == "ctrl":
            continue
        background = member.get("background_wrfout")
        if not background:
            raise RuntimeError(f"{manifest}: member {key} records no background_wrfout")
        resolved = str(Path(background).resolve())
        if resolved in by_background:
            raise RuntimeError(f"{manifest}: members {by_background[resolved][0]} and {key} share background "
                               f"{background}")
        by_background[resolved] = (key, member)
    cases, wrfinputs, used = {}, {}, set()
    for label, frame in frames.items():
        match = by_background.get(str(Path(frame).resolve()))
        if match is None:
            raise RuntimeError(f"{manifest}: no analysed member has loop member {label}'s background {frame}")
        if match[0] in used:
            raise RuntimeError(f"{manifest}: analysed member {match[0]} matches two loop members")
        used.add(match[0])
        cases[label], wrfinputs[label] = Path(match[1]["case"]), Path(match[1]["wrfinput"])
    unused = sorted({key for key, _ in by_background.values()} - used)
    if unused:
        raise RuntimeError(f"{manifest}: analysed members {unused} have no loop member")
    return cases, wrfinputs


def _window_steps(cfg: Mapping[str, Any]) -> int:
    from fractions import Fraction

    from gpuwrf.io.namelist_parser import parse_namelist

    domains = parse_namelist(Path(cfg["source_case"]) / "namelist.input")["domains"]
    first = lambda k, d=0: (domains.get(k, d)[0] if isinstance(domains.get(k, d), list) else domains.get(k, d))  # noqa: E731
    dt = Fraction(str(first("time_step"))) + Fraction(str(first("time_step_fract_num", 0))) / Fraction(
        str(first("time_step_fract_den", 1) or 1))
    steps = Fraction(int(cfg["window_hours"]) * 3600) / dt
    if steps.denominator != 1:
        raise ValueError("window_hours must hold an integral number of root steps")
    return int(steps)


def _seam(cfg: Mapping[str, Any], arm: str) -> str:
    return cfg["arms"][arm].get("seam", "file")


def run_cycles(cfg: Mapping[str, Any], *, driver: MemberDriver | None = None) -> dict[str, Any]:
    """Advance every arm's members over ``cycles`` windows; see the module docstring for the layout.

    File-seam arms share window 0 (stream ``shared``) and restart each window from analysis cases.
    Restart-seam arms (``seam: restart``) keep one checkpointed stream per member (``streams/<arm>/<m>``)
    and continue it across the analysis with restart_inject (``--resume-checkpoint --extend-run``).
    """
    out = Path(cfg["out"])
    if out.exists():
        raise ValueError(f"{out}: new output directory required")
    # Everything the loop reads itself (window-0 cases, templates, the source case); the analysis API guards its own.
    refuse_protected_inputs("cycle inputs", [cfg["source_case"], *filter(None, [cfg.get("template_wrfinput")]),
                                             *(cfg.get("initial_members") or {}).values(),
                                             *(cfg.get("initial_cases") or {}).values()], runtime=True)
    out.mkdir(parents=True)
    hours = int(cfg["window_hours"])
    cycles = int(cfg["cycles"])
    restart_arms = [arm for arm in cfg["arms"] if _seam(cfg, arm) == "restart"]
    file_arms = [arm for arm in cfg["arms"] if _seam(cfg, arm) == "file"]
    window_steps = _window_steps(cfg) if restart_arms else None
    receipt: dict[str, Any] = {"schema": 1, "kind": "gpuwrf da cycle", "config": cfg, "windows": [],
                               "started_utc": datetime.now(timezone.utc).isoformat()}
    if cfg.get("require_platform"):
        import jax

        platform = jax.devices()[0].platform
        if platform != cfg["require_platform"]:
            raise RuntimeError(f"cycle loop requires platform {cfg['require_platform']!r}, process has {platform!r}")
        receipt["platform_asserted"] = platform
    driver = driver or MemberDriver(out / ".scratch", share_executables=bool(cfg.get("share_executables", True)))
    start = time.perf_counter()
    try:
        streams: dict[str, dict[str, Any]] = {}
        root0 = out / "c00" / "shared"
        root0.mkdir(parents=True)
        # One first-window case per member (forcing_hours records) for every arm; a restart stream keeps it
        # as its input for the whole horizon.
        cases0, first_steps = first_window_cases(cfg, root0)
        # Window 0 is identical for every arm: it runs once per member, checkpointed in the first restart arm's
        # stream when there is one (that arm then resumes it in place), else as the plain stream 'shared'.
        # Further restart arms need their own checkpointed window 0 (a stream resumes only in its own dir).
        if restart_arms:
            for arm in restart_arms:
                streams[arm] = {"seam": "restart", "cases": cases0, "resume": None}
            first_fc = out / "streams" / restart_arms[0]
        else:
            streams["shared"] = {"seam": "file", "cases": cases0}
            first_fc = out / "c00" / "shared" / "fc"
        labels = list(next(iter(streams.values()))["cases"])
        for cycle in range(cycles):
            window = {"cycle": cycle, "t_start": _time(cfg, cycle), "t_end": _time(cfg, cycle + 1), "streams": {}}
            if cycle == 0:
                window["first_window"] = first_steps
            for name, stream in streams.items():
                t_stream = time.perf_counter()
                runs = {}
                resume_of = dict(stream.get("resume") or {})
                for label in labels:
                    if stream["seam"] == "file":
                        fc_root = out / f"c{cycle:02d}" / name / "fc"
                        runs[label] = driver.run(label, stream["cases"][label], fc_root / label, hours)
                    else:
                        fc_root = out / "streams" / name
                        resume = stream["resume"][label] if stream["resume"] else None
                        runs[label] = driver.run(
                            label, stream["cases"][label], fc_root / label, hours * (cycle + 1),
                            checkpoint_dir=out / "checkpoints" / name / label,
                            checkpoint_interval_steps=window_steps,
                            resume_checkpoint=resume, extend_run=resume is not None)
                window["streams"][name] = {
                    "seam": stream["seam"], "fc_root": str(fc_root), "wall_seconds": time.perf_counter() - t_stream,
                    "members": {label: {"case": str(stream["cases"][label]), "wall_seconds": r["wall_seconds"],
                                        "resume_checkpoint": str(resume_of[label]) if resume_of.get(label) else None,
                                        "wrfout": r["wrfout"], "aot": r["aot"]} for label, r in runs.items()}}
            receipt["windows"].append(window)
            if cycle + 1 == cycles:
                break
            t_a = _time(cfg, cycle + 1)
            prepared = {}
            for arm in cfg["arms"]:
                if _seam(cfg, arm) == "file":
                    fc_root = first_fc if cycle == 0 else out / f"c{cycle:02d}" / arm / "fc"
                else:
                    fc_root = out / "streams" / arm
                arm_root = out / f"c{cycle + 1:02d}" / arm
                arm_root.mkdir(parents=True)
                if _seam(cfg, arm) == "restart" and cfg["arms"][arm]["kind"] == "zero_update":
                    # The zero-update analysis IS the background frame: inject its (exactly zero)
                    # increment from the frame itself; no analysis files or cases are needed.
                    frames = _member_frames(fc_root, labels, t_a)
                    prepared[arm] = {"analysis": "background frame (zero increment)", "analysis_wrfinputs": frames}
                    cases = None
                else:
                    cases, prepared[arm] = prepare_next(cfg, arm, fc_root, labels, t_a, arm_root / "prep",
                                                        arm_root / "analysis_work")
                if _seam(cfg, arm) == "file":
                    streams[arm] = {"seam": "file", "cases": cases}
                else:
                    prepared[arm]["injection"] = inject_next(out, arm, fc_root, labels, t_a, arm_root,
                                                             prepared[arm]["analysis_wrfinputs"])
                    streams[arm]["resume"] = {label: Path(entry["generation"]) for label, entry in
                                              prepared[arm]["injection"]["members"].items()}
            streams.pop("shared", None)
            window["prepare_next"] = prepared
    finally:
        receipt["driver"] = driver.receipt()
        receipt["wall_seconds"] = time.perf_counter() - start
        (out / "CYCLE_RECEIPT.json").write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str) + "\n")
    return receipt


def inject_next(out: Path, arm: str, fc_root: Path, labels, t_a: str, arm_root: Path,
                wrfinputs: Mapping[str, str]) -> dict[str, Any]:
    """Restart seam: add each member's analysis increment to its end-of-window generation (CPU child)."""
    members = {}
    for label in labels:
        members[label] = {"store": str(out / "checkpoints" / arm / label),
                          "analysis_wrfinput": str(wrfinputs[label]),
                          "background_frame": str(fc_root / label / f"wrfout_d01_{t_a}")}
    spec = {"t_a": t_a, "members": members, "receipt": str(arm_root / "injection_receipt.json")}
    spec_path = arm_root / "injection_spec.json"
    spec_path.write_text(json.dumps(spec, indent=1) + "\n")
    step = _cpu_child(["-m", "gpuwrf.assimilation.letkf.restart_inject", str(spec_path)], arm_root / "injection.log")
    step["members"] = json.loads(Path(spec["receipt"]).read_text())
    return step


# ----------------------------------------------------------------------------- CPU child steps

def _require_cpu():
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise SystemExit("host-only cycle step: set JAX_PLATFORMS=cpu")


def first_window(spec_path: Path) -> int:
    """Publish each member wrfinput (valid at t0, land unchanged) as a window case (land='cold')."""
    _require_cpu()
    from .cycle_case import write_cycle_case

    spec = json.loads(Path(spec_path).read_text())
    refuse_protected_inputs("first-window inputs", [spec["source_case"], *spec["members"].values()], runtime=True)
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=False)
    source = Path(spec["source_case"])
    with refuse_protected_opens():
        for label, wrfinput in spec["members"].items():
            safe_label("first-window member", label)
            write_cycle_case(source, {"d01": Path(wrfinput)}, spec["t_a"], int(spec["hours"]), out / label,
                             land="cold")
            print(label, "case", out / label, flush=True)
    return 0


def zero_update(spec_path: Path) -> int:
    """Control arm: every member's analysis is its own background frame (zero increment), land cycled."""
    _require_cpu()
    from .member_cycle import ANALYSED, REBALANCED, write_member_cycle
    from .state_io import read_member_frames

    spec = json.loads(Path(spec_path).read_text())
    labels = [safe_label("zero-update member", k) for k in spec["members"]]
    paths = [spec["members"][k] for k in labels]
    refuse_protected_inputs("zero-update inputs", [*paths, spec["template_wrfinput"], spec["source_case"]],
                            runtime=True)
    names = ANALYSED + REBALANCED
    with refuse_protected_opens():
        frames = read_member_frames(paths, spec["t_a"], names)
        fields = {name: frames[name] for name in names}
        manifest = write_member_cycle(paths, fields, spec["t_a"], spec["template_wrfinput"], spec["source_case"],
                                      int(spec["hours"]), spec["out"], member_ids=labels,
                                      extra_provenance={"arm": "zero_update", "lane": "w41-da-cycle"})
    print("manifest", manifest, flush=True)
    return 0


# ----------------------------------------------------------------------------- CLI

def letkf_step(spec_path: Path) -> int:
    """LETKF arm: one ``api.analyse_case`` on the member frames (the cycle file's analysis settings)."""
    _require_cpu()
    from . import api, config as dacfg

    spec = json.loads(Path(spec_path).read_text())
    loaded = dacfg.load(spec["da_config"])
    data = json.loads(json.dumps(loaded.data))
    for key in ("cycle", "control_file"):
        data.pop(key, None)
    exp = next(e for e in data["experiments"] if e["name"] == spec["experiment"])
    exp.update(write_mean=False, write_members={"source_case_dir": spec["source_case"], "hours": int(spec["hours"])})
    ensemble = {k: v for k, v in data["ensemble"].items()
                if k not in ("member_glob", "member_files", "runs_glob", "max_age_days", "max_members")}
    ensemble.update(kind="members", member_files=list(spec["member_files"]))
    t_a = datetime.strptime(spec["t_a"], WRF_TIME).strftime("%Y-%m-%dT%H:%M:%SZ")
    data.update(t_a=t_a, ensemble=ensemble, experiments=[exp], template_wrfinput=spec["template_wrfinput"])
    cfg = dacfg.validate(data, base_dir=Path(spec["da_config"]).parent, source=spec["da_config"])
    summary = api.analyse_case(cfg, Path(spec["out"]))
    written = summary["experiments"][exp["name"]].get("member_cycle_manifest") or {}
    result = {"summary": str(Path(spec["out"]) / "summary.json"), "experiment": exp["name"],
              "admissible": summary["experiments"][exp["name"]]["admissible"],
              "refusal_reasons": summary["experiments"][exp["name"]].get("refusal_reasons"),
              "manifest": written.get("path"), "config_sha256": cfg.sha256}
    (Path(spec["out"]) / "cycle_step.json").write_text(json.dumps(result, indent=1) + "\n")
    if not written.get("path"):
        print(f"analysis refused: {result['refusal_reasons']}", flush=True)
        return 1
    print("manifest", written["path"], flush=True)
    return 0


# ----------------------------------------------------------------------------- CLI

def register(subparsers) -> None:
    """Mount ``cycle`` and ``members`` under the ``gpuwrf da`` parser (w41-da-core owns ``da``)."""
    from .member_driver import add_members_arguments, members_command

    cycle = subparsers.add_parser(
        "cycle", help="forecast/analysis cycles: N d01 members per window in ONE process (experimental)",
        description="Cycle config: a gpuwrf.da.v1 file with a 'cycle' section (LETKF arms use its analysis "
                    "settings) or a gpuwrf.da.cycle.v1 file (zero-update arms only). See docs/da-cycle.md.")
    cycle.add_argument("--config", type=Path, required=True, help="gpuwrf.da.v1 (+cycle) or gpuwrf.da.cycle.v1 JSON")
    cycle.add_argument("--out", type=Path, required=True, help="new output directory (must not exist)")
    cycle.add_argument("--dry-run", action="store_true", help="validate the config and print the plan; run nothing")
    cycle.set_defaults(func=_cycle_command)
    members = subparsers.add_parser("members", help="run the members of an ENSEMBLE_MANIFEST.json in one process")
    add_members_arguments(members)
    members.set_defaults(func=members_command)


def plan(cfg: Mapping[str, Any]) -> dict[str, Any]:
    n = len(cfg.get("initial_members") or cfg.get("initial_cases"))
    restart = sum(_seam(cfg, arm) == "restart" for arm in cfg["arms"])
    first = max(1, restart)  # window 0 runs once, plus once more per extra restart arm
    return {"windows": [[_time(cfg, c), _time(cfg, c + 1)] for c in range(cfg["cycles"])],
            "arms": {k: {"kind": v["kind"], "seam": _seam(cfg, k)} for k, v in cfg["arms"].items()},
            "members": list(cfg.get("initial_members") or cfg.get("initial_cases")),
            "forcing_hours": forcing_hours(cfg), "member_runs": n * (first + (cfg["cycles"] - 1) * len(cfg["arms"])),
            "da_config": cfg.get("da_config")}


def _cycle_command(args) -> int:
    try:
        cfg = load_config(args.config, out=args.out)
    except (ValueError, OSError) as exc:
        print(f"gpuwrf da cycle: error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(plan(cfg), indent=1))
        return 0
    receipt = run_cycles(cfg)
    print(json.dumps({"wall_seconds": receipt["wall_seconds"], "out": cfg["out"],
                      "member_runs": receipt["driver"]["n_runs"]}, indent=1))
    return 0


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m gpuwrf.assimilation.letkf.cycle_loop")
    sub = parser.add_subparsers(dest="command", required=True)
    register(sub)
    for name, func in (("first-window", first_window), ("zero-update", zero_update), ("letkf-step", letkf_step)):
        step = sub.add_parser(name, help="internal host-only step (CPU child)")
        step.add_argument("spec", type=Path)
        step.set_defaults(func=lambda a, f=func: f(a.spec))
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
