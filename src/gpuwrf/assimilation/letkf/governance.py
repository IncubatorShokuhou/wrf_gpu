"""Read governance for ``gpuwrf da``: every input is planned before anything is opened, then read inside a
policy session that records (or restricts) what is actually opened.

Flow (``api.analyse_case``): ``plan_reads(cfg)`` lists every file the run will open with its role — no
payload is opened, only directories are listed for globs → ``policy.refine(plan)`` may drop candidates
(e.g. archive frames on embargoed days) → ``with policy.session(plan, receipt_dir) as session:`` the
analysis runs → ``session.receipt`` (also written to ``receipt_dir/governance.json``).

Plug-ins (config ``governance.plugin``):

``none`` (default)  provenance only: size/mtime/SHA256 of every planned input (``hash_inputs``) and every
                    file actually opened (Python ``open`` audit hook + ``netCDF4.Dataset`` wrapper), flagging
                    opens that were not planned. Refuses any planned path inside the ALISIOS data roots —
                    those require the ``alisios`` plug-in.
``alisios``         the ALISIOS data rule (``alisios_guard``): guard scan of every valid day, DENY days dropped
                    from archive candidates BEFORE any open, then either ALISIOS's own reader facade
                    (``alisios.guarded_read`` sealed at ``alisios_guard.FACADE_COMMIT``: every ALISIOS input is
                    scanned, admitted, read once into a private stage with READ_SPY, and the analysis runs on the
                    stage while every ALISIOS open is refused) or, if the facade is unavailable or a planned
                    ALISIOS role is not staged (source case files, 4D frames, inflation state), the fallback
                    ``guarded_session`` (``guard_call.admit`` on the exact read set + refusing read spy).
                    ``options.facade``: ``auto`` (default) | ``require`` | ``off``. Observation files inside the
                    ALISIOS data roots are refused (use a guarded extract). Nothing is imported unless selected.
``pkg.module:factory``  ``factory(options) -> Policy`` for a site's own rules.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import glob as globmod
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys

from . import alisios_guard


@dataclass(frozen=True)
class ReadEntry:
    path: str
    role: str                       # control | template | member | archive | namelist | obs | case | fourd | inflation_state
    valid_days: tuple = ()          # YYYYMMDD (filled by policies that need them)
    meta: dict = field(default_factory=dict)


@dataclass
class ReadPlan:
    entries: list
    dropped: list = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    def paths(self, role=None):
        return [e.path for e in self.entries if role is None or e.role == role]

    def to_dict(self):
        return {"entries": [asdict(e) for e in self.entries], "dropped": self.dropped, "notes": self.notes}


class PlanError(ValueError):
    """A planned input is missing or forbidden; nothing has been opened."""


# ------------------------------------------------------------------------------------------- plan

def _glob(pattern):
    return sorted(str(Path(p)) for p in globmod.glob(pattern))


def _exists(path, role):
    if not Path(path).is_file():
        raise PlanError(f"{role} file does not exist: {path}")
    return str(Path(path).resolve())


def _member_paths(ens):
    if ens["kind"] != "members":
        return []
    paths = ens.get("member_files") or _glob(ens["member_glob"])
    if len(paths) < 2:
        raise PlanError(f"ensemble needs >= 2 member files, found {len(paths)} "
                        f"({ens.get('member_glob') or 'member_files'})")
    return paths


def plan_reads(cfg) -> ReadPlan:
    """Every file the analysis of ``cfg`` (a validated DaConfig) will open — listing only, no payload opens."""
    from . import archive_ensemble

    data = cfg.data
    t_a = cfg.t_a
    domain = data["domain"]
    entries, notes = [], {}

    def add(path, role, **meta):
        entries.append(ReadEntry(_exists(path, role), role, (), meta))

    if "control_file" in data:
        add(data["control_file"], "control")
        namelist = Path(data["control_file"]).parent / "namelist.input"
        if namelist.is_file():   # read_geometry reads spec_bdy_width from it
            add(namelist, "namelist")
    if "template_wrfinput" in data:
        add(data["template_wrfinput"], "template")
    ens = data["ensemble"]
    members = _member_paths(ens)
    for i, path in enumerate(members):
        add(path, "member", index=i)
    if members and "control_file" not in data:
        namelist = Path(members[0]).parent / "namelist.input"
        if namelist.is_file():
            add(namelist, "namelist")
    if ens["kind"] == "same_phase":
        runs = _glob(ens["runs_glob"])
        cands = archive_ensemble.candidate_frames(runs, domain, t_a, exclude=(str(Path(data["control_file"])),),
                                                  max_age_days=ens.get("max_age_days"))
        if len(cands) < 2:
            raise PlanError(f"same_phase ensemble: {len(cands)} causal archive frames at {t_a:%H}Z "
                            f"within {ens.get('max_age_days')} days in {ens['runs_glob']}")
        for path, stamp, run in cands:
            add(path, "archive", valid=stamp.strftime("%Y%m%d%H"), run=run)
        notes["archive_candidates"] = len(cands)
    if ens.get("fourd"):
        runs = _glob(ens["fourd"]["member_run_glob"])
        if [Path(r).name for r in runs] != [Path(m).parent.name for m in members]:
            raise PlanError("fourd member runs must match the ensemble members in order")
        start = t_a.timestamp() - ens["fourd"]["window_hours"] * 3600.0
        for run in runs:
            for path in sorted(Path(run).glob(f"wrfout_{domain}_*")):
                try:
                    stamp = datetime.strptime(path.name.split("_", 2)[2], "%Y-%m-%d_%H:%M:%S").replace(
                        tzinfo=timezone.utc)
                except ValueError:
                    continue
                if start - 1800.0 <= stamp.timestamp() <= t_a.timestamp() + 1800.0:
                    add(path, "fourd", run=run)
    for source in data["obs"]["sources"]:
        add(source["path"], "obs", format=source["format"])
        if source.get("isd_history"):
            add(source["isd_history"], "obs", format="isd_history")
    from . import inflation as infl
    if data.get("inflation") is not None and hasattr(infl, "read_plan"):
        for path in infl.read_plan(data["inflation"]):         # state files of the inflation plug-in
            add(path, "inflation_state")
    for exp in data["experiments"]:
        if "write_members" in exp:
            src = Path(exp["write_members"]["source_case_dir"])
            for name in ("namelist.input", f"wrfbdy_{domain}", f"wrflowinp_{domain}", f"wrfinput_{domain}",
                         "freezeH2O.dat", "qr_acr_qg_V4.dat", "qr_acr_qsV2.dat"):
                if (src / name).exists():
                    add(src / name, "case", experiment=exp["name"])
    seen, unique = set(), []
    for e in entries:
        key = (e.path, e.role)
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return ReadPlan(unique, [], notes)


def limit_archive(plan: ReadPlan, max_members) -> ReadPlan:
    """Keep only the newest ``max_members`` archive frames AFTER the policy filtered candidates, so that the read
    set a policy admits is exactly what ``archive_ensemble.same_phase_members`` will open."""
    archive = [e for e in plan.entries if e.role == "archive"]
    if max_members is None or len(archive) <= max_members:
        return plan
    keep = {e.path for e in archive[-int(max_members):]}
    entries = [e for e in plan.entries if e.role != "archive" or e.path in keep]
    dropped = plan.dropped + [{"path": e.path, "reason": "beyond max_members (older)"} for e in archive
                              if e.path not in keep]
    notes = dict(plan.notes)
    if "issue_days" in notes:
        notes["issue_days"] = sorted({m.group(1) for e in entries
                                      if alisios_guard.is_alisios_data(e.path) and (m := _RUN_RE.search(e.path))})
    return ReadPlan(entries, dropped, notes)


# ------------------------------------------------------------------------------------------- policies

def _file_record(path, hash_inputs):
    st = os.stat(path)
    rec = {"size": st.st_size, "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}
    if hash_inputs:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 22), b""):
                digest.update(block)
        rec["sha256"] = digest.hexdigest()
    return rec


class Policy:
    """Base policy: no restriction, no recording (subclass for real use)."""

    name = "base"

    def __init__(self, options=None, *, hash_inputs=True):
        self.options = dict(options or {})
        self.hash_inputs = hash_inputs

    def refine(self, plan: ReadPlan, cfg) -> ReadPlan:
        return plan

    @contextmanager
    def session(self, plan: ReadPlan, receipt_dir, *, output_dir=None):
        session = _Recorder(plan, output_dir)
        yield session
        session.receipt = {"plugin": self.name}


# --- a recording (non-refusing) spy for the default policy

_RECORDER = None
_HOOKED = False


class _Recorder:
    def __init__(self, plan, output_dir=None):
        self.planned = {str(Path(p).resolve()) for p in plan.paths()}
        self.output_dir = None if output_dir is None else Path(output_dir).resolve()
        self.opened, self.own_outputs = {}, {}
        self.receipt = None

    def note(self, path, kind):
        try:
            resolved = Path(os.fsdecode(path)).resolve()
        except (TypeError, ValueError):
            return
        if self.output_dir is not None and (resolved == self.output_dir or self.output_dir in resolved.parents):
            self.own_outputs[str(resolved)] = self.own_outputs.get(str(resolved), 0) + 1
            return
        key = str(resolved)
        row = self.opened.setdefault(key, {"kind": kind, "open_count": 0, "planned": key in self.planned})
        row["open_count"] += 1


_IGNORED_PREFIXES = (sys.prefix, sys.base_prefix, "/proc", "/sys", "/dev", "/usr", "/etc")


def _audit(event, args):
    if event != "open" or _RECORDER is None:
        return
    file, mode, flags = args
    if not isinstance(file, (str, bytes, os.PathLike)):
        return
    if isinstance(mode, str) and not ("r" in mode or "+" in mode):
        return                                                  # write-only Python open
    if mode is None and isinstance(flags, int) and (flags & os.O_ACCMODE) == os.O_WRONLY:
        return                                                  # write-only os.open (e.g. bytecode cache)
    text = os.fsdecode(file)
    if (text.endswith((".py", ".pyc", ".so")) or "__pycache__" in text or text.startswith(_IGNORED_PREFIXES)):
        return
    _RECORDER.note(file, "python_open")


class OpenPolicy(Policy):
    """Default: provenance receipt; refuses ALISIOS data (needs the 'alisios' plug-in)."""

    name = "none"

    def refine(self, plan, cfg):
        forbidden = [e.path for e in plan.entries if alisios_guard.is_alisios_data(e.path)]
        if forbidden:
            raise PlanError(f"{len(forbidden)} planned input(s) are ALISIOS data (e.g. {forbidden[0]}); "
                            "set governance.plugin = 'alisios'")
        return plan

    @contextmanager
    def session(self, plan, receipt_dir, *, output_dir=None):
        """Record opens; files under ``output_dir`` (this run's own outputs) are counted separately."""
        global _RECORDER, _HOOKED
        import netCDF4

        if _RECORDER is not None:
            raise RuntimeError("nested governance session")
        receipt_dir = Path(receipt_dir)
        receipt_dir.mkdir(parents=True, exist_ok=True)
        inputs = {e.path: {"role": e.role, **_file_record(e.path, self.hash_inputs)} for e in plan.entries}
        recorder = _Recorder(plan, output_dir)
        original = netCDF4.Dataset

        def recording_dataset(filename, *args, **kwargs):
            recorder.note(filename, "netcdf_open")
            return original(filename, *args, **kwargs)

        if not _HOOKED:
            sys.addaudithook(_audit)
            _HOOKED = True
        _RECORDER = recorder
        netCDF4.Dataset = recording_dataset
        try:
            yield recorder
        finally:
            netCDF4.Dataset = original
            _RECORDER = None
            unplanned = sorted(p for p, row in recorder.opened.items() if not row["planned"])
            recorder.receipt = {"schema": "gpuwrf.da.governance.v1", "plugin": self.name, "inputs": inputs,
                                "opened": recorder.opened, "unplanned_opens": unplanned,
                                "own_output_opens": len(recorder.own_outputs),
                                "dropped": plan.dropped, "notes": plan.notes}
            (receipt_dir / "governance.json").write_text(json.dumps(recorder.receipt, indent=1, default=str) + "\n")


_RUN_RE = re.compile(r"wg_(\d{8})_(\d{2})z_a\d+")


def _days_between(first, last):
    """YYYYMMDD of every calendar day from ``first`` to ``last`` (inclusive)."""
    from datetime import timedelta

    days, day = [], first.date()
    while day <= last.date():
        days.append(day.strftime("%Y%m%d"))
        day += timedelta(days=1)
    return days
_WRFOUT_RE = re.compile(r"wrfout_d\d\d_(\d{4})-(\d{2})-(\d{2})_")


# roles whose files the analysis reads by their planned path only (a staged copy is a complete substitute)
STAGEABLE_ROLES = frozenset({"control", "template", "namelist", "member", "archive"})


def _localizer(staged: dict):
    """(cfg, plan) -> the same run reading the staged copies (config hash and source unchanged)."""
    import copy

    from .config import DaConfig

    def localize(cfg, plan):
        data = copy.deepcopy(cfg.data)
        for key in ("control_file", "template_wrfinput"):
            if key in data and str(Path(data[key]).resolve()) in staged:
                data[key] = staged[str(Path(data[key]).resolve())]
        entries = [ReadEntry(staged[e.path], e.role, e.valid_days, {**e.meta, "original": e.path})
                   if e.path in staged else e for e in plan.entries]
        return DaConfig(data, cfg.source, cfg.sha256), ReadPlan(entries, plan.dropped, plan.notes)

    return localize


class AlisiosPolicy(Policy):
    """ALISIOS data rule via ``alisios_guard``: guard scan + DENY drop, then the sealed ALISIOS reader facade
    (preferred) or ``guarded_session`` (admit + refusing read spy)."""

    name = "alisios"

    def backend(self, plan):
        """("guarded_read", None) or ("guarded_session", reason) for this plan."""
        choice = self.options.get("facade", "auto")
        if choice not in ("auto", "require", "off"):
            raise PlanError(f"governance.options.facade must be auto|require|off, got {choice!r}")
        alisios = [e for e in plan.entries if alisios_guard.is_alisios_data(e.path)]
        roles = sorted({e.role for e in alisios} - STAGEABLE_ROLES)
        if choice == "off":
            reason = "governance.options.facade = 'off'"
        elif not alisios:
            reason = "no ALISIOS input planned"
        elif roles:
            reason = f"planned ALISIOS roles {roles} are read outside their planned paths (not staged)"
        else:
            reason = alisios_guard.facade(self.options.get("sealed_dir"))[2].get("unavailable")
        if reason and choice == "require":
            raise PlanError(f"governance.options.facade = 'require' but {reason}")
        return ("guarded_session", reason) if reason else ("guarded_read", None)

    def _run_window(self, run_dir, issue_day, receipt_dir):
        """Time window of a run from its ``namelist.input``, read as admitted metadata (the run's issue day):
        start per domain, end, history interval (minutes, None if unset) and frames per history file per domain.
        WRF Registry defaults apply to absent keys; an unknown start or run length refuses the plan."""
        from datetime import timedelta
        from gpuwrf.io.namelist_parser import parse_namelist

        namelist = Path(run_dir) / "namelist.input"
        if not namelist.is_file():
            raise PlanError(f"cannot establish the valid-day coverage of the ALISIOS run {run_dir}: no namelist.input")
        with alisios_guard.metadata_session(namelist, issue_day, receipt_dir,
                                            purpose="gpuwrf da: run window metadata (namelist)",
                                            prefer_facade=self.options.get("facade", "auto") != "off",
                                            sealed_root=self.options.get("sealed_dir")):
            nml = parse_namelist(namelist)
        tc, max_dom = nml.get("time_control", {}), nml.get("domains", {}).get("max_dom", 1)
        max_dom = int(max_dom[0] if isinstance(max_dom, list) else max_dom)

        def per(key, d, default=None):
            value = tc.get(key, default)
            if isinstance(value, list):
                value = value[d] if d < len(value) else value[0]
            return value

        def when(prefix, d):
            parts = [per(f"{prefix}_{k}", d) for k in ("year", "month", "day")]
            if any(v is None for v in parts):
                return None
            return datetime(*(int(v) for v in parts), int(per(f"{prefix}_hour", d, 0)),
                            int(per(f"{prefix}_minute", d, 0)), int(per(f"{prefix}_second", d, 0)))

        starts = [when("start", d) for d in range(max_dom)]
        if starts[0] is None:
            raise PlanError(f"{namelist}: start date unknown; cannot establish the valid-day coverage")
        length = timedelta(days=int(per("run_days", 0, 0)), hours=int(per("run_hours", 0, 0)),
                           minutes=int(per("run_minutes", 0, 0)), seconds=int(per("run_seconds", 0, 0)))
        end = starts[0] + length if length > timedelta(0) else when("end", 0)
        if end is None or end < starts[0]:
            raise PlanError(f"{namelist}: run length unknown; cannot establish the valid-day coverage")
        intervals, frames = [], []
        for d in range(max_dom):
            split = [int(per(f"history_interval_{u}", d, 0)) for u in ("d", "h", "m", "s")]
            minutes = (1440 * split[0] + 60 * split[1] + split[2] + split[3] / 60.0 if any(split)
                       else int(per("history_interval", d, 0)))
            intervals.append(minutes or None)
            frames.append(int(per("frames_per_outfile", d, 10)))
        return {"namelist": str(namelist), "start": starts, "end": end, "history_interval_min": intervals,
                "frames_per_outfile": frames, "max_dom": max_dom}

    def _run_days(self, run_dir, issue_day, receipt_dir):
        """Every valid day of a run (start of domain 1 .. end) from its admitted namelist."""
        window = self._run_window(run_dir, issue_day, receipt_dir)
        return _days_between(window["start"][0], window["end"])

    @staticmethod
    def _coverage(entry, window):
        """Complete valid-day coverage of one ALISIOS file of a run (``window`` from ``_run_window``)."""
        from datetime import timedelta

        name = Path(entry.path).name
        dom = re.search(r"_d(\d\d)", name)
        d = int(dom.group(1)) - 1 if dom else 0
        start = window["start"][d] if d < len(window["start"]) and window["start"][d] else window["start"][0]
        if re.match(r"wrfinput_d\d\d$", name):
            return [start.strftime("%Y%m%d")]                       # a wrfinput holds its domain's start time
        stamp = re.match(r"wrfout_d\d\d_(\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)$", name)
        if stamp:
            t0 = datetime.strptime(stamp.group(1), "%Y-%m-%d_%H:%M:%S")
            if not start <= t0 <= window["end"]:
                raise PlanError(f"{entry.path}: history time outside its run window {start}..{window['end']}")
            minutes, frames = (window["history_interval_min"][min(d, len(window["frames_per_outfile"]) - 1)],
                               window["frames_per_outfile"][min(d, len(window["frames_per_outfile"]) - 1)])
            last = window["end"] if minutes is None else min(window["end"], t0 + timedelta(minutes=minutes * (frames - 1)))
            return _days_between(t0, last)                         # every frame the file can hold
        return _days_between(window["start"][0], window["end"])    # boundary/lower-boundary files, tables

    def refine(self, plan, cfg):
        """Complete guard coverage BEFORE any payload open: archive candidates are pre-filtered by their file-name
        valid day and issue day; every remaining ALISIOS file gets its complete valid-day coverage from its run's
        admitted namelist; then every covered day and issue day is scanned: archive frames with a non-ALLOW day
        are dropped, any other non-ALLOW day refuses the plan."""
        import tempfile

        entries, dropped, issues = [], list(plan.dropped), set()
        metadata_receipts = Path(self.options.get("receipt_dir") or tempfile.mkdtemp(prefix="gpuwrf-da-guard-"))
        metadata_receipts = metadata_receipts / "guard_metadata"
        windows = {}
        archive = [e for e in plan.entries if e.role == "archive"]
        kept = set()
        if archive:
            # an archive frame needs BOTH its valid day and its run's issue day ALLOWed (scan only, nothing opened)
            pairs = [(e, e.meta["valid"][:8]) for e in archive]
            pairs += [(e, m.group(1)) for e in archive if (m := _RUN_RE.search(e.path))]
            _, verdicts = alisios_guard.drop_deny_days(pairs)
            kept = {e.path for e in archive}
            for e, day in pairs:
                if verdicts.get(day) != "ALLOW":
                    kept.discard(e.path)
            dropped += [{"path": e.path, "reason": "guard non-ALLOW valid or issue day"} for e in archive
                        if e.path not in kept]
        for e in plan.entries:
            if not alisios_guard.is_alisios_data(e.path):
                entries.append(e)
                continue
            if e.role == "obs":
                raise PlanError(f"observation source points into ALISIOS data ({e.path}); use a guarded extract")
            if e.role == "archive" and e.path not in kept:
                continue
            run = _RUN_RE.search(e.path)
            if not run:
                raise PlanError(f"cannot establish the valid-day coverage of ALISIOS file {e.path}: "
                                "not inside a wg_<issue>_<HH>z_a<N> run")
            issues.add(run.group(1))
            if e.role == "namelist":
                days = [run.group(1)]                              # run metadata, admitted under the issue day
            else:
                key = str(Path(e.path).parent)
                if key not in windows:
                    target, n = metadata_receipts / run.group(0), 1
                    while target.exists():                       # READ_SPY files are never overwritten
                        target, n = metadata_receipts / f"{run.group(0)}.{n}", n + 1
                    windows[key] = self._run_window(Path(e.path).parent, run.group(1), target)
                days = self._coverage(e, windows[key])
            entries.append(ReadEntry(e.path, e.role, tuple(days), e.meta))
        alisios = [e for e in entries if alisios_guard.is_alisios_data(e.path)]
        pairs = [(e, d) for e in alisios for d in sorted(set(e.valid_days) | {_RUN_RE.search(e.path).group(1)})]
        _, verdicts = alisios_guard.drop_deny_days(pairs) if pairs else ([], {})
        bad = {}
        for e, day in pairs:
            if verdicts.get(day) != "ALLOW":
                bad.setdefault(e.path, []).append(day)
        refused = [e for e in alisios if e.path in bad and e.role != "archive"]
        if refused:
            e = refused[0]
            raise PlanError(f"{e.path} ({e.role}) covers non-ALLOW day(s) {bad[e.path]} (guard); nothing was opened")
        dropped += [{"path": e.path, "reason": f"coverage day(s) {bad[e.path]} not ALLOW"} for e in alisios
                    if e.path in bad]
        entries = [e for e in entries if e.path not in bad]
        notes = {**plan.notes, "issue_days": sorted(issues), "metadata_receipts": str(metadata_receipts),
                 "run_windows": {k: {**w, "start": [str(t) for t in w["start"]], "end": str(w["end"])}
                                 for k, w in windows.items()}}
        return ReadPlan(entries, dropped, notes)

    @contextmanager
    def session(self, plan, receipt_dir, *, output_dir=None):
        """Yields a session; ``session.localize(cfg, plan)`` (facade backend only) gives the staged run."""
        import shutil

        purpose = self.options.get("purpose", "gpuwrf da analysis")
        receipt_dir = Path(receipt_dir)
        kind, reason = self.backend(plan)
        alisios = [e for e in plan.entries if alisios_guard.is_alisios_data(e.path)]
        if kind == "guarded_session":
            allowed = {e.path: list(e.valid_days) for e in alisios}
            with alisios_guard.guarded_session(allowed, plan.notes.get("issue_days", []),
                                               receipt_dir / "alisios_guard", purpose=purpose) as guarded:
                yield guarded
            backend = {"kind": kind, "fallback_reason": reason}
        else:
            targets = {}
            for e in alisios:
                run = _RUN_RE.search(e.path)
                issue, valid = targets.get(e.path, ((), ()))
                targets[e.path] = (tuple(issue) + ((run.group(1),) if run else tuple(e.valid_days)),
                                   tuple(valid) + tuple(e.valid_days))
            stage_root = receipt_dir / "alisios_stage"
            try:
                staged, stage_receipt = alisios_guard.facade_stage(
                    targets, stage_root, receipt_dir / "alisios_guarded_read", purpose=purpose,
                    sealed_root=self.options.get("sealed_dir"))
                # analysis phase: every ALISIOS open is refused (all ALISIOS inputs are staged)
                with alisios_guard.guarded_session({}, [], receipt_dir / "alisios_guard",
                                                   purpose=purpose + " (staged; ALISIOS opens refused)") as guarded:
                    guarded.localize = _localizer(staged)
                    yield guarded
            finally:
                if stage_root.is_dir() and stage_root.name == "alisios_stage" and stage_root.parent == receipt_dir:
                    shutil.rmtree(stage_root)
            backend = {"kind": kind, **stage_receipt, "stage_removed": not stage_root.exists()}
        guarded.receipt = {"schema": "gpuwrf.da.governance.v1", "plugin": self.name, "backend": backend,
                           "alisios_guard": guarded.receipt, "dropped": plan.dropped, "notes": plan.notes,
                           "non_alisios_inputs": sorted(e.path for e in plan.entries
                                                        if not alisios_guard.is_alisios_data(e.path))}
        (receipt_dir / "governance.json").write_text(json.dumps(guarded.receipt, indent=1, default=str) + "\n")


def make_policy(spec: dict, *, receipt_dir=None) -> Policy:
    plugin = spec.get("plugin", "none")
    options = dict(spec.get("options", {}))
    if receipt_dir is not None:
        options.setdefault("receipt_dir", str(receipt_dir))
    if plugin == "none":
        return OpenPolicy(options, hash_inputs=spec.get("hash_inputs", True))
    if plugin == "alisios":
        return AlisiosPolicy(options, hash_inputs=spec.get("hash_inputs", True))
    module_name, attr = plugin.split(":")
    factory = getattr(importlib.import_module(module_name), attr)
    policy = factory(options)
    if not isinstance(policy, Policy):
        raise TypeError(f"governance plug-in {plugin} must return a governance.Policy")
    return policy
