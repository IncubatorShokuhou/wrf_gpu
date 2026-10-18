"""Run the d01 members of a DA cycle in ONE process (lane w41-da-cycle; ALISIOS idea A1).

A fresh ``gpuwrf run`` pays ~17 s of fixed cost per process (imports, backend, executable load,
initialisation; proofs/a1_da/COST_MODEL.md: 84 % of a warm 6-h d01 member on the GPU). The
driver runs each member as the ordinary CLI forecast, ``gpuwrf.cli.main(["run", ...])``, inside
one process, so a member writes exactly the history a separate ``gpuwrf run`` of its case writes,
while imports, the backend and (``share_executables``) the loaded step executables are paid once:
``runtime.domain_tree.set_process_aot_reuse`` lets every later member reuse the executable that
the first member loaded or compiled for the same program address (domain, cheap key / HLO).

Import-time switches are pinned by the first member (the namelist's cloud-overlap flag, GPUWRF_*
switches read at import). Every member of one driver must therefore share the namelist outside
``&time_control`` and the process environment may not change between members; both are refused.
Nothing here changes the forecast program; the default ``gpuwrf run`` path is untouched.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

# The CLI rewrites these per run (scratch/tmp locations); they never reach a traced program
# (aot_cheap_key.HLO_AFFECTING_ENV_DENYLIST) and may differ between members.
_PER_RUN_ENV = frozenset({"GPUWRF_SCRATCH", "GPUWRF_TMPDIR"})
_PINNED_ENV_PREFIXES = ("GPUWRF_", "XLA_", "JAX_", "TF_", "CUDA_", "NVIDIA_TF32")
_EVENTS: list[dict[str, Any]] | None = None
_LISTENING = False
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def safe_label(where: str, label) -> str:
    """A member/arm label names directories under the output root: it must be ONE plain path component."""
    if not isinstance(label, str) or not _LABEL.fullmatch(label):
        raise ValueError(f"{where}: label {label!r} must be one path component matching "
                         "[A-Za-z0-9][A-Za-z0-9_.-]* (at most 64 characters)")
    return label


class ProtectedInputError(ValueError):
    """An input the driver or the cycle loop would read directly lies under the ALISIOS data roots."""


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [s for item in value for s in _strings(item)]
    if isinstance(value, dict):
        return [s for item in value.values() for s in _strings(item)]
    return []


def _namelist_strings(path: Path) -> list[str]:
    """Every character value of a namelist file: parsed values plus every quoted string of the raw text (a
    malformed or unusual namelist is still scanned)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    found = [a or b for a, b in re.findall(r"'([^']*)'|\"([^\"]*)\"", text)]
    try:
        from gpuwrf.io.namelist_parser import parse_namelist

        found += _strings(parse_namelist(path))
    except Exception:  # noqa: BLE001 - the raw scan above already covers it
        pass
    return found


def _path_candidates(token: str, base: Path) -> list[Path]:
    """Where the run may open ``token``: like the readers (``run_dir / name``; an absolute name stays absolute),
    relative to the working directory too, with WRF ``<domain>`` expanded; the parent directory covers names
    that keep other templates (``<date>``)."""
    token = token.strip().strip("'\"")
    if not token:
        return []
    out = []
    for name in {token, token.replace("<domain>", "01")}:
        for root in (base, Path.cwd()):
            path = root / name
            out += [path, path.parent]
    return out


# Scratch roots the run only writes: checked as paths, never walked (TMPDIR may be the shared /tmp).
_PATH_ONLY_ENV = frozenset({"TMPDIR", "GPUWRF_TMPDIR", "GPUWRF_SCRATCH"})
_RUNTIME_SCANS: dict[tuple, list[str]] = {}


def _environment_paths() -> list[tuple[Path, bool]]:
    """(path, walk) for every absolute path the run's environment names (GPUWRF_*_ROOT/_DIR/..., XLA_FLAGS paths,
    JAX/Triton/CUDA cache and toolkit directories)."""
    keys = sorted(k for k in os.environ if k.startswith(_PINNED_ENV_PREFIXES) or k in
                  ("TMPDIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"))
    return [(Path(part), k not in _PATH_ONLY_ENV) for k in keys
            for part in re.split(r"[\s,;:=]+", os.environ[k]) if part.startswith("/")]


def _runtime_roots() -> list[tuple[Path, bool]]:
    """Every directory the run reads beyond the case, through the runtime's own resolvers (so defaults and
    environment overrides are both covered): the WRF ``run/`` table directory, the runtime-asset cache, the
    package data root, the compile cache; plus every path the environment names."""
    from gpuwrf import assets
    from gpuwrf.config import paths

    return [(root(), True) for root in (paths.wrf_run_dir, assets.cache_root, paths.data_root,
                                        paths.jax_cache_dir)] + _environment_paths()


def _walk(root: Path, check, *, namelists: bool = True) -> None:
    """Visit every entry below ``root``: a link is resolved and checked, a directory link is followed once
    (cycle-safe, never into a protected target); a regular entry below a real unprotected directory cannot be
    protected itself. With ``namelists``, file names in ``namelist*`` files are resolved like the readers do
    (case and input directories; the runtime roots hold example namelists the run never reads)."""
    if check(root):
        return
    seen: set[str] = set()
    stack = [root]
    while stack:
        directory = stack.pop()
        real = os.path.realpath(directory)
        if real in seen:
            continue
        seen.add(real)
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except (NotADirectoryError, FileNotFoundError, PermissionError):
            continue
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink() and check(path):
                continue
            if entry.is_dir():
                stack.append(path)
            elif namelists and entry.name.startswith("namelist"):
                for value in _namelist_strings(path):
                    for candidate in _path_candidates(value, path.parent):
                        check(candidate)


def refuse_protected_inputs(where: str, paths: Iterable, *, outputs: Iterable = (), runtime: bool = False) -> None:
    """Fail closed on ALISIOS data. The member driver and the cycle loop read cases, wrfinputs and manifests
    directly (outside the da-core guarded analysis API), so anything under the ALISIOS roots is refused before
    a forecast or writer starts (``refuse_protected_opens`` is the open-layer backstop while they run).

    Walked (``_walk``): every input path; with ``runtime`` also every runtime root (``_runtime_roots``: WRF table
    directory, asset cache, data root, compile cache, every directory the environment names - cached per process
    for the same environment). Every character value of every ``namelist*`` file found is resolved as the run
    resolves file names (``auxinput4_inname``, ``gfdda_inname``, ... any key). ``outputs`` and scratch roots are
    checked as paths only. Nothing under the ALISIOS roots is listed or opened."""
    from .alisios_guard import is_alisios_data

    hits: list[str] = []

    def check(path) -> bool:
        if is_alisios_data(path):
            hits.append(str(path))
            return True
        return False

    for path in paths:
        _walk(Path(path), check)
    for path in outputs:
        check(path)
    if runtime:
        roots = _runtime_roots()
        walked = tuple(sorted({str(root) for root, walk in roots if walk}))
        if walked not in _RUNTIME_SCANS:      # the same roots in one process: walked once (env is pinned per driver)
            found: list[str] = []

            def runtime_check(path) -> bool:
                if is_alisios_data(path):
                    found.append(str(path))
                    return True
                return False

            for root in walked:
                _walk(Path(root), runtime_check, namelists=False)
            _RUNTIME_SCANS[walked] = found
        hits += _RUNTIME_SCANS[walked]
        for root, walk in roots:
            if not walk:
                check(root)
    if hits:
        raise ProtectedInputError(f"{where}: ALISIOS data is read only through the da-core guard "
                                  f"(scan/admit/READ_SPY); stage it outside the ALISIOS roots first: "
                                  f"{sorted(set(hits))[:3]}")


_OPEN_SPY: dict[str, list | None] = {"refused": None}
_OPEN_HOOKED = False


def _open_audit(event, args) -> None:
    refused = _OPEN_SPY["refused"]
    if refused is None or event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    from .alisios_guard import is_alisios_data

    if is_alisios_data(args[0]):
        refused.append(os.fsdecode(args[0]))
        raise ProtectedInputError(f"refused open of ALISIOS path {os.fsdecode(args[0])} (member driver)")


@contextlib.contextmanager
def refuse_protected_opens():
    """Open-layer backstop while a member forecast or a case writer runs: any Python-level open (audit hook, every
    thread) and any ``netCDF4.Dataset`` open (C-level, so wrapped; readers resolve it at call time through
    ``gpuwrf.io.netcdf_lock``) of a path under the ALISIOS roots raises ``ProtectedInputError``. Yields the list
    of refused paths (a reader that swallows the error still leaves its path there)."""
    global _OPEN_HOOKED
    import netCDF4

    from .alisios_guard import is_alisios_data

    if not _OPEN_HOOKED:
        sys.addaudithook(_open_audit)
        _OPEN_HOOKED = True
    refused: list[str] = []
    previous, original = _OPEN_SPY["refused"], netCDF4.Dataset

    def checked_dataset(filename, *args, **kwargs):
        if isinstance(filename, (str, bytes, os.PathLike)) and is_alisios_data(filename):
            refused.append(os.fsdecode(filename))
            raise ProtectedInputError(f"refused netCDF open of ALISIOS path {os.fsdecode(filename)} (member driver)")
        return original(filename, *args, **kwargs)

    _OPEN_SPY["refused"] = refused
    netCDF4.Dataset = checked_dataset
    try:
        yield refused
    finally:
        netCDF4.Dataset = original
        _OPEN_SPY["refused"] = previous


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def namelist_signature(case_dir: Path) -> str:
    """sha256 of the parsed namelist without ``&time_control`` (what import-time flags may read)."""
    from gpuwrf.io.namelist_parser import parse_namelist

    groups = parse_namelist(Path(case_dir) / "namelist.input")
    kept = {name: groups[name] for name in sorted(groups) if name != "time_control"}
    return hashlib.sha256(json.dumps(kept, sort_keys=True, default=repr).encode()).hexdigest()


def _pinned_environment() -> dict[str, str]:
    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith(_PINNED_ENV_PREFIXES) and k not in _PER_RUN_ENV}


def _listen_compile_events() -> None:
    """One process-wide JAX duration listener; events go to the member that is running."""
    global _LISTENING
    if _LISTENING:
        return
    import jax

    def duration(event, seconds, **_metadata):
        if _EVENTS is not None and ("compil" in event or "cache" in event):
            _EVENTS.append({"event": str(event), "seconds": float(seconds)})

    jax.monitoring.register_event_duration_secs_listener(duration)
    _LISTENING = True


def _event_categories(events: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    # Separate categories: compile_time_saved is avoided cost, not elapsed time (FINDINGS E54).
    out: dict[str, dict[str, float]] = {}
    for event in events:
        entry = out.setdefault(event["event"], {"calls": 0, "seconds": 0.0})
        entry["calls"] += 1
        entry["seconds"] += float(event["seconds"])
    return out


def _aot_domains() -> dict[str, Any]:
    from gpuwrf.runtime import domain_tree

    keep = ("loaded", "source", "cached", "real_load", "load_wall_seconds", "cheap_key", "hlo_sha256", "error")
    domains = domain_tree.NESTED_AOT_STATUS.get("domains", {}) or {}
    return {name: {k: status.get(k) for k in keep if k in status} for name, status in domains.items()}


class MemberDriver:
    """Sequential in-process member forecasts with shared executables (one instance per process)."""

    def __init__(self, scratch_root: Path, *, share_executables: bool = True):
        import gpuwrf

        if getattr(gpuwrf, "_OFFLINE_COMMAND", False):
            # An offline entry skips x64/command buffers/compile cache: members would run another program.
            raise RuntimeError("member forecasts need the gpuwrf package bootstrap; this process imported "
                               "gpuwrf as an offline command (see gpuwrf/__init__.py _OFFLINE_COMMAND)")
        self.scratch_root = Path(scratch_root)
        self.share_executables = bool(share_executables)
        self.namelist_signature: str | None = None
        self.environment: dict[str, str] | None = None
        self.platform: str | None = None
        self.runs: list[dict[str, Any]] = []
        self.created = time.perf_counter()
        if self.share_executables:
            from gpuwrf.runtime import domain_tree

            domain_tree.set_process_aot_reuse(True)

    def _admit(self, case_dir: Path) -> None:
        signature = namelist_signature(case_dir)
        environment = _pinned_environment()
        if self.namelist_signature is None:
            self.namelist_signature, self.environment = signature, environment
            return
        if signature != self.namelist_signature:
            raise ValueError(f"{case_dir}: namelist outside &time_control differs from the first member of "
                             "this process (import-time switches are pinned); run it in another process")
        if environment != self.environment:
            changed = sorted(set(environment.items()) ^ set((self.environment or {}).items()))
            raise ValueError(f"process environment changed between members: {[k for k, _ in changed]}")

    def run(self, member: str, case_dir: Path, member_root: Path, hours: int, *,
            checkpoint_dir: Path | None = None, checkpoint_interval_steps: int = 0,
            resume_checkpoint: Path | None = None, extend_run: bool = False,
            aot_prefetch: bool | None = None) -> dict[str, Any]:
        """One ``gpuwrf run`` of ``case_dir`` (d01, max_dom 1); history files land in ``member_root``."""
        global _EVENTS
        case_dir, member_root = Path(case_dir), Path(member_root)
        if isinstance(hours, bool) or int(hours) != hours or int(hours) <= 0:
            raise ValueError("hours must be a positive integer")
        if member_root.exists() and resume_checkpoint is None:
            raise ValueError(f"{member_root}: new member directory required")
        safe_label("member", member)
        refuse_protected_inputs(f"member {member}", [case_dir, *filter(None, [resume_checkpoint])],
                                outputs=[member_root, self.scratch_root, *filter(None, [checkpoint_dir])],
                                runtime=True)
        self._admit(case_dir)
        _listen_compile_events()
        member_root.mkdir(parents=True, exist_ok=resume_checkpoint is not None)
        output_dir, proof_dir = member_root, member_root / "proofs"
        scratch = self.scratch_root / f"{member}-{len(self.runs):04d}"
        argv = ["run", "--input-dir", str(case_dir), "--output-dir", str(output_dir), "--proof-dir", str(proof_dir),
                "--scratch-dir", str(scratch), "--domain", "d01", "--max-dom", "1", "--hours", str(int(hours))]
        if checkpoint_dir is not None:
            argv += ["--checkpoint-dir", str(checkpoint_dir), "--checkpoint-interval-steps",
                     str(int(checkpoint_interval_steps))]
        if resume_checkpoint is not None:
            argv += ["--resume-checkpoint", str(resume_checkpoint)]
            if extend_run:
                argv.append("--extend-run")
        # After the first member the executables are already resident: a speculative prefetch
        # would only load a second copy of the same blob in the background.
        if aot_prefetch is None:
            aot_prefetch = not (self.share_executables and self.runs)
        if not aot_prefetch:
            argv.append("--no-aot-prefetch")
        from gpuwrf.cli import main as cli

        stdout_path = member_root / f"cli_stdout_{len(self.runs):04d}.json"
        _EVENTS = []
        sys.stderr.write(f"[gpuwrf:da-members] member={member} case={case_dir} hours={int(hours)}\n")
        sys.stderr.flush()
        start, error, rc, refused = time.perf_counter(), None, 1, []
        try:
            with stdout_path.open("w") as stream, contextlib.redirect_stdout(stream), \
                    refuse_protected_opens() as refused:
                rc = int(cli(argv))
            if refused:
                raise ProtectedInputError(f"member {member}: the forecast tried to open ALISIOS data {refused[:3]}")
        except BaseException as exc:  # recorded, then re-raised: a failed member is never silent
            error = {"type": type(exc).__name__, "message": str(exc), "refused_opens": list(refused)}
            raise
        finally:
            wall = time.perf_counter() - start
            events, _EVENTS = _EVENTS, None
            import shutil

            shutil.rmtree(scratch, ignore_errors=True)
            payload_path = proof_dir / "nested_pipeline_run.json"
            verdict = None
            if payload_path.is_file():
                verdict = json.loads(payload_path.read_text()).get("verdict")
            import jax

            platform = jax.devices()[0].platform
            if self.platform is None:
                self.platform = platform
            files = sorted(output_dir.glob("wrfout_d01_*")) if output_dir.is_dir() else []
            record = {
                "member": member, "case": str(case_dir), "member_root": str(member_root), "hours": int(hours),
                "argv": argv, "rc": rc, "verdict": verdict, "error": error, "wall_seconds": wall,
                "platform": platform, "index_in_process": len(self.runs),
                "seconds_since_driver_start": time.perf_counter() - self.created,
                "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "compile_events": _event_categories(events or []),
                "aot": _aot_domains(),
                "case_manifest_sha256": (sha256(case_dir / "cycle_case_manifest.json")
                                         if (case_dir / "cycle_case_manifest.json").is_file() else None),
                "wrfout": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files},
            }
            self.runs.append(record)
            (member_root / f"member_receipt_{record['index_in_process']:04d}.json").write_text(
                json.dumps(record, indent=1, sort_keys=True) + "\n")
        if rc != 0 or verdict != "PIPELINE_GREEN":
            raise RuntimeError(f"member {member}: gpuwrf run rc={rc} verdict={verdict} (see {stdout_path})")
        return record

    def receipt(self) -> dict[str, Any]:
        walls = [r["wall_seconds"] for r in self.runs]
        return {"schema": 1, "kind": "gpuwrf da members (one process)", "platform": self.platform,
                "share_executables": self.share_executables, "namelist_signature": self.namelist_signature,
                "pinned_environment": self.environment, "n_runs": len(self.runs),
                "member_wall_seconds": walls, "driver_wall_seconds": time.perf_counter() - self.created,
                "runs": self.runs}


def run_members(driver: MemberDriver, members: Mapping[str, Path], out_root: Path, hours: int) -> dict[str, Any]:
    """Run every member case (label → case dir) into ``out_root/<label>``; stop at the first failure."""
    out_root = Path(out_root)
    results = {}
    for label, case in members.items():
        results[label] = driver.run(label, Path(case), out_root / label, hours)
    return results


def members_from_manifest(path: Path, *, include_control: bool = False) -> dict[str, Path]:
    """Member cases of an ENSEMBLE_MANIFEST.json (a1-io / member_cycle format), label order."""
    refuse_protected_inputs("members manifest", [path])
    manifest = json.loads(Path(path).read_text())
    members = manifest["members"]
    cases = {safe_label("manifest member", label): Path(members[label]["case"]) for label in sorted(members)
             if include_control or label != "ctrl"}
    refuse_protected_inputs("members manifest cases", cases.values())
    return cases


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="gpuwrf da members", description=__doc__.split("\n\n")[0])
    add_members_arguments(parser)
    return members_command(parser.parse_args(argv))


def add_members_arguments(parser) -> None:
    parser.add_argument("--manifest", type=Path, required=True, help="ENSEMBLE_MANIFEST.json (members -> case)")
    parser.add_argument("--hours", type=int, required=True, help="forecast hours of every member")
    parser.add_argument("--out", type=Path, required=True, help="new output root (one directory per member)")
    parser.add_argument("--include-control", action="store_true", help="also run the manifest's 'ctrl' case")
    parser.add_argument("--no-share-executables", action="store_true",
                        help="reload the step executable for every member (A/B control)")


def members_command(args) -> int:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out}: new output root required")
    out.mkdir(parents=True)
    driver = MemberDriver(out / ".scratch", share_executables=not args.no_share_executables)
    members = members_from_manifest(args.manifest, include_control=args.include_control)
    try:
        run_members(driver, members, out, args.hours)
    finally:
        receipt = driver.receipt()
        receipt["manifest"] = {"path": str(args.manifest), "sha256": sha256(args.manifest)}
        (out / "MEMBERS_RECEIPT.json").write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
