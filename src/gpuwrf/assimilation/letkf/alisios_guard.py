"""Guarded reads of ALISIOS data (binding rule, wrf_gpu2 manager 2026-10-17 01:25Z; ALISIOS slot conditions).

No ALISIOS data may be read outside ``alisios.guard_call`` scan+admit with a read spy, and NO guard-DENY day may be
accessed for any purpose. Usage::

    with guarded_session({path: "YYYYMMDD", ...}, issue_days, receipt_dir) as session:
        ... read exactly those files ...
    session.receipt  # guard receipt + every observed open

Before anything is opened the guard scans every requested day and refuses the whole session if one is DENY
(``drop_deny_days`` filters candidates first). Then ``guard_call.admit`` runs on the exact read set. Inside the
session, opens of paths under the ALISIOS DATA roots are allowed only for the listed files and recorded: Python
``open`` via an audit hook, netCDF via a checking wrapper around ``netCDF4.Dataset`` (C-level opens are invisible to
audit hooks; every netCDF reader in this package and ``gpuwrf.io.netcdf_lock`` resolves ``netCDF4.Dataset`` at call
time). Code under the ALISIOS source tree is not payload. Values are never logged.

ALISIOS's own reader facade (``alisios.guarded_read``, stage 1) is preferred where it fits: ``facade()`` imports it
from a copy sealed out of the ALISIOS git object store at ``FACADE_COMMIT`` (never the working tree; every file
checked against its git blob id) and ``facade_stage`` copies admitted files to a private stage inside one
``GuardedRead`` session (scan -> admit -> open + READ_SPY). The guard script itself stays the live ALISIOS one, as
for ``guard_call``; its SHA-256 is recorded.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ALISIOS_SRC = "<USER_HOME>/src/alisios/src"
ALISIOS_REPO = "<USER_HOME>/src/alisios"
DATA_ROOTS = tuple(Path(p).resolve() for p in ("<DATA_ROOT>/server/work/src/alisios", "<DATA_ROOT>/server/work/alisios", "<DATA_ROOT>/alisios"))
MODE = "CALIBRATION_READ"
GUARD = Path(ALISIOS_REPO) / "ops/guards/blocklist_guard_v1_10.py"   # = guard_call.GUARD of the ALISIOS checkout
FACADE_COMMIT = "80b076c9fb507aa9f8034b8fcc6effd1c3e2454f"         # ALISIOS main, guarded_read stage 1 (2026-10-17)
FACADE_FILES = ("src/alisios/__init__.py", "src/alisios/guard_call.py", "src/alisios/guarded_read.py")
FACADE_MODULES = ("duckdb", "pyarrow.parquet", "pandas", "netCDF4", "xarray", "numpy", "pyarrow.dataset")
_ACTIVE = None
_ADMITTED_SESSION = ContextVar("alisios_admitted_session", default=None)
_PATH_RESOLUTION = ContextVar("alisios_path_resolution", default=False)
_AUDITED_OS_OPEN = ContextVar("alisios_audited_os_open", default=None)
_HOOKED = False
_GUARD_SETUP = ContextVar("alisios_guard_setup", default=0)  # guard metadata, not payload
_FACADE: dict = {}


@contextmanager
def _guard_setup():
    token = _GUARD_SETUP.set(_GUARD_SETUP.get() + 1)
    try:
        yield
    finally:
        _GUARD_SETUP.reset(token)


def in_guard_setup() -> bool:
    return _GUARD_SETUP.get() > 0


def admitted_session():
    """The scan/admit-completed session in this context, never an environment bypass."""
    return _ADMITTED_SESSION.get()


def resolving_path() -> bool:
    """Only the resolver's own stat/readlink calls may bypass metadata checks."""
    return _PATH_RESOLUTION.get()


def resolve_path(path, *, dir_fd=None) -> Path:
    """Canonical path, including already-open descriptors and openat-style paths."""
    token = _PATH_RESOLUTION.set(True)
    try:
        if isinstance(path, int):
            path = f"/proc/self/fd/{path}"
        p = Path(os.fsdecode(path))
        if dir_fd is not None and not p.is_absolute():
            p = Path(f"/proc/self/fd/{dir_fd}") / p
        return p.resolve()
    finally:
        _PATH_RESOLUTION.reset(token)


@contextmanager
def audited_os_open(path, *, dir_fd=None):
    """Supply dir_fd context missing from CPython's open audit event."""
    token = _AUDITED_OS_OPEN.set((os.fsdecode(path), resolve_path(path, dir_fd=dir_fd)))
    try:
        yield
    finally:
        _AUDITED_OS_OPEN.reset(token)


def audit_open_path(path):
    opened = _AUDITED_OS_OPEN.get()
    if opened is not None and not isinstance(path, int) and os.fsdecode(path) == opened[0]:
        return opened[1]
    return path


def _guard_call():
    if ALISIOS_SRC not in sys.path:
        sys.path.insert(0, ALISIOS_SRC)
    from alisios import guard_call
    return guard_call


def is_alisios_data(path, *, roots=None) -> bool:
    try:
        p = resolve_path(path)
    except (TypeError, ValueError):
        return False
    return any(p == root or root in p.parents for root in (DATA_ROOTS if roots is None else roots))


def drop_deny_days(candidates, *, mode=MODE):
    """candidates: iterable of (item, 'YYYYMMDD'). Returns (allowed items, verdicts) — scan only, nothing opened."""
    candidates = list(candidates)
    days = sorted({d for _, d in candidates})
    if not days:
        return [], {}
    with _guard_setup():
        verdicts = _guard_call().scan(days, mode=mode)
    return [item for item, d in candidates if verdicts.get(d) == "ALLOW"], verdicts


class _Session:
    def __init__(self, allowed, issue_days, receipt_dir, purpose):
        self.allowed = {str(resolve_path(p)): ([d] if isinstance(d, str) else sorted(set(d))) for p, d in allowed.items()}
        self.issue_days = sorted({str(d) for d in issue_days})
        self.receipt_dir = Path(receipt_dir)
        self.purpose = purpose
        self.observed, self.refused = {}, []
        self.receipt = None

    def check(self, path, kind):
        if not is_alisios_data(path):
            return
        p = str(resolve_path(path))
        if p not in self.allowed:
            self.refused.append(p)
            raise _guard_call().GuardRefused(f"a1 read spy refused an unlisted ALISIOS payload path: {p}")
        row = self.observed.setdefault(p, {"days": self.allowed[p], "kind": kind, "open_count": 0})
        row["open_count"] += 1


def _audit(event, args):
    if event != "open" or _ACTIVE is None:
        return
    file, mode, flags = args
    if not isinstance(file, (str, bytes, os.PathLike, int)):
        return
    if isinstance(mode, str) and not ("r" in mode or "+" in mode):
        return
    _ACTIVE.check(audit_open_path(file), "python_open")


@contextmanager
def guarded_session(allowed: dict, issue_days, receipt_dir, *, purpose="A1 LETKF DA probe (development cases)", mode=MODE):
    """Guard-admitted read session for exactly ``allowed`` = {path: valid day YYYYMMDD or list of days}."""
    global _ACTIVE, _HOOKED
    import netCDF4

    if _ACTIVE is not None:
        raise RuntimeError("nested guarded_session")
    gc = _guard_call()
    session = _Session(allowed, issue_days, receipt_dir, purpose)
    valid_days = sorted({d for days in session.allowed.values() for d in days})
    session.receipt_dir.mkdir(parents=True, exist_ok=True)
    admit_path = session.receipt_dir / "guard_admit.json"
    if not session.allowed:
        # Nothing from ALISIOS is planned: no guard call (it refuses empty day lists); the spy below still refuses
        # every ALISIOS payload open inside the session.
        admitted = {"result": "NO_ALISIOS_READS_PLANNED"}
    else:
        with _guard_setup():
            verdicts = gc.scan(sorted(set(valid_days) | set(session.issue_days)), mode=mode)
        deny = {d: v for d, v in verdicts.items() if v != "ALLOW"}
        if deny:
            raise gc.GuardRefused(f"requested read set contains non-ALLOW days {deny}; nothing was opened")
        with _guard_setup():
            admitted = gc.admit(session.issue_days, valid_days, mode=mode, receipt_out=str(admit_path))
    original = netCDF4.Dataset

    def checked_dataset(filename, *args, **kwargs):
        session.check(filename, "netcdf_open")
        return original(filename, *args, **kwargs)

    if not _HOOKED:
        sys.addaudithook(_audit)
        _HOOKED = True
    _ACTIVE = session
    token = _ADMITTED_SESSION.set(session)
    # The test-suite class proxy already performs this exact session check,
    # including cached Dataset aliases, and preserves xarray's isinstance API.
    netCDF4.Dataset = original if getattr(original, "_alisios_read_spy", False) else checked_dataset
    try:
        yield session
    finally:
        netCDF4.Dataset = original
        _ADMITTED_SESSION.reset(token)
        _ACTIVE = None
        session.receipt = {"schema": "a1_da.guarded_session.v1", "purpose": session.purpose, "mode": mode,
                           "issue_days": session.issue_days, "valid_days": valid_days,
                           "guard_admit_receipt": str(admit_path), "guard_result": admitted.get("result"),
                           "spy": {"method": "python audit hook + netCDF4.Dataset wrapper", "observed": session.observed,
                                   "refused_paths": session.refused, "listed_paths": len(session.allowed),
                                   "values_logged": False}}
        (session.receipt_dir / "guarded_session.json").write_text(json.dumps(session.receipt, indent=1, default=str) + "\n")


# ------------------------------------------------------------------------------- ALISIOS reader facade

def _git(repo, *args) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, timeout=60).stdout


def _git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def seal_facade(sealed_root=None, *, repo=ALISIOS_REPO, commit=FACADE_COMMIT) -> tuple[Path, dict]:
    """Write ``FACADE_FILES`` at ``commit`` from the git object store to ``sealed_root/<commit>/``.

    Returns (directory, {file: blob id}). A present file is reused only if it hashes to its blob id."""
    base = Path(sealed_root) if sealed_root else (
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "gpuwrf" / "alisios_sealed")
    dest = base / commit
    blobs = {}
    for rel in FACADE_FILES:
        blob = _git(repo, "rev-parse", f"{commit}:{rel}").decode().strip()
        target = dest / rel
        if not (target.is_file() and _git_blob_id(target.read_bytes()) == blob):
            data = _git(repo, "cat-file", "blob", blob)
            if _git_blob_id(data) != blob:
                raise RuntimeError(f"{rel}@{commit}: git blob does not hash to its id")
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)
        blobs[rel] = blob
    return dest, blobs


def _import_sealed(src: Path):
    """Import alisios.guarded_read/guard_call from ``src`` without touching an already imported ``alisios``."""
    names = lambda: [k for k in sys.modules if k == "alisios" or k.startswith("alisios.")]  # noqa: E731
    saved = {k: sys.modules.pop(k) for k in names()}
    sys.path.insert(0, str(src))
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        reader = importlib.import_module("alisios.guarded_read")
        guard = sys.modules["alisios.guard_call"]
        for module in (reader, guard):
            if Path(module.__file__).resolve().parent != (src / "alisios").resolve():
                raise ImportError(f"{module.__name__} resolved outside the sealed copy: {module.__file__}")
    finally:
        sys.dont_write_bytecode = dont_write
        sys.path.remove(str(src))
        for k in names():
            del sys.modules[k]
        sys.modules.update(saved)
    return reader, guard


def facade(sealed_root=None):
    """(guarded_read module, its guard_call module, info) of the sealed facade, or (None, None, info) with
    ``info['unavailable']`` = why the fallback (``guarded_session``) has to be used. Cached per sealed root."""
    key = str(sealed_root)
    if key not in _FACADE:
        info = {"commit": FACADE_COMMIT, "repo": ALISIOS_REPO}
        try:
            missing = [m for m in FACADE_MODULES if importlib.util.find_spec(m) is None]
            if missing:
                raise ImportError(f"optional modules missing for alisios.guarded_read: {missing}")
            dest, blobs = seal_facade(sealed_root)
            reader, guard = _import_sealed(dest / "src")
            info.update(sealed=str(dest), blobs=blobs)
            _FACADE[key] = (reader, guard, info)
        except (OSError, ImportError, RuntimeError, subprocess.SubprocessError) as exc:
            _FACADE[key] = (None, None, {**info, "unavailable": f"{type(exc).__name__}: {exc}"})
    return _FACADE[key]


def stage_path(path, stage_root) -> Path:
    """Stage location keeping the layout below its data root (a namelist stays next to its wrfout)."""
    p = resolve_path(path)
    for k, root in enumerate(DATA_ROOTS):
        if p == root or root in p.parents:
            return Path(stage_root) / f"root{k}" / p.relative_to(root)
    raise ValueError(f"not ALISIOS data: {p}")


def facade_stage(targets: dict, stage_root, receipt_dir, *, purpose, mode=MODE, sealed_root=None):
    """Copy ``targets`` = {path: (issue_days, valid_days)} (complete day coverage of each whole file) into
    ``stage_root`` inside ONE ``alisios.guarded_read.GuardedRead`` session; returns ({path: staged path}, receipt).

    The facade scans and admits every file before the first payload open (a DENY anywhere refuses the whole batch)
    and logs every open to READ_SPY; this module's refusing spy additionally covers all ``DATA_ROOTS``."""
    global _ACTIVE, _HOOKED
    reader_mod, guard_mod, info = facade(sealed_root)
    if reader_mod is None:
        raise RuntimeError(f"alisios.guarded_read unavailable: {info['unavailable']}")
    paths = {str(resolve_path(p)): (sorted(set(i)), sorted(set(v))) for p, (i, v) in targets.items()}
    receipt_dir, stage_root = Path(receipt_dir), Path(stage_root).resolve()
    receipt_dir.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, mode=0o700)
    spy = _Session({p: v for p, (_, v) in paths.items()}, [d for i, _ in paths.values() for d in i], receipt_dir, purpose)
    roots = sorted({str(Path(p).parent) for p in paths})
    session = reader_mod.GuardedRead(
        [reader_mod.Target(p, tuple(i), tuple(v), complete=True) for p, (i, v) in paths.items()],
        protected_roots=roots, read_spy=receipt_dir / "READ_SPY.jsonl", mode=mode, guard=str(GUARD))
    with _guard_setup():                       # scan + admit (guard metadata reads) run inside __enter__
        session.__enter__()
    staged, files = {}, {}
    if not _HOOKED:
        sys.addaudithook(_audit)
        _HOOKED = True
    token = _ADMITTED_SESSION.set(spy)
    try:
        _ACTIVE = spy
        for p in paths:
            target = stage_path(p, stage_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, size = hashlib.sha256(), 0
            with open(p, "rb") as src, open(target, "xb") as dst:
                for block in iter(lambda: src.read(1 << 22), b""):
                    digest.update(block)
                    dst.write(block)
                    size += len(block)
            staged[p] = str(target)
            files[p] = {"staged": str(target.relative_to(stage_root)), "sha256": digest.hexdigest(), "size": size,
                        "issue_days": paths[p][0], "valid_days": paths[p][1]}
    except BaseException as exc:
        _ACTIVE = None
        session.__exit__(type(exc), exc, exc.__traceback__)
        raise
    finally:
        _ADMITTED_SESSION.reset(token)
    _ACTIVE = None
    session.__exit__(None, None, None)
    admits = receipt_dir / "guard_admit_receipts.json"
    admits.write_text(json.dumps(session.receipts, indent=1, default=str) + "\n")
    receipt = {"schema": "gpuwrf.da.alisios_facade.v1", "purpose": purpose, "mode": mode,
               "facade": {k: info[k] for k in ("commit", "repo", "sealed", "blobs")},
               "guard": {"path": str(GUARD), "sha256": _sha256(GUARD) if GUARD.is_file() else None},
               "protected_roots": roots, "read_spy": str(receipt_dir / "READ_SPY.jsonl"),
               "counters": dict(session.counters), "guard_admit_receipts": str(admits),
               "files": files, "spy": {"observed": spy.observed, "refused_paths": spy.refused},
               "values_logged": False}
    (receipt_dir / "facade_stage.json").write_text(json.dumps(receipt, indent=1, default=str) + "\n")
    return staged, receipt


@contextmanager
def metadata_session(path, issue_day, receipt_dir, *, purpose, prefer_facade=True, sealed_root=None, mode=MODE):
    """Admitted read of one metadata file (a run's namelist) under its run's issue day: through the sealed facade
    when available (one GuardedRead, READ_SPY), else ``guarded_session``. Yields; read ``path`` inside."""
    global _ACTIVE, _HOOKED
    reader_mod = facade(sealed_root)[0] if prefer_facade else None
    if reader_mod is None:
        with guarded_session({str(path): [issue_day]}, [issue_day], receipt_dir, purpose=purpose, mode=mode) as s:
            yield s
        return
    receipt_dir = Path(receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    resolved = str(Path(path).resolve())
    spy = _Session({resolved: [issue_day]}, [issue_day], receipt_dir, purpose)
    session = reader_mod.GuardedRead([reader_mod.Target(resolved, (issue_day,), (issue_day,), complete=True)],
                                     protected_roots=[str(Path(resolved).parent)],
                                     read_spy=receipt_dir / "READ_SPY.jsonl", mode=mode, guard=str(GUARD))
    with _guard_setup():
        session.__enter__()
    if not _HOOKED:
        sys.addaudithook(_audit)
        _HOOKED = True
    try:
        _ACTIVE = spy
        yield spy
    except BaseException as exc:
        _ACTIVE = None
        session.__exit__(type(exc), exc, exc.__traceback__)
        raise
    _ACTIVE = None
    session.__exit__(None, None, None)
    spy.receipt = {"schema": "gpuwrf.da.alisios_metadata.v1", "backend": "guarded_read", "purpose": purpose,
                   "path": resolved, "issue_day": issue_day, "counters": dict(session.counters),
                   "read_spy": str(receipt_dir / "READ_SPY.jsonl"), "observed": spy.observed}
    (receipt_dir / "metadata_session.json").write_text(json.dumps(spy.receipt, indent=1, default=str) + "\n")
