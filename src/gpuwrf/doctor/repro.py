"""Streaming input hashes and read-only reproducibility verification."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess

from .common import case_inputs, parse_namelist, support_report

ENV_KEYS = ("JAX_PLATFORMS", "JAX_ENABLE_X64", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR", "OMP_NUM_THREADS",
            "CUDA_VISIBLE_DEVICES")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity():
    from gpuwrf import __version__
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    commit = None
    root = package.parent.parent
    if (root / ".git").exists():
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            commit = result.stdout.strip()
    return {"version": __version__, "commit": commit, "python_source_sha256": digest.hexdigest()}


def environment():
    from gpuwrf._fast_defaults import FAST_PATH_DEFAULTS
    # Explicit allowlist: never capture arbitrary credentials in the environment.
    enabled = os.environ.get("GPUWRF_FAST_DEFAULTS", "1").strip().lower() not in {"0", "false", "no", "off"}
    flags = {key: os.environ.get(key, default if enabled else None) for key, default in FAST_PATH_DEFAULTS.items()}
    # Include opt-in controls declared by this installed source, including those
    # outside the release-default bundle; unknown environment names are excluded.
    for path in Path(__file__).resolve().parents[1].rglob("*.py"):
        for key in re.findall(r'[\'\"](GPUWRF_[A-Z0-9_]+)[\'\"]', path.read_text()):
            if any(word in key for word in ("SECRET", "PASSWORD", "TOKEN", "CREDENTIAL", "API_KEY")):
                continue
            if key not in flags and key in os.environ:
                flags[key] = os.environ[key]
    flags["GPUWRF_FAST_DEFAULTS"] = os.environ.get("GPUWRF_FAST_DEFAULTS", "1")
    packages = {}
    for name in ("numpy", "netCDF4", "jax", "jaxlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "system": platform.platform(), "packages": packages,
            "flags": flags, "variables": {key: os.environ.get(key) for key in ENV_KEYS}}


def create_manifest(case_dir, namelist=None):
    case = Path(case_dir).resolve()
    nlpath = Path(namelist).resolve() if namelist else case / "namelist.input"
    nl = parse_namelist(nlpath)
    paths = case_inputs(case, nlpath)
    inputs = []
    for path in paths:
        try:
            relative = str(path.relative_to(case))
        except ValueError:
            relative = None
        inputs.append({"path": str(path), "relative_path": relative, "bytes": path.stat().st_size,
                       "sha256": sha256(path), "role": "namelist" if path == nlpath else "input"})
    return {"schema": "GpuwrfReproducibility", "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(), "case_dir": str(case),
            "inputs": inputs, "namelist": nl, "namelist_text": nlpath.read_text(),
            "gpuwrf": identity(), "environment": environment(), "support": support_report(nl),
            "scope": "native input files and case-local parameter tables; dry-run only; not a replay guarantee"}


def reproduce(manifest_path, case_dir=None):
    manifest = json.loads(Path(manifest_path).read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != "GpuwrfReproducibility" or manifest.get("schema_version") != 1:
        raise ValueError("unsupported reproducibility manifest schema/version")
    if not isinstance(manifest.get("inputs"), list) or not manifest["inputs"]:
        raise ValueError("manifest has no input inventory")
    case = Path(case_dir).resolve() if case_dir else Path(manifest["case_dir"]).resolve()
    checks = []
    nlpath = None
    expected = set()
    for entry in manifest["inputs"]:
        relative = entry["relative_path"]
        if relative is not None:
            part = Path(relative)
            if part.is_absolute() or ".." in part.parts:
                raise ValueError("manifest relative_path must stay inside case_dir")
            path = case / part
        else:
            path = Path(entry["path"])
        if path in expected:
            raise ValueError("duplicate input inventory entry")
        expected.add(path)
        if entry["role"] == "namelist":
            if nlpath is not None:
                raise ValueError("multiple namelists in manifest")
            nlpath = path
        matches = path.is_file() and path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"]
        checks.append(dict(check="input_sha256", path=str(path), matches=matches))
    if nlpath is None:
        raise ValueError("manifest has no namelist")
    actual = set(case_inputs(case, nlpath))
    checks.append(dict(check="input_inventory", matches=actual == expected,
                       added=sorted(map(str, actual - expected)), missing=sorted(map(str, expected - actual))))
    if nlpath.is_file():
        nl = parse_namelist(nlpath)
        checks.append(dict(check="namelist", matches=nl == manifest["namelist"] and nlpath.read_text() == manifest["namelist_text"]))
        checks.append(dict(check="support", matches=support_report(nl) == manifest["support"]))
    current_identity = identity()
    for key in ("version", "python_source_sha256"):
        checks.append(dict(check=f"gpuwrf_{key}", matches=current_identity[key] == manifest["gpuwrf"][key]))
    env = environment()
    for key in ("flags", "variables", "packages", "python"):
        checks.append(dict(check=f"environment_{key}", matches=env[key] == manifest["environment"][key]))
    return {"schema": "GpuwrfReproduceCheck", "schema_version": 1, "dry_run": True,
            "status": "PASS" if all(c["matches"] for c in checks) else "FAIL", "checks": checks,
            "recorded_commit": manifest["gpuwrf"].get("commit"), "current_commit": current_identity["commit"],
            "scope": "same source bytes checked; commit and host system disclosed; no forecast launched"}
