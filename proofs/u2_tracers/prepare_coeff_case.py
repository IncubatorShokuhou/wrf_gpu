"""Stage/audit history-only CPU-WRF coefficient diagnostics without source edits.

Usage: python prepare_coeff_case.py OUTPATH [--extra-fields AL,ALB] [--audit]
Stage refuses an existing OUTPATH. Run the original WRF there separately, then
--audit checks all original history variables byte-for-byte and removes the
three disposable private Thompson tables. TR01 reference is read-only.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys


REFERENCE = Path("<USER_HOME>/wrf_gpu2_lanes/u2-tracers/TR01_original_cpu_case")
# The iofields request uses the Registry identifier 'alt'; ordinary state I/O
# uses uppercase DataName='ALT' (generated inc/allocs_1.F:3598-3599). Unlike
# 4-D scalar tr17 names, this field passes through WRF's uppercase generator.
DIAGNOSTICS = ("EXCH_H", "EXCH_M", "ALT")
_SOURCE = Path(__file__).with_name("prepare_cpu_tracer_case.py")
_SPEC = importlib.util.spec_from_file_location("_original_cpu_stage_helpers", _SOURCE)
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)


def reference_hashes() -> dict[str, str]:
    # Include all regular reference artifacts, and the linked boundary input.
    # Read-only radiation/table symlinks need not be rehashed for preservation.
    paths = [p for p in REFERENCE.iterdir() if p.is_file() and not p.is_symlink()]
    paths.append(REFERENCE / "wrfbdy_d01")
    return {p.name: _HELPERS.file_hash(p) for p in sorted(paths)}


def prepare(root: Path, *, extra_fields: tuple[str, ...] = ()) -> dict:
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"refusing existing diagnostic case: {root}")
    diagnostics = (*DIAGNOSTICS, *extra_fields)
    if (len(set(diagnostics)) != len(diagnostics)
            or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) for name in diagnostics)):
        raise ValueError("diagnostic field names must be unique valid WRF names")
    if len(list(REFERENCE.glob("wrfout_d01_*"))) != 4:
        raise ValueError("original TR01 four-frame reference required")
    _HELPERS.check_original_executable(REFERENCE / "wrf.exe")
    original = json.loads((REFERENCE / "PREPARATION.json").read_text())
    before = reference_hashes()
    if before["wrf.exe"] != original["executable_sha256"]:
        raise ValueError("original executable hash does not match TR01 provenance")
    if before["wrfinput_d01"] != original["input_sha256"]:
        raise ValueError("original input hash does not match TR01 provenance")
    copies = {"wrf.exe", "wrfinput_d01", "wrfbdy_d01", *_HELPERS.TABLE_NAMES}
    for name in copies:
        if not (REFERENCE / name).is_file():
            raise FileNotFoundError(f"original private source missing: {name}")
    root.mkdir(parents=True)
    for name in copies:
        shutil.copyfile(REFERENCE / name, root / name)
    (root / "wrf.exe").chmod(0o755)
    for path in REFERENCE.iterdir():
        if (path.name in copies or path.name.startswith((
            "wrfout", "wrfrst", "rsl.", "namelist", "PREPARATION", "tracer_coeff_fields"
        ))):
            continue
        if path.is_file():
            (root / path.name).symlink_to(path.resolve())
    text = _HELPERS.edit((REFERENCE / "namelist.input").read_text(), "time_control", {
        "iofields_filename": "'tracer_coeff_fields.txt'"
    })
    (root / "namelist.input").write_text(text)
    requested = ("EXCH_H", "EXCH_M", "alt", *extra_fields)
    (root / "tracer_coeff_fields.txt").write_text("+:h:0:" + ",".join(requested) + "\n")
    receipt = {
        "schema": "u2-tracers-original-coefficient-history-v1", "reference": str(REFERENCE),
        "original_reference_sha256": before,
        "copied_source_sha256": {name: _HELPERS.file_hash(root / name) for name in sorted(copies)},
        "diagnostics": diagnostics, "requested_iofields": requested,
        "namelist_sha256": _HELPERS.file_hash(root / "namelist.input"),
        "iofields_sha256": _HELPERS.file_hash(root / "tracer_coeff_fields.txt"),
        "steps": 3, "dt_s": 18, "history_interval_s": 18,
        "wrf_body_or_source_edits": False, "ran_wrf": False,
    }
    for name in copies:
        if receipt["copied_source_sha256"][name] != before[name]:
            raise ValueError(f"copied source differs: {name}")
    if reference_hashes() != before:
        raise ValueError("original reference changed during staging")
    (root / "COEFFICIENT_PREPARATION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def audit(root: Path) -> dict:
    import numpy as np

    preparation = json.loads((root / "COEFFICIENT_PREPARATION.json").read_text())
    diagnostics = tuple("ALT" if name == "alt" else name for name in preparation["diagnostics"])
    Dataset = _HELPERS.locked_dataset()
    frames, failures = [], []
    original_paths = sorted(REFERENCE.glob("wrfout_d01_*"))
    candidate_paths = sorted(root.glob("wrfout_d01_*"))
    if [p.name for p in candidate_paths] != [p.name for p in original_paths]:
        raise ValueError("diagnostic case does not have the four matching reference frames")
    for original_path in original_paths:
        path = root / original_path.name
        mismatched, nonfinite, coefficients = [], [], {}
        with Dataset(original_path) as original, Dataset(path) as candidate:
            original.set_auto_maskandscale(False)
            candidate.set_auto_maskandscale(False)
            for name, variable in original.variables.items():
                if name not in candidate.variables:
                    mismatched.append(name + ":missing")
                    continue
                other = candidate.variables[name]
                expected, actual = np.asarray(variable[:]), np.asarray(other[:])
                if (variable.dimensions != other.dimensions or expected.dtype != actual.dtype
                        or expected.shape != actual.shape or expected.tobytes() != actual.tobytes()):
                    mismatched.append(name)
                if actual.dtype.kind in "fiu" and not np.isfinite(actual).all():
                    nonfinite.append(name)
            for name in diagnostics:
                if name not in candidate.variables:
                    failures.append(f"{path.name}:{name}:missing")
                    continue
                variable = candidate.variables[name]
                values = np.asarray(variable[:])
                finite = bool(np.isfinite(values).all())
                coefficients[name] = {
                    "shape": values.shape, "dimensions": variable.dimensions,
                    "stagger": str(getattr(variable, "stagger", "")),
                    "units": str(getattr(variable, "units", "")),
                    "MemoryOrder": str(getattr(variable, "MemoryOrder", "")),
                    "dtype": str(values.dtype), "finite": finite,
                    "nonzero": int(np.count_nonzero(values)),
                    "min": float(np.min(values)), "max": float(np.max(values)),
                }
                if not finite:
                    nonfinite.append(name)
            if mismatched or nonfinite:
                failures.append(f"{path.name}:original_byte={mismatched}:nonfinite={nonfinite}")
            frames.append({
                "file": str(path), "sha256": _HELPERS.file_hash(path),
                "original_variable_count": len(original.variables),
                "original_variable_mismatches": mismatched, "nonfinite_variables": nonfinite,
                "coefficients": coefficients,
            })
    unchanged = reference_hashes() == preparation["original_reference_sha256"]
    if not unchanged:
        failures.append("original reference file hashes changed")
    success = "SUCCESS COMPLETE WRF" in (root / "rsl.error.0000").read_text()
    if not success:
        failures.append("original WRF did not report SUCCESS COMPLETE WRF")
    result = {
        "schema": "u2-tracers-original-coefficient-audit-v1", "frames": frames,
        "original_reference_unchanged": unchanged, "original_wrf_success": success,
        "all_original_variables_byte_identical": not any(f["original_variable_mismatches"] for f in frames),
        "failures": failures, "accepted": not failures,
        "jax_imported": any(n == "jax" or n.startswith("jax.") for n in sys.modules),
        "port_parity_claim": False, "wrf_body_or_source_edits": False,
    }
    if result["jax_imported"]:
        result["failures"].append("unexpected JAX import")
        result["accepted"] = False
    # This script owns only the three explicitly named copied writable tables.
    cleaned = []
    for name in _HELPERS.TABLE_NAMES:
        path = root / name
        if path.is_file() and not path.is_symlink():
            path.unlink()
            cleaned.append(name)
    result["cleaned_private_tables"] = cleaned
    result["private_tables_absent"] = all(not (root / name).exists() for name in _HELPERS.TABLE_NAMES)
    (root / "COEFFICIENT_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpath", type=Path)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--extra-fields", default="", help="comma-separated additional WRF output names")
    args = parser.parse_args()
    if args.audit and args.extra_fields:
        parser.error("--audit reads field names from the preparation receipt")
    root = args.outpath.resolve()
    extra_fields = tuple(args.extra_fields.split(",")) if args.extra_fields else ()
    result = audit(root) if args.audit else prepare(root, extra_fields=extra_fields)
    print(json.dumps(result, indent=2))
    if result.get("accepted") is False:
        raise SystemExit(1)
