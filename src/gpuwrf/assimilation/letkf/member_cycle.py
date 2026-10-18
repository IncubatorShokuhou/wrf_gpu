"""Per-member analysis wrfinputs + next-window cycle cases for real ensemble cycling (lane a1-io; CPU-WRF route).

For member m: analysis wrfinput from ITS OWN background wrfout frame (passive fields and Noah-MP land from that frame,
land marker via ``cycling.core.cycle``) with member m's analysed U/V/T/QVAPOR and rebalanced P/PH
(``state_io.write_analysis_wrfinput``), then ``cycle_case.write_cycle_case`` (land='cycled'). The manifest has the
ENSEMBLE_MANIFEST.json format of the 06Z IC set (``members`` dict → ``case``), consumed by the lead's
``proofs/a1_da/cpuwrf/run_members.sh``. Host NumPy only; no model run.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .state_io import (
    boundary_taper,
    read_frame,
    read_geometry,
    rebalance_report,
    sha256,
    write_analysis_wrfinput,
    wrf_time,
)

ANALYSED = ("U", "V", "T", "QVAPOR")
REBALANCED = ("P", "PH")
_STAGGER = {"U": "x", "V": "y"}


def spec_zone_changes(frame: Mapping[str, np.ndarray], fields: Mapping[str, np.ndarray], spec_bdy_width: int = 5) -> dict:
    """Cells changed (vs the member frame) inside the WRF specified+relaxation zone, per field."""
    out = {}
    for name, value in fields.items():
        ref = np.asarray(frame[name])
        ny = ref.shape[-2] - (_STAGGER.get(name) == "y")
        nx = ref.shape[-1] - (_STAGGER.get(name) == "x")
        zone = boundary_taper(ny, nx, spec_bdy_width, 0, _STAGGER.get(name)) == 0
        changed = np.asarray(value).astype(ref.dtype) != ref
        out[name] = int(changed[..., zone].sum())
    return out


def write_member_cycle(member_wrfouts: Sequence, analysis_member_fields: Mapping[str, np.ndarray], valid_time,
                       template_wrfinput, source_case_dir, hours: int, out_root, *, member_ids=None,
                       spec_bdy_width: int | None = None, extra_provenance: Mapping | None = None,
                       source_labels: Mapping[str, str] | None = None) -> Path:
    """Write one analysis wrfinput + one cycle case per member; return the ENSEMBLE_MANIFEST.json path.

    ``analysis_member_fields``: name → (N, …) for U, V, T, QVAPOR and the rebalanced P, PH (``analysis.analyse``
    output). MU is frozen. Refuses: N mismatch, missing P/PH (rebalance first), a member frame not valid exactly at
    ``valid_time``, any change inside the specified/relaxation zone (shared wrfbdy), an existing ``out_root``.
    """
    from .cycle_case import write_cycle_case

    labels = dict(source_labels or {})          # staged input -> original path (governance stage)
    t_a = wrf_time(valid_time)
    frames = [Path(p).resolve() for p in member_wrfouts]
    n = len(frames)
    missing = [k for k in ANALYSED + REBALANCED if k not in analysis_member_fields]
    if missing:
        raise ValueError(f"analysis fields missing {missing} (P/PH must come from rebalance_increment)")
    for name in ANALYSED + REBALANCED:
        if np.shape(analysis_member_fields[name])[0] != n:
            raise ValueError(f"{name}: leading axis {np.shape(analysis_member_fields[name])[0]} != {n} member frames")
    ids = [f"m{i:02d}" for i in range(n)] if member_ids is None else [str(m) for m in member_ids]
    if len(ids) != n or len(set(ids)) != n or "ctrl" in ids:
        raise ValueError("member_ids must be N distinct labels other than 'ctrl'")
    out_root = Path(out_root)
    if out_root.exists():
        raise ValueError(f"{out_root}: new output directory required")
    template = Path(template_wrfinput).resolve()
    source = Path(source_case_dir).resolve()
    geo = read_geometry(frames[0], base_wrfinput=template)
    from .analysis import resolve_spec_bdy_width
    spec_bdy_width, width_source = resolve_spec_bdy_width(geo, spec_bdy_width)   # the case zone, never a default 5
    names = ANALYSED + REBALANCED
    members = {}
    for i, (label, frame_path) in enumerate(zip(ids, frames)):
        frame = read_frame(frame_path, t_a, names + ("MU",))
        fields = {k: np.asarray(analysis_member_fields[k][i]).astype(frame[k].dtype) for k in names}
        zone = spec_zone_changes(frame, fields, spec_bdy_width)
        if any(zone.values()):
            raise ValueError(f"{label}: analysis changes the specified/relaxation zone {zone} (shared wrfbdy)")
        report = rebalance_report(frame, {**frame, **fields}, fields, geo)
        target = out_root / "members" / label / "wrfinput_d01"
        rec = write_analysis_wrfinput(template, frame_path, t_a, fields, target, source_labels=labels,
                                      extra_provenance={"kind": "member analysis (ensemble cycling)", "member": label,
                                                        "background": labels.get(str(frame_path), str(frame_path)),
                                                        **dict(extra_provenance or {})})
        case = write_cycle_case(source, {"d01": target}, t_a, hours, out_root / "cases" / label)
        adm = case["admission"]
        members[label] = {"wrfinput": str(target), "sha256": rec["output_sha256"], "case": case["output_case"],
                          "case_wrfinput_sha256": case["outputs"]["wrfinput_d01"]["sha256"],
                          "case_manifest_sha256": sha256(out_root / "cases" / label / "cycle_case_manifest.json"),
                          "background_wrfout": labels.get(str(frame_path), str(frame_path)),
                          "background_sha256": rec["member_sha256"],
                          "n_changed": {k: v["n_changed"] for k, v in rec["analysis_fields"].items()},
                          "spec_zone_changed": zone, "rebalance": report,
                          "admission": {k: (v.get("status") if isinstance(v, dict) else v) for k, v in adm.items()
                                        if k != "wrfbdy_coverage"}}
    manifest = {"schema": 1, "t_a": t_a, "hours": int(hours), "n_members": n,
                "spec_bdy_width": {"value": spec_bdy_width, "source": width_source,
                                   "namelist": labels.get(str(geo.get("SPEC_BDY_WIDTH_source")),
                                                          geo.get("SPEC_BDY_WIDTH_source"))},
                "mode": "member analyses (ensemble cycling: own background frame, land cycled)",
                "source_case": str(source), "template_wrfinput": labels.get(str(template), str(template)),
                "template_sha256": sha256(template),
                "created_utc": datetime.now(timezone.utc).isoformat(), "extra": dict(extra_provenance or {}),
                "members": members}
    path = out_root / "ENSEMBLE_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return path


def member_cycle_read_plan(member_wrfouts: Sequence, template_wrfinput, source_case_dir, valid_time) -> tuple[dict, list]:
    """(allowed {path: [days]}, issue_days) for ``alisios_guard.guarded_session`` around :func:`write_member_cycle`.
    Only paths under the ALISIOS data roots are listed (member backgrounds from our own CPU-WRF runs are not
    ALISIOS data). Listing only; nothing is opened."""
    from . import alisios_guard
    from . import archive_ensemble as ae
    from .ensemble import case_read_set, run_days_from_name

    day = wrf_time(valid_time)[:10].replace("-", "")
    allowed = {}
    for path in member_wrfouts:
        if alisios_guard.is_alisios_data(path):
            allowed[str(Path(path).resolve())] = [day]
    template = Path(template_wrfinput)
    if alisios_guard.is_alisios_data(template):
        allowed[str(template.resolve())] = run_days_from_name(template.parent)[:1]
    if alisios_guard.is_alisios_data(source_case_dir):
        allowed.update({p: d for p, d in case_read_set(source_case_dir).items() if p not in allowed})
    issues = sorted({m.group(1) for p in allowed if (m := ae._RUN_RE.search(p))})
    return allowed, issues
