"""Guarded staging of the G2 CPU-twin inputs (lane w41-nestqual) — CPU only, no JAX.

Reads ALISIOS paths ONLY inside one ``alisios_guard.facade_stage`` session (sealed ``alisios.guarded_read``:
scan -> admit -> open + READ_SPY, plus our refusing spy over every ALISIOS data root) and copies them ONCE into a
private stage outside the ALISIOS roots; everything downstream (twin builder, CPU-WRF, scorer) reads only the stage.

Read set for one issue (rd-s6 ARMS_FOR_WRFGPU2.json, manager 2026-10-17T17:35Z):
  * ARMS_FOR_WRFGPU2.json itself (metadata: arm namelist paths, input dir, 16-hex input SHA prefixes)
  * per arm: the 0444 namelist copy listed in ARMS (``namelist``) and the namelist the GPU CLI reads
    (``g2/runs/<issue>/<arm>/in/namelist.input``) -> both staged, their SHA-256 must agree (E184)
  * the WIN-store inputs of the issue (wrfinput_d01-03, wrfbdy_d01, wrflowinp_d01-03), same for every arm
Days: issue day; inputs are valid on issue day .. issue day + 3 (72-h inputs). The guard is scanned on that whole
range first; a non-ALLOW day refuses everything before any payload open (manager: DENY = stop and report).

usage: JAX_PLATFORMS=cpu PYTHONPATH=<tree>/src python stage_inputs.py <issue YYYYMMDD_HHz> <stage_root> <receipt_dir> [arm ...]
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from gpuwrf.assimilation.letkf import alisios_guard as ag

G2 = Path("<DATA_ROOT>/alisios/scratch/rd-s6-wrf-cloud-attribution/g2")
ARMS = G2 / "ARMS_FOR_WRFGPU2.json"
ARMS_SHA = "2d4aeb461b736f853d3cf2c191c67f50a031e48cb487b39914823662c792a320"   # manager 17:35Z (sha 2d4aeb46)
INPUTS = ("wrfinput_d01", "wrfinput_d02", "wrfinput_d03", "wrfbdy_d01",
          "wrflowinp_d01", "wrflowinp_d02", "wrflowinp_d03")
PURPOSE = ("wrf_gpu2 w41-nestqual: original CPU-WRF 4.7.1 twins of the G2 nested PBL arms (D6 nest gate, "
           "model inputs only, no observations)")


def days(issue: str, n: int) -> list[str]:
    d0 = datetime.strptime(issue[:8], "%Y%m%d")
    return [(d0 + timedelta(days=k)).strftime("%Y%m%d") for k in range(n)]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main(issue, stage_root, receipt_dir, arms):
    stage_root, receipt_dir = Path(stage_root), Path(receipt_dir)
    issue_day, span = issue[:8], days(issue, 4)
    # 1) scan only (nothing opened): every day of the read set must be ALLOW
    _, verdicts = ag.drop_deny_days([(d, d) for d in span])
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / "guard_scan.json").write_text(json.dumps(dict(mode=ag.MODE, days=span, verdicts=verdicts), indent=1) + "\n")
    deny = {d: v for d, v in verdicts.items() if v != "ALLOW"}
    if deny or set(verdicts) != set(span):
        raise SystemExit(f"GUARD REFUSED {deny or verdicts}: nothing opened, stop and report")
    # 2) ARMS json first (metadata) in its own admitted stage, so the payload read set is derived from it
    meta_stage = stage_root / "meta"
    staged, _ = ag.facade_stage({str(ARMS): ([issue_day], [issue_day])}, meta_stage, receipt_dir / "meta",
                                purpose=PURPOSE + " — ARMS metadata")
    arms_copy = Path(staged[str(ARMS.resolve())])
    assert sha256(arms_copy) == ARMS_SHA, "ARMS_FOR_WRFGPU2.json drifted from the manager-announced sha 2d4aeb46"
    spec = json.loads(arms_copy.read_text())
    arms = arms or spec["recommended_twins"]["arms"]
    runs = {arm: spec["runs"][f"{issue}:{arm}"] for arm in arms}
    input_dirs = {r["input_dir"] for r in runs.values()}
    assert len(input_dirs) == 1, f"arms use different input stores: {input_dirs}"
    input_dir = Path(input_dirs.pop())
    prefixes = runs[arms[0]]["inputs"]
    assert all(r["inputs"] == prefixes for r in runs.values()), "arms list different input hashes"
    targets = {str(input_dir / f): ([issue_day], [issue_day] if f.startswith("wrfinput") else span) for f in INPUTS}
    for arm, r in runs.items():
        targets[r["namelist"]] = ([issue_day], [issue_day])
        targets[str(G2 / "runs" / issue / arm / "in" / "namelist.input")] = ([issue_day], [issue_day])
    # 3) payload: one admitted session for the whole read set
    staged, receipt = ag.facade_stage(targets, stage_root / "payload", receipt_dir / "payload", purpose=PURPOSE)
    files = receipt["files"]
    checks = {}
    for f in INPUTS:
        row = files[str((input_dir / f).resolve())]
        checks[f] = dict(sha256=row["sha256"], arms_prefix=prefixes[f], prefix_match=row["sha256"].startswith(prefixes[f]))
    namelists = {}
    for arm, r in runs.items():
        listed = files[str(Path(r["namelist"]).resolve())]["sha256"]
        gpu_in = files[str((G2 / "runs" / issue / arm / "in" / "namelist.input").resolve())]["sha256"]
        namelists[arm] = dict(listed_path=r["namelist"], listed_sha256=listed, arms_sha256=r["namelist_sha"],
                              gpu_cli_input_namelist_sha256=gpu_in,
                              match=listed == r["namelist_sha"] == gpu_in,
                              staged=staged[str(Path(r["namelist"]).resolve())],
                              diff_vs_store=r["namelist_diff_vs_store"])
    ok = all(c["prefix_match"] for c in checks.values()) and all(n["match"] for n in namelists.values())
    out = dict(schema="w41-nestqual.stage.v1", issue=issue, arms=arms, arms_json=str(ARMS), arms_json_sha256=ARMS_SHA,
               input_dir=str(input_dir), inputs={f: dict(staged=staged[str((input_dir / f).resolve())], **checks[f])
                                                 for f in INPUTS},
               namelists=namelists, guard_days=span, guard_verdicts=verdicts,
               facade_receipt=str(receipt_dir / "payload" / "facade_stage.json"), verdict="OK" if ok else "MISMATCH")
    (receipt_dir / "STAGE.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(dict(verdict=out["verdict"], arms=arms, files=len(files),
                          spy_refused=receipt["spy"]["refused_paths"], counters=receipt["counters"])))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]))
