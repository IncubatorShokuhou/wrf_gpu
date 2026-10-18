"""Guarded staging of the WN3 0227 3-nest icloud=2 arm inputs (lane w41-icloud2) — CPU only, no JAX.

ALISIOS model inputs are read ONLY inside one ``alisios_guard.facade_stage`` session (sealed
``alisios.guarded_read``: scan -> admit -> open + READ_SPY, plus the refusing spy over every ALISIOS data root)
and copied ONCE into CASE/stage; GPU arm, CPU-WRF reference and scorer read only CASE/input (symlinks into the
stage + the edited namelist), never an ALISIOS path (wave rule, alisios_mgr C1745).

Read set: wg_20260227_18z_a1/run/run/{namelist.input, wrfinput_d01-03, wrfbdy_d01, wrflowinp_d01-03}.
Days: issue 20260227; wrfinput/namelist valid 20260228, wrfbdy/wrflowinp 20260228..20260303 (same coverage as
w41-bdyrec's guarded copy, receipt 18:08Z). The guard is scanned on 20260227..20260303 first; any non-ALLOW day
refuses everything before a payload open.

Twin identity checks (E184): the 7 model inputs must be byte-identical to the inputs of the merged icloud=3 arm
(w41-icloud cpu_run.json input_sha256). The original run-dir namelist is the ALISIOS 72 h one; the icloud=3 twin
differs from it ONLY in run_hours 72->24, restart_interval 180->100000 and the inserted icloud line (checked), so the
icloud=2 namelist = the twin namelist with icloud 3->2 (checked: exactly one line differs from the twin, and versus the
original only those three keys). Only then STAGE.json says OK.

usage: JAX_PLATFORMS=cpu PYTHONPATH=<tree>/src python stage_inputs.py CASE RECEIPT_DIR [--scan-only | --finalize]
  --finalize: rebuild input/namelist.input + STAGE.json from the existing private stage (no ALISIOS read; every
  staged file is re-hashed against the facade receipt first).
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

from gpuwrf.assimilation.letkf import alisios_guard as ag

RUN = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
INPUTS = ("wrfinput_d01", "wrfinput_d02", "wrfinput_d03", "wrfbdy_d01",
          "wrflowinp_d01", "wrflowinp_d02", "wrflowinp_d03")
ISSUE, SPAN = "20260227", ["20260227", "20260228", "20260301", "20260302", "20260303"]
TWIN = Path("<USER_HOME>/wrf_gpu2_lanes/w41-icloud/wn3_icloud3_receipts")   # merged icloud=3 arm receipts
PURPOSE = ("wrf_gpu2 w41-icloud2: WN3 0227 3-nest 24 h icloud=2 GPU arm + original CPU-WRF 4.7.1 icloud=2 "
           "reference (D6 gate; model inputs only, no observations)")
ICLOUD_LINE = re.compile(rb"^( *icloud *= *)(\d+)(,.*)$", re.M)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


TWIN_EDITS = {"run_hours", "restart_interval", "icloud"}   # run-control + the arm's option (no other key may move)


def _changed_keys(a: bytes, b: bytes) -> tuple[list[str], set[str]]:
    diff = [l for l in difflib.unified_diff(a.decode().splitlines(), b.decode().splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    return diff, {l[1:].split("=")[0].strip().lower() for l in diff}


def icloud2_namelist(original: bytes, twin: bytes) -> tuple[bytes, dict]:
    """icloud=2 namelist = the icloud=3 twin with icloud 3->2, after proving the twin == original + TWIN_EDITS."""
    twin_diff, twin_keys = _changed_keys(original, twin)
    nl = ICLOUD_LINE.sub(rb"\g<1>2\g<3>", twin, count=1)
    diff, keys = _changed_keys(twin, nl)
    orig_diff, orig_keys = _changed_keys(original, nl)
    one_line = (len(diff) == 2 and ICLOUD_LINE.match(diff[0][1:].encode()).group(2) == b"3"
                and ICLOUD_LINE.match(diff[1][1:].encode()).group(2) == b"2")
    ok = one_line and twin_keys <= TWIN_EDITS and orig_keys <= TWIN_EDITS and "icloud" in orig_keys
    return nl, dict(diff_vs_icloud3_twin=diff, only_icloud_differs_from_twin=one_line,
                    twin_vs_original=twin_diff, icloud2_vs_original=orig_diff, ok=ok)


def finish(case, receipt_dir, verdicts, staged, files, receipt):
    twin_run = json.loads((TWIN / "cpu_run.json").read_text())
    (case / "input").mkdir(exist_ok=True)
    checks = {}
    for f in INPUTS:
        src = Path(staged[str((RUN / f).resolve())])
        link = case / "input" / f
        if not link.is_symlink():
            link.symlink_to(src)
        assert link.resolve() == src.resolve(), (link, src)
        got = files[str((RUN / f).resolve())]["sha256"]
        checks[f] = dict(staged=str(src), sha256=got, icloud3_twin_sha256=twin_run["input_sha256"][f],
                         match=got == twin_run["input_sha256"][f])
    original = Path(staged[str((RUN / "namelist.input").resolve())]).read_bytes()
    twin_nl = (TWIN / "input/namelist.input").read_bytes()
    assert sha256(TWIN / "input/namelist.input") == twin_run["input_sha256"]["namelist.input"], "twin namelist copy drifted"
    nl, nl_checks = icloud2_namelist(original, twin_nl)
    (case / "input/namelist.input").write_bytes(nl)
    ok = all(c["match"] for c in checks.values()) and nl_checks["ok"] and not receipt["spy"]["refused_paths"]
    out = dict(schema="w41-icloud2.stage.v2", run_dir=str(RUN), guard_days=SPAN, guard_verdicts=verdicts,
               inputs=checks, namelist=dict(original_sha256=files[str((RUN / "namelist.input").resolve())]["sha256"],
                                            icloud2_sha256=sha256(case / "input/namelist.input"),
                                            icloud3_twin_sha256=twin_run["input_sha256"]["namelist.input"],
                                            **nl_checks),
               facade_receipt=str(receipt_dir / "payload" / "facade_stage.json"), counters=receipt["counters"],
               spy_refused=receipt["spy"]["refused_paths"], verdict="OK" if ok else "MISMATCH")
    (receipt_dir / "STAGE.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(dict(verdict=out["verdict"], files=len(files), diff_vs_twin=nl_checks["diff_vs_icloud3_twin"],
                          vs_original=nl_checks["icloud2_vs_original"], counters=receipt["counters"])))
    return 0 if ok else 1


def main(case, receipt_dir, mode=None):
    case, receipt_dir = Path(case).resolve(), Path(receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    if mode == "--finalize":           # private stage only: re-hash every staged file against the facade receipt
        receipt = json.loads((receipt_dir / "payload" / "facade_stage.json").read_text())
        verdicts = json.loads((receipt_dir / "guard_scan.json").read_text())["verdicts"]
        staged = {p: str(case / "stage" / row["staged"]) for p, row in receipt["files"].items()}
        for p, row in receipt["files"].items():
            assert sha256(staged[p]) == row["sha256"], f"staged copy drifted: {staged[p]}"
        return finish(case, receipt_dir, verdicts, staged, receipt["files"], receipt)
    # 1) scan only (nothing opened): every day of the read set must be ALLOW
    _, verdicts = ag.drop_deny_days([(d, d) for d in SPAN])
    (receipt_dir / "guard_scan.json").write_text(json.dumps(dict(mode=ag.MODE, days=SPAN, verdicts=verdicts), indent=1) + "\n")
    deny = {d: v for d, v in verdicts.items() if v != "ALLOW"}
    if deny or set(verdicts) != set(SPAN):
        raise SystemExit(f"GUARD REFUSED {deny or verdicts}: nothing opened, stop and report")
    if mode == "--scan-only":
        print(json.dumps(dict(scan=verdicts)))
        return 0
    if (case / "input").exists() or (case / "stage").exists():
        raise SystemExit(f"refusing existing {case}/input or {case}/stage (use --finalize)")
    # 2) payload: one admitted session for the whole read set
    targets = {str(RUN / "namelist.input"): ([ISSUE], [SPAN[1]])}
    targets.update({str(RUN / f): ([ISSUE], [SPAN[1]] if f.startswith("wrfinput") else SPAN[1:]) for f in INPUTS})
    staged, receipt = ag.facade_stage(targets, case / "stage", receipt_dir / "payload", purpose=PURPOSE)
    return finish(case, receipt_dir, verdicts, staged, receipt["files"], receipt)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
