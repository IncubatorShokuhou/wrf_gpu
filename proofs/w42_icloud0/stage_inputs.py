"""Stage the WN3 0227 3-nest icloud=0 arm inputs (lane w42-icloud0) — CPU only, no JAX, NO ALISIOS read.

Wave rule (alisios_mgr C1745, _WAVE_V041_COMMON): prefer copies outside the ALISIOS roots that were made once under the
guard.  Source = the private guarded stage of the merged icloud=2 twin (w41-icloud2: one alisios_guard.facade_stage
session, 8 authorized reads, 0 refused, guard days 20260227..20260303 ALLOW; receipts in its stage/payload).  Every file
is re-hashed against that STAGE.json AND the icloud=3 twin's CPU receipt before it is copied (real copies, not links:
the other lane's scratch may be cleaned).  The icloud=0 namelist = the icloud=2 twin namelist with icloud 2 -> 0;
exactly that one line may differ (the twin itself differs from the ALISIOS original only in run_hours 72->24,
restart_interval 180->100000 and the icloud line, proven and recorded by w41-icloud2).  Only then STAGE.json says OK.

usage: python3 stage_inputs.py CASE RECEIPT_DIR
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

SRC_STAGE = Path("<USER_HOME>/wrf_gpu2_lanes/w41-icloud2/stage/STAGE.json")
SRC_NAMELIST = Path("<DATA_ROOT>/wrf_gpu2/scratch/w41-icloud2/wn3_icloud2/input/namelist.input")
TWIN3 = Path("<USER_HOME>/wrf_gpu2_lanes/w41-icloud/wn3_icloud3_receipts/cpu_run.json")
ROOTS = ("<DATA_ROOT>/alisios/", "<DATA_ROOT>/server/work/src/alisios/", "<DATA_ROOT>/server/work/alisios/")
ICLOUD_LINE = re.compile(rb"^( *icloud *= *)(\d+)(,.*)$", re.M)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main(case: Path, receipt_dir: Path) -> int:
    src = json.loads(SRC_STAGE.read_text())
    twin3 = json.loads(TWIN3.read_text())["input_sha256"]
    assert src["verdict"] == "OK" and set(src["guard_verdicts"].values()) == {"ALLOW"}, "source stage not OK"
    (case / "input").mkdir(parents=True, exist_ok=False)
    checks = {}
    for name, row in src["inputs"].items():
        staged = Path(row["staged"])
        real = str(staged.resolve())
        assert not real.startswith(ROOTS), f"{name}: source under an ALISIOS root"
        digest = sha256(staged)
        ok = digest == row["sha256"] == row["icloud3_twin_sha256"] == twin3[name]
        dst = case / "input" / name
        shutil.copyfile(staged, dst)
        checks[name] = dict(source=real, sha256=digest, copy_sha256=sha256(dst), ok=ok and sha256(dst) == digest)
    nl2 = SRC_NAMELIST.read_bytes()
    nl2_ok = hashlib.sha256(nl2).hexdigest() == src["namelist"]["icloud2_sha256"]
    nl0 = ICLOUD_LINE.sub(rb"\g<1>0\g<3>", nl2, count=1)
    diff = [l for l in difflib.unified_diff(nl2.decode().splitlines(), nl0.decode().splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    one_line = (len(diff) == 2 and ICLOUD_LINE.match(diff[0][1:].encode()).group(2) == b"2"
                and ICLOUD_LINE.match(diff[1][1:].encode()).group(2) == b"0")
    (case / "input" / "namelist.input").write_bytes(nl0)
    verdict = "OK" if nl2_ok and one_line and all(c["ok"] for c in checks.values()) and len(checks) == 7 else "FAIL"
    stage = dict(schema="w42-icloud0.stage.v1", verdict=verdict, alisios_reads=0,
                 provenance=dict(source_stage=str(SRC_STAGE), source_stage_sha256=sha256(SRC_STAGE),
                                 guard_days=src["guard_days"], guard_verdicts=src["guard_verdicts"],
                                 source_receipts=str(SRC_STAGE.parent / "payload")),
                 inputs=checks,
                 namelist=dict(icloud2_source=str(SRC_NAMELIST), icloud2_sha256=hashlib.sha256(nl2).hexdigest(),
                               icloud2_matches_stage=nl2_ok, icloud0_sha256=hashlib.sha256(nl0).hexdigest(),
                               diff_vs_icloud2_twin=diff, only_icloud_differs=one_line))
    receipt_dir.mkdir(parents=True, exist_ok=True)
    for target in (case / "STAGE.json", receipt_dir / "STAGE.json"):
        target.write_text(json.dumps(stage, indent=1) + "\n")
    print(json.dumps({"verdict": verdict, "diff": diff}))
    return 0 if verdict == "OK" else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
