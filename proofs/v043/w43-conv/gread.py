"""Guarded reads for lane w43-conv: alisios.guarded_read.GuardedRead (sealed facade 80b076c9) with the
alisios_mgr admission A-148-FIDELITY-ANALYSIS-WRFGPU2 (CALIBRATION_READ, days 20261018-21). Values never logged."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpuwrf.assimilation.letkf import alisios_guard as ag

ADMISSION_BASE = {"path": "<DATA_ROOT>/alisios/state/manager/cross/TO_WRF_GPU2_148_ADMISSION_20261020.json",
                  "sha256": "b0c718cd4a7260ab474a0b98d84c37a5a0120ac05976f7a1ebc64eda52e0a4df"}
ADMISSION_ADD = {"path": "<DATA_ROOT>/alisios/state/manager/cross/TO_WRF_GPU2_148_ADMISSION_ADD_20261020.json",
                 "sha256": "dfd73f98f250a48a17e86fa3ca0b0f0722171d0d48ac0ede3b3014e7aa6aa90d"}
ADMISSION_ADD2 = {"path": "<DATA_ROOT>/alisios/state/manager/cross/TO_WRF_GPU2_148_ADMISSION_ADD2_20261020.json",
                  "sha256": "8b0499c2b3dd51b9c0ba6f11828c4b85f443b53793d9d3a4b8d8f5e450edf41f"}
ADMISSION = ADMISSION_BASE


def use_add2():
    """twin-cost reads: the ADD2 admission (A-148-...-ADD2) lists rank-0 rsl + run configs; refuse anything else."""
    global ADMISSION
    ADMISSION = ADMISSION_ADD2
    doc = json.load(open(ADMISSION_ADD2["path"]))
    return {f["path"]: f.get("sha256") for f in doc["files"]}


def use_add():
    """wrfout reads: the ADD admission (A-148-...-ADD1) lists every wrfout path; refuse any target it does not list."""
    global ADMISSION
    ADMISSION = ADMISSION_ADD
    doc = json.load(open(ADMISSION_ADD["path"]))
    return {f["path"]: f["sha256"] for f in doc["files"]}
ISSUE = ("20261018",)
VALID = ("20261018", "20261019", "20261020", "20261021")
RECEIPTS = Path(__file__).resolve().parent / "receipts"


def session(paths, label):
    reader, guard, info = ag.facade()
    if reader is None:
        raise RuntimeError(info["unavailable"])
    rdir = RECEIPTS / label
    rdir.mkdir(parents=True, exist_ok=False)
    paths = [str(Path(p).resolve()) for p in paths]
    if ADMISSION is ADMISSION_ADD or ADMISSION is ADMISSION_ADD2:
        listed = use_add() if ADMISSION is ADMISSION_ADD else use_add2()
        missing = [p for p in paths if p not in listed]
        if missing:
            raise RuntimeError(f"targets not listed in ADD admission: {missing[:5]}")
    roots = sorted({str(Path(p).parent) for p in paths})
    s = reader.GuardedRead([reader.Target(p, ISSUE, VALID, complete=True) for p in paths],
                           protected_roots=roots, read_spy=rdir / "READ_SPY.jsonl", mode="CALIBRATION_READ",
                           admission=ADMISSION, guard=str(ag.GUARD))
    (rdir / "plan.json").write_text(json.dumps({"label": label, "admission": ADMISSION, "facade": {k: info.get(k) for k in ("commit", "sealed", "blobs")},
                                                "n_targets": len(paths), "targets": paths, "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    return s, rdir


def close_receipt(s, rdir):
    (rdir / "counters.json").write_text(json.dumps({"counters": dict(s.counters), "n_admit_receipts": len(s.receipts),
                                                    "admit_results": sorted({r.get("result") for r in s.receipts}),
                                                    "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
