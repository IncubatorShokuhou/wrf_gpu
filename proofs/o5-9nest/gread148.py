"""Guarded reads for lane o5-9nest: alisios.guarded_read.GuardedRead (sealed facade) with admission A-O5-9NEST-148.
Every target must be listed in the admission (path + sha256); values are never written to receipts."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gpuwrf.assimilation.letkf import alisios_guard as ag  # noqa: E402

ADMISSION = {"path": "<DATA_ROOT>/alisios/state/manager/cross/TO_WRF_GPU2_ADMISSION_o5-9nest_20261024T152208.json",
             "sha256": "c0a2eb9609a1ae86e6c2da731d5ac84f95f4a56af8e9801ea3ed7b1c089d2477"}
RECEIPTS = Path("<USER_HOME>/wrf_gpu2_lanes/o5-9nest/receipts")


def admitted():
    doc = json.load(open(ADMISSION["path"]))
    return {f["path"]: f for f in doc["files"]}, doc


def session(paths, label):
    reader, guard, info = ag.facade()
    if reader is None:
        raise RuntimeError(info["unavailable"])
    listed, doc = admitted()
    paths = [str(Path(p).resolve()) for p in paths]
    missing = [p for p in paths if p not in listed]
    if missing:
        raise RuntimeError(f"targets not listed in A-O5-9NEST-148: {missing[:5]}")
    issue = tuple(d for d in doc["issue_days"])
    valid = tuple(doc["valid_days"])
    rdir = RECEIPTS / label
    rdir.mkdir(parents=True, exist_ok=False)
    roots = sorted({str(Path(p).parent) for p in paths})
    s = reader.GuardedRead([reader.Target(p, issue, valid, complete=True) for p in paths], protected_roots=roots,
                           read_spy=rdir / "READ_SPY.jsonl", mode="CALIBRATION_READ", admission=ADMISSION, guard=str(ag.GUARD))
    (rdir / "plan.json").write_text(json.dumps({"label": label, "admission": ADMISSION, "facade": {k: info.get(k) for k in ("commit", "sealed", "blobs")},
                                                "n_targets": len(paths), "targets": paths,
                                                "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    return s, rdir, listed


def close_receipt(s, rdir, extra=None):
    rec = {"counters": dict(s.counters), "n_admit_receipts": len(s.receipts),
           "admit_results": sorted({str(r.get("result")) for r in s.receipts}),
           "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rec.update(extra or {})
    (rdir / "counters.json").write_text(json.dumps(rec, indent=1))
