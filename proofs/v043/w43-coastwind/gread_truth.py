"""Guarded read of the ALISIOS trusted export for w43-coastwind (admission A-V043-COASTWIND-T1-TRUTH-EXPORT, CALIBRATION_READ).
Sealed facade GuardedRead + guard_call.admit with the admission {path, sha256}; targets = the 2 export files, issue days = the
36 admitted issues, valid days = the 95 admitted days. Copies the tables to the lane scratch (values never logged)."""
import json, os, sys, time, hashlib
from pathlib import Path
assert os.environ.get("JAX_PLATFORMS") == "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpuwrf.assimilation.letkf import alisios_guard as ag
ADM = {"path": "<DATA_ROOT>/alisios/state/manager/cross/TO_WRF_GPU2_ADMISSION_w43-coastwind_20261020T224828.json",
       "sha256": "da328750692ff6c38a1a37e7b8018f65d8c69faed7be640d2db6950adf314b0f"}
doc = json.load(open(ADM["path"]))
assert hashlib.sha256(open(ADM["path"], "rb").read()).hexdigest() == ADM["sha256"]
issues = tuple(sorted({i[:8] for i in doc["issues"]})); valid = tuple(sorted(doc["days"]))
files = {f["path"]: f["sha256"] for f in doc["files"]}
reader, guard, info = ag.facade()
rdir = Path(__file__).resolve().parent / "receipts" / "R01_truth_export"; rdir.mkdir(parents=True, exist_ok=False)
paths = [str(Path(p).resolve()) for p in files]
s = reader.GuardedRead([reader.Target(p, issues, valid, complete=True) for p in paths], protected_roots=sorted({str(Path(p).parent) for p in paths}),
                       read_spy=rdir / "READ_SPY.jsonl", mode="CALIBRATION_READ", admission=ADM, guard=str(ag.GUARD))
out = Path("<DATA_ROOT>/wrf_gpu2/scratch/w43-coastwind/truth"); out.mkdir(parents=True, exist_ok=True)
with s:
    import pandas as pd
    shas = {}
    for p in paths:
        raw = open(p, "rb").read(); shas[p] = hashlib.sha256(raw).hexdigest()
        assert shas[p] == files[p], p
        df = pd.read_parquet(p)
        df.to_parquet(out / Path(p).name)
(rdir / "receipt.json").write_text(json.dumps({"admission": ADM, "admission_id": doc["admission_id"], "files": shas, "issues": issues, "n_valid": len(valid),
    "counters": dict(s.counters), "admit_results": sorted({str(r.get("result")) for r in s.receipts}), "utc": time.strftime("%FT%TZ", time.gmtime()),
    "facade": {k: info.get(k) for k in ("commit", "sealed")}, "values_logged": False}, indent=1, default=str))
print(json.dumps({"counters": dict(s.counters), "sha_ok": True}))
