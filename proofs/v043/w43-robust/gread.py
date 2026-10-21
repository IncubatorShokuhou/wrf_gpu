"""Guarded reads for lane w43-robust: sealed ALISIOS facade GuardedRead (80b076c9) with admission
A-V043-ROBUST-E2-20250531 (CALIBRATION_READ). Every target must be listed in the admission. Values never logged."""
import datetime as dt, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpuwrf.assimilation.letkf import alisios_guard as ag

ADMISSION = {"path": "<DATA_ROOT>/alisios/state/manager/cross/TO_WRF_GPU2_ADMISSION_w43-robust_20261020T220311.json",
             "sha256": "8c3b84f99ed1ad449f92169dd9ccf3c68f7974a35ae054f4d2306612ef37ae12"}
RECEIPTS = Path(__file__).resolve().parent / "receipts"
ROOT = "<DATA_ROOT>/alisios/state/manager/runs/mgr_wave_20261009/rd-e2-massdays/"


def listed():
    return {f["path"]: f["sha256"] for f in json.load(open(ADMISSION["path"]))["files"]}


def days(path):
    """issue day = case init date in the path (series-level files: 20250531, the case under diagnosis); valid = +0..3 d."""
    m = re.search(r"/(\d{8})_18z", path)
    d0 = dt.date.fromisoformat(f"{(m.group(1) if m else '20250531')[:4]}-{(m.group(1) if m else '20250531')[4:6]}-{(m.group(1) if m else '20250531')[6:]}")
    return (d0.strftime("%Y%m%d"),), tuple((d0 + dt.timedelta(days=k)).strftime("%Y%m%d") for k in range(4))


def stage(paths, label):
    """copy admitted targets into analysis/data/<label>/ inside ONE GuardedRead session; returns {path: local copy}."""
    reader, guard, info = ag.facade()
    if reader is None:
        raise RuntimeError(info["unavailable"])
    adm = listed()
    paths = sorted({str(Path(p).resolve()) for p in paths})
    missing = [p for p in paths if p not in adm]
    if missing:
        raise RuntimeError(f"targets not listed in admission: {missing[:5]}")
    rdir = RECEIPTS / label
    rdir.mkdir(parents=True, exist_ok=False)
    out = Path(__file__).resolve().parent / "data" / label
    targets = [reader.Target(p, *days(p), complete=True) for p in paths]
    (rdir / "plan.json").write_text(json.dumps({"label": label, "admission": ADMISSION, "facade": {k: info.get(k) for k in ("commit", "sealed")},
                                                "targets": paths, "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    s = reader.GuardedRead(targets, protected_roots=sorted({str(Path(p).parent) for p in paths}), read_spy=rdir / "READ_SPY.jsonl",
                           mode="CALIBRATION_READ", admission=ADMISSION, guard=str(ag.GUARD))
    import hashlib
    copies, bad = {}, []
    with s:
        for p in paths:
            data = open(p, "rb").read()
            if hashlib.sha256(data).hexdigest() != adm[p]:
                bad.append(p)
            dst = out / p[len(ROOT):] if p.startswith(ROOT) else out / p.lstrip("/")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            copies[p] = str(dst)
    (rdir / "counters.json").write_text(json.dumps({"counters": dict(s.counters), "n_admit_receipts": len(s.receipts),
                                                    "admit_results": sorted({str(r.get("result")) for r in s.receipts}),
                                                    "sha_mismatch_vs_admission": bad,
                                                    "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    return copies, bad


if __name__ == "__main__":
    c, bad = stage(list(listed()), sys.argv[1])
    print("staged", len(c), "sha mismatches", len(bad))
