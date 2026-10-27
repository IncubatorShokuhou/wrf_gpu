"""Stage admitted ALISIOS files into a private wrf_gpu2 stage (lane w44-gint; copy of w43-teacher analysis/stage.py: request list + receipt root changed only) — CPU only, values never logged.

usage: JAX_PLATFORMS=cpu python3 analysis/stage.py ADMISSION_JSON ADMISSION_SHA256 STAGE_ROOT LABEL REGEX
Every requested file = a line of <USER_HOME>/wrf_gpu2_lanes/w44-gint/GINT_ADMISSION_PATHS.txt matching REGEX; each must be LISTED in the admission
(status ISSUED, sha256 of the admission file pinned, valid_until in the future, mode CALIBRATION_READ). All files are opened
inside ONE alisios.guarded_read.GuardedRead session (sealed facade, guard scan + admit before the first payload open, READ_SPY).
Copies keep the path below its data root: STAGE_ROOT/root<k>/<relative path>; SHA-256 computed while copying and checked
against the admission's sha256 when it carries one. Writes STAGE_ROOT/STAGE_<LABEL>.json + receipts/<LABEL>/.
"""
import hashlib, json, os, re, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from gpuwrf.assimilation.letkf import alisios_guard as ag  # noqa: E402

assert os.environ.get("JAX_PLATFORMS") == "cpu"
adm_path, adm_sha, stage_root, label, regex = sys.argv[1:6]
raw = Path(adm_path).read_bytes()
assert hashlib.sha256(raw).hexdigest() == adm_sha, "admission sha mismatch"
adm = json.loads(raw)
assert adm["status"] == "ISSUED" and "CALIBRATION_READ" in adm["modes"]
assert datetime.fromisoformat(adm["valid_until_utc"].replace("Z", "+00:00")) > datetime.now(timezone.utc)
listed = {f["path"]: f.get("sha256") for f in adm["files"]}
req = [l for l in Path("<USER_HOME>/wrf_gpu2_lanes/w44-gint/GINT_ADMISSION_PATHS.txt").read_text().split("\n") if l and re.search(regex, l)]
assert req, "nothing selected"
missing = [p for p in req if p not in listed]
assert not missing, f"not admitted: {missing[:3]} (+{len(missing)})"


def days(path):
    """issue day of the run directory + every day the 72 h run covers (issue 18z -> start 00z next day -> +72 h)."""
    m = re.search(r"(20\d{6})_18z", path)
    if m:
        d0 = datetime.strptime(m.group(1), "%Y%m%d")
        return [m.group(1)], [(d0 + timedelta(days=k)).strftime("%Y%m%d") for k in range(5)]
    return sorted(adm["days"]), sorted(adm["days"])   # multi-issue metadata (E2 JSON): the admission's day list


rdir = HERE.parent / "receipts" / label
rdir.mkdir(parents=True, exist_ok=False)
reader, guard, info = ag.facade()
assert reader is not None, info.get("unavailable")
targets = {p: days(p) for p in req}
admission = {"path": adm_path, "sha256": adm_sha}
sess = reader.GuardedRead([reader.Target(p, tuple(i), tuple(v), complete=True) for p, (i, v) in targets.items()],
                          protected_roots=sorted({str(Path(p).parent) for p in req}), read_spy=rdir / "READ_SPY.jsonl",
                          mode="CALIBRATION_READ", admission=admission, guard=str(ag.GUARD))
(rdir / "plan.json").write_text(json.dumps({"label": label, "admission": admission, "regex": regex, "n": len(req),
                                            "facade": {k: info.get(k) for k in ("commit", "sealed")},
                                            "utc_start": time.strftime("%FT%TZ", time.gmtime())}, indent=1))
index, bad = {}, []
with sess:
    for p in req:
        target = ag.stage_path(p, stage_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        with open(p, "rb") as src, open(target, "xb") as dst:
            for b in iter(lambda: src.read(1 << 22), b""):
                h.update(b)
                dst.write(b)
        d = h.hexdigest()
        if listed[p] and listed[p] != d:
            bad.append(p)
        index[p] = {"staged": str(target), "sha256": d, "admitted_sha256": listed[p]}
counters = {"counters": dict(sess.counters), "admit_results": sorted({str(r.get("result")) for r in sess.receipts}),
            "utc_end": time.strftime("%FT%TZ", time.gmtime()), "sha_mismatch": bad}
(rdir / "counters.json").write_text(json.dumps(counters, indent=1))
Path(stage_root, f"STAGE_{label}.json").write_text(json.dumps({"admission": admission, "files": index, **counters}, indent=1))
print(json.dumps({"label": label, "staged": len(index), "sha_mismatch": len(bad), **{k: counters[k] for k in ("admit_results",)}}))
sys.exit(1 if bad else 0)
