"""Run ONE frozen CPU tool in-process inside ONE sealed alisios.guarded_read.GuardedRead session that reads admitted ALISIOS
files IN PLACE (no staging copy; manager 15:54Z 'bevorzugt direkt lesen statt kopieren'). Same facade, guard and admission check
as tools/stage.py (w43-teacher stage.py lineage); the tool is executed with runpy in the session's owner thread AFTER the
session patched netCDF4.Dataset, so the tool's own `from netCDF4 import Dataset` binds the checking wrapper and every open is
authorised against the target list and written to READ_SPY. Values are never logged here.

usage: JAX_PLATFORMS=cpu python3 guarded_run.py ADMISSION_JSON ADMISSION_SHA256 LABEL REGEX -- TOOL.py [ARGS...]
  REGEX selects the session targets from <USER_HOME>/wrf_gpu2_lanes/w44-gint/GINT_ADMISSION_PATHS.txt (each must be LISTED in
  the admission with status ISSUED / CALIBRATION_READ / valid). Receipts: proofs/w44_gint/receipts/<LABEL>/ (plan, READ_SPY,
  counters incl. the tool's exit code). Exit = the tool's exit code (1 if the session refused anything).
"""
import hashlib, json, os, re, runpy, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from gpuwrf.assimilation.letkf import alisios_guard as ag  # noqa: E402

assert os.environ.get("JAX_PLATFORMS") == "cpu"
k = sys.argv.index("--")
adm_path, adm_sha, label, regex = sys.argv[1:5]
tool, targs = sys.argv[k + 1], sys.argv[k + 2:]
raw = Path(adm_path).read_bytes()
assert hashlib.sha256(raw).hexdigest() == adm_sha, "admission sha mismatch"
adm = json.loads(raw)
assert adm["status"] == "ISSUED" and "CALIBRATION_READ" in adm["modes"]
assert datetime.fromisoformat(adm["valid_until_utc"].replace("Z", "+00:00")) > datetime.now(timezone.utc)
listed = {f["path"]: f.get("sha256") for f in adm["files"]}
req = [l for l in Path("<USER_HOME>/wrf_gpu2_lanes/w44-gint/GINT_ADMISSION_PATHS.txt").read_text().split("\n")
       if l and re.search(regex, l)]
assert req, "nothing selected"
missing = [p for p in req if p not in listed]
assert not missing, f"not admitted: {missing[:3]} (+{len(missing)})"


def days(path):
    m = re.search(r"(20\d{6})_18z", path)
    d0 = datetime.strptime(m.group(1), "%Y%m%d")
    return [m.group(1)], [(d0 + timedelta(days=i)).strftime("%Y%m%d") for i in range(5)]


rdir = HERE.parent / "receipts" / label
rdir.mkdir(parents=True, exist_ok=False)
reader, guard, info = ag.facade()
assert reader is not None, info.get("unavailable")
admission = {"path": adm_path, "sha256": adm_sha}
sess = reader.GuardedRead([reader.Target(p, tuple(i), tuple(v), complete=True) for p, (i, v) in ((p, days(p)) for p in req)],
                          protected_roots=sorted({str(Path(p).parent) for p in req}), read_spy=rdir / "READ_SPY.jsonl",
                          mode="CALIBRATION_READ", admission=admission, guard=str(ag.GUARD))
(rdir / "plan.json").write_text(json.dumps({"label": label, "admission": admission, "regex": regex, "n_targets": len(req),
                                            "tool": tool, "tool_sha256": hashlib.sha256(Path(tool).read_bytes()).hexdigest(),
                                            "args": targs, "facade": {kk: info.get(kk) for kk in ("commit", "sealed")},
                                            "utc_start": time.strftime("%FT%TZ", time.gmtime())}, indent=1))
rc, err = 0, None
with sess:
    sys.argv = [tool, *targs]
    try:
        runpy.run_path(tool, run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except Exception as e:  # noqa: BLE001 - record and fail closed (traceback = code lines/paths, no field values)
        import traceback
        traceback.print_exc()
        rc, err = 1, f"{type(e).__name__}"
counters = {"counters": dict(sess.counters), "admit_results": sorted({str(r.get("result")) for r in sess.receipts}),
            "tool_rc": rc, "tool_error_type": err, "utc_end": time.strftime("%FT%TZ", time.gmtime())}
(rdir / "counters.json").write_text(json.dumps(counters, indent=1))
print(json.dumps({"label": label, **counters}), file=sys.stderr)
sys.exit(1 if sess.counters.get("refused") else rc)
