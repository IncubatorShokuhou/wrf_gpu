"""w41-rw3add: stage the WN3 0227 3-nest model inputs (72 h coverage) through the ALISIOS guard.

usage: JAX_PLATFORMS=cpu python stage_0227.py STAGE_ROOT RECEIPT_DIR CASE_DIR
ALISIOS model inputs are opened ONLY inside alisios_guard.facade_stage (one GuardedRead session: scan + admit every day
before the first payload open, every open logged to READ_SPY). File list + sealed SHA-256 from the F20 archive
inputs.json (its input/ links are only readlink'ed). Days as in proofs/w41_bdyrec/b1/stage_wn3.py: issue day = the 18Z
run date; namelist/wrfinput valid on the start day (issue + 1); wrfbdy/wrflowinp valid start day .. +3 (72 h).
CASE_DIR gets hard links to the staged copies (flat CLI input dir); every file is re-hashed there.
"""
import datetime as dt, hashlib, importlib.util, json, os, sys
from pathlib import Path

assert os.environ.get("JAX_PLATFORMS") == "cpu"
G = Path("<USER_HOME>/src/wrf_gpu2/src/gpuwrf/assimilation/letkf/alisios_guard.py")
spec = importlib.util.spec_from_file_location("alisios_guard_main", G); ag = importlib.util.module_from_spec(spec)
sys.modules["alisios_guard_main"] = ag; spec.loader.exec_module(ag)
F20 = Path("<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016")
SEALED = Path("<USER_HOME>/wrf_gpu2_lanes/w41-bdyrec/alisios_sealed/80b076c9fb507aa9f8034b8fcc6effd1c3e2454f")
CASE, ISSUE = "WN3-0227", "20260227"
stage_root, receipt_dir, case_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def days(issue, n):
    start = dt.date(int(issue[:4]), int(issue[4:6]), int(issue[6:])) + dt.timedelta(days=1)
    return [(start + dt.timedelta(days=k)).strftime("%Y%m%d") for k in range(n)]


targets, expect = {}, {}
for name, digest in json.loads((F20 / CASE / "inputs.json").read_text())["input_sha256"].items():
    source = os.readlink(F20 / CASE / "input" / name)        # the ALISIOS path; not opened here
    n = 4 if name.startswith(("wrfbdy", "wrflowinp")) else 1
    targets[source] = ([ISSUE], days(ISSUE, n))
    expect[source] = (name, digest)
staged, receipt = ag.facade_stage(targets, stage_root, receipt_dir, sealed_root=SEALED,
                                  purpose="w41-rw3add v0.4.1 addendum: clean RW3 timing, WN3 0227 3-nest model inputs "
                                          "(N/O sealed-bundle arms, contract kernel 081f17f070)")
case_dir.mkdir(parents=True)
bad, index = [], {}
for source, (name, digest) in expect.items():
    key = str(ag.resolve_path(source))
    os.link(staged[key], case_dir / name)
    h = hashlib.sha256((case_dir / name).read_bytes()).hexdigest()
    if receipt["files"][key]["sha256"] != digest or h != digest:
        bad.append(name)
    index[name] = dict(staged=str(staged[key]), sha256=h)
(receipt_dir / "STAGED_INDEX.json").write_text(json.dumps({"case_dir": str(case_dir), "index": index, "sha_mismatch": bad},
                                                          indent=1) + "\n")
print(json.dumps({"staged": len(index), "sha_mismatch": bad, "counters": receipt["counters"],
                  "refused": receipt["spy"]["refused_paths"]}))
sys.exit(1 if bad else 0)
