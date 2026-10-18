"""B1 release benchmark: stage the WN3 0502 + 0614 model inputs through the ALISIOS guard (0227 is already staged).

usage: JAX_PLATFORMS=cpu python stage_wn3.py STAGE_ROOT RECEIPT_DIR
ALISIOS model inputs are opened ONLY inside alisios_guard.facade_stage (kernel 3a997be576): one GuardedRead session
scans + admits every day before the first payload open and logs every open (READ_SPY). File list + sealed SHA-256 from
the F20 archive inputs.json (its input/ links are only readlink'ed). Days follow the WN3 run layout (as for 0227): issue
day = the 18Z run date; namelist/wrfinput valid on the start day (issue + 1); wrfbdy/wrflowinp valid start day .. +3.
"""
import datetime as dt, importlib.util, json, os, sys
from pathlib import Path

assert os.environ.get("JAX_PLATFORMS") == "cpu"
G = Path("<USER_HOME>/src/wrf_gpu2/src/gpuwrf/assimilation/letkf/alisios_guard.py")
spec = importlib.util.spec_from_file_location("alisios_guard_main", G); ag = importlib.util.module_from_spec(spec)
sys.modules["alisios_guard_main"] = ag; spec.loader.exec_module(ag)
F20 = Path("<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016")
SEALED = Path("<USER_HOME>/wrf_gpu2_lanes/w41-bdyrec/alisios_sealed/80b076c9fb507aa9f8034b8fcc6effd1c3e2454f")
CASES = {"WN3-0502": "20260502", "WN3-0614": "20260614"}
stage_root, receipt_dir = Path(sys.argv[1]), Path(sys.argv[2])


def days(issue, n):
    start = dt.date(int(issue[:4]), int(issue[4:6]), int(issue[6:])) + dt.timedelta(days=1)
    return [(start + dt.timedelta(days=k)).strftime("%Y%m%d") for k in range(n)]


targets, expect = {}, {}
for case, issue in CASES.items():
    for name, digest in json.loads((F20 / case / "inputs.json").read_text())["input_sha256"].items():
        source = os.readlink(F20 / case / "input" / name)        # the ALISIOS path; not opened here
        n = 4 if name.startswith(("wrfbdy", "wrflowinp")) else 1
        targets[source] = ([issue], days(issue, n))
        expect[source] = (case, name, digest)
staged, receipt = ag.facade_stage(targets, stage_root, receipt_dir, sealed_root=SEALED,
                                  purpose="w41-bdyrec B1 v0.4.1 release benchmark: WN3 0502/0614 3-nest model inputs "
                                          "(parallel 24 h benchmark, manager 2026-10-18T05:31:27Z)")
index, bad = {}, []
for source, (case, name, digest) in expect.items():
    key = str(ag.resolve_path(source))
    if receipt["files"][key]["sha256"] != digest:
        bad.append(f"{case}/{name}")
    index.setdefault(case, {})[name] = staged[key]
(receipt_dir / "STAGED_INDEX.json").write_text(json.dumps({"index": index, "sha_mismatch": bad}, indent=1) + "\n")
print(json.dumps({"staged": {c: len(v) for c, v in index.items()}, "sha_mismatch": bad,
                  "counters": receipt["counters"], "refused": receipt["spy"]["refused_paths"]}))
sys.exit(1 if bad else 0)
