"""w44-rc v0.4.4 RW3 (copy of the v0.4.2 w42-bundle042 stage_cpu72.py; ONLY changes: guard module = the pinned snapshot's
alisios_guard.py (argv 3), purpose text): stage the ORIGINAL CPU-WRF 72 h reference frames of WN3 0227 (d01-d03, leads 0-72)
through the ALISIOS guard. Needs ALISIOS admission + a manager DISK-PLAN (~22 GB staged; v0.4.2 staged 219 frames).

usage: JAX_PLATFORMS=cpu python stage_cpu72.py STAGE_ROOT RECEIPT_DIR SNAPSHOT
Frame list = the CPU side of the v0.3.3 W3P full72 D6 pairing (wrf_gpu2 receipt, metadata only; the paths are not opened here).
ALISIOS CPU-WRF outputs are opened ONLY inside alisios_guard.facade_stage (one GuardedRead session: scan + admit every day
before the first payload open, every open logged in READ_SPY). Leads 0-24 must equal the sealed SHA-256 of the F20 archive
CPU_OUTPUTS.json (v0.4.0 24 h gate refs). STAGE_ROOT must not exist yet.
"""
import importlib.util, json, os, sys
from pathlib import Path

assert os.environ.get("JAX_PLATFORMS") == "cpu"
G = Path(sys.argv[3]) / "src/gpuwrf/assimilation/letkf/alisios_guard.py"
spec = importlib.util.spec_from_file_location("alisios_guard_lane", G); ag = importlib.util.module_from_spec(spec)
sys.modules["alisios_guard_lane"] = ag; spec.loader.exec_module(ag)
W3P = Path("<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate/arms0227/W3P/d6_72h/summary.json")
F20 = Path("<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/WN3-0227/CPU_OUTPUTS.json")
stage_root, receipt_dir = Path(sys.argv[1]), Path(sys.argv[2])
summary = json.loads(W3P.read_text())
targets, rows = {}, []
for dom in ("d01", "d02", "d03"):
    pairs = summary["domains"][dom]["pairing"]["pairs"]
    assert len(pairs) == 73 and [p["lead_h"] for p in pairs] == list(range(73)), dom
    for p in pairs:
        src = p["cpu_file"]
        assert ag.is_alisios_data(src), src
        day = Path(src).name.split("_")[2].replace("-", "")
        targets[src] = (["20260227"], [day])
        rows.append((dom, p["lead_h"], src))
sealed24 = {r["file"]: r["sha256"] for r in json.loads(F20.read_text())}
staged, receipt = ag.facade_stage(targets, stage_root, receipt_dir,
                                  purpose="w44-rc RW3 real-world gate: original CPU-WRF 72 h reference frames WN3 0227 "
                                          "d01-d03 (same frames as the v0.3.3 W3P full72 gate and the v0.4.1/v0.4.2 RW3) for the v0.4.4 "
                                          "72 h D6 + integrity (contract lanes/w44-rc.md; manager admission + DISK-PLAN required)")
index, bad, checked24 = {}, [], 0
for dom, lead, src in rows:
    key = str(ag.resolve_path(src)); name = Path(src).name
    digest = receipt["files"][key]["sha256"]
    if name in sealed24:
        checked24 += 1
        if sealed24[name] != digest:
            bad.append(name)
    index.setdefault(dom, {})[name] = dict(staged=str(staged[key]), sha256=digest, lead_h=lead, source=src)
(receipt_dir / "STAGED_INDEX.json").write_text(json.dumps({"index": index, "sealed24_checked": checked24,
                                                           "sha_mismatch_vs_f20": bad}, indent=1) + "\n")
print(json.dumps({"staged": {d: len(v) for d, v in index.items()}, "sealed24_checked": checked24, "sha_mismatch": bad,
                  "counters": receipt["counters"], "refused": receipt["spy"]["refused_paths"]}))
sys.exit(1 if bad or checked24 != 75 else 0)
