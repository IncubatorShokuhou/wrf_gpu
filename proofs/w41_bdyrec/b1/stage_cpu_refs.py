"""B1 nested D6 gate: stage the original CPU-WRF 24 h reference frames of WN3 0502 + 0614 through the ALISIOS guard.

usage: JAX_PLATFORMS=cpu python stage_cpu_refs.py STAGE_ROOT RECEIPT_DIR
(0227's frames are already staged for the composed requal: requal_refs_receipt/STAGED_INDEX.json.)
ALISIOS CPU-WRF outputs are opened ONLY inside alisios_guard.facade_stage (kernel 3a997be576): one GuardedRead
session scans + admits every day before the first payload open and logs every open (READ_SPY). The frame list and
sealed SHA-256 come from the F20 archive's CPU_OUTPUTS.json (wrf_gpu2 metadata; its cpu/ links are only readlink'ed,
never opened). Every staged copy must equal its sealed SHA. STAGE_ROOT must not exist yet (facade_stage creates it 0700).
"""
import importlib.util, json, os, sys
from pathlib import Path

assert os.environ.get("JAX_PLATFORMS") == "cpu"
G = Path("<USER_HOME>/src/wrf_gpu2/src/gpuwrf/assimilation/letkf/alisios_guard.py")
spec = importlib.util.spec_from_file_location("alisios_guard_main", G); ag = importlib.util.module_from_spec(spec)
sys.modules["alisios_guard_main"] = ag; spec.loader.exec_module(ag)
F20 = Path("<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016")
SEALED = Path("<USER_HOME>/wrf_gpu2_lanes/w41-bdyrec/alisios_sealed/80b076c9fb507aa9f8034b8fcc6effd1c3e2454f")
ISSUE = {"WN3-0502": "20260502", "WN3-0614": "20260614"}
stage_root, receipt_dir = Path(sys.argv[1]), Path(sys.argv[2])

targets, expect = {}, {}
for case, issue in ISSUE.items():
    for row in json.loads((F20 / case / "CPU_OUTPUTS.json").read_text()):
        link = F20 / case / "cpu" / row["file"]
        source = os.readlink(link)                      # the ALISIOS path; not opened here
        assert any(source.startswith(str(root)) for root in ag.DATA_ROOTS) or "alisios" in source, (link, source)
        day = row["file"].split("_")[2].replace("-", "")  # wrfout_dNN_YYYY-MM-DD_hh:mm:ss -> valid day
        targets[source] = ([issue], [day])
        expect[source] = (case, row["file"], row["sha256"])
staged, receipt = ag.facade_stage(targets, stage_root, receipt_dir, sealed_root=SEALED,
                                  purpose="w41-bdyrec B1 v0.4.1 release benchmark: original CPU-WRF 24 h reference frames "
                                          "WN3 0502/0614 d01-d03 for the par24 nested D6 gate (manager 2026-10-18T08:23:19Z)")
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
