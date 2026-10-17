"""Read-only independent rescans of retained deviceless HLO and pinned GPU source frames."""
from pathlib import Path
import collections
import contextlib
import gc
import hashlib
import importlib.util
import io
import json
import sys

root, output = map(lambda p: Path(p).resolve(), sys.argv[1:3])
sys.path.insert(0, str(root / "src"))
spec = importlib.util.spec_from_file_location("critic_hlo_scan", root / "scripts/e41/hlo_scan.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)
base = Path("<DATA_ROOT>/wrf_gpu2/scratch/t-cfix-mynn/e41_hlo")
result = {"scope": "retained CPU deviceless artifacts; actual GPU executable scan follows verdict", "arms": {}}
names = ("DL_e7e3", "DL_9a6f", "DL_c5b_PROD_d01", "DL_c5b_PROD_d02",
         "DL_c5b_WN3-0227_d01", "DL_c5b_WN3-0227_fused_d02", "DL_c5b_Swiss_d01")
for name in names:
    folder = base / name
    rec = json.loads((folder / "receipt.json").read_text())
    assert rec["phase"] == "done" and not rec["gpu_executed"] and not rec["source_dirty"]
    raw = (folder / "optimized.hlo").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == rec["optimized_hlo_sha256"]
    rows = scan.scan_text(raw.decode())["transpose_hero_fusions"]
    bad = [(r["fusion"], r["n_instr"]) for r in rows if r["kind"] == "kInput" and r["n_instr"] > 500]
    assert sorted(bad) == sorted(map(tuple, rec["kinput_transpose_over500"]))
    max_size = max(r["n_instr"] for r in rows if r["kind"] == "kInput")
    native = []
    native_hashes = {}
    for path in sorted((folder / "e41").glob("cubin_scan_*.json")):
        native_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        native.extend(json.loads(path.read_text()))
    assert native and len(native) == rec["kernels"]
    assert not any(k["stack"] > 256 or k.get("nonleaf_over1k") or k.get("callgraph_unknown") for k in native)
    assert max(k["stack"] for k in native) == rec["max_stack_B"]
    row = {"source": rec["source_commit"], "optimized_hlo_sha256": sha, "transpose_fusions": len(rows),
           "max_kinput_hlo": max_size, "over500": bad, "max_stack_B": rec["max_stack_B"],
           "kernels": len(native), "native_report_sha256": native_hashes}
    result["arms"][name] = row
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(name, max_size, len(bad), rec["max_stack_B"], flush=True)
    del raw, rows, native
    gc.collect()
assert sorted(n for _, n in result["arms"]["DL_e7e3"]["over500"]) == [694, 803]
assert all(not r["over500"] for n, r in result["arms"].items() if n != "DL_e7e3")
# Redirect only the attribution tool's output destination; retained owner artifacts stay untouched.
script = root / "proofs/t_cfix_mynn/e41/fusion_sources.py"
source = script.read_text()
anchor = 'Path(sys.argv[1] + ".fusion_sources.json").write_text'
assert source.count(anchor) == 1
source = source.replace(anchor, f'Path({str(output.parent / "gpu_source_frames.json")!r}).write_text')
sys.argv = [str(script), str(base / "f20_d01_jit.hlo"),
            "input_add_broadcast_clamp_compare_divide_log_maximum_minimum_multiply_power_select_sqrt_transpose_fusion",
            "input_and_divide_select_transpose_fusion"]
log = io.StringIO()
with contextlib.redirect_stdout(log):
    exec(compile(source, str(script), "exec"), {"__file__": str(script), "__name__": "__main__"})
(output.parent / "gpu_source_frames.log").write_text(log.getvalue())
frames = json.loads((output.parent / "gpu_source_frames.json").read_text())
assert sorted(r["n_instr"] for r in frames.values()) == [694, 803]
result["gpu_frame_attribution_sha256"] = hashlib.sha256((output.parent / "gpu_source_frames.json").read_bytes()).hexdigest()
output.write_text(json.dumps(result, indent=2) + "\n")
