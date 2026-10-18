"""Compare retained before/after compiler evidence for one actual composite."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

assert os.environ["JAX_PLATFORMS"] == "cpu"
assert not Path("/tmp/wrf_gpu2_quiet").exists()
case = sys.argv[1]
assert case in ("c10", "c18")
lane = Path("<USER_HOME>/wrf_gpu2_lanes/w41-wdm57")
before, after = [lane / (arm + "-" + case) for arm in ("before", "after")]
left, right = [json.loads((root / "summary.json").read_text()) for root in (before, after)]
assert left["inputs"] == right["inputs"]
assert left["programs"][0]["static_kwargs"] == right["programs"][0]["static_kwargs"]
assert left["physical_backend"] == right["physical_backend"] == "cpu"
assert not left["gpu_executed"] and not right["gpu_executed"]
temporaries = []
try:
    for arm, root in (("before", before), ("after", after)):
        target = lane / (case + "-" + arm + ".hlo")
        temporaries.append(target)
        target.write_bytes(gzip.decompress((root / "optimized.hlo.gz").read_bytes()))
    tool = Path(__file__).resolve().parents[1] / "t_cfix_cu/c11_ptxas/canon_diff.py"
    subprocess.run([sys.executable, str(tool), *map(str, temporaries), str(lane / (case + "-canonical.json"))], check=True)
finally:
    for target in temporaries:
        target.unlink(missing_ok=True)
result = {"case": case, "same_inputs_and_work": True, "scope": left["scope"]}
for field in ("source_commit", "n_fusions", "ptxas_sampled_span_s", "ptxas_peak_rss_KiB", "e41_cpu_prescreen"):
    result[field] = {"before": left[field], "after": right[field]}
for field in ("compile_s", "memory"):
    result[field] = {"before": left["programs"][0][field], "after": right["programs"][0][field]}
result["kinput_transpose_over500"] = {name: [(r["fusion"], r["n_instr"]) for r in summary["kinput_transpose_over500"]]
                                        for name, summary in (("before", left), ("after", right))}
result["wdm_max_transpose_instr"] = {name: max(r["n_instr"] for r in summary["wdm_transpose_fusions"])
                                       for name, summary in (("before", left), ("after", right))}
(lane / (case + "-comparison.json")).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
