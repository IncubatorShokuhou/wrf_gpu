"""Freeze small CPU compile receipts, scan the retained real executable, no GPU."""
import gzip
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
assert os.environ["JAX_PLATFORMS"] == "cpu"
assert not Path("/tmp/wrf_gpu2_quiet").exists()
lane = Path("<USER_HOME>/wrf_gpu2_lanes/w41-wdm57")
arm = sys.argv[1]
assert arm in ("before-c10", "after-c10", "before-c18", "after-c18")
out = lane / arm
trace = (out / "scanner-stacks.log").open("w")
faulthandler.dump_traceback_later(30, repeat=True, file=trace)
rec = json.loads((out / "receipt.json").read_text())
dump = lane / (arm + "-work/dump")
files = list(dump.glob("*after_optimizations.txt"))
assert len(files) == 1, files
hlo = files[0].read_bytes()
print("STAGE HLO read", len(hlo), flush=True)
(out / "optimized.hlo.gz").write_bytes(gzip.compress(hlo, mtime=0))
for pattern in ("*memory-usage-report.txt", "*buffer-assignment.txt"):
    matches = list(dump.glob(pattern))
    assert len(matches) == 1, matches
    with gzip.open(out / (pattern[1:] + ".gz"), "wb") as stream:
        stream.write(matches[0].read_bytes())
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import wave_cubin_gate
from e41 import hlo_scan

scan = hlo_scan.scan_text(hlo.decode())
print("STAGE HLO parsed", scan["n_fusions"], flush=True)
# Resolve the *same text's* stack IDs; op_name alone omits the WDM source.
lines = hlo.decode().splitlines()
heads = {"FileNames", "FunctionNames", "FileLocations", "StackFrames"}


def table(name):
    index, rows = lines.index(name) + 1, {}
    while index < len(lines) and lines[index] and lines[index] not in heads:
        key, value = lines[index].split(" ", 1)
        rows[int(key)] = value
        index += 1
    return rows


files = {k: v.strip('"') for k, v in table("FileNames").items()}
functions = {k: v.strip('"') for k, v in table("FunctionNames").items()}
locations = {k: dict(re.findall(r"(\w+)=(\d+)", v)) for k, v in table("FileLocations").items()}
frames = {k: dict(re.findall(r"(\w+)=(\d+)", v)) for k, v in table("StackFrames").items()}


def stack(frame):
    result, visited = [], set()
    while frame and frame not in visited:
        visited.add(frame)
        row = frames[frame]
        loc = locations[int(row["file_location_id"])]
        result.append((files[int(loc["file_name_id"])], functions[int(loc["function_name_id"])], int(loc["line"])))
        frame = int(row.get("parent_frame_id", 0))
    return result


wdm_transposes = []
for fusion in scan["transpose_hero_fusions"]:
    body = lines[fusion["comp_lines"][0] - 1:fusion["comp_lines"][1]]
    sources = [source for line in body for match in re.finditer(r"stack_frame_id=(\d+)", line)
               for source in stack(int(match[1]))]
    wdm = sorted({source for source in sources if "wdm" in source[0] or "wdm" in source[1]})
    if wdm:
        wdm_transposes.append({**fusion, "wdm_sources": wdm})
gate = wave_cubin_gate.run_gate(out / "cache", out / "e41")
print("STAGE native scanned", gate["e41_gate"], flush=True)
ptxas = list(rec["ptxas"].values())
assert ptxas, "missing assembler observations"
summary = {
    "arm": arm, "source_commit": rec["source_commit"], "compile": rec["compile"],
    "physical_backend": rec["physical_backend"], "gpu_executed": rec["gpu_executed"],
    "native_trace": rec["native_trace"], "inputs": rec["input_sha256"],
    "programs": rec["programs"], "peak_host_rss_KiB": rec["peak_rss_KiB"],
    "ptxas_sample_interval_s": 1.0,
    "ptxas_sampled_span_s": sum(r["last"] - r["first"] for r in ptxas),
    "ptxas_peak_rss_KiB": max(r["peak_rss_KiB"] for r in ptxas),
    "ptxas_jobs_observed": len(ptxas),
    "optimized_hlo_sha256": hashlib.sha256(hlo).hexdigest(),
    "n_fusions": scan["n_fusions"],
    "wdm_transpose_fusions": wdm_transposes,
    "kinput_transpose_over500": [r for r in scan["transpose_hero_fusions"]
                                if r["kind"] == "kInput" and r["n_instr"] > 500],
    "e41_cpu_prescreen": gate["e41_gate"],
    "scope": "CPU CompileOnly sm_120a AUTOTUNE0 prescreen; no GPU speed/fidelity claim",
}
(out / "hlo_scan.json").write_text(json.dumps(scan, indent=2) + "\n")
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k not in (
    "programs", "inputs", "native_trace", "wdm_transpose_fusions")}, indent=2))
faulthandler.cancel_dump_traceback_later()
trace.close()
