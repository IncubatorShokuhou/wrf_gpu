"""CPU follow-up of the w41-bdyrec GPU requalification (after the lock is released).

usage: JAX_PLATFORMS=cpu python gpu_followup.py ROOT [--skip-d6] [--skip-e41]
Writes ROOT/RESULT.json:
  phase 1  per case: model rc, frames, E41 (scripts/wave_cubin_gate.py of the candidate on the
           case cache, with the run proof), frozen original-CPU D6 per domain (sealed F20 scorer,
           SHA-pinned), frames 0-3 whole-file SHA vs the sealed v0.4.0 bundle frames (WN3/Swiss)
  phase 2  per case: new cache entries of the truncated-input arm (names), AOT loaded lines,
           frames == phase-1 frames of the same hours
  phase 3  release control frames == bundle frames (E99 control); nsys copy/sync counts and the
           warm wall of release vs candidate
Never edits a receipt it did not write; missing evidence is reported as such, never as PASS.
"""
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

assert os.environ.get("JAX_PLATFORMS") == "cpu"
ROOT = Path(sys.argv[1]).resolve()
SKIP = set(sys.argv[2:])
plan = json.loads((ROOT / "PLAN.json").read_text())
CAND = Path(plan["candidate"])
TOOLS = ROOT / "tools"
PINS = {"score.py": "4c025af8382c9626e7f7e0de1e888a91bd0270376f5c837431d1000c70c2c21d",
        "score_with_handles.py": "7407b0ddc611640b318e930f29b62673015f894fbc39d76f34d42856f92d8d19"}
ENV = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH=str(CAND / "src"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frames(path):
    path = Path(path)
    if not path.exists():
        return {}
    return dict(line.split()[::-1] for line in path.read_text().splitlines() if line.strip())


def rc_of(log):
    match = re.findall(r"arm rc=(\d+)", Path(log).read_text()) if Path(log).exists() else []
    return int(match[-1]) if match else None


def wall_of(log):
    match = re.findall(r"Elapsed \(wall clock\) time .*: (\S+)", Path(log).read_text()) if Path(log).exists() else []
    if not match:
        return None
    parts = [float(x) for x in match[-1].split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


def first_hours(table, hours):
    """The first hours+1 frames of every domain (stamps sort chronologically; dt 54 s alarms included)."""
    keep = {}
    for domain in sorted({name.rsplit("_", 2)[0] for name in table}):
        names = sorted(n for n in table if n.rsplit("_", 2)[0] == domain)[:hours + 1]
        keep.update({n: table[n] for n in names})
    return keep


def _csv_rows(text):
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if "Count" in line or "Num Calls" in line), None)
    return [] if header is None else list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))


def nsys_counts(report):
    """HtoD/DtoH memcpy counts and sync/host-function API calls from an .nsys-rep (whole process)."""
    if not Path(report).exists():
        return {"missing": str(report)}
    keep = {}
    for name in ("cuda_gpu_mem_size_sum", "cuda_api_sum"):
        text = subprocess.run(["nsys", "stats", "--report", name, "--format", "csv", str(report)],
                              capture_output=True, text=True, timeout=1800).stdout
        for row in _csv_rows(text):
            if name == "cuda_gpu_mem_size_sum":
                keep[row.get("Operation", "?")] = int(float(row.get("Count", 0)))
            elif any(k in row.get("Name", "") for k in ("Synchronize", "Memcpy", "LaunchHostFunc")):
                keep[row["Name"]] = int(float(row.get("Num Calls", 0)))
    return keep


result = {"plan": plan, "phase1": {}, "phase2": {}, "phase3": {}}
for name, pin in PINS.items():
    assert sha(TOOLS / name) == pin, f"scorer pin {name}"
for case_name, arm in plan["arms"].items():
    case = Path(arm["case"])
    row = {"rc": rc_of(case / "run24/run.log"), "wall_s": wall_of(case / "run24/run.log"),
           "frames": len(frames(case / "run24/wrfout.sha256"))}
    if "--skip-e41" not in SKIP and not (case / "e41/wave_gate.json").exists():
        command = [sys.executable, str(CAND / "scripts/wave_cubin_gate.py"), str(case / "cache/jax"), str(case / "e41")]
        proof = case / "run24/proof/nested_pipeline_run.json"
        if proof.exists():
            command += ["--run-proof", str(proof)]
        subprocess.run(command, cwd=CAND, env=ENV, timeout=1800)
    if (case / "e41/wave_gate.json").exists():
        gate = json.loads((case / "e41/wave_gate.json").read_text())
        row["E41"] = gate.get("e41_gate"), gate.get("coverage", {}).get("status")
    row["D6"] = {}
    for domain in arm["domains"]:
        target = case / f"D6_{domain}.json"
        if "--skip-d6" not in SKIP and not target.exists():
            subprocess.run([sys.executable, str(TOOLS / "score_with_handles.py"), str(CAND), str(case),
                            "--domains", domain], cwd=CAND, env=ENV, timeout=3600)
        if target.exists():
            d6 = json.loads(target.read_text())
            row["D6"][domain] = {k: d6.get(k) for k in ("verdict", "paired_frames", "coverage_complete")} | {
                "bounded_misses": len(d6.get("failures", [])), "nonfinite": len(d6.get("nonfinite", []))}
    if "release_frames" in arm:
        mine = first_hours(frames(case / "run24/wrfout.sha256"), 3)
        ref = arm["release_frames"]
        row["byte_vs_release_bundle"] = {"compared": sorted(ref), "equal": sorted(n for n in ref if mine.get(n) == ref[n]),
                                         "differ": sorted(n for n in ref if mine.get(n) != ref[n])}
    result["phase1"][case_name] = row
    short = Path(arm["short"])
    before = (short / "inventory_before.txt")
    after = (short / "inventory_after.txt")
    two = {"rc": rc_of(short / "run3/run.log"), "wall_s": wall_of(short / "run3/run.log")}
    if before.exists() and after.exists():
        old = {line.split()[0] for line in before.read_text().splitlines()}
        new = [line for line in after.read_text().splitlines() if line.split()[0] not in old]
        two["new_cache_entries"] = new
        two["new_step_programs"] = [line for line in new if re.match(r"(aot/|jit__advance_chunk|jit_fused)", line)]
    log = short / "run3/run.log"
    if log.exists():
        two["aot_lines"] = [line for line in log.read_text().splitlines() if "nested-aot" in line][:20]
    ours = frames(short / "run3/wrfout.sha256")
    full = frames(case / "run24/wrfout.sha256")
    two["frames_equal_phase1"] = sorted(n for n in ours if full.get(n) == ours[n])
    two["frames_differ_phase1"] = sorted(n for n in ours if full.get(n) != ours[n])
    result["phase2"][case_name] = two
control = Path(plan["release_control"]["case"])
ref = plan["arms"]["WN3-0227"]["release_frames"]
ctrl = frames(control / "cold3/wrfout.sha256")
result["phase3"] = {
    "release_control_rc": rc_of(control / "cold3/run.log"),
    "release_control_equal_bundle": sorted(n for n in ref if ctrl.get(n) == ref[n]),
    "release_control_differ_bundle": sorted(n for n in ref if ctrl.get(n) != ref[n]),
    "release_nsys": nsys_counts(control / "warm3_nsys/release.nsys-rep"),
    "candidate_nsys": nsys_counts(Path(plan["arms"]["WN3-0227"]["case"]) / "warm3_nsys/candidate.nsys-rep"),
    "release_warm_wall_s": wall_of(control / "warm3_nsys/run.log"),
    "candidate_warm_wall_s": wall_of(Path(plan["arms"]["WN3-0227"]["case"]) / "warm3_nsys/run.log"),
    "scope": "nsys walls are profiled walls (E185), not S1; counts are whole-process (init + 3 h)",
}
(ROOT / "RESULT.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
print(json.dumps({"phase1": result["phase1"], "phase2": {k: {x: v.get(x) for x in ("rc", "new_step_programs",
      "frames_differ_phase1")} for k, v in result["phase2"].items()}, "phase3": {k: v for k, v in result["phase3"].items()
      if "nsys" not in k}}, indent=1, default=str))
