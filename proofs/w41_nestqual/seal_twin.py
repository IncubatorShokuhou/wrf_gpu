"""Seal a finished original CPU-WRF twin (lane w41-nestqual) in the v040-original-cpu-reference-v1 layout read by
proofs/v040/final_validation/score.py: <case>/CPU_OUTPUTS.json (file, bytes, sha256, times, fields) + <case>/cpu_run.json.
Refuses unless WRF printed SUCCESS COMPLETE and every domain has hours+1 hourly frames. Adds the CPU timing of the run
(wall from the launcher, per-domain inclusive d01 'Timing for main' sum = s/sim-h, E30) and the actual rank layout.
usage: python seal_twin.py <case_dir> [hours]
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import netCDF4

case = Path(sys.argv[1]).resolve()
hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
cpu = case / "cpu"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


assert not (case / "cpu_run.json").exists(), "never overwrite a reference receipt"
rsl0 = (cpu / "rsl.out.0000").read_text(errors="replace")
assert "SUCCESS COMPLETE WRF" in rsl0, "no SUCCESS COMPLETE WRF"
prep = json.loads((case / "PREPARE.json").read_text())
rows = []
for path in sorted(cpu.glob("wrfout_d0*")):
    with netCDF4.Dataset(path) as ds:
        times = [b"".join(t).decode() for t in ds["Times"][:]]
        fields = {k: dict(dimensions=list(v.dimensions), shape=list(v.shape), dtype=v.dtype.str)
                  for k, v in ds.variables.items()}
    rows.append(dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path), times=times, fields=fields,
                     mtime=path.stat().st_mtime))
domains = sorted({r["file"][7:10] for r in rows})
for d in domains:
    n = sum(r["file"].startswith(f"wrfout_{d}_") for r in rows)
    assert n == hours + 1, f"{d}: {n} frames, expected {hours + 1}"
manifest = case / "CPU_OUTPUTS.json"
manifest.write_text(json.dumps([{k: v for k, v in r.items() if k != "mtime"} for r in rows], indent=2) + "\n")
inputs = {p.name: sha(p) for p in sorted((case / "input").iterdir()) if p.is_file()}
for name, row in prep["inputs"].items():
    assert inputs[name] == row["sha256"], f"{name} changed since PREPARE"
# timing (not a CPU benchmark: shared host, nice 19)
d01 = [float(x) for x in re.findall(r"Timing for main: time \S+ on domain\s+1:\s+([0-9.]+) elapsed", rsl0)]
wall = json.loads((case / "cpu_wall.json").read_text()) if (case / "cpu_wall.json").exists() else {}
d01_frames = sorted(r["mtime"] for r in rows if r["file"].startswith("wrfout_d01_"))
ranks = len(list(cpu.glob("rsl.out.*")))
timing = dict(wall_s=wall.get("wall_s"), d01_inclusive_step_sum_s=sum(d01), d01_steps=len(d01),
              s_per_sim_hour_d01_inclusive=sum(d01) / hours if hours else None,
              s_per_sim_hour_frames=(d01_frames[-1] - d01_frames[0]) / hours if hours and len(d01_frames) > 1 else None,
              ranks=ranks, cores=(case / "cpu_cores").read_text().strip() if (case / "cpu_cores").exists() else None,
              scope=f"original WRF 4.7.1 dmpar, {ranks} ranks pinned one per core, nice 19, shared host; not a CPU benchmark")
receipt = dict(schema="v040-original-cpu-reference-v1", case=case.name, rc=0, success_marker=True,
               status="COMPLETE", frame_count=len(rows), domains=domains, executable=prep.get("executable"),
               input_sha256=inputs, actual_input_sha256=inputs, prepare=prep, timing=timing,
               output_manifest=str(manifest), output_manifest_sha256=sha(manifest),
               timing_scope=timing["scope"], sealed_by="proofs/w41_nestqual/seal_twin.py", sealer_sha256=sha(__file__))
(case / "cpu_run.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(dict(case=case.name, frames=len(rows), domains=domains, timing=timing,
                      manifest_sha256=receipt["output_manifest_sha256"])))
