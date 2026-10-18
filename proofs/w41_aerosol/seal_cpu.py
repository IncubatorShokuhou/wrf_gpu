"""Seal a finished original CPU-WRF reference for proofs/v040/final_validation/score.py (w41-aerosol composite arms).

usage: JAX_PLATFORMS=cpu python seal_cpu.py CASE      (CASE/input = model inputs + namelist, CASE/cpu = the WRF run dir)
Writes CASE/CPU_OUTPUTS.json (every wrfout: name, bytes, sha256, Times) and CASE/cpu_run.json (rc, wall, executable,
ranks/cores as launched, input and namelist SHA-256, output manifest SHA).  Refuses an unfinished run (rc.txt).
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from netCDF4 import Dataset

case = Path(sys.argv[1]).resolve()
run = case / "cpu"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
rc = (run / "rc.txt").read_text().strip()
assert rc == "rc=0", rc
assert "SUCCESS COMPLETE WRF" in (run / "rsl.error.0000").read_text()
rows = []
for f in sorted(run.glob("wrfout_d0*")):
    with Dataset(f) as ds:
        times = [b"".join(t).decode() for t in ds["Times"][:]]
    rows.append({"file": f.name, "bytes": f.stat().st_size, "sha256": sha(f), "times": times})
(case / "CPU_OUTPUTS.json").write_text(json.dumps(rows, indent=1) + "\n")
t0 = datetime.fromisoformat((run / "start_utc.txt").read_text().strip()[:26])
t1 = datetime.fromisoformat((run / "end_utc.txt").read_text().strip()[:26])
exe = (run / "wrf.exe").resolve()
receipt = {"schema": "w41-aerosol-original-cpu-reference-v1", "rc": 0, "wall_s": round((t1 - t0).total_seconds(), 1),
           "wrfout_files": len(rows), "executable": str(exe), "executable_sha256": sha(exe),
           "launch": (run / "run_ref.sh").read_text(), "rank_pinning": (run / "pin_rank.sh").read_text(),
           "namelist_sha256": sha(run / "namelist.input"),
           "input_sha256": {p.name: sha(p) for p in sorted((case / "input").iterdir())},
           "output_manifest_sha256": sha(case / "CPU_OUTPUTS.json")}
assert receipt["input_sha256"]["namelist.input"] == receipt["namelist_sha256"], "CPU and GPU namelists differ"
(case / "cpu_run.json").write_text(json.dumps(receipt, indent=1) + "\n")
print(json.dumps({k: receipt[k] for k in ("rc", "wall_s", "wrfout_files")}))
