"""Seal a finished original CPU-WRF reference of a topo_wind case (lane w41-topowind) in the
v040-original-cpu-reference-v1 layout read by proofs/v040/final_validation/score.py:
<case>/CPU_OUTPUTS.json (file, bytes, sha256, times, fields) + <case>/cpu_run.json.
Refuses unless every expected history frame exists and WRF printed SUCCESS COMPLETE.
usage: python seal_cpu.py <case_dir> [hours]
"""
import hashlib
import json
import sys
from pathlib import Path

import netCDF4

case = Path(sys.argv[1]).resolve()
hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
cpu = case / 'cpu'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (case / 'cpu_run.json').exists(), 'never overwrite a reference receipt'
logs = sorted(cpu.glob('rsl.out.0000')) or [case / 'cpu.log']
success = any('SUCCESS COMPLETE WRF' in p.read_text(errors='replace') for p in logs if p.exists())
assert success, f'no SUCCESS COMPLETE WRF in {logs}'
prep = json.loads((case / 'PREPARE.json').read_text())
domains = sorted({p.name[7:10] for p in cpu.glob('wrfout_d0*')})
rows = []
for path in sorted(cpu.glob('wrfout_d0*')):
    with netCDF4.Dataset(path) as ds:
        times = [b''.join(t).decode() for t in ds['Times'][:]]
        fields = {k: dict(dimensions=list(v.dimensions), shape=list(v.shape), dtype=v.dtype.str)
                  for k, v in ds.variables.items()}
    rows.append(dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path), times=times, fields=fields))
for d in domains:
    n = sum(r['file'].startswith(f'wrfout_{d}_') for r in rows)
    assert n == hours + 1, f'{d}: {n} frames, expected {hours + 1}'
manifest = case / 'CPU_OUTPUTS.json'
manifest.write_text(json.dumps(rows, indent=2) + '\n')
inputs = {p.name: sha(p) for p in sorted((case / 'input').iterdir()) if p.is_file()}
receipt = dict(schema='v040-original-cpu-reference-v1', case=case.name, rc=0, success_marker=True,
               status='COMPLETE', frame_count=len(rows), domains=domains, executable=prep.get('executable'),
               input_sha256=inputs, actual_input_sha256=inputs, prepare=prep,
               output_manifest=str(manifest), output_manifest_sha256=sha(manifest),
               timing_scope='original WRF 4.7.1 on shared nice19 lane/loaned cores; not a CPU benchmark',
               sealed_by='proofs/w41_topowind/seal_cpu.py', launcher_sha256=sha(__file__))
(case / 'cpu_run.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(case=case.name, frames=len(rows), domains=domains, manifest_sha256=receipt['output_manifest_sha256'])))
