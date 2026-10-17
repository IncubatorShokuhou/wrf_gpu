"""Capture real c45 initializer operands and same-state noon solar operands."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

source, case, out = map(Path, sys.argv[1:4])
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert subprocess.run(['git', '-C', str(source), 'symbolic-ref', '--quiet', 'HEAD'], capture_output=True).returncode
sys.path[:0] = [str(source/'src'), str(source/'proofs/o1_nlbind')]
import jax
import numpy as np
import cpu_safe
cpu_safe.install()
from gpuwrf.contracts import state as sc
from gpuwrf.coupling import goddard_radiation as gr
from gpuwrf.integration import nested_pipeline as pipeline
assert all(d.platform == 'cpu' for d in jax.devices())
sc._gpu_device = lambda: jax.devices('cpu')[0]
out.mkdir(parents=True, exist_ok=False)
manifest = json.loads((case/'inputs.json').read_text())
hashes = {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (case/'input').iterdir() if f.is_file()}
assert hashes == manifest['input_sha256']
original = gr.solve_goddard_columns
captures = []
def capture(columns, julday, solcon):
    values = {k:np.asarray(v) for k,v in columns.items()}
    values.update(julday=np.asarray(julday), solcon=np.asarray(solcon))
    assert all(np.isfinite(v).all() for v in values.values())
    name = 'night' if not np.any(values['cosz'] > np.float32(.0001)) else 'day'
    path = out/(name+'.npz')
    assert not path.exists()
    np.savez(path, **values)
    captures.append(dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ncol=values['T'].shape[0], nlev=values['T'].shape[1],
        day_columns=int(np.sum(values['cosz'] > np.float32(.0001))),
        shapes={k:list(v.shape) for k,v in values.items()}))
    return original(columns, julday, solcon)
gr.solve_goddard_columns = capture
started = time.monotonic()
cfg = pipeline.NestedPipelineConfig(case/'input', out/'unused_output', out/'unused_proof', hours=1, max_dom=1)
_, bundles, _, _, _, carries = pipeline._load_domains(cfg, ('d01',))
nl = bundles['d01'].namelist
c = carries['d01']
assert (nl.ra_lw_physics, nl.ra_sw_physics, nl.mp_physics) == (5, 5, 8)
# Real thermodynamic initialization held fixed; noon astronomy is a component
# regime check, explicitly not a simulated h12 state or a fidelity trajectory.
from gpuwrf.runtime.operational_mode import _lake_radiation_state
gr.paired_radiation(c.state, nl.grid, time_utc=nl.time_utc, lead_seconds=43200.,
    solar_lead_seconds=43200., radiation_static=nl.radiation_static,
    land_state=_lake_radiation_state(c,nl))
rec = dict(source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
    input_sha256=hashes, captures=captures, wall_s=time.monotonic()-started,
    peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    scope='c45 real initializer night + same initialized thermodynamics/noon astronomy; CPU capture, no forecast',
    resolved_work=dict(dt_s=float(nl.dt_s), cadence=int(nl.radiation_cadence_steps), ra_lw=5, ra_sw=5, mp=8),
    cache='NONE GPUWRF_JAX_CACHE=0')
(out/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps(rec,indent=2), flush=True)
os._exit(0)
