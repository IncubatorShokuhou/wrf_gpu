"""Bisect dynamic-parameter constantness on retained real Noah-MP columns.

CPU regression diagnosis only; pristine-WRF oracles remain the physics reference.
The public refusal is bypassed only within this process, never in product code.
"""
import argparse
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--cold', action='store_true')
p.add_argument('--coefficients', action='store_true', help='causal coefficient intervention before shared clock hook')
p.add_argument('--arms', default='off,all,hvt,hvb,z0sno,snow_emis,smcmax,dksat,bexp,psisat,rsmin')
a = p.parse_args()
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert os.environ['GPUWRF_FAST_DEFAULTS'] == '1'
sys.path[:0] = [str(a.source/'src'), str(a.source/'proofs/w44_lsmparams')]
import jax
import jax.numpy as jnp
import numpy as np
from columns import build, source_columns, frozen_soil_columns
from gpuwrf.io.noahmp_options import NoahMPOptions
from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
from gpuwrf.runtime.operational_mode import OperationalNamelist, build_clock_base
from gpuwrf.surface_params import controls as sc

assert jax.devices()[0].platform == 'cpu'
sc.UNAVAILABLE_SURFACE_CONTROLS = frozenset()  # local research fixture
rows = source_columns()[4:]+frozen_soil_columns() if a.cold else source_columns()[:4]
options = NoahMPOptions(opt_crs=2)
land, forcing, static, ep, rp, fixture, rows = build(rows=rows, options=options)
prior = sc.prior_controls(fixture, sorted(sc.LSM_CONTROLS), device=False)
nl = OperationalNamelist(grid=None, tendencies=None, metrics=None, dt_s=float(rows[0]['dt']),
                        use_noahmp=True, noahmp_static=static, noahmp_land=land,
                        noahmp_energy_params=ep, noahmp_rad_params=rp, noahmp_options=options)
record = dict(scope=__doc__, platform='cpu', gpu_executed=False, rows=[r['name'] for r in rows],
              cold=a.cold, source=str(a.source), cache='NONE', arms={})
started = time.monotonic()
baseline = None

def save():
    record.update(wall_s=time.monotonic()-started, peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=2)+'\n')

for label in a.arms.split(','):
    selected = () if label == 'off' else sorted(prior) if label == 'all' else label.split('+')
    def program(land, forcing, clock):
        s,e,r = static,ep,rp
        if selected:
            s,e,r = sc.apply_surface_controls(s,e,r,clock.surface_controls)
        return noah_mp_step(land,forcing,s,nl.dt_s,energy_params=e,rad_params=r,options=options,return_diag=True)
    bound = nl if not selected else dataclasses.replace(nl,surface_controls={n:jnp.asarray(prior[n]) for n in selected})
    fn = jax.jit(program)
    clock = build_clock_base(bound)
    if a.coefficients:
        from gpuwrf.surface_params.lsmparams import clock_controls
        clock = clock._replace(surface_controls=clock_controls(bound))
    output = {jax.tree_util.keystr(path):np.asarray(v).copy()
              for path,v in jax.tree_util.tree_flatten_with_path(fn(land,forcing,clock))[0]}
    assert all(np.isfinite(v).all() for v in output.values())
    if baseline is None:
        assert label == 'off'
        baseline = output
    differences = []
    for path,x in baseline.items():
        y = output[path]
        if x.tobytes() == y.tobytes():
            continue
        delta = np.abs(x.astype(np.float64)-y.astype(np.float64))
        differences.append(dict(path=path,cells=int(np.count_nonzero(x!=y)),max_abs=float(delta.max()),
                                off_max=float(np.abs(x.astype(np.float64)).max()),dtype=str(x.dtype)))
    record['arms'][label] = dict(differences=differences,
        sha256={path:hashlib.sha256(v.tobytes()).hexdigest() for path,v in output.items()})
    save()
    print(label, len(differences), round(record['wall_s'],1), flush=True)
    del fn, output, bound
    jax.clear_caches()
    gc.collect()
