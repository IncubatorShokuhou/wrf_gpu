"""Original-WRF two-domain comparison; release kernels interpreted on CPU.

Only the public *admission* refusal is bypassed for this staging experiment.
All runtime guards and the actual coupled parent/child runner remain enabled.
CPU PTX/Thomas shims are evidence tooling, never a GPU executable claim.
"""
from datetime import timedelta
from pathlib import Path
import hashlib
import json
import os
import sys
import time

assert os.environ['JAX_PLATFORMS'] == 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'o1_smag3d'))
import cpu_safe
import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset
from jax.experimental import pallas as pl
import jax.experimental.pallas.triton as pt

cpu_safe.install()
real_pallas_call = pl.pallas_call


def interpret_call(*args, **kwargs):
    kwargs['interpret'] = True
    kwargs.pop('compiler_params', None)
    return real_pallas_call(*args, **kwargs)


def cpu_inline_asm(asm, *, args, result_shape_dtypes, **kwargs):
    op = asm.split()[0].split('.')[0]
    a, b = (jnp.asarray(x, jnp.float64) for x in args)
    exact = {'add': a+b, 'sub': a-b, 'mul': a*b, 'div': a/b}[op]
    dtype = result_shape_dtypes[0].dtype
    if jnp.dtype(dtype) == jnp.float32:
        exact = jax.lax.reduce_precision(exact, exponent_bits=8, mantissa_bits=23)
    return [exact.astype(dtype)]


pl.pallas_call = interpret_call
pt.elementwise_inline_asm = cpu_inline_asm
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.io import namelist_check
from gpuwrf.io.wrfout_writer import _theta_dry_for_output
from gpuwrf.nesting.scalar_species import extra_boundary_species
from gpuwrf.runtime.domain_tree import DomainTree, run_operational_domain_tree

state_contract._gpu_device = lambda: jax.devices('cpu')[0]
assert jax.devices()[0].platform == 'cpu'
assert jax.config.x64_enabled, 'WRF DOUBLE islands and the CPU rounding shim require JAX_ENABLE_X64=1'
namelist_check.refuse_single_domain_mp_nested = lambda *args: None
CASE = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
plan = json.loads((CASE / 'PLAN.json').read_text())
frozen = Path(__file__).resolve().parents[1] / 'v014/grid_delta_atlas/tolerance_manifest_candidate.json'
manifest = json.loads(frozen.read_text())['fields']
report = {'reference_plan': plan, 'manifest_sha256': hashlib.sha256(frozen.read_bytes()).hexdigest(),
          'mode': 'release CPU Pallas interpretation + RN32 PTX emulation + Thomas harness solver',
          'public_admission_override': 'staging only: extra-species nested preflight refusal',
          'root_steps_requested': STEPS, 'frames': [], 'status': 'LOADING', 'cache': 'NONE'}
started = time.monotonic()


def save():
    report['wall_s'] = time.monotonic()-started
    OUT.write_text(json.dumps(report, indent=2)+'\n')


def regions(shape):
    ny, nx = shape[-2:]
    y, x = np.indices((ny, nx))
    ring = np.minimum(np.minimum(y, ny-1-y), np.minimum(x, nx-1-x))
    return {'boundary_0_4': ring < 5, 'interior_5_plus': ring >= 5}


VARIABLES = {'qh':'QHAIL', 'Nc':'QNCLOUD', 'Ns':'QNSNOW', 'Ng':'QNGRAUPEL', 'Nh':'QNHAIL',
             'Nn':'QNCCN', 'qvolg':'QVGRAUPEL', 'qvolh':'QVHAIL', 'nwfa':'QNWFA', 'nifa':'QNIFA'}


def observe(name, step, carry):
    jax.block_until_ready(carry)
    state = carry.state
    valid = run_start + timedelta(seconds=step*dts[name])
    reference = CASE / f'wrfout_{name}_{valid:%Y-%m-%d_%H:%M:%S}'
    frame = {'domain': name, 'own_step': step, 'reference': str(reference), 'fields': {},
             'nonfinite_carry_values': sum(int((~np.isfinite(np.asarray(x))).sum())
                 for x in jax.tree.leaves(carry) if np.issubdtype(np.asarray(x).dtype, np.floating))}
    arrays = {k: np.asarray(getattr(state, v)) for k,v in {
        'U':'u','V':'v','W':'w','QVAPOR':'qv','QCLOUD':'qc','QRAIN':'qr',
        'QICE':'qi','QSNOW':'qs','QGRAUP':'qg','QNICE':'Ni','QNRAIN':'Nr','P':'p_perturbation',
        'PH':'ph_perturbation','MU':'mu_perturbation','RAINNC':'rain_acc','RAINC':'rainc_acc'}.items()}
    arrays['T'] = _theta_dry_for_output(state, np.asarray(state.theta), np.asarray(state.qv))-300.
    if step and carry.history_diagnostics is not None:
        arrays.update({key:np.asarray(getattr(carry.history_diagnostics,attr))
                       for key,attr in {'T2':'t2','U10':'u10','V10':'v10','PSFC':'psfc'}.items()})
    if carry.base_state is not None:
        arrays.update({k: np.asarray(getattr(carry.base_state, v))
                       for k,v in {'PB':'pb','PHB':'phb','MUB':'mub'}.items()})
    for species in extra_boundary_species(plan['mp_physics']):
        key = 'QNDROP' if plan['mp_physics']==18 and species=='Nc' else VARIABLES[species]
        arrays[key] = np.asarray(getattr(state, species))
    with Dataset(reference) as ds:
        for key, actual in arrays.items():
            if key not in ds.variables:
                frame['fields'][key] = {'status':'MISSING_REFERENCE'}
                continue
            expected = np.asarray(ds[key][0])
            if actual.shape != expected.shape:
                raise ValueError(f'{name}/{key}: {actual.shape} != WRF {expected.shape}')
            delta = actual.astype(np.float64)-expected.astype(np.float64)
            row = {'actual_dtype':str(actual.dtype), 'shape':list(actual.shape),
                   'actual_nonzero':int(np.count_nonzero(actual)),
                   'reference_nonzero':int(np.count_nonzero(expected)),
                   'reference_sha256':hashlib.sha256(expected.tobytes()).hexdigest(),
                   'finite':bool(np.isfinite(actual).all() and np.isfinite(expected).all()), 'regions':{}}
            for region, mask in regions(actual.shape).items():
                a,b,d = actual[...,mask],expected[...,mask],delta[...,mask]
                stats = {'rmse':float(np.sqrt(np.mean(d*d))), 'max_abs':float(np.abs(d).max())}
                bounds = manifest.get(key,{})
                gates = [stats[k] <= bounds[k] for k in ('rmse','max_abs') if k in bounds]
                if key in VARIABLES.values() or key=='QNDROP':
                    family = 'extra_mass' if key=='QHAIL' else ('extra_volumes' if key in ('QVGRAUPEL','QVHAIL') else 'extra_numbers')
                    limits = plan['gate'][family]
                    stats['max_normalized'] = float(np.max(np.abs(d)/(limits['atol']+limits['rtol']*np.abs(b))))
                    gates.append(stats['max_normalized'] <= 1.)
                stats['gate'] = 'PASS' if gates and all(gates) and row['finite'] else ('FAIL' if gates else 'REPORT_ONLY')
                row['regions'][region] = stats
            frame['fields'][key] = row
    if carry.census is not None:
        frame['census_guards'] = np.asarray(carry.census.guards).tolist()
        from gpuwrf.diagnostics.census import F2_FIELDS, F2_PHASES, F2_EVENTS
        frame['f2_axes'] = {'phases':F2_PHASES,'fields':F2_FIELDS,'events':F2_EVENTS}
        for field in ('f2_counts','f2_field_observations','water_changes','unknown_water_changes',
                      'f2_first_value','f2_first_step','f2_first_index',
                      'f2_min_negative','f2_min_step','f2_min_index','f2_negative_location'):
            values = getattr(carry.census,field)
            frame[field] = None if values is None else np.asarray(values).tolist()
    report['frames'].append(frame)
    report['status'] = 'RUNNING'
    save()
    failures = [(k,r) for k,v in frame['fields'].items() for r,s in v.get('regions',{}).items() if s['gate']=='FAIL']
    print(json.dumps({'domain':name,'step':step,'nonfinite':frame['nonfinite_carry_values'],
                      'failures':failures,'wall_s':report['wall_s']}), flush=True)


observe.wants_carry = True
try:
    config = pipeline.NestedPipelineConfig(CASE, OUT.parent/'history', OUT.parent/'proof', hours=1, max_dom=2)
    hierarchy, bundles, meta, run_start, dts, carries = pipeline._load_domains(config, ('d01','d02'))
    report['load_s'] = time.monotonic()-started
    report['namelists'] = {name:{key:getattr(bundle.namelist,key) for key in
        ('mp_physics','dt_s','moist_adv_opt','scalar_adv_opt','use_mp_re','ra_lw_physics','ra_sw_physics')}
        for name,bundle in bundles.items()}
    report['persistent_state_bytes'] = {name:int(carry.state.bytes()) for name,carry in carries.items()}
    report['extra_boundary_bytes'] = {name:sum(int(getattr(carry.state,f'{species}_bdy').nbytes)
        for species in extra_boundary_species(plan['mp_physics'])
        if getattr(carry.state,f'{species}_bdy') is not None) for name,carry in carries.items()}
    for name, carry in carries.items():
        observe(name,0,carry)
    if STEPS:
        tree = DomainTree.from_domains(hierarchy,bundles,feedback_enabled=False)
        result = run_operational_domain_tree(tree,root_steps=STEPS,carries=carries,output=observe,
            output_cadence_steps={'d01':1,'d02':1},block_between=True,event_aware_fusion_k=0)
        report['own_steps'] = result.own_steps
    report['status'] = 'COMPLETE'
    save()
except Exception as exc:
    report['status'] = 'ERROR'
    report['error'] = f'{type(exc).__name__}: {exc}'
    save()
    raise
