"""Original WRF full-column comparison; tolerances frozen before P310."""
import argparse
import json
from pathlib import Path
import time
from functools import partial

import jax
import numpy as np

from gpuwrf.physics.p3.column import step

from comparison import (RTOL, ComparisonError, checked_inventory, checked_limit,
                        compare_field, tolerance)


def compiled_limit(name, data, out, n, mp, independent):
    rtol, atol = tolerance(name)
    limit = checked_limit(atol + rtol * np.abs(data[name][..., n].astype(np.float64)),
                          f'{name}.strict_limit')
    if independent:
        # Pre-registered L-infinity WRF-vs-WRF contraction spread on identical
        # initial operands (P348). Both complete inventories are already checked.
        literal, fma = independent
        spread = checked_limit(np.max(np.abs(fma[name][...,1:].astype(np.float64) -
                                             literal[name][...,1:])), f'{name}.WRF_spread')
        limit = np.maximum(limit, spread)
        if name == 're_cloud':
            # Preserve the registered two-path gamma32 cancellation bound and
            # WRF get_cloud_dsd2 radius maximum verbatim (COMPILED_BOUND.md).
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                water = sum(data[k][...,0].astype(np.float64) for k in
                            ('qv','qc','qr','qi') + (('qi2',) if mp==52 else ()))
                unit = np.finfo(np.float32).eps/2
                density = (data['p'][...,0].astype(np.float64) /
                           (287.15*data['th'][...,0]*data['exner'][...,0]) *
                           (.622/(.622+data['qv'][...,0])))
                mass_weight = checked_limit(density*data['dz'][...,0], 'cloud.mass_weight')
                inventory = (np.sum(water*mass_weight, axis=-1, keepdims=True) /
                             np.min(mass_weight, axis=-1, keepdims=True))
                residue = checked_limit((2*32*unit/(1-32*unit))*inventory, 'cloud.residue')
                tiny = (np.abs(np.asarray(out['qc'])) + np.abs(data['qc'][...,n])) <= residue
                limit = np.where(tiny, np.maximum(limit, 1/30000), limit)
    return checked_limit(limit, f'{name}.derived_limit')


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--schemes',type=int,nargs='+',choices=(50,51,52,53),default=[50,51,52,53])
    p.add_argument('--steps',type=int,choices=(1,2,3),default=3)
    p.add_argument('--fixtures',type=Path,default=Path(__file__).with_name('fixtures'))
    p.add_argument('--backend',choices=('jax','pallas-interpret'),default='jax')
    p.add_argument('--wrf-spread', type=Path,
                   help='Independent pristine WRF FMA/no-FMA arms under wave policy')
    a=p.parse_args(argv)
    assert jax.devices()[0].platform=='cpu'
    call = step
    if a.backend == 'pallas-interpret':
        from gpuwrf.kernels.phys_p3 import full_column
        @partial(jax.jit, static_argnames=('mp',))
        def call(fields, dt, timestep, *, mp):
            return full_column(fields, dt, timestep, mp=mp, interpret=True)
    report={'backend':'cpu','implementation':a.backend,'rtol':float(RTOL),'scenarios':'original WRF constructed active columns','schemes':{}}
    report.update(validation_failures=0, nonfinite_failures=0)
    try:
        for mp in a.schemes:
            with np.load(a.fixtures/f'columns_mp{mp}.npz',allow_pickle=False) as f:
                data={k:f[k] for k in f.files}
            checked_inventory(data, f'fixture.mp{mp}')
            fields={k:v[...,0] for k,v in data.items() if v.ndim==3}
            if mp!=52:
                for k in ('qi2','ni2','qir2','bir2','vm2','di2','rhoi2'):fields.pop(k,None)
            if mp!=53:fields.pop('qzi',None)
            independent = None
            if a.wrf_spread:
                arms = []
                for arm in ('literal', 'fma'):
                    with np.load(a.wrf_spread/f'{arm}_mp{mp}.npz', allow_pickle=False) as f:
                        arrays = {k: f[k] for k in f.files}
                    checked_inventory(arrays, f'{arm}.mp{mp}')
                    arms.append(arrays)
                independent = tuple(arms)
                if any(not np.array_equal(value, data[name]) for name,value in independent[0].items()):
                    raise ComparisonError(f'literal.mp{mp}: differs from frozen fixture')
            scheme={'steps':{},'compile_first_s':None}
            saved = {}
            report['schemes'][str(mp)]=scheme
            for n in range(1,a.steps+1):
                t=time.monotonic()
                out=call(fields,18.,n,mp=mp)
                jax.block_until_ready(out)
                elapsed=time.monotonic()-t
                if n==1:scheme['compile_first_s']=elapsed
                checked_inventory(out, f'candidate.mp{mp}.step{n}', mp=mp, reference=data)
                scores={}
                for name,value in out.items():
                    ref=data[name][...,n]
                    value=np.asarray(value)
                    limit = compiled_limit(name, data, out, n, mp, independent)
                    score = compare_field(name, value, ref, limit=limit)
                    scores[name] = {key: score[key] for key in
                                    ('finite','max_abs','bad','strict_bad','n','rtol','atol')}
                    saved[f'{name}__{n}'] = value
                scheme['steps'][str(n)]={'elapsed_s':elapsed,'scores':scores}
                a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                print(mp,n,round(elapsed,3),{k:(v['bad'],v['max_abs']) for k,v in scores.items() if v['bad']},flush=True)
                # Original wrapper updates all carried prognostic/old fields.
                # Keep the original wrapper input pytree stable between calls;
                # precipitation diagnostics are outputs, never new column inputs.
                fields={k:out.get(k,value) for k,value in fields.items()}
            jax.clear_caches()
            np.savez_compressed(a.out.with_name(a.out.stem+f'_mp{mp}.npz'), **saved)
    except ComparisonError as error:
        report['validation_failures'] += error.count
        report['nonfinite_failures'] += error.nonfinite
        report['validation_error'] = str(error)
    failures=sum(v['bad'] for s in report['schemes'].values() for r in s['steps'].values() for v in r['scores'].values())
    failures += report['validation_failures']
    report['failure_count'] = failures
    report['verdict']='PASS' if failures==0 else 'FAIL'
    report['compiled_policy'] = str(a.wrf_spread) if a.wrf_spread else None
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(report['verdict'],failures,flush=True)
    return 0 if failures==0 else 1


if __name__=='__main__':
    raise SystemExit(main())
