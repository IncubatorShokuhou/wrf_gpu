"""Changed native output projection vs existing pristine WRF bounds on CPU.

P3 uses its frozen field tolerances. Jensen reports the independent input-ULP
and FMA enclosure and compares old outputs with the retained native receipt;
an excess is reported, never converted into a widened bound or a PASS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.contracts.ice_diagnostics import ice_diagnostic_mapping


def main():
    p = argparse.ArgumentParser(); p.add_argument('--mp', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--wrf-spread', type=Path, default=Path('<USER_HOME>/wrf_gpu2_lanes/s2-p3/P348_WRF_contract'))
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]; mp = a.mp
    assert jax.devices()[0].platform == 'cpu'
    report = {'mp': mp, 'GPU': False, 'platform': 'cpu', 'fields': {},
              'source': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
              'scope': 'changed operational native output projection; CPU Pallas interpreter versus pristine WRF'}
    start = time.monotonic()
    if mp == 55:
        from gpuwrf.kernels.phys_jensen import full_column
        from gpuwrf.physics.jensen_ishmael.registry import output_mapping
        folder = root / 'proofs/s2_mpmore/jensen55'
        manifest = json.loads((folder / 'column_controls/manifest.json').read_text())
        old_names = (*output_mapping(), 'th')
        # Exact operational adapter ABI: diagnostics extend mapping BEFORE th.
        names = (*output_mapping(), *ice_diagnostic_mapping(mp), 'th')
        rows = [json.loads((folder / f'fixtures/fp32_{i}.json').read_text()) for i in range(1,18)]
        f = {n: jnp.asarray(v, jnp.float32) for n,v in rows[0]['input'].items()}
        fn = jax.jit(lambda f,d,i: full_column(f,d,i,interpret=True,output_names=names))
        t = time.monotonic(); ex = fn.lower(f,jnp.float32(rows[0]['dt']),jnp.int32(rows[0]['it'])).compile()
        report['CPU_compile_s'] = time.monotonic()-t
        print('COMPILED',mp,report['CPU_compile_s'],flush=True)
        memory = ex.memory_analysis()
        actual = {n: [] for n in names}; old_values = []
        for row in rows:
            f = {n: jnp.asarray(v,jnp.float32) for n,v in row['input'].items()}
            out, precip = jax.tree.map(np.asarray, ex(f,jnp.float32(row['dt']),jnp.int32(row['it'])))
            assert all(np.isfinite(v).all() for v in (*out.values(),*precip.values()))
            old_values.append(np.concatenate([*(out[n].ravel() for n in old_names),
                np.asarray([precip[n] for n in manifest['precipitation_names']],np.float32)]))
            for n in names: actual[n].append(out[n])
            print('COLUMN',mp,len(old_values),flush=True)
        old = np.load(folder / 'native_operational_observations.npz')['outputs']
        report['existing_native_outputs_BYTE'] = np.stack(old_values).tobytes() == old.tobytes()
        np.savez_compressed(a.out.with_suffix('.npz'),**{n:np.stack(v) for n,v in actual.items()})
        nz = len(rows[0]['input']['qv']); refs = []
        for mode in ('noFMA','FMA'):
            with np.load(folder / f'column_controls/fp32_{mode}.npz') as data:
                refs.append(data['outputs'])
        for n in ice_diagnostic_mapping(mp):
            index = manifest['output_names'].index(n)
            sl = slice(index*nz,(index+1)*nz)
            low = np.minimum.reduce([x.min(axis=0)[:,sl] for x in refs])
            high = np.maximum.reduce([x.max(axis=0)[:,sl] for x in refs])
            value = np.stack(actual[n]); bad = (value<low)|(value>high)
            report['fields'][n] = dict(nonfinite=int((~np.isfinite(value)).sum()),
                outside_original_enclosure=int(bad.sum()),
                max_excess=float(np.max(np.maximum(low-value,value-high).clip(0))),
                max_abs=float(np.max(np.abs(value-refs[0][0][:,sl]))),
                min=float(value.min()), max=float(value.max()), nonzero=int(np.count_nonzero(value)))
        report['status'] = ('REGRESSION_REVIEW_REQUIRED' if not report['existing_native_outputs_BYTE'] else
                            'PASS' if not any(x['outside_original_enclosure'] for x in report['fields'].values())
                            else 'DISCLOSED_EXCESS')
    else:
        from gpuwrf.kernels.phys_p3 import full_column
        from gpuwrf.physics.p3.adapter import output_mapping
        sys.path.insert(0,str(root / 'proofs/s2_p3'))
        from comparison import tolerance
        from run_column_gate import compiled_limit
        with np.load(root / f'proofs/s2_p3/fixtures/columns_mp{mp}.npz') as data:
            data = dict(data)
        inputs=('qv','qc','qr','qi','ni','nr','nc','qir','bir','th_old','qv_old','th','p','dz')
        inputs += ('qi2','ni2','qir2','bir2') if mp==52 else ('qzi',) if mp==53 else ()
        fields = {n:jnp.asarray(data[n][...,0]) for n in inputs}
        independent = []
        for tag in ('literal', 'fma'):
            path = a.wrf_spread / f'{tag}_mp{mp}.npz'
            report[tag + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as original:
                independent.append(dict(original))
        assert all(np.array_equal(v, independent[0][n]) for n,v in data.items())
        report['bound'] = 'Frozen P3 field tolerance or independently measured original WRF FMA/no-FMA spread (same P348/P360 rule)'
        names = (*output_mapping(mp),'th','rainncv','snowncv')
        fn = jax.jit(lambda f,d,i:full_column(f,d,i,mp=mp,interpret=True,output_names=names))
        t=time.monotonic(); ex=fn.lower(fields,jnp.float32(18),jnp.int32(1)).compile()
        report['CPU_compile_s']=time.monotonic()-t;memory=ex.memory_analysis()
        print('COMPILED',mp,report['CPU_compile_s'],flush=True)
        saved={}
        for it in (1,2,3):
            out=jax.tree.map(np.asarray,ex(fields,jnp.float32(18),jnp.int32(it)))
            saved.update({f'{name}__{it}':value for name,value in out.items()})
            assert all(np.isfinite(v).all() for v in out.values())
            for n in out:
                ref=data[n][...,it];value=out[n];rtol,atol=tolerance(n)
                error=np.abs(value.astype(np.float64)-ref.astype(np.float64));strict=atol+rtol*np.abs(ref)
                spread=np.max(np.abs(independent[1][n][...,1:].astype(np.float64) - independent[0][n][...,1:].astype(np.float64)))
                limit=compiled_limit(n,data,out,it,mp,independent)
                report['fields'][f'{n}__{it}']=dict(nonfinite=int((~np.isfinite(value)).sum()),
                    outside_frozen_tolerance=int(np.count_nonzero(error>limit)),
                    outside_strict_tolerance=int(np.count_nonzero(error>strict)),original_WRF_spread=float(spread),
                    max_abs=float(error.max()),min=float(value.min()),max=float(value.max()),
                    nonzero=int(np.count_nonzero(value)))
            fields={n:jnp.asarray(out.get(n,v)) for n,v in fields.items()}
            print('STEP',mp,it,flush=True)
        np.savez_compressed(a.out.with_suffix('.npz'),**saved)
        report['status']='PASS' if not any(x['outside_frozen_tolerance'] for x in report['fields'].values()) else 'FAIL'
    report['memory'] = dict(arguments_B=memory.argument_size_in_bytes,outputs_B=memory.output_size_in_bytes,
                           temps_B=memory.temp_size_in_bytes)
    report['wall_s']=time.monotonic()-start
    report['maxrss_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    if report['status']=='FAIL': raise SystemExit(1)


if __name__=='__main__': main()
