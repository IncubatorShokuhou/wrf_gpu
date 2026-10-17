"""Pristine compiler-init sensitivity and WRF FMA/no-FMA spread (CPU only).

The registered numerical bounds in oracle.py remain unchanged. Every excess
over WRF's own per-value compiler spread is reported, including small ones.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).with_name('oracle.py')
    for arm,flags in (('zero',()),('poison',('--init','poison')),('fma',('--optimization','native'))):
        subprocess.run([sys.executable,str(script),'--out',str(args.out/arm),*flags],check=True,timeout=120)
    import oracle
    cases=oracle.inputs()
    truth=np.load(args.out/'zero/oracle.npz');poison=np.load(args.out/'poison/oracle.npz');fma=np.load(args.out/'fma/oracle.npz')
    before=np.load(Path(__file__).with_name('oracle.npz'))
    result={'admitted_cases':len(cases),'initialized_vs_original_byte':{},'init_sensitivity_byte':{},'compiled':{},
            'scope':'original WRF computational routines; REAL4 literal and native compiler contraction; no GPU claim'}
    for key in truth.files:
        result['initialized_vs_original_byte'][key]=bool(np.array_equal(truth[key].view(np.uint32),before[key].view(np.uint32)))
        result['init_sensitivity_byte'][key]=bool(np.array_equal(truth[key].view(np.uint32),poison[key].view(np.uint32)))
    import jax
    import jax.numpy as jnp
    from gpuwrf.io.windfarm import WindFarm
    from gpuwrf.physics.wind_fitch import dragforce
    assert jax.devices()[0].platform=='cpu'
    outputs=[]
    for factor in (0.,.25,1.):
        farm=WindFarm(cases[0]['turbines'],factor)
        fn=jax.jit(lambda u,v,z,dz,q,du,dv:dragforce(u,v,z,dz,q,du,dv,dx=1000.,dt=18.,farm=farm))
        for case in cases:
            if case['factor']==factor:outputs.append(fn(*(jnp.asarray(case[k]) for k in ('u','v','z','dz','qke','du','dv'))))
    for i,key in enumerate(('du','dv','qke','power')):
        actual=np.stack([np.asarray(x[i]) for x in outputs]).astype(np.float64)
        reference=truth[key].astype(np.float64)
        spread=np.abs(fma[key].astype(np.float64)-reference)
        error=np.abs(actual-reference)
        result['compiled'][key]=dict(max_jax_abs_error=float(error.max()),max_wrf_compiler_spread=float(spread.max()),
                per_value_excess_count=int(np.count_nonzero(error>spread)),max_excess=float(np.maximum(error-spread,0).max()),
                excess_disposition='disclosed REAL4 compiler/intrinsic deviation; frozen component tolerances and full D6 remain binding')
    assert all(result['initialized_vs_original_byte'].values()) and all(result['init_sensitivity_byte'].values())
    (args.out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
