"""Five independent pristine REAL4 witnesses for Fitch's asymmetric ELSE sum.

Run with the proof-only no-FMA flags documented in UPSTREAM_WRF_BUGS.md.
--mutant restores the rejected grouping in memory and runs the same BYTE gate.
"""
from pathlib import Path
import argparse
import hashlib
import json
import types

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.wind_fitch import turbine_area


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path)
    p.add_argument('--mutant',action='store_true')
    args=p.parse_args()
    assert jax.devices()[0].platform=='cpu'
    root=Path(__file__).resolve().parents[2]
    source=(root/'src/gpuwrf/physics/wind_fitch.py').read_text()
    fixture=json.loads((Path(__file__).with_name('area_order_witnesses.json')).read_text())
    fn=turbine_area
    if args.mutant:
        assert source.count('(fd+c1)+c2')==1
        changed=source.replace('(fd+c1)+c2','fd+fc')
        module=types.ModuleType('fitch_area_order_mutant')
        exec(compile(changed,'<Fitch ELSE association mutant>','exec'),module.__dict__)
        fn=module.turbine_area
    exe=jax.jit(fn)
    actual=[]
    for case in fixture['witnesses']:
        z1,z2,diameter,_=case['inputs']
        assert z1<diameter/2<z2 and diameter/2-z1!=z2-diameter/2
        actual.append(np.asarray(exe(*(jnp.float32(x) for x in case['inputs']))))
    actual=np.asarray(actual,np.float32)
    reference=np.array([case['WRF'] for case in fixture['witnesses']],np.float32)
    report=dict(backend='cpu',source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                fixture_sha256=hashlib.sha256(Path(__file__).with_name('area_order_witnesses.json').read_bytes()).hexdigest(),
                cases=len(reference),byte_equal=int(np.sum(actual.view(np.uint32)==reference.view(np.uint32))),
                actual=actual.tolist(),reference=reference.tolist(),product_files_changed=False,
                mutant=args.mutant)
    try:
        np.testing.assert_array_equal(actual.view(np.uint32),reference.view(np.uint32))
    except AssertionError:
        if not args.mutant:raise
        report['killed']=True
        assert report['byte_equal']==0
    else:
        assert not args.mutant,'ELSE association mutant survived the BYTE gate'
    if args.out:args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
