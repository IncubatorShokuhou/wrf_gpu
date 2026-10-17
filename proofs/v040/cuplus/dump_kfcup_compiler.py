"""Inspect the unchanged compiled CuP artifact without replaying science."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from compare_kfcup_parcel import parse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    sys.path.insert(0,str(args.source/'src'))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpuwrf.kernels.phys_kfcup_cps import kfcup_fresh_column
    assert all(d.platform=='cpu' for d in jax.devices())
    row=json.loads(args.oracle.read_text())['records'][0]
    fields,scalars=parse(row['stdout'])
    names=('T','QV','P','DZ','RHO','W0AVG','U','V','TH','Z','ZW','PI')
    arrays=tuple(jnp.asarray(fields[key],jnp.float32) for key in names)
    arrays+=(jnp.asarray([54.,9000.,1500.,6.,150.,1e-4,1.,0.],jnp.float32),)
    def one(*values):
        controls=values[12]
        return kfcup_fresh_column(*values[:12],dt=controls[0],dx=controls[1],pblh=controls[2],
                                  kpbl=controls[3],hfx=controls[4],qfx=controls[5],land=controls[6],
                                  terrain=controls[7],num_bins=5,interpret=True)
    start=time.monotonic()
    lowered=jax.jit(one).lower(*arrays)
    stable=lowered.as_text()
    compiled=lowered.compile()
    optimized=compiled.as_text()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'stablehlo.txt').write_text(stable)
    (args.output_dir/'optimized_hlo.txt').write_text(optimized)
    memory=compiled.memory_analysis()
    record=dict(source=str(args.source),backend=jax.default_backend(),model_steps_executed=0,
                input_abi='12xREAL32[40]+REAL32[8]runtime controls, matching full CPS comparison',
                compile_s=time.monotonic()-start,stablehlo_f64_tokens=stable.count('f64'),
                optimized_f64_tokens=optimized.count('f64'),
                cost=compiled.cost_analysis(),memory={key:getattr(memory,key) for key in (
                    'argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')},
                source_sha256={key:hashlib.sha256((args.source/key).read_bytes()).hexdigest() for key in (
                    'src/gpuwrf/kernels/phys_kfcup_column.py','src/gpuwrf/kernels/phys_kfcup_cps.py',
                    'src/gpuwrf/physics/cumulus_kfcup_support.py','src/gpuwrf/physics/cumulus_kfcup_tables.py')})
    (args.output_dir/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({key:record[key] for key in ('source','backend','model_steps_executed','compile_s','stablehlo_f64_tokens','optimized_f64_tokens','memory')}))


if __name__=='__main__':
    main()
