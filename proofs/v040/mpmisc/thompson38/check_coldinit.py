"""True original coldinit BYTE plus host init timing/memory shape receipt."""
import ast
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    root=Path(__file__).resolve().parents[4]
    # Load the standalone host module without importing the JAX package: this
    # executes its exact committed code, not a separate initialization formula.
    path=root/'src/gpuwrf/physics/thompson_hail/init.py'
    namespace={'__name__':'original_host_init_probe'}
    exec(compile(path.read_text(),str(path),'exec'),namespace)
    reference=json.loads((Path(__file__).parent/'original_coldinit.json').read_text())
    fields=reference['fields']
    start=time.monotonic()
    result=namespace['initial_profiles'](np.asarray(fields['HGT'],np.float32))
    wall=time.monotonic()-start
    rows={}
    for name,actual,expected in zip(('NWFA','NIFA','NWFA2D'),result,(fields['NWFA'],fields['NIFA'],reference['emission'])):
        expected=np.asarray(expected,np.float32)
        rows[name]=dict(byte_equal=bool(np.array_equal(actual,expected)),matching=int(np.count_nonzero(actual==expected)),total=int(actual.size),max_abs=float(np.max(np.abs(actual-expected))))
    # Timing only; geometric repetition does not supply extra fidelity cases.
    hgt=np.broadcast_to(np.asarray(fields['HGT'][0],np.float32),(42,42,40)).copy()
    start=time.monotonic();large=namespace['initial_profiles'](hgt);timing=time.monotonic()-start
    receipt=dict(scope='Exact production host init against independently compiled pristine whole-module coldinit',rows=rows,reference_wall_s=wall,timing_only=dict(shape=list(hgt.shape),wall_s=timing,output_bytes=sum(a.nbytes for a in large)),allocations='Two output REAL profiles; REAL height/offset/exponent temporaries at init only; no timestep launches or device transfers')
    (Path(__file__).parent/'coldinit_literal_byte.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    assert all(row['byte_equal'] for row in rows.values())


if __name__=='__main__': main()
