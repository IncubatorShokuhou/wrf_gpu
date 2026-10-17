"""Independent pristine-WRF propagation envelope for one input binary32 ULP.

This only calls the unchanged Fortran oracle. No candidate or JAX is imported.
The final compiled candidate is admitted against the sealed envelope afterward.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def main(fixtures,build,target):
    data=json.loads(fixtures.read_text());manifest=json.loads((build/'manifest.json').read_text())
    width=manifest['width'];fields=manifest['fields'];lib=ctypes.CDLL(str(build/'glacier_zero.so'))
    lib.glacier_eval.argtypes=[np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')]*2+[np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')]
    integer=np.zeros(width,bool);inputs=np.zeros(width,bool)
    for f in fields:
        sl=slice(f['offset'],f['offset']+f['extent'])
        integer[sl]=f['integer'];inputs[sl]=f['intent']!='OUT'
    rng=np.random.default_rng(20261014)
    lows=[];highs=[]
    for fixture in data['fixtures']:
        x=np.asarray(fixture['input'],np.float32);options=np.asarray(fixture['options'],np.int32)
        ref=np.asarray(fixture['reference'],np.float32);lo=ref.copy();hi=ref.copy()
        mask=inputs & (~integer) & (x!=0.) & np.isfinite(x)
        low=np.where(mask,np.nextafter(x,np.float32(-np.inf)),x)
        high=np.where(mask,np.nextafter(x,np.float32(np.inf)),x)
        variants=[low,high]
        # Independent directions catch cancellation and branch sensitivity;
        # also vary each input alone to avoid correlated perturbation gaps.
        for pos in np.flatnonzero(mask):
            for direction in (-1,1):
                a=x.copy();a[pos]=low[pos] if direction<0 else high[pos];variants.append(a)
        for _ in range(128):
            choice=rng.integers(0,3,size=width)
            variants.append(np.where(choice==0,low,np.where(choice==1,high,x)))
        for variant in variants:
            out=np.zeros(width,np.float32);lib.glacier_eval(np.ascontiguousarray(variant),out,options)
            if not np.isfinite(out).all():raise AssertionError('nonfinite independent WRF ULP ensemble')
            lo=np.minimum(lo,out);hi=np.maximum(hi,out)
        lows.append(lo);highs.append(hi)
    np.savez_compressed(target,lower=np.asarray(lows),upper=np.asarray(highs))
    receipt=dict(reference='unchanged pristine WRF; no candidate imported',source_sha256=manifest['source_sha256'],
        fixture_sha256=hashlib.sha256(fixtures.read_bytes()).hexdigest(),cases=len(lows),
        sample_rule='all inputs individually +/-1 ULP, all-low/all-high, 128 seeded independent mixed directions; nonzero REAL IN/INOUT only',
        seed=20261014,gate='compiled output within pristine min/max envelope expanded by 1 output binary32 ULP for libm rounding; reject nonfinite',
        envelope_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    target.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
