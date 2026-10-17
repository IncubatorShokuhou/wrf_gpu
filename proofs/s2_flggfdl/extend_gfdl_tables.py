"""Add native Ferrier/qualified Gaussian tables, preserving prior table bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main(asset,ferrier,support,receipt):
    prior=hashlib.sha256(asset.read_bytes()).hexdigest()
    with np.load(asset,allow_pickle=False) as f:
        tables={k:f[k].copy() for k in f.files}
    prior_arrays={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in tables.items()}
    with ferrier.open('rb') as f:
        coefficients=np.fromfile(f,np.float32,2)
        fpvs=np.fromfile(f,np.float32,7501)
        assert not f.read(1)
    with support.open('rb') as f:
        gaussian=np.fromfile(f,np.float32,311)
    assert coefficients.size==2 and fpvs.size==7501 and gaussian.size==311
    assert all(np.isfinite(x).all() for x in (coefficients,fpvs,gaussian))
    tables.update(ferrier_coefficients=coefficients,ferrier_tbpvs=fpvs,axsd=gaussian[1:])
    np.savez_compressed(asset,**tables)
    with np.load(asset,allow_pickle=False) as f:
        assert all(hashlib.sha256(f[k].tobytes()).hexdigest()==v for k,v in prior_arrays.items())
    receipt.write_text(json.dumps(dict(prior_asset_sha256=prior,
        prior_arrays_BYTE=True,prior_array_sha256=prior_arrays,
        asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        native_ferrier_sha256=hashlib.sha256(ferrier.read_bytes()).hexdigest(),
        native_support_sha256=hashlib.sha256(support.read_bytes()).hexdigest(),
        scope='original Ferrier GPVS; GAUSIN x=.01..3.10 exactly same init nodes, FG19 source-qualified'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('asset','ferrier','support','receipt'):p.add_argument(k,type=Path)
    a=p.parse_args();main(a.asset,a.ferrier,a.support,a.receipt)
