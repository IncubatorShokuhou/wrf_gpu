"""Actual original-WRF public two-call columns, including active mixed phases."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'proofs/s2_ntu'))
from literal_numpy import LiteralNumpy
from gpuwrf.physics.ntu.column import step, PROGNOSTICS, SURFACE


@pytest.mark.parametrize('kind',('r4','r8'))
def test_literal_original_two_step_columns(kind):
    folder=ROOT/'proofs/s2_ntu/fixtures'
    file=folder/f'columns_{kind}.npz'; meta=json.loads((folder/f'columns_{kind}.json').read_text())
    assert hashlib.sha256(file.read_bytes()).hexdigest()==meta['npz_sha256']
    data=np.load(file); xp=LiteralNumpy(); cases=(3,8,11)  # full12-case qualification receipts are retained separately
    fields={n:xp.asarray(np.stack([data[f'c{i:02}_s0_{n}'] for i in cases])) for n in ['th']+PROGNOSTICS}
    fields.update({n:xp.asarray(np.concatenate([data[f'c{i:02}_s0_{n}'] for i in cases])) for n in SURFACE})
    fields.update({n:xp.asarray(np.stack([data[f'c{i:02}_input_{n}'] for i in cases])) for n in ('p','pii','dz','w')})
    assert all(np.count_nonzero(fields[n]) for n in PROGNOSTICS)
    with np.errstate(all='ignore'):
        for stage in (1,2):
            outputs=step(fields,np.dtype(fields['th'].dtype).type(18),xp=xp)
            for n,actual in outputs.items():
                expected=(np.concatenate if n in SURFACE else np.stack)([data[f'c{i:02}_s{stage}_{n}'] for i in cases])
                assert np.isfinite(actual).all() and np.isfinite(expected).all()
                assert np.asarray(actual).tobytes()==expected.tobytes(),(kind,stage,n,np.max(np.abs(actual-expected)))
            fields.update(outputs)


def test_all_runtime_and_oracle_assets_are_tracked():
    import subprocess
    files=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    for kind in ('r4','r8'):
        assert f'src/gpuwrf/physics/ntu/data/aerosol_{kind}.npz' in files
        for family in ('columns','scalars','activation','initialization'):
            assert f'proofs/s2_ntu/fixtures/{family}_{kind}.npz' in files
