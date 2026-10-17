"""Actual pristine before/after adaptive fallout; literal REAL operation order."""
from pathlib import Path
import hashlib,json
import numpy as np
import pytest

DATA=Path(__file__).resolve().parents[2]/'proofs/s2_mpmore/jensen55/column_stages'


def load(mode,case,stage):
    manifest=json.loads((DATA/'manifest.json').read_text())
    row=next(r for r in manifest['rows'] if (r['mode'],r['case'],r['stage'])==(mode,case,stage))
    p=DATA/f'{mode}_{case}_{stage}.npz'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'] and row['nonfinite']==0
    with np.load(p) as data:return {k:data[k].copy() for k in data.files}


@pytest.mark.parametrize('mode',['fp32','fp64'])
@pytest.mark.parametrize('case',range(1,18))
def test_original_adaptive_fallout_literal_BYTE(mode,case):
    from gpuwrf.physics.jensen_ishmael.sedimentation import sedimentation
    fields=load(mode,case,1);ref=load(mode,case,2)
    got=sedimentation(fields,xp=np)
    failures=[]
    for name,value in ref.items():
        observed=np.asarray(got[name])
        assert np.isfinite(observed).all() and observed.dtype==value.dtype
        if observed.tobytes()!=value.tobytes():
            bad=np.flatnonzero(observed!=value)
            failures.append((name,len(bad),bad[:6]))
    assert not failures,failures
    assert int(got['sedimentation_steps'])>=1
