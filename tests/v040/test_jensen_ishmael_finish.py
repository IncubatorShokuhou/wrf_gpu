"""Pristine final vertical traversal, including held aggregate density."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
import pytest

DATA=Path(__file__).resolve().parents[2]/'proofs/s2_mpmore/jensen55'
spec=importlib.util.spec_from_file_location('finish_libm',DATA/'libm_numpy.py')
libm=importlib.util.module_from_spec(spec);spec.loader.exec_module(libm)


def load(mode,case,stage):
    folder=DATA/'column_stages';manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['E26_BYTE_columns']==34 and manifest['prior_stage_fields_BYTE']
    row=next(r for r in manifest['rows'] if (r['mode'],r['case'],r['stage'])==(mode,case,stage))
    p=folder/f'{mode}_{case}_{stage}.npz'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'] and row['nonfinite']==0
    with np.load(p) as data:return {k:data[k].copy() for k in data.files}


@pytest.mark.parametrize('mode',['fp32','fp64'])
@pytest.mark.parametrize('case',range(1,18))
def test_original_final_vertical_traversal_literal_BYTE(mode,case):
    from gpuwrf.physics.jensen_ishmael.finish import finish
    fields=load(mode,case,2);ref=load(mode,case,3)
    with np.load(DATA/'stages'/(mode+'.npz')) as data:tables={'gamma_tab':data['gamma_table'].copy()}
    with np.errstate(all='ignore'):got=finish(fields,tables,fields['held_rhobar'],xp=libm.backend)
    failures=[]
    for name,value in ref.items():
        # These four arrays are non-returned WRF scratch, assigned before use
        # on every active level. The production scan needs only RHOBAR carry.
        if name in ('held_ani','held_cni','held_rni','held_deltastr'):continue
        observed=np.asarray(got[name])
        assert np.isfinite(observed).all() and observed.dtype==value.dtype,(name,observed.dtype,value.dtype)
        if observed.tobytes()!=value.tobytes():
            bad=np.flatnonzero(observed!=value)
            failures.append((name,len(bad),bad[:8]))
    assert not failures,failures
