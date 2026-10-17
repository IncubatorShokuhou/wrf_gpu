"""Original actual aggregation boundaries, independent of port arithmetic."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'proofs/s2_mpmore/jensen55'
spec=importlib.util.spec_from_file_location('aggregation_libm',DATA/'libm_numpy.py')
libm=importlib.util.module_from_spec(spec);spec.loader.exec_module(libm)


@pytest.mark.parametrize('mode',['fp32','fp64'])
def test_actual_original_aggregation_literal_BYTE(mode,tmp_path):
    from gpuwrf.physics.jensen_ishmael.aggregation import aggregation
    m=json.loads((DATA/'aggregation/manifest.json').read_text());row=next(r for r in m['rows'] if r['mode']==mode)
    p=DATA/'aggregation'/(mode+'.npz');assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    assert row['E26_BYTE_columns']==17 and row['nonfinite']==0
    with np.load(p) as a:x=a['inputs'];ref=a['outputs']
    with np.load(DATA/'stages'/(mode+'.npz')) as a:tables=(a['coltab'],a['coltabn'])
    fields={k:x[:,i] for i,k in enumerate(m['input_names'])}
    backend=libm.aggregation_backend(tmp_path)
    with np.errstate(all='ignore'):got=aggregation(fields,*tables,xp=backend)
    failures=[]
    for name,value in got.items():
        expected=ref[:,m['output_names'].index(name)]
        assert np.isfinite(value).all() and value.dtype==expected.dtype
        if value.tobytes()!=expected.tobytes():
            bad=np.flatnonzero(value!=expected)
            failures.append((name,len(bad),bad[:6],value[bad[:6]],expected[bad[:6]]))
    assert not failures,failures
    assert np.count_nonzero(ref[:,m['output_names'].index('nagg3')])>0
