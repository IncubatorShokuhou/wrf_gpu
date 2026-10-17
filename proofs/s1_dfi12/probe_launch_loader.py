"""Actual native loader preserves DFL filtered W with zero surface W; deletion control.

CPU mechanical gate only, no artificial state is claimed as a scientific oracle.
Usage: python probe_launch_loader.py IMMUTABLE_SOURCE FILTERED_INPUT_DIR OUT_JSON
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile

source, filtered, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0,str(source/'src'))
sys.path.insert(0,str(source/'proofs/o1_nlbind'))
import jax
import numpy as np
import cpu_safe
cpu_safe.install()
from jax.experimental import pallas as pl
pc = pl.pallas_call
def interpreted(*a, **kw):
    kw['interpret']=True; kw.pop('compiler_params',None)
    return pc(*a,**kw)
pl.pallas_call=interpreted
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration import d02_replay as dr
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.io.netcdf_lock import Dataset
assert jax.devices()[0].platform=='cpu'
assert str(dr.__file__).startswith(str(source))
state_contract._gpu_device=lambda:jax.devices('cpu')[0]
with tempfile.TemporaryDirectory(dir=output.parent,prefix='loader-') as tmp:
    case=Path(tmp)
    for file in filtered.iterdir():
        if file.name!='wrfinput_d01': (case/file.name).symlink_to(file.resolve())
    shutil.copy2(filtered/'wrfinput_d01',case/'wrfinput_d01')
    with Dataset(case/'wrfinput_d01','r+') as ds:
        expected=np.asarray(ds['W'][0],np.float32)
        expected[0]=0.
        expected[1:-1]+=.125  # nonzero interior, with zero surface: old gate would reset it.
        ds['W'][0]=expected
    cfg=pipeline.NestedPipelineConfig(case,case/'out',case/'proof',hours=.85,max_dom=1)
    def load_w():
        _h,_b,_m,_start,_dt,carries=pipeline._load_domains(cfg,('d01',))
        return np.asarray(carries['d01'].state.w)
    actual=load_w()
    np.testing.assert_array_equal(actual,expected)
    assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
    original_is_file=Path.is_file
    def absent_marker(path):
        return False if path.name=='gpuwrf_dfi_applied.json' and path.parent==case else original_is_file(path)
    Path.is_file=absent_marker
    killed=False
    try:
        old=load_w()
        try:
            np.testing.assert_array_equal(old,expected)
        except AssertionError as exc:
            killed='Arrays are not equal' in str(exc)
            delta=float(np.abs(old-expected).max())
    finally:
        Path.is_file=original_is_file
    assert killed, 'removing the marker must fail the actual loader W-byte assertion'
    result=dict(backend='cpu',source=str(source),filtered_input=str(filtered),
                actual_native_loader_w_byte_equal=True,deletion_mutant_assertion_killed=killed,
                mutant_max_abs_w_delta=delta,shape=list(expected.shape))
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
