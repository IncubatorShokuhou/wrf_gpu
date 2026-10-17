"""Pristine WRF 4ice source parity, active hail, continuation and native path."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import pytest
import jax
from gpuwrf.physics.goddard4ice.column import column
from gpuwrf.physics.goddard4ice.native import native_column

HERE=Path(__file__).parent.parent
ROOT=HERE.parents[2]
INPUT=('TH_IN','QV_IN','QC_IN','QR_IN','QI_IN','QS_IN','QG_IN','QH_IN','DEN','PII','P','DELZ','Z','W')
CORE=('theta','qv','qc','qr','qi','qs','qg','qh')
RADII=tuple('re_'+name+'_gsfc' for name in ('cloud','rain','ice','snow','graupel','hail'))
AC=tuple('acphys'+name for name in 'cedsmf')
NAMES=(*CORE,*RADII,*AC,*(('phys'+name) for name in 'cedsmf'),
       'preci3d','precs3d','precg3d','prech3d','precr3d','rainnc','snownc','graupelnc','hailnc','sr')


def load():
 path=HERE/'goddard4ice_columns.npz';meta=json.loads(path.with_suffix('.json').read_text())
 assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
 return meta,np.load(path,allow_pickle=False)


def literal():
 path=ROOT/'proofs/v040/mpmisc/goddard4ice/probe.py'
 spec=importlib.util.spec_from_file_location('goddard4ice_literal',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module.literal()


def run(group,case,fn,native=False):
 meta,data=load();cfg=meta['groups'][group]['caller'];prefix=f'{group}_{case}__'
 args=[data[prefix+key] for key in INPUT]
 diagnostics=None
 for step in range(1,cfg['steps']+1):
  if native:
   result=fn(*(a[:,None,None] for a in args),cfg['dt'],cfg['dx'],cfg['land'],0.,step,
             None if diagnostics is None else {k:v[:,None,None] for k,v in diagnostics.items()},interpret=True,full=True)
   result={k:np.asarray(v)[:,0,0] if np.asarray(v).ndim==3 else np.asarray(v)[0,0] for k,v in result.items()}
  else:
   # Only inactive vectorized NumPy branches can produce these diagnostics.
   # Final original-WRF fields remain explicitly finite-checked below.
   with np.errstate(all='ignore'):
    result=fn(*args,cfg['dt'],cfg['dx'],cfg['land'],0.,step,diagnostics)
  args[:8]=[np.asarray(result[k]) for k in CORE]
  diagnostics={k:result[k] for k in AC}
 return result,{key:data[prefix+key] for key in NAMES},{key:data[prefix+key+'_bound'] for key in NAMES}


@pytest.mark.parametrize('group',['base','water18','land54'])
def test_literal_all_original_wrf_columns_byte(group):
 fn=literal()
 for case in range(1,7):
  actual,expected,_=run(group,case,fn)
  for name in NAMES:
   value=np.asarray(actual[name]);ref=expected[name]
   assert value.dtype==np.float32 and np.isfinite(value).all(),(group,case,name)
   assert value.shape==ref.shape and value.tobytes()==ref.tobytes(),(group,case,name)


def test_active_graupel_hail_independent_categories():
 meta,data=load();group=meta['groups']['base']['cases']['4']
 assert group['species_nonzero']['QG_IN']>0 and group['species_nonzero']['QH_IN']>0
 assert group['hail_surface_mm']>0
 actual,expected,_=run('base',4,literal())
 assert not np.array_equal(actual['qg'],actual['qh'])
 assert np.count_nonzero(expected['qh'])>0


def test_genuine_hail_sedimentation_mutant_is_rejected():
 fn=literal()
 fn.__globals__['C']=dict(fn.__globals__['C'],vhc=np.float32(0))
 actual,expected,_=run('base',4,fn)
 assert np.isfinite(actual['qh']).all()
 with pytest.raises(AssertionError):
  assert np.asarray(actual['qh']).tobytes()==expected['qh'].tobytes()


def test_native_complete_column_active_hail():
 actual,expected,bounds=run('base',4,native_column,native=True)
 for name in NAMES:
  got=np.asarray(actual[name]);ref=expected[name]
  assert got.shape==ref.shape and got.dtype==np.float32 and np.isfinite(got).all()
  # Independent input-ULP and original compiler-contraction envelope. Tiny
  # disclosed excess is governed separately by the measured regression receipt.
  scale=np.maximum(np.max(np.abs(ref)),1. if name=='sr' else 1e-12)
  assert np.all(np.abs(got-ref)<=bounds[name]+128*np.finfo(np.float32).eps*(scale+np.abs(ref))),name


@pytest.mark.parametrize('group',['base','water18','land54'])
def test_default_compiled_all_original_wrf_columns(group):
 fn=jax.jit(column)
 receipt=[]
 for case in range(1,7):
  actual,expected,bounds=run(group,case,fn)
  for name in NAMES:
   got=np.asarray(actual[name]);ref=expected[name]
   assert got.shape==ref.shape and got.dtype==np.float32 and np.isfinite(got).all()
   absolute=np.abs(got.astype(float)-ref.astype(float))
   excess=np.maximum(0.,absolute-bounds[name])
   if np.any(excess):receipt.append({'case':case,'field':name,'cells':int(np.count_nonzero(excess)),'maxabs':float(np.max(absolute)),'maxexcess':float(np.max(excess))})
   # Independently staged bound plus a fixed REAL32 regression budget.
   # This extra budget is not reported as strict original-FMA envelope parity.
   scale=np.maximum(np.max(np.abs(ref)),1. if name=='sr' else 1e-12)
   assert np.all(np.abs(got-ref)<=bounds[name]+128*np.finfo(np.float32).eps*(scale+np.abs(ref))),(group,case,name)

 if os.environ.get('GPUWRF_GODDARD_RECEIPT_DIR'):
  (Path(os.environ['GPUWRF_GODDARD_RECEIPT_DIR'])/(group+'_compiled_excess.json')).write_text(json.dumps(receipt,indent=2))
