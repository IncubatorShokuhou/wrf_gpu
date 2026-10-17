import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.surface_layer import _surface_layer_impl
from gpuwrf.physics.sfclay_revised_mm5 import SfclayRevisedMM5ColumnState,sfclay_revised_mm5_run
from gpuwrf.physics.sfclay_old_mm5 import sfclay_old_mm5_columns

DATA=Path(__file__).resolve().parents[1]/'proofs/s2_oml/fixtures/flux.json'
OUTPUTS=('ust','mol','rmol','zol','regime','psim','psih','br','flhc','flqc','hfx','qfx','lh','qsfc','qgh',
         'chs','chs2','cqs2','ch','wspd','gz1oz0','u10','v10','th2','t2','q2','cpm','wstar','qstar','znt','ustm')
INPUTS=('u','v','temperature','qv','pressure','dz','rho','u2','v2','dz2','mavail','pblh','xland','tsk','psfc',
        'qcg','snowh','znt','ust','mol','qsfc','hfx','qfx')


@pytest.mark.parametrize('opt',[1,2])
@pytest.mark.parametrize('first',[False,True])
@pytest.mark.parametrize('sms',[False,True])
def test_mynn_pristine_hurricane(opt,first,sms):
    fixture=json.loads((DATA.with_name('cold_flux.json') if first else DATA).read_text())[f'mynn_{opt}']
    inputs=np.asarray(fixture['inputs'],np.float32)
    inp=dict(zip(INPUTS,inputs.T,strict=True))
    surface=('rho','mavail','pblh','xland','psfc','mol','qsfc','hfx','qfx')
    view=SimpleNamespace(**{k:jnp.asarray(inp[k])[:,None] for k in surface})
    for dst,src in (('t_skin','tsk'),('roughness_m','znt'),('ustar','ust')):
        setattr(view,dst,jnp.asarray(inp[src])[:,None])
    for dst,src in (('u','u'),('v','v'),('theta','temperature'),('t_air','temperature'),('qv','qv'),('p','pressure'),('dz','dz')):
        setattr(view,dst,jnp.asarray(inp[src])[:,None,None])
    view.lakemask=jnp.zeros((len(inputs),1),jnp.float32)
    view.dx_m=9000.
    if sms:view.sms_ustm=jnp.full((len(inputs),1),fixture['ustm_in'],jnp.float32)
    diag,ustm=_surface_layer_impl(view,first,jnp.float32,isftcflx=opt,ustm=jnp.full((len(inputs),1),0. if sms else fixture['ustm_in'],jnp.float32),_return_hurricane=True)
    expected=dict(zip(OUTPUTS,np.asarray(fixture['output'],np.float32).T,strict=True))
    bound=dict(zip(OUTPUTS,np.asarray(fixture['input_ulp_bound'],np.float32).T,strict=True))
    fma=fixture['fma_output'] if first else json.loads(DATA.with_name('flux_fma.json').read_text())[f'mynn_{opt}']['output']
    spread=dict(zip(OUTPUTS,np.abs(np.asarray(fma,np.float32)-np.asarray(fixture['output'],np.float32)).T,strict=True))
    actual={k:np.asarray(getattr(diag,k)).ravel() for k in ('hfx','lh','t2','q2','u10','v10','qsfc','mol','znt')}
    actual.update(ust=np.asarray(diag.fluxes.ustar).ravel(),ustm=np.asarray(ustm).ravel())
    for name,value in actual.items():
        assert np.isfinite(value).all(),name
        assert np.all(np.abs(value-expected[name])<=bound[name]+spread[name]),(name,value,expected[name],bound[name]+spread[name])


@pytest.mark.parametrize('scheme',['revised','old'])
@pytest.mark.parametrize('opt',[1,2])
def test_mm5_pristine_hurricane(scheme,opt):
    d=json.loads(DATA.read_text())[f'{scheme}_{opt}']
    fma=json.loads(DATA.with_name('flux_fma.json').read_text())[f'{scheme}_{opt}']
    a=lambda k:jnp.asarray(d[k],jnp.float32)
    core=[a(k) for k in ('U','V','T','QV','P','DZ')]
    if scheme=='revised':
        state=SfclayRevisedMM5ColumnState(*core,*(a(k) for k in ('PSFC','TSK','XLAND','LAKEMASK','MAVAIL','ZNT_IN','UST_IN',
            'MOL_IN','HFX_IN','QFX_IN','QSFC_IN','PBLH','DX','WATER_DEPTH')))
        actual=sfclay_revised_mm5_run(state,isftcflx=opt)._asdict()
    else:
        actual=sfclay_old_mm5_columns(*core,*(a(k) for k in ('PSFC','TSK','XLAND','ZNT_IN','UST_IN','MOL_IN','QSFC_IN',
            'PBLH','MAVAIL','LAKEMASK','DX','HFX_IN','QFX_IN')),isftcflx=opt)
    for key in ('hfx','qfx','lh','ust','znt','mol','u10','v10','t2','q2','chs2','cqs2'):
        assert np.isfinite(actual[key]).all(),key
        oracle='TSTAR' if key=='mol' and scheme=='revised' else key.upper()
        # ES23.15 decimal text is round-trippable to REAL4; retaining a Python
        # DOUBLE decimal residue invents nonzero errors against a zero bound.
        reference=np.asarray(d[oracle],np.float32)
        bound=np.asarray(d['input_ulp_bound'][oracle],np.float32)+np.abs(np.asarray(fma[oracle],np.float32)-reference)
        assert np.all(np.abs(np.asarray(actual[key])-reference)<=bound),(scheme,key,actual[key],reference,bound)
