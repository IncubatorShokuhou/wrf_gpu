"""Registry-based Ni ownership; inactive shared leaves are not scalar sources."""
from pathlib import Path
import dataclasses
import hashlib
import importlib.util
import json
import re
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import operational_mode as op

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mp', op._SUPPORTED_SCHEME_VALUES['mp_physics'] if hasattr(op, '_SUPPORTED_SCHEME_VALUES') else
                         (0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,18,19,21,22,24,26,28,38,40,50,51,52,53,55,97))
def test_pbl_qni_flag_matches_original_registry(mp):
    source = (ROOT/'data/wrf_pristine/WRF/Registry/Registry.EM_COMMON').read_text()
    if mp in (17,18,19,21,22):
        # check_a_mundo.F:3397-3430 normalizes legacy NSSL options; the
        # nssl2mconc package is enabled only for nssl_2moment_on==1.
        active = mp in (17,18,22)
        assert re.search(r'^package\s+nssl2mconc\s+nssl_2moment_on==1\s+.*scalar:[^;\n]*\bqni\b', source, re.M)
    else:
        row = re.search(r'^package\s+\w+\s+mp_physics=='+str(mp)+r'\s+([^\n]*)',source,re.M)
        assert row is not None
        active = bool(re.search(r'scalar:[^;\n]*\bqni\b',row[1]))
    assert op._pbl_qni_active(mp) == active


def test_unowned_physics_source_still_refused():
    # No blanket skip in the consumer: even zero-valued unowned sources fail.
    with pytest.raises(NotImplementedError,match='non-advected.*Ni'):
        op._with_physics_sc(('qv',),(jnp.zeros((2,3,4)),),{'Ni':jnp.zeros((2,3,4))},())


@pytest.mark.parametrize('mp',(17,18,22))
def test_active_nssl_ni_source_not_dropped(mp):
    assert op._pbl_qni_active(mp)
    rate = jnp.arange(24,dtype=jnp.float32).reshape(2,3,4)
    got, = op._with_physics_sc(('Ni',),(jnp.zeros_like(rate),),{'Ni':rate},())
    np.testing.assert_array_equal(got,rate)


@pytest.mark.parametrize('bl',(5,9))
@pytest.mark.parametrize('mp',(1,3,4,5,6,7,8,13,14,15,16,18,19,21,22,24,26,97))
def test_real_pbl_dispatch_respects_registry_qni(mp,bl,monkeypatch):
    import jax
    from gpuwrf.contracts import state as sc
    from gpuwrf.coupling import scan_adapters
    from gpuwrf.runtime.operational_state import initial_operational_carry
    spec=importlib.util.spec_from_file_location('nsslni_dispatch',ROOT/'tests/test_rrtm_lw_operational_wiring.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    monkeypatch.setattr(sc,'_gpu_device',lambda:jax.devices('cpu')[0])
    monkeypatch.setenv('GPUWRF_PHYS_TEND_RK_WRF','1')
    grid=fixture._grid(nz=8,ny=3,nx=4)
    state=fixture._state(grid)
    nl=dataclasses.replace(fixture._namelist(grid),mp_physics=mp,bl_pbl_physics=bl,
                           cu_physics=0,run_physics=True,rad_rk_tendf=1)
    monkeypatch.setattr(op,'surface_adapter',lambda s,*a,**kw:s)
    zeros=jnp.zeros_like(state.qv)
    if bl==5:
        def mynn(s,*a,**kw):
            return SimpleNamespace(state=s.replace(Ni=s.Ni+17.),rthblten=zeros,rqvblten=zeros,
                rublten=zeros,rvblten=zeros,rqcblten=zeros,rqiblten=zeros)
        monkeypatch.setattr(op,'mynn_adapter_with_source_leaves',mynn)
    else:
        def cam(s,*a,**kw):
            return s,None,dict(rthblten=zeros,rqvblten=zeros,rublten=zeros,rvblten=zeros,
                              rqcblten=zeros,rqiblten=zeros,rqniblten=jnp.ones_like(zeros)*17.)
        monkeypatch.setattr(scan_adapters,'camuw_pbl_adapter',cam)
    forcing=op._physics_step_forcing(initial_operational_carry(state),nl,0.,run_radiation=False)
    active=mp in (8,18,22)
    assert ('Ni' in forcing.dry_tendencies.moist_tendf)==active
    np.testing.assert_array_equal(forcing.state.Ni,state.Ni)
    if active:
        assert np.any(np.asarray(forcing.dry_tendencies.moist_tendf['Ni'])!=0)


@pytest.mark.parametrize('variant',(19,21))
def test_two_real_swiss_post_rk_calls_preserve_pristine_inactive_ni(variant,monkeypatch):
    import jax
    from gpuwrf.contracts import state as sc
    proof=ROOT/'proofs/t_nsslni'
    manifest=json.loads((proof/'manifest.json').read_text())
    assert hashlib.sha256((proof/'swiss.npz').read_bytes()).hexdigest()==manifest['fixture_sha256']
    assert manifest['nonzero_cci']>0
    spec=importlib.util.spec_from_file_location('nsslni_state',ROOT/'tests/test_rrtm_lw_operational_wiring.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    monkeypatch.setattr(sc,'_gpu_device',lambda:jax.devices('cpu')[0])
    with np.load(proof/'swiss.npz') as data:
        ncol,nz=data['THM'].shape
        grid=fixture._grid(nz=nz,ny=1,nx=ncol)
        convert=lambda a:jnp.asarray(a.T[:,None,:],jnp.float32)
        state=fixture._state(grid).ensure_conditional_leaves(mp_physics=variant)
        updates={leaf:convert(data[name]) for leaf,name in
                 {'theta':'THM','p':'P','p_total':'P','ph':'PH','ph_total':'PH','qv':'QV','qc':'QC','qr':'QR',
                  'qi':'QI','qs':'QS','qg':'QH','Ni':'CCI'}.items()}
        updates['w']=jnp.concatenate((convert(data['W']),jnp.zeros((1,1,ncol),jnp.float32)))
        # The generic wiring fixture starts with wide Ni. Operational carries
        # are REAL4; do not let replace's preserve-current-dtype rule widen it.
        state=state.replace(_cast=False,**updates)
        nl=dataclasses.replace(fixture._namelist(grid),dt_s=18.,mp_physics=variant,run_physics=True)
        call=jax.jit(lambda s,step:op._apply_post_rk_microphysics(s,nl,step_index=step))
        before=np.asarray(state.Ni).copy()
        qv_before=np.asarray(state.qv).copy()
        for step in (1,2):
            state=call(state,jnp.int32(step))
            ni=np.asarray(state.Ni)
            ref=data[f'v{variant}'][step-1,:,manifest['fields'].index('CCI'),:].T[:,None,:]
            assert ni.dtype==ref.dtype==before.dtype==np.dtype('float32')
            assert np.isfinite(ni).all() and np.isfinite(ref).all()
            assert ni.tobytes()==ref.tobytes()==before.tobytes()
            for name in ('theta','qv','qc','qr','qi','qs','qg'):
                assert np.isfinite(np.asarray(getattr(state,name))).all(),(variant,step,name)
        assert np.any(np.asarray(state.qv)!=qv_before), 'actual microphysics must evolve'
