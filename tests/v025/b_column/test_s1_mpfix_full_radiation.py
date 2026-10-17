"""Actual full pristine radiation_driver oracle on active real WN3 columns.

Use frozen RE01 bounds: flux 1 W/m2 +5%, heating 1e-4 K/s +5%.
The same held-radii vectors feed each new scheme; producer algorithms have
separate scheme-specific pristine gates. This isolates the caller handoff.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.kernels import rad_mcica

from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState, solve_rrtmg_lw_column
from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState, solve_rrtmg_sw_column

ROOT=Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def original_overlap(monkeypatch):
    monkeypatch.setattr(rad_mcica, "_MAXRAND", True)
    jax.clear_caches()
    yield
    jax.clear_caches()


def test_actual_full_driver_radii_and_fluxes():
    jax.config.update('jax_enable_x64',True)
    ref=json.loads((ROOT/'proofs/s1_mpfix/fixtures/full_radiation_driver.json').read_text())
    fx=np.load(ROOT/'tests/v025/fid_q2/fixtures/rrtmg_mp_re_0227_v1.npz')
    rows=[r for r in ref['rows'] if r['mp']==18 and r['use_mp_re']==1]
    idx=ref['input_columns']
    a=lambda n:jnp.asarray(fx[n][idx],jnp.float64)
    expected=np.array([r['scalars'] for r in rows])
    profiles=np.array([r['profiles'] for r in rows])
    common=dict(T=a('in_t'),p=a('in_p_hyd'),qv=a('in_qv'),qc=a('in_qc_rad'),qi=a('in_qi_rad'),
                qs=a('in_qs'),qg=a('in_qg'),cloud_fraction=jnp.asarray(profiles[:,0]),
                dz=a('in_dz8w')[:,:44],rho=a('in_rho'),pressure_interfaces=a('in_p_hyd_w'),
                temperature_interfaces=a('in_t8w'),re_cloud=a('col_re_cloud'),re_ice=a('col_re_ice'),
                re_snow=a('col_re_snow'),xland=a('in_xland'))
    lw=solve_rrtmg_lw_column(RRTMGLWColumnState(**common,surface_temperature=a('in_tsk'),
              surface_emissivity=a('in_emiss'),top_pressure_pa=float(fx['in_p_top'])))
    sw=solve_rrtmg_sw_column(RRTMGSWColumnState(**common,surface_albedo=a('in_albedo'),
              coszen=jnp.asarray(expected[:,5]),solar_source_scale=jnp.asarray(expected[:,4])/1368.22))
    for candidate,original in ((lw.surface_down,expected[:,0]),(lw.toa_up,expected[:,3]),
                               (sw.surface_down,expected[:,1]),(sw.toa_up,expected[:,7])):
        candidate=np.asarray(candidate)
        assert np.isfinite(candidate).all()
        assert np.all(np.abs(candidate-original)<=1.+.05*np.abs(original)),(candidate,original)
    for candidate,original in ((np.asarray(lw.heating_rate)/fx['in_pi'][idx],profiles[:,1]),
                               (np.asarray(sw.heating_rate)/fx['in_pi'][idx],profiles[:,2])):
        assert np.isfinite(candidate).all()
        assert np.all(np.abs(candidate-original)<=1e-4+.05*np.abs(original))
    # Each exact full-driver call for18/24/28 used flags derived from physics_init
    # and retained the supplied radius vectors. This is never a GPU-vs-GPU gate.
    for mp in (18,24,28):
        for i,operand in enumerate(idx,1):
            row=next(r for r in ref['rows'] if r['mp']==mp and r['column']==i and r['use_mp_re']==1)
            for radius,name in zip(row['profiles'][3:],('col_re_cloud','col_re_ice','col_re_snow')):
                np.testing.assert_array_equal(np.array(radius,np.float32),fx[name][operand])
            assert row['scalars']==rows[i-1]['scalars']
    assert np.count_nonzero(profiles[:,0])>10
    assert np.count_nonzero(fx['in_qi_rad'][idx])>0 and np.count_nonzero(fx['in_qs'][idx])>0
    # The original WRF disabled-radius arm must change an active physical result.
    disabled=np.array([r['scalars'] for r in ref['rows'] if r['mp']==18 and r['use_mp_re']==0])
    assert np.max(np.abs(disabled[:,:2]-expected[:,:2]))>1.
