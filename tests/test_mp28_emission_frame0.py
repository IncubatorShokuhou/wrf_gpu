"""Original CPU-WRF c19/mp28 frame-zero field, including mountain threshold.

Frozen input PH/PHB and CPU output are independent of the port. Eager startup
allows four float32 ULPs; compiled startup uses the existing 32eps32 FMA bound.
Neither bound qualifies a GPU forecast or the separate 3-D aerosol profile.
"""
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import physics_couplers as pc
from test_v016_thompson_aero_threading import _tiny_state

ROOT = Path(__file__).resolve().parents[1]/'proofs/t_writer_mp28_emission'


def original():
    with np.load(ROOT/'original_frame0.npz') as a:
        # Sum in the loader's double precision, then the producer must round
        # that sum and divide by REAL4 g just as start_em does.
        ph = a['input_PH'].astype('f8')+a['input_PHB'].astype('f8')
        return ph, a['mp28_cpu_QNWFA2D'], a['mp28_cpu_QNIFA2D']


@pytest.mark.parametrize('compiled', (False, True))
def test_original_cpu_frame0_emission(compiled):
    ph, water, dust = original()
    def compute(ph):
        return pc._aerosol_surface_emission_columns(SimpleNamespace(ph=ph))
    actual, actual_dust = (jax.jit(compute) if compiled else compute)(jnp.asarray(ph))
    actual = np.asarray(actual)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    if compiled:
        np.testing.assert_allclose(actual, water, rtol=32*np.finfo('f4').eps, atol=0)
    else:
        np.testing.assert_array_max_ulp(actual, water, maxulp=4)
    np.testing.assert_array_equal(actual_dust, dust)


def test_coldstart_holds_original_emission_after_geometry_changes():
    ph, water, dust = original()
    ny,nx = water.shape
    state = _tiny_state(ny=ny,nx=nx)
    nz = state.qv.shape[0]
    upper = ph[1:]+np.arange(1,nz)[:,None,None]*300*9.81
    state = state.replace(ph_total=jnp.asarray(np.concatenate((ph,upper))))
    out = pc.thompson_aero_coldstart_init(state)
    np.testing.assert_array_max_ulp(np.asarray(out.nwfa2d), water, maxulp=4)
    np.testing.assert_array_equal(out.nifa2d, dust)
    again = pc.thompson_aero_coldstart_init(out.replace(ph_total=out.ph_total*1.1))
    np.testing.assert_array_equal(again.nwfa2d, out.nwfa2d)
    np.testing.assert_array_equal(again.nifa2d, dust)
