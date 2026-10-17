"""Canopy geometry choices use pristine TWOSTREAM and preserve optional history."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]


def matrix(opt_rad,engine=jnp):
    from gpuwrf.physics.noahmp.radiation_options import canopy_gaps
    fx=json.loads((ROOT/"proofs/u2_land/gap_oracle.json").read_text())
    cases=[c for c in fx["cases"] if c["opt_rad"]==opt_rad]
    values=engine.asarray([[c[k] for k in ("cosz","vai","fveg","rc","hvt","hvb")] for c in cases],engine.float32)
    def compute(values):return engine.stack(canopy_gaps(*values.T,opt_rad=opt_rad),axis=-1)
    return np.asarray(jax.jit(compute)(values) if engine is jnp else compute(values)),cases,fx


@pytest.mark.parametrize("opt_rad",[1,2,3])
def test_original_gap_geometry_runtime_inputs(opt_rad):
    actual,cases,fx=matrix(opt_rad)
    assert actual.dtype==np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual,[c["expected"] for c in cases],**fx["nominal_tolerance"])


def test_gap_history_is_only_added_when_active():
    from types import SimpleNamespace
    from gpuwrf.runtime.history_accumulators import seed_history,as_packed,LAND_FLUX_FIELDS
    state=SimpleNamespace(t_skin=jnp.zeros((2,3)))
    old,_=seed_history(state);new,_=seed_history(state,canopy_gaps=True)
    assert tuple(old)==LAND_FLUX_FIELDS
    assert tuple(new)==(*LAND_FLUX_FIELDS,"BGAP","WGAP")
    assert tuple(as_packed(new,LAND_FLUX_FIELDS))==tuple(new)


def test_canopy_radiation_options_admission():
    from gpuwrf.io.noahmp_options import selected_options,option_refusals
    for code in (1,2):
        assert selected_options({"noah_mp":{"opt_rad":code}}).opt_rad==code
        assert not option_refusals({"noah_mp":{"opt_rad":code}})
