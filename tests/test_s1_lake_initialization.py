"""Compare the standalone lakeini component to original CPU-WRF columns."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
from jax import numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.lake_initialization import (
    LakeColumnState, LakeInitConfig, LakeInitInputs, LakeInitialization,
    empty_lake_columns, lake_initial_state,
)

FIXTURE = Path(__file__).resolve().parents[1] / "proofs/s1_lake/fixtures"
INIT = jax.jit(lake_initial_state, static_argnames=("config", "restart"))


def inputs(case):
    return LakeInitInputs(*(jnp.asarray(case[key], jnp.int32 if key in ("veg", "soil") else jnp.float32)
                            for key in ("veg", "soil", "ht", "snow", "skin", "mask", "xice", "xland", "depth")))


def config(case):
    return LakeInitConfig(lakeflag=case["maskflag"], lake_depth_flag=case["depthflag"],
                          use_lakedepth=case["use_depth"], lakedepth_default=case["default_depth"],
                          lake_min_elev=case["min_elev"])


def previous(inp):
    return LakeInitialization(empty_lake_columns(inp.tsk.shape, fill=-777.0), inp.ivgtyp,
                              inp.isltyp, inp.xland, inp.xice, inp.lakemask,
                              jnp.zeros_like(inp.tsk, dtype=bool), jnp.int32(1))


MANIFEST = json.loads((FIXTURE / "manifest.json").read_text())
CASES = {case["name"]: case for case in MANIFEST["cases"]}


@pytest.mark.parametrize("name", CASES)
def test_lakeini_pristine_all_fields(name):
    """No self-comparison: every expected value comes from pristine Fortran."""
    inp = inputs(CASES[name])
    actual = INIT(inp, config=config(CASES[name]), previous=previous(inp))
    with np.load(FIXTURE / "columns.npz", allow_pickle=False) as oracle:
        for field, value in actual.columns._asdict().items():
            expected = oracle[f"{name}/step0/{field}"]
            value = np.asarray(value).reshape(-1)
            assert value.dtype == np.float32
            assert np.isfinite(value).all() and np.isfinite(expected).all()
            np.testing.assert_allclose(value, expected, rtol=2e-7, atol=0.0,
                                       err_msg=f"pristine lakeini {name}/{field}")
        meta = np.array([actual.ivgtyp, actual.isltyp, actual.xland, actual.xice,
                         actual.lakemask, actual.lake_depth_flag, actual.lake], dtype=np.float32)
        np.testing.assert_array_equal(meta, oracle[f"{name}/step0/meta"])


def test_regimes_are_active_and_provenance_bound():
    assert len(CASES) == 55
    assert not MANIFEST["failures"]
    assert MANIFEST["source_sha256"] and MANIFEST["driver_sha256"]
    proof = Path(__file__).resolve().parents[1] / "proofs/s1_lake"
    for name, key in (("oracle_driver.F90", "driver_sha256"), ("wrf_stubs.F90", "stubs_sha256")):
        assert hashlib.sha256((proof / name).read_bytes()).hexdigest() == MANIFEST[key]
    assert hashlib.sha256((FIXTURE / "columns.npz").read_bytes()).hexdigest() == MANIFEST["files"][0]["sha256"]
    assert {case["census"]["initial_snl"] for case in CASES.values()} >= {0., -1., -2., -3., -4., -5.}
    assert CASES["freeze"]["census"]["ice_formation"]
    assert CASES["freeze"]["census"]["initial_max_ice"] == 0.0
    assert CASES["freeze"]["census"]["final_max_ice"] > 0.0
    assert CASES["snowfall"]["census"]["final_snow"] > 0.0
    assert {CASES[f"soil_{i:02d}"]["soil"] for i in range(1, 20)} == set(range(1, 20))


def test_restart_preserves_payload_exactly():
    inp = inputs(CASES["open_deep"])
    old = previous(inp)
    # Missing-depth configuration would fail cold init; WRF returns before
    # inspecting those flags on restart.
    assert lake_initial_state(inp, config=LakeInitConfig(lake_depth_flag=0),
                              restart=True, previous=old) is old
    with pytest.raises(ValueError, match="existing lake payload"):
        lake_initial_state(inp, restart=True)


def test_missing_requested_depth_refused():
    with pytest.raises(ValueError, match="lake-depth information"):
        lake_initial_state(inputs(CASES["open_deep"]), config=LakeInitConfig(lake_depth_flag=0))


def test_batched_columns_preserve_nonlake_seeds():
    lake = inputs(CASES["open_deep"])
    land = inputs(CASES["nonlake"])
    inp = LakeInitInputs(*(jnp.stack((a, b)) for a, b in zip(lake, land)))
    actual = INIT(inp, previous=previous(inp))
    assert actual.columns.t_lake3d.shape == (2, 10)
    assert actual.columns.t_soisno3d.shape == (2, 15)
    assert actual.columns.zi3d.shape == (2, 16)
    assert len(actual.columns) == len(LakeColumnState._fields) == 22
    np.testing.assert_array_equal(actual.columns.t_lake3d[1], np.full(10, -999.0, np.float32))
    np.testing.assert_array_equal(actual.columns.zi3d[1], np.full(16, -777.0, np.float32))
    assert float(actual.columns.savedtke12d[1]) == -777.0
    assert not bool(actual.lake[1])


def test_all_land_preserves_shared_depth_flag():
    inp = inputs(CASES["nonlake"])
    actual = INIT(inp, config=LakeInitConfig(use_lakedepth=0), previous=previous(inp))
    assert int(actual.lake_depth_flag) == 1
