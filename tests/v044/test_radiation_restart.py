"""Public restart formats preserve every diagnostic byte and refuse old schemas."""
from dataclasses import replace
from pathlib import Path
import pickle
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from test_restart_schema_held_physics import case, TIMES
from test_online_diagnostics_restart_store import identity, rewrite
from gpuwrf.coupling.physics_couplers import RRTMGRadiationDiagnostics
from gpuwrf.diagnostics.radiation import RadiationHistory
from gpuwrf.io import restart as rp, wrfrst_netcdf as nc
from gpuwrf.runtime import checkpoint as cp
from gpuwrf.runtime.restart_store import RestartStore


@pytest.fixture
def populated(case):
    grid, nl, carry = case
    zero = jnp.zeros(carry.state.t_skin.shape, jnp.float64)
    diag = RRTMGRadiationDiagnostics(*(zero for _ in range(23)))
    values = RadiationHistory(*(jnp.full(carry.state.theta.shape, (i+1)*1.e-5, jnp.float32) for i in range(8)))
    return grid, replace(nl, rad_diag=1), carry.replace(radiation_diagnostics=diag._replace(
        cloud_fraction=jnp.zeros_like(carry.state.theta), rad_diag=values))


@pytest.mark.parametrize("fmt", ["pickle", "netcdf", "checkpoint", "native"])
def test_public_rad_diag_roundtrip_and_schema_refusal(populated, tmp_path, fmt):
    grid, nl, carry = populated
    path = tmp_path / fmt
    if fmt == "pickle":
        rp.write_restart(carry, nl, grid, 2, path)
        read = lambda: rp.read_restart(path, namelist=nl)[0]
        actual = read()
        payload = pickle.loads(path.read_bytes())
        payload["carry"].pop("rad_diag_schema_version")
        path.write_bytes(pickle.dumps(payload))
    elif fmt == "netcdf":
        nc.write_wrfrst_carry(carry, grid, nl, path, **TIMES)
        read = lambda: nc.read_wrfrst_carry(path, namelist=nl)[0]
        actual = read()
        with Dataset(path, "a") as ds:
            ds.delncattr("GPUWRF_RAD_DIAG_SCHEMA_VERSION")
    elif fmt == "checkpoint":
        cp.write_checkpoint(carry.state, nl, grid, 2, path, runtime_state=carry)
        read = lambda: cp.read_checkpoint_with_runtime_state(path, namelist=nl)[4]
        actual = read()
        payload = pickle.loads(path.read_bytes())
        payload.pop("rad_diag_schema_version")
        path.write_bytes(pickle.dumps(payload))
    else:
        config = dict(identity(nl), rad_diag_config={"d01": 1})
        store = RestartStore(path, 10**8, 2, 0)
        generation, _ = store.save({"d01": carry}, {"d01": 2}, {"d01": nl.dt_s}, config)
        read = lambda: store.read(generation)[0]["carries"]["d01"]
        actual = read()
        rewrite(generation, lambda payload, manifest: payload.pop("rad_diag_schema_version"))
    for expected, got in zip(carry.radiation_diagnostics.rad_diag, actual.radiation_diagnostics.rad_diag):
        expected, got = np.asarray(expected), np.asarray(got)
        assert (expected.dtype, expected.shape, expected.tobytes()) == (got.dtype, got.shape, got.tobytes())
    with pytest.raises(ValueError, match="rad_diag.*E78"):
        read()


def test_active_native_rad_diag_requires_selection_binding(populated, tmp_path):
    _grid,nl,carry=populated
    store=RestartStore(tmp_path/'missing_selection',10**8,2,0)
    with pytest.raises(ValueError,match='rad_diag configuration is missing.*E78'):
        store.save({'d01':carry},{'d01':2},{'d01':nl.dt_s},identity(nl))
