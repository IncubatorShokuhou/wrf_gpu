"""Fail-closed restart wiring of the held cu 5/93 WRF RTHFTEN/RQVFTEN pair (E78).

Every checkpoint format must keep the pair byte-exact, refuse an active cu_physics
file without it (older files / dropped group) and refuse the group under an inactive
selector, at write time and again at read time against the saved and the requested
selection.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import sys

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v022_camuw_pbl import _grid, _namelist, _state  # noqa: E402

from gpuwrf.coupling import cu_forcing as cu_forcing_module  # noqa: E402
from gpuwrf.coupling.cu_forcing import CU_FORCING_SCHEMES, cu_forcing_required  # noqa: E402
from gpuwrf.io.cu_forcing_schema import SCHEMA_VERSION, validate  # noqa: E402

FORMATS = ("restart", "wrfrst", "checkpoint", "store")
MISSING = "RTHFTEN/RQVFTEN restart carry missing"
INACTIVE = "inactive cumulus selector"


def _pair(shape, dtype=np.float32):
    # Distinct signed nonzero values expose silent resets, swaps or narrowing.
    n = int(np.prod(shape))
    rth = (np.arange(n).reshape(shape) * 1.25e-5 - 3.0e-4).astype(dtype)
    rqv = np.linspace(-2.0e-7, 3.0e-7, n).reshape(shape).astype(dtype)
    return jnp.asarray(rth), jnp.asarray(rqv)


def _case(scheme=5, dtype=np.float32):
    from gpuwrf.runtime.operational_state import initial_operational_carry
    grid = _grid()
    state = _state(grid)
    nml = _namelist(grid, cu_physics=scheme)
    carry = initial_operational_carry(state)
    if cu_forcing_required(scheme):
        carry = carry.replace(cu_forcing=_pair(state.theta.shape, dtype))
    return grid, state, nml, carry


def _exact(expected, actual):
    assert isinstance(actual, tuple) and len(actual) == 2
    for x, y in zip(expected, actual, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        assert x.dtype == y.dtype and x.shape == y.shape
        assert x.tobytes() == y.tobytes()


def _identity(scheme, shape):
    return {"sas_config": {"d01": {"cu_physics": int(scheme), "osas_domain_id": 1,
                                   "domain_shape": [int(n) for n in shape]}}}


def _write(fmt, tmp_path, grid, state, nml, carry, *, identity=None):
    if fmt == "restart":
        from gpuwrf.io.restart import write_restart
        path = tmp_path / "carry.pkl"
        write_restart(carry, nml, grid, 7, path)
        return path
    if fmt == "wrfrst":
        from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry
        path = tmp_path / "wrfrst.nc"
        write_wrfrst_carry(carry, grid, nml, path, valid_time="2023-01-15_13:00:00",
                           run_start="2023-01-15_12:00:00", step_index=7)
        return path
    if fmt == "checkpoint":
        from gpuwrf.runtime.checkpoint import write_checkpoint
        path = tmp_path / "single.checkpoint"
        write_checkpoint(state, nml, grid, 7, path, runtime_state=carry)
        return path
    from gpuwrf.runtime.restart_store import RestartStore
    store = RestartStore(tmp_path / "store", 10**8, 2, 0)
    if identity is None:
        identity = _identity(nml.cu_physics, state.theta.shape)
    generation, _ = store.save({"d01": carry}, {"d01": 7}, {"d01": 10.0}, identity)
    return generation


def _read(fmt, path, namelist=None):
    if fmt == "restart":
        from gpuwrf.io.restart import read_restart
        return read_restart(path, namelist=namelist)[0].cu_forcing
    if fmt == "wrfrst":
        from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry
        return read_wrfrst_carry(path, namelist=namelist)[0].cu_forcing
    if fmt == "checkpoint":
        from gpuwrf.runtime.checkpoint import read_checkpoint_with_runtime_state
        return read_checkpoint_with_runtime_state(path, namelist=namelist)[4].cu_forcing
    from gpuwrf.runtime.restart_store import RestartStore
    snapshot, _ = RestartStore(path.parent, 10**8, 2, 0).read(path)
    return snapshot["carries"]["d01"].cu_forcing


def _rewrite_pickle(path, mutate):
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    mutate(payload)
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _rewrite_store(generation, mutate):
    """Re-seal a mutated generation so only the schema admission can refuse it."""
    _rewrite_pickle(generation / "snapshot.pkl", mutate)
    data = (generation / "snapshot.pkl").read_bytes()
    manifest = json.loads((generation / "manifest.json").read_text())
    manifest.update(payload_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    (generation / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


# --------------------------------------------------------------------------- validator


@pytest.mark.parametrize("scheme", CU_FORCING_SCHEMES)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("container", (tuple, list))
def test_validator_accepts_active_pair_and_inactive_absence(scheme, dtype, container):
    shape = (4, 3, 2)
    pair = container(_pair(shape, dtype))
    for namelist in ({"cu_physics": scheme}, {"physics": {"cu_physics": scheme}},
                     replace(_namelist(_grid()), cu_physics=scheme)):
        validate(pair, namelist, shape, SCHEMA_VERSION)
    for namelist in (None, {}, {"cu_physics": 0}, {"cu_physics": 1}):
        validate(None, namelist, shape)


@pytest.mark.parametrize("bad,match", [
    ("missing", MISSING),
    ("inactive_version", "inactive cumulus forcing restart has schema metadata"),
    ("inactive_present", INACTIVE),
    ("version_none", "schema version mismatch"),
    ("version_other", "schema version mismatch"),
    ("single", "must be the \\(RTHFTEN, RQVFTEN\\) pair"),
    ("triple", "must be the \\(RTHFTEN, RQVFTEN\\) pair"),
    ("mapping", "must be the \\(RTHFTEN, RQVFTEN\\) pair"),
    ("rthften_shape", "RTHFTEN shape/dtype mismatch"),
    ("rqvften_shape", "RQVFTEN shape/dtype mismatch"),
    ("rthften_dtype", "RTHFTEN shape/dtype mismatch"),
    ("rqvften_integer", "RQVFTEN shape/dtype mismatch"),
    ("rthften_nan", "RTHFTEN contains nonfinite"),
    ("rqvften_inf", "RQVFTEN contains nonfinite"),
])
def test_validator_refuses_every_malformed_branch(bad, match):
    shape = (4, 3, 2)
    rth, rqv = (np.asarray(a) for a in _pair(shape))
    value, namelist, version = (rth, rqv), {"cu_physics": 5}, SCHEMA_VERSION
    if bad == "missing":
        value, version = None, None
    elif bad == "inactive_version":
        value, namelist = None, {"cu_physics": 0}
    elif bad == "inactive_present":
        namelist = {"cu_physics": 0}
    elif bad == "version_none":
        version = None
    elif bad == "version_other":
        version = SCHEMA_VERSION + 1
    elif bad == "single":
        value = (rth,)
    elif bad == "triple":
        value = (rth, rqv, rqv)
    elif bad == "mapping":
        value = {"RTHFTEN": rth, "RQVFTEN": rqv}
    elif bad == "rthften_shape":
        value = (rth[:, :, :-1], rqv)
    elif bad == "rqvften_shape":
        value = (rth, rqv[:-1])
    elif bad == "rthften_dtype":
        value = (rth.astype(np.float16), rqv)
    elif bad == "rqvften_integer":
        value = (rth, rqv.astype(np.int32))
    elif bad == "rthften_nan":
        rth = rth.copy(); rth[1, 2, 1] = np.nan; value = (rth, rqv)
    elif bad == "rqvften_inf":
        rqv = rqv.copy(); rqv[0, 0, 0] = np.inf; value = (rth, rqv)
    with pytest.raises(ValueError, match=match):
        validate(value, namelist, shape, version)


def test_validator_follows_cu_forcing_required_when_the_scheme_set_grows(monkeypatch):
    shape = (4, 3, 2)
    extra = next(c for c in range(1, 1000) if not cu_forcing_required(c))
    with pytest.raises(ValueError, match=INACTIVE):
        validate(_pair(shape), {"cu_physics": extra}, shape, SCHEMA_VERSION)
    monkeypatch.setattr(cu_forcing_module, "CU_FORCING_SCHEMES", cu_forcing_module.CU_FORCING_SCHEMES + (extra,))
    validate(_pair(shape), {"cu_physics": extra}, shape, SCHEMA_VERSION)
    with pytest.raises(ValueError, match=MISSING):
        validate(None, {"cu_physics": extra}, shape)


# --------------------------------------------------------------------------- round trips


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("scheme", CU_FORCING_SCHEMES)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_every_format_roundtrips_the_pair_byte_exact(tmp_path, fmt, scheme, dtype):
    grid, state, nml, carry = _case(scheme, dtype)
    path = _write(fmt, tmp_path, grid, state, nml, carry)
    for requested in ((None,) if fmt == "store" else (None, nml)):
        _exact(carry.cu_forcing, _read(fmt, path, requested))


@pytest.mark.parametrize("fmt", FORMATS)
def test_inactive_scheme_roundtrips_without_the_group(tmp_path, fmt):
    grid, state, nml, carry = _case(0)
    assert carry.cu_forcing is None
    path = _write(fmt, tmp_path, grid, state, nml, carry)
    assert _read(fmt, path) is None


# --------------------------------------------------------------------------- E78 write side


@pytest.mark.parametrize("fmt", FORMATS)
def test_active_scheme_without_the_group_is_refused_at_write(tmp_path, fmt):
    grid, state, nml, carry = _case(5)
    with pytest.raises(ValueError, match=MISSING):
        _write(fmt, tmp_path, grid, state, nml, carry.replace(cu_forcing=None))
    assert not list(tmp_path.rglob("generation-*"))


@pytest.mark.parametrize("fmt", FORMATS)
def test_inactive_scheme_with_the_group_is_refused_at_write(tmp_path, fmt):
    grid, state, nml, carry = _case(5)
    with pytest.raises(ValueError, match=INACTIVE):
        _write(fmt, tmp_path, grid, state, replace(nml, cu_physics=0), carry)
    assert not list(tmp_path.rglob("generation-*"))


def test_unbound_legacy_store_refuses_a_populated_group(tmp_path):
    grid, state, nml, carry = _case(5)
    with pytest.raises(ValueError, match=INACTIVE):
        _write("store", tmp_path, grid, state, nml, carry, identity={})


def test_state_only_wrfrst_writer_refuses_an_active_scheme(tmp_path):
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_state
    grid, state, nml, _ = _case(5)
    with pytest.raises(ValueError, match=MISSING):
        write_wrfrst_state(state, grid, nml, tmp_path / "state.nc", valid_time="2023-01-15_13:00:00",
                           run_start="2023-01-15_12:00:00", step_index=7)


# --------------------------------------------------------------------------- E78 read side


@pytest.mark.parametrize("fmt", ("restart", "wrfrst", "checkpoint"))
def test_requested_inactive_scheme_refuses_a_saved_group(tmp_path, fmt):
    grid, state, nml, carry = _case(5)
    path = _write(fmt, tmp_path, grid, state, nml, carry)
    with pytest.raises(ValueError, match=INACTIVE):
        _read(fmt, path, replace(nml, cu_physics=0))


@pytest.mark.parametrize("fmt", ("restart", "wrfrst", "checkpoint"))
def test_requested_active_scheme_refuses_a_file_without_the_group(tmp_path, fmt):
    grid, state, nml, carry = _case(0)
    path = _write(fmt, tmp_path, grid, state, nml, carry)
    assert _read(fmt, path) is None
    with pytest.raises(ValueError, match=MISSING):
        _read(fmt, path, replace(nml, cu_physics=5))


def _drop_restart_group(payload):
    # Exactly what a file written before the group existed holds.
    payload["carry"]["carry_field_order"].remove("cu_forcing")
    del payload["carry"]["extra_fields"]["cu_forcing"]
    del payload["carry"]["cu_forcing_schema_version"]


def _drop_wrfrst_group(path):
    from netCDF4 import Dataset
    with Dataset(path, "r+") as dataset:
        for attribute in ("GPUWRF_OPTIONAL_CARRY_FIELD_ORDER", "GPUWRF_OPTIONAL_CARRY_KIND"):
            manifest = json.loads(str(dataset.getncattr(attribute)))
            del manifest["cu_forcing"]
            dataset.setncattr(attribute, json.dumps(manifest))
        for attribute in ("GPUWRF_CU_FORCING_CONFIG", "GPUWRF_CU_FORCING_SCHEMA_VERSION"):
            if attribute in dataset.ncattrs():
                dataset.delncattr(attribute)


@pytest.mark.parametrize("fmt", ("restart", "wrfrst"))
def test_older_file_without_the_group_resumes_only_when_inactive(tmp_path, fmt):
    for scheme in (0, 5):
        grid, state, nml, carry = _case(scheme)
        path = _write(fmt, tmp_path / str(scheme), grid, state, nml, carry)
        if fmt == "restart":
            _rewrite_pickle(path, _drop_restart_group)
        else:
            _drop_wrfrst_group(path)
        if scheme == 0:
            assert _read(fmt, path) is None and _read(fmt, path, nml) is None
            continue
        for requested in (None, nml):
            with pytest.raises(ValueError, match=MISSING):
                _read(fmt, path, requested)


def test_checkpoint_with_dropped_group_is_refused(tmp_path):
    grid, state, nml, carry = _case(5)
    path = _write("checkpoint", tmp_path, grid, state, nml, carry)

    def drop(payload):
        payload["runtime_state"] = payload["runtime_state"].replace(cu_forcing=None)
        payload.pop("cu_forcing_schema_version")

    _rewrite_pickle(path, drop)
    for requested in (None, nml):
        with pytest.raises(ValueError, match=MISSING):
            _read("checkpoint", path, requested)


@pytest.mark.parametrize("fmt", ("restart", "checkpoint"))
def test_pickled_group_without_schema_version_is_refused(tmp_path, fmt):
    grid, state, nml, carry = _case(5)
    path = _write(fmt, tmp_path, grid, state, nml, carry)

    def unversion(payload):
        (payload["carry"] if fmt == "restart" else payload)["cu_forcing_schema_version"] = None

    _rewrite_pickle(path, unversion)
    with pytest.raises(ValueError, match="schema version mismatch"):
        _read(fmt, path)


@pytest.mark.parametrize("change", ("config", "version"))
def test_wrfrst_saved_selection_and_version_bind_the_group(tmp_path, change):
    from netCDF4 import Dataset
    grid, state, nml, carry = _case(5)
    path = _write("wrfrst", tmp_path, grid, state, nml, carry)
    with Dataset(path, "r+") as dataset:
        assert json.loads(dataset.GPUWRF_CU_FORCING_CONFIG) == {"cu_physics": 5}
        assert int(dataset.GPUWRF_CU_FORCING_SCHEMA_VERSION) == SCHEMA_VERSION
        if change == "config":
            dataset.GPUWRF_CU_FORCING_CONFIG = json.dumps({"cu_physics": 0})
        else:
            dataset.delncattr("GPUWRF_CU_FORCING_SCHEMA_VERSION")
    with pytest.raises(ValueError, match=INACTIVE if change == "config" else "schema version mismatch"):
        _read("wrfrst", path)


@pytest.mark.parametrize("mutation", ("dropped", "unversioned", "nonfinite"))
def test_store_readback_refuses_a_malformed_group(tmp_path, mutation):
    grid, state, nml, carry = _case(5)
    generation = _write("store", tmp_path, grid, state, nml, carry)
    _exact(carry.cu_forcing, _read("store", generation))

    def mutate(snapshot):
        payload = snapshot["carries"]["d01"]
        if mutation == "dropped":
            payload["extra_fields"]["cu_forcing"] = None
            payload["cu_forcing_schema_version"] = None
        elif mutation == "unversioned":
            payload["cu_forcing_schema_version"] = None
        else:
            rth, rqv = payload["extra_fields"]["cu_forcing"]
            rth = np.array(rth); rth[0, 0, 0] = np.nan
            payload["extra_fields"]["cu_forcing"] = (rth, rqv)

    _rewrite_store(generation, mutate)
    match = {"dropped": MISSING, "unversioned": "schema version mismatch", "nonfinite": "nonfinite"}[mutation]
    with pytest.raises(ValueError, match=match):
        _read("store", generation)


@pytest.mark.parametrize("reader_name", ("read_checkpoint", "read_checkpoint_with_land_state"))
def test_reduced_checkpoint_readers_refuse_discarding_the_group(tmp_path, reader_name):
    from gpuwrf.runtime import checkpoint
    grid, state, nml, carry = _case(5)
    path = _write("checkpoint", tmp_path, grid, state, nml, carry)
    with pytest.raises(ValueError, match="cu_forcing checkpoint requires read_checkpoint_with_runtime_state"):
        getattr(checkpoint, reader_name)(path, namelist=nml)


def test_moving_nest_refuses_an_unshifted_group():
    from gpuwrf.nesting.moving_driver import _CARRY_FAIL_CLOSED_SUBTREES
    assert "cu_forcing" in _CARRY_FAIL_CLOSED_SUBTREES


def test_cpu_wrfrst_hydration_seeds_held_cumulus_forcing():
    """E78 sibling: an original-WRF restart carries RTHFTEN/RQVFTEN ('r'); the hydrator copies them
    for cu 3/5/6/16/93, refuses an unseeded target, and refuses seeding an inactive scheme."""
    import ast
    import inspect
    from types import SimpleNamespace

    import numpy as np
    from gpuwrf.validation import wrf_restart_hydration as H

    old = (np.zeros((3, 2, 2), np.float32), np.zeros((3, 2, 2), np.float32))
    raw = {'RTHFTEN': np.full((3, 2, 2), 1.5e-5), 'RQVFTEN': np.full((3, 2, 2), -2.0e-9)}
    put = lambda o, r, label, origin: np.asarray(r, o.dtype)  # noqa: E731
    for cu in (3, 5, 6, 16, 93):
        out = H._hydrate_cu_forcing(SimpleNamespace(cu_forcing=old), cu, raw.__getitem__, put)
        np.testing.assert_array_equal(out['cu_forcing'][0], raw['RTHFTEN'].astype(np.float32))
        np.testing.assert_array_equal(out['cu_forcing'][1], raw['RQVFTEN'].astype(np.float32))
        with pytest.raises(ValueError, match='unseeded'):
            H._hydrate_cu_forcing(SimpleNamespace(cu_forcing=None), cu, raw.__getitem__, put)
    assert H._hydrate_cu_forcing(SimpleNamespace(cu_forcing=None), 1, raw.__getitem__, put) == {}
    with pytest.raises(ValueError, match='inactive'):
        H._hydrate_cu_forcing(SimpleNamespace(cu_forcing=old), 1, raw.__getitem__, put)
    # wiring: the release hydrator applies it, and read_wrfrst reads both fields
    calls = [n for n in ast.walk(ast.parse(inspect.getsource(H.hydrate_wrfrst)))
             if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == '_hydrate_cu_forcing']
    assert len(calls) == 1
    reader = inspect.getsource(H.read_wrfrst)
    assert "'RTHFTEN'" in reader and "'RQVFTEN'" in reader
