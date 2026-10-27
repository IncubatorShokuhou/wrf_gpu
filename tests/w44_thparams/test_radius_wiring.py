"""Nt_c adapter/held-radius wiring using pinned ORIGINAL WRF radius outputs.

Only full_column is stubbed in the healthy path: it returns the same real
history column used by the isolated WRF radius oracle, with zero precipitation.
This tests radius production and LW/SW input wiring, not a microphysics step or
radiation solve. External oracle absence skips explicitly; changed evidence
fails. No JAX-vs-JAX comparison is presented as WRF fidelity.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.coupling import thompson_runtime_adapter as A
from gpuwrf.physics import rrtmg_mp_re as R
from gpuwrf.physics import thompson_nt_c as N
from gpuwrf.physics.thompson_column import ThompsonColumnState, density_from_pressure_temperature


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/v025/b_thompson/fixtures/mixed_phase_columns.npz"
FIXTURE_SHA = "f3978ef7f7992b7c3aa855ab1ce5206b04a239cf8048048972f1059dc6ba98f3"
ORACLE = Path(os.environ.get("GPUWRF_THOMPSON_RADIUS_ORACLE_ROOT",
                             "<USER_HOME>/wrf_gpu2_lanes/w44-thparams/NT_RADIUS"))
SUMMARY_SHA = "a74cf64f6a0d28b61262bc01693f029758edc00b10109c76d9fa6c0225d6f0b8"
REFERENCES = {
    50: ("0ca825623d257d275ce0b444cf60c37aa8d4721a83f8ebbff3780ddfe574d56a",
         "5adebfc6c7fb4eed4d2ec66ccd6ffce238f13f86c4729d6b3044f89ba7abb22d"),
    100: ("22ca7286ed7980ff02c3f443dfda761a3b1d1d2c79c89f86d3eef4ae6d6ead98",
          "fc123a2fa536e6c4e28fb333df982b74ed72f7389d0ccc2af36b4227381f9b37"),
    150: ("0ac27f81886025a8ca8b66d6a0756cfc16af1651124cb879be8dc33c9246e45f",
          "235b8babff8aa65c4a9398d26744a8a14260f93dc4107c0cb633ff73d5c266bb"),
}
FIELDS = {"cloud": "re_cloud", "ice": "re_ice", "snow": "re_snow"}
BOUNDS = {"cloud": {"real_ulp": 8}, "ice": {"real_ulp": 8}, "snow": {"relative": 1e-5}}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def oracle_data():
    if not (ORACLE / "SUMMARY.json").is_file():
        pytest.skip("external ORIGINAL WRF Nt_c radius oracle is not installed")
    assert _sha(ORACLE / "SUMMARY.json") == SUMMARY_SHA
    summary = json.loads((ORACLE / "SUMMARY.json").read_text())
    assert summary["fixture_sha256"] == FIXTURE_SHA and summary["bounds"] == BOUNDS
    assert _sha(FIXTURE) == FIXTURE_SHA
    with np.load(FIXTURE, allow_pickle=False) as loaded:
        fixture = {name: np.asarray(loaded[name]) for name in loaded.files}
    assert all(np.isfinite(value).all() for name, value in fixture.items() if name.startswith(("in_", "wrf_")))
    references, names = {}, None
    for nt_million, (reference_sha, receipt_sha) in REFERENCES.items():
        directory = ORACLE / f"nt{nt_million}"
        assert _sha(directory / "reference.npz") == reference_sha
        assert _sha(directory / "receipt.json") == receipt_sha
        receipt = json.loads((directory / "receipt.json").read_text())
        assert receipt["nt_c"] == nt_million * 1.e6 and receipt["bounds"] == BOUNDS
        assert receipt["fixture_sha256"] == FIXTURE_SHA
        names = receipt["selected_columns"] if names is None else names
        assert receipt["selected_columns"] == names
        with np.load(directory / "reference.npz", allow_pickle=False) as loaded:
            references[nt_million] = {name: np.asarray(loaded[name]) for name in ("T", *FIELDS)}
        assert all(np.isfinite(value).all() for value in references[nt_million].values())
    assert len(names) == len(set(names)) == 24
    indices = np.array([fixture["names"].tolist().index(name) for name in names])
    assert np.all(fixture["dt"][indices] == 18.)
    data = {name: value[indices] for name, value in fixture.items() if name.startswith("in_")}
    assert all(np.count_nonzero(data["in_" + name]) for name in ("qc", "qi", "qs", "qg"))
    expected_t = (data["in_th"] * data["in_pii"]).astype(np.float32)
    for reference in references.values():
        assert all(value.shape == (24, 44) and value.dtype == np.float32 for value in reference.values())
        assert reference["T"].tobytes() == expected_t.tobytes()
    return data, references


def _original_radius_bound(actual, reference, name):
    a, b = np.asarray(actual), np.asarray(reference)
    assert a.shape == b.shape and a.dtype == b.dtype == np.float32, f"{name} radius schema"
    assert np.isfinite(a).all() and np.isfinite(b).all(), f"non-finite {name} radius"
    assert (a > 0).all() and (b > 0).all(), f"nonpositive {name} radius"
    if name in ("cloud", "ice"):
        distance = np.abs(a.view(np.uint32).astype(np.int64) - b.view(np.uint32).astype(np.int64))
        assert int(distance.max()) <= BOUNDS[name]["real_ulp"], f"{name} WRF bound: {distance.max()} REAL ULP"
    else:
        relative = np.abs(a.astype(np.float64) - b.astype(np.float64)) / np.abs(b.astype(np.float64))
        assert float(relative.max()) <= BOUNDS[name]["relative"], f"{name} WRF bound: relative {relative.max()}"


def _real_state(data):
    ncol, nz = data["in_qv"].shape
    shape_grid = SimpleNamespace(nx=ncol, ny=1, nz=nz)
    state = State(**{name: jnp.zeros(shape, DEFAULT_DTYPES.dtype_for(name))
                     for name, shape in _state_field_shapes(shape_grid).items()})
    column = lambda name: jnp.asarray(data["in_" + name], jnp.float32)[None]
    mass = lambda name: jnp.moveaxis(column(name), -1, 0)
    temp = (column("th") * column("pii")).astype(jnp.float32)
    zero = jnp.zeros_like(temp)
    height = jnp.concatenate((jnp.zeros_like(column("dz8w")[..., :1]), jnp.cumsum(column("dz8w"), axis=-1)), axis=-1)
    state = state.replace(p=mass("p"), theta=mass("th"), qv=mass("qv"), qc=mass("qc"), qr=mass("qr"),
                          qi=mass("qi"), qs=mass("qs"), qg=mass("qg"), Ni=mass("ni"), Nr=mass("nr"),
                          Ns=jnp.moveaxis(zero, -1, 0), Ng=jnp.moveaxis(zero, -1, 0),
                          ph=jnp.moveaxis(height * jnp.float32(9.81), -1, 0),
                          t_skin=temp[..., 0], xland=jnp.ones_like(temp[..., 0]))
    # Surface/geometry fields are structural wiring inputs; no surface/radiation
    # fidelity is claimed. The radius oracle uses only the real fields below.
    out = ThompsonColumnState(qv=column("qv"), qc=column("qc"), qr=column("qr"), qi=column("qi"), qs=column("qs"),
                              qg=column("qg"), Ni=column("ni"), Nr=column("nr"), Ns=zero, Ng=zero,
                              T=temp, p=column("p"), rho=density_from_pressure_temperature(column("p"), temp, column("qv")),
                              dz=column("dz8w"), w=column("w"))
    return state.ensure_conditional_leaves(mp_physics=8, use_mp_re=1), out


@pytest.fixture
def native(monkeypatch):
    for suffix in ("NATIVE_REAL", "FULL_COLUMN", "MIXED_PHASE_WRF", "COLUMN_SED", "SED_FP32", "WRF_CONSTANTS"):
        monkeypatch.setenv("GPUWRF_THOMPSON_" + suffix, "1")
    monkeypatch.setenv("GPUWRF_THOMPSON_IMPLICIT_SED", "0")
    monkeypatch.setenv("GPUWRF_MP_COLUMN_TILING", "0")
    monkeypatch.setenv("GPUWRF_MICROPHYSICS_WRF_ORDER", "1")
    monkeypatch.setattr(R, "_MP_RE", True)


def _adapter_output(monkeypatch, data, nt_million, *, omit_nt=False):
    before, known_out = _real_state(data)
    assert all(getattr(before, name) is not None for name in FIELDS.values())
    nt_c = jnp.float32(nt_million * 1.e6)
    calls = {"kernel": [], "writeback": [], "radius": []}
    real_writeback, real_radius = A._state_from_thompson_output, N.effective_radii

    def known_full_column(column, dt, *, parameters):
        assert column.qv.shape == known_out.qv.shape and dt == 18.
        assert parameters["nt_c"].shape == () and parameters["nt_c"].dtype == jnp.float32
        calls["kernel"].append(float(np.asarray(parameters["nt_c"])))
        return known_out, {name: jnp.zeros(known_out.qv.shape[:-1], jnp.float32)
                           for name in ("rain", "snow", "graupel", "ice", "cloudw")}

    def writeback(state, out, precip):
        # This is the actual original copyback, observed at the point where
        # its fixed-Nt producer would run if the local suppression were lost.
        assert state.re_cloud is state.re_ice is state.re_snow is None
        calls["writeback"].append(True)
        return real_writeback(state, out, precip)

    def radius(out, value):
        assert out is known_out
        calls["radius"].append(float(np.asarray(value)))
        if omit_nt:
            # Explicit deletion control: the former fixed-Nt producer.
            return R.calc_thompson_effective_radii(out.T, out.p, out.qv, out.qc, out.qi, out.Ni, out.qs)
        return real_radius(out, value)

    monkeypatch.setattr(A, "full_column", known_full_column)
    monkeypatch.setattr(A, "_state_from_thompson_output", writeback)
    monkeypatch.setattr(N, "effective_radii", radius)
    adapter = A.selected_adapter(SimpleNamespace(thompson_parameters={"nt_c": nt_c}, mp_physics=8, run_physics=True))
    after = adapter(before, 18.)
    assert calls == {"kernel": [float(nt_million * 1.e6)], "writeback": [True], "radius": [float(nt_million * 1.e6)]}
    assert all(getattr(after, name) is not None and getattr(after, name).dtype == jnp.float32 for name in FIELDS.values())
    return after


@pytest.mark.parametrize("nt_million", (50, 150))
def test_nt_radii_written_once_and_held_for_lw_and_sw(oracle_data, native, monkeypatch, nt_million):
    data, references = oracle_data
    after = _adapter_output(monkeypatch, data, nt_million)
    expected = references[nt_million]
    for name, field in FIELDS.items():
        _original_radius_bound(np.asarray(C._to_columns(getattr(after, field)))[0], expected[name], name)
    # Next-step moisture changes must not cause either consumer to recompute
    # radii from a different state. Exercise the real common LW/SW builder.
    later = after.replace(qc=jnp.zeros_like(after.qc), qi=jnp.zeros_like(after.qi), qs=jnp.zeros_like(after.qs))
    sw, lw, *_ = C._rrtmg_column_inputs(later, None, time_utc="2026-02-28_00:00:00")
    for consumer in (sw, lw):
        for name, field in FIELDS.items():
            value = getattr(consumer, field)
            assert value is not None, f"missing held {field}"
            radius = np.asarray(value)
            held = np.asarray(C._to_columns(getattr(after, field)))
            assert radius.shape == held.shape and radius.dtype == held.dtype
            assert np.isfinite(radius).all() and radius.tobytes() == held.tobytes(), field
            _original_radius_bound(radius[0], expected[name], name)


@pytest.mark.parametrize("nt_million", (50, 150))
def test_omitting_runtime_nt_is_killed_by_original_cloud_radius(oracle_data, native, monkeypatch, nt_million):
    data, references = oracle_data
    fixed = _adapter_output(monkeypatch, data, nt_million, omit_nt=True)
    cloud = np.asarray(C._to_columns(fixed.re_cloud))[0]
    _original_radius_bound(cloud, references[100]["cloud"], "cloud")
    with pytest.raises(AssertionError, match="cloud WRF bound"):
        _original_radius_bound(cloud, references[nt_million]["cloud"], "cloud")


@pytest.mark.parametrize("name", FIELDS)
@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
@pytest.mark.parametrize("side", ("candidate", "reference"))
def test_radius_gate_rejects_nonfinite_before_reduction(oracle_data, name, bad, side):
    reference = oracle_data[1][50][name].copy()
    candidate = reference.copy()
    (candidate if side == "candidate" else reference)[-1, -1] = bad
    with pytest.raises(AssertionError, match="non-finite"):
        _original_radius_bound(candidate, reference, name)
