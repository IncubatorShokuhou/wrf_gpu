"""Direct history fallback: all WRF moist masses, including hail and extra ice.

The independent reference compiles unchanged pristine WRF phy_prep, REAL4.
These controlled columns qualify the writer seam, not a forecast trajectory.
"""
from datetime import datetime
from pathlib import Path
import os
import shutil
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.io import wrfout_writer as writer
from test_m7_netcdf_writer import authenticated_source_grid, synthetic_case

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proofs/t_qhmass"))
from wrf_moist_oracle import build_oracle


def _require_original_inputs():
    """Skip, naming what is missing, when the pristine source or a Fortran compiler is unavailable.

    Compiler: $FC, else gfortran on PATH (build_oracle's own order).
    """
    source = ROOT / "data/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F"
    if not source.is_file():
        pytest.skip(f"pristine WRF source unavailable: {source} (link data/wrf_pristine)")
    compiler = os.environ.get("FC") or shutil.which("gfortran")
    if not compiler or not (os.path.isfile(compiler) and os.access(compiler, os.X_OK)) and not shutil.which(compiler):
        pytest.skip(f"Fortran compiler for the pristine WRF oracle unavailable: FC={compiler!r}, "
                    "no gfortran on PATH (set FC)")

MASS = ("qv", "qc", "qr", "qi", "qs", "qg", "qh", "qi2", "qi3")
ALIASES = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL", "QICE2", "QICE3")


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    _require_original_inputs()
    return build_oracle(ROOT, tmp_path_factory.mktemp("original_writer_psfc"))


def column(optional=(), *, upper=False):
    state, grid, nl = synthetic_case(nx=3, ny=2, nz=4)
    del state.psfc
    shape = state.qv.shape
    # Binary-exact operands make REAL4 downward sweep and the existing writer
    # vertical reduction identical, without introducing any tolerance.
    cell = np.indices(shape, dtype=np.float32).sum(axis=0) + 1
    for index, name in enumerate(MASS):
        setattr(state, name, (cell * (index + 1) / 4096).astype(np.float32)
                if name in MASS[:6] or name in optional else None)
    state.mu_total = np.full(shape[1:], 65536, np.float32)
    state.mu_perturbation = np.zeros(shape[1:], np.float32)
    from types import SimpleNamespace
    grid.metrics = SimpleNamespace(c1h=np.ones(shape[0], np.float32),
                                   c2h=np.zeros(shape[0], np.float32),
                                   dnw=np.full(shape[0], -.25, np.float32),
                                   p_top=np.float32(8192))
    # Non-mass scalar slots must NEVER enter the moist loading.
    for name in ("Ni", "Nr", "Nh", "qvolg", "qvolh", "qir", "qib", "nwfa", "nifa"):
        setattr(state, name, np.full(shape, 1.e9, np.float32))
    moist = tuple(getattr(state, name) for name in MASS if getattr(state, name) is not None)
    if upper:
        for name, alias in zip(MASS, ALIASES):
            setattr(state, alias, getattr(state, name))
            delattr(state, name)
    return state, grid, nl, moist


def prepare(tmp_path, state, grid, nl, subset, diagnostics=None):
    return writer.prepare_wrfout_payload(
        state, grid, nl, tmp_path / "wrfout_d01", domain="d01",
        domain_authority=writer.bind_wrfout_domain_authority(
            "d01", authenticated_source_grid(grid, "d01"), grid),
        run_start=datetime(2026, 10, 15), valid_time=datetime(2026, 10, 15, 1),
        lead_hours=1., variable_subset=("PSFC",) if subset else None,
        diagnostics=diagnostics)


@pytest.mark.parametrize("subset", [False, True], ids=["full", "psfc_only"])
@pytest.mark.parametrize("upper", [False, True], ids=["state", "wrf_alias"])
@pytest.mark.parametrize("optional", [(), ("qh",), ("qi2", "qi3"), ("qh", "qi2", "qi3")])
def test_direct_writer_all_masses_match_original(original, tmp_path, optional, upper, subset):
    state, grid, nl, moist = column(optional, upper=upper)
    m = grid.metrics
    _, faces, _ = original(moist, state.mu_total, m.c1h, m.c2h, m.dnw, m.p_top)
    payload = prepare(tmp_path, state, grid, nl, subset)
    np.testing.assert_array_equal(payload.fields["PSFC"], faces[0])
    if subset:
        assert set(payload.fields) == {"PSFC"}


@pytest.mark.parametrize("subset", [False, True])
@pytest.mark.parametrize("source", ["state", "diagnostics"])
def test_producer_psfc_wins_over_fallback(tmp_path, subset, source):
    state, grid, nl, _ = column(("qh",))
    held = np.full(state.mu_total.shape, 98765.5, np.float32)
    diagnostics = {"PSFC": held} if source == "diagnostics" else None
    if source == "state":
        state.psfc = held
    payload = prepare(tmp_path, state, grid, nl, subset, diagnostics)
    assert payload.fields["PSFC"].tobytes() == held.tobytes()


@pytest.mark.parametrize("subset", [False, True])
def test_no_metrics_keeps_existing_extrapolation(tmp_path, subset):
    state, grid, nl, _ = column(("qh",))
    del grid.metrics
    p = state.p_total
    phi = state.ph_total
    weight = (phi[0] - .5*(phi[1]+phi[2])) / (.5*(phi[0]+phi[1]) - .5*(phi[1]+phi[2]))
    expected = weight*p[0] + (1-weight)*p[1]
    np.testing.assert_array_equal(prepare(tmp_path, state, grid, nl, subset).fields["PSFC"], expected)


def test_psfc_only_batches_all_active_mass_leaves(tmp_path, monkeypatch):
    state, grid, nl, _ = column(("qh", "qi2", "qi3"))
    for name in MASS:
        setattr(state, name, jnp.asarray(getattr(state, name)))
    calls = []
    device_get = jax.device_get
    def capture(values):
        calls.append(tuple(id(value) for value in values))
        return device_get(values)
    monkeypatch.setattr(jax, "device_get", capture)
    prepare(tmp_path, state, grid, nl, True)
    assert len(calls) == 1
    assert {id(getattr(state, name)) for name in MASS} <= set(calls[0])


def test_hail_fallback_survives_netcdf_publication(original, tmp_path):
    from netCDF4 import Dataset
    state, grid, nl, moist = column(("qh",))
    m = grid.metrics
    _, faces, _ = original(moist, state.mu_total, m.c1h, m.c2h, m.dnw, m.p_top)
    payload = prepare(tmp_path, state, grid, nl, True)
    writer.write_prepared_wrfout(payload, expected_domain="d01",
        expected_domain_authority=payload.domain_authority, variable_subset=("PSFC",))
    with Dataset(payload.target) as nc:
        assert np.asarray(nc["PSFC"][0]).tobytes() == faces[0].tobytes()


@pytest.mark.parametrize("missing_frozen", [False, True])
def test_no_hail_full_writer_keeps_legacy_bytes(tmp_path, missing_frozen):
    state, grid, nl, _ = column()
    rng = np.random.default_rng(20261015)
    for name in MASS[:6]:
        setattr(state, name, rng.uniform(0, .01, state.qv.shape).astype(np.float32))
    if missing_frozen:
        del state.qs, state.qg
    qtot = state.qv + state.qc + state.qr + state.qi
    for name in ("qs", "qg"):
        if hasattr(state, name):
            qtot = qtot + getattr(state, name)
    m = grid.metrics
    # Preserve the actual parent writer's f32 species / f64 vertical arithmetic,
    # independently of the new shared helper (regression, not a WRF oracle).
    dp = (m.c1h.astype(np.float64)[:, None, None] * state.mu_total[None]
          + m.c2h.astype(np.float64)[:, None, None]) * (-m.dnw.astype(np.float64)[:, None, None])
    terms = (1.0 + qtot) * dp
    total = np.zeros_like(terms[0])
    comp = np.zeros_like(total)
    for term in terms:
        y = term - comp
        updated = total + y
        comp = (updated - total) - y
        total = updated
    expected = np.asarray(float(m.p_top) + (total - comp), np.float32)
    assert prepare(tmp_path, state, grid, nl, False).fields["PSFC"].tobytes() == expected.tobytes()
