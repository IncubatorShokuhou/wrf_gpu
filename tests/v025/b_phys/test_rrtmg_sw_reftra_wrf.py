"""Original dated RRTMG-SW clear-sky on real N36 hydrated columns."""
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.physics import rrtmg_sw as SW, rrtmg_lw as LW
from gpuwrf.kernels import rad_sw_band_sums as BS

KEY = "GPUWRF_RRTMG_SW_REFTRA_WRF"
DATA = Path(__file__).parent / "fixtures/rrtmg_sw_reftra_n36.npz"


def finite(*arrays):
    for array in arrays:
        assert np.isfinite(np.asarray(array)).all(), "non-finite original or candidate"


@pytest.fixture(autouse=True)
def context(monkeypatch):
    # E144: explicitly select the released McICA/REAL path on CPU. Imports
    # may already have occurred during collection, so bind captured switches
    # as well as trace-time environment flags, restoring everything on exit.
    from gpuwrf._fast_defaults import FAST_PATH_DEFAULTS
    from gpuwrf.kernels import rad_mcica
    from gpuwrf.physics import rrtmg_constants
    monkeypatch.setenv("GPUWRF_FAST_DEFAULTS", "1")
    for name,value in FAST_PATH_DEFAULTS.items():
        if name.startswith(("GPUWRF_MCICA_", "GPUWRF_RRTMG_")):
            monkeypatch.setenv(name,value)
    monkeypatch.setattr(SW, "_MCICA_JUMPAHEAD", True)
    monkeypatch.setattr(LW, "_MCICA_JUMPAHEAD", True)
    monkeypatch.setattr(rad_mcica, "_MAXRAND", True)
    monkeypatch.setattr(rrtmg_constants, "_RRTMG_REAL_CONSTANTS", True)
    monkeypatch.setenv(KEY, "1")
    monkeypatch.setattr(SW, "_FUSED_QUADRATURE", True)
    monkeypatch.setattr(LW, "_FUSED_TRANSFER", True)
    from jax.experimental import pallas as pl
    original = pl.pallas_call
    monkeypatch.setattr(pl, "pallas_call", lambda *a, **kw: original(*a, **dict(kw, interpret=True)))
    jax.clear_caches()
    yield
    jax.clear_caches()


@pytest.mark.parametrize("domain", ("d03", "d02"))
@pytest.mark.parametrize("band_mode", (0, 1))
def test_original_clear_sw_real_columns(domain, monkeypatch, band_mode):
    monkeypatch.setenv("GPUWRF_RRTMG_SW_BAND_SUMS", str(band_mode))
    with np.load(DATA, allow_pickle=False) as file:
        fields = {k.split("__", 1)[1]: file[k] for k in file.files if k.startswith(domain + "__")}
    expected = fields.pop("reference"); fields.pop("reference_allsky", None)
    finite(expected, *fields.values())
    parameters = inspect.signature(SW.RRTMGSWColumnState.__init__).parameters
    actual = []
    for i in range(8):
        kwargs = {name: jnp.asarray(value if value.ndim == 0 else value[i:i+1])
                  for name, value in fields.items() if name in parameters}
        with jax.enable_x64(False):
            state = SW.RRTMGSWColumnState(**kwargs)
            result = SW.solve_rrtmg_sw_column(state, with_clear_sky=True)
            jax.block_until_ready(result)
        finite(*jax.tree.leaves(result))
        down = np.asarray(result.clear_flux_down).reshape(-1)
        up = np.asarray(result.clear_flux_up).reshape(-1)
        actual.append([down[-1], up[-1], down[0], up[0]])
    actual = np.asarray(actual, np.float32)
    finite(actual)
    error = actual.astype(float) - expected.astype(float)
    assert np.all(np.abs(error) <= .02 + 2e-5 * np.abs(expected)), error
    assert abs(error[:, 2].mean()) <= .05


@pytest.mark.parametrize("domain", ("d03", "d02", "thick_d03", "thick_d02"))
def test_original_allsky_real_cloud_columns(domain, monkeypatch):
    monkeypatch.setenv("GPUWRF_RRTMG_SW_BAND_SUMS", "1")
    with np.load(DATA, allow_pickle=False) as file:
        fields = {k.split("__", 1)[1]: file[k] for k in file.files if k.startswith(domain + "__")}
    expected = fields.pop("reference_allsky"); fields.pop("reference", None)
    finite(expected, *fields.values())
    parameters = inspect.signature(SW.RRTMGSWColumnState.__init__).parameters
    actual = []
    for i in range(len(expected)):
        kwargs = {name: jnp.asarray(value if value.ndim == 0 else value[i:i+1])
                  for name, value in fields.items() if name in parameters}
        with jax.enable_x64(False):
            state = SW.RRTMGSWColumnState(**kwargs)
            result = SW.solve_rrtmg_sw_column(state, with_clear_sky=True)
            jax.block_until_ready(result)
        finite(*jax.tree.leaves(result))
        down = np.asarray(result.flux_down).reshape(-1)
        up = np.asarray(result.flux_up).reshape(-1)
        actual.append([down[-1], up[-1], down[0], up[0]])
    actual = np.asarray(actual, np.float32)
    finite(actual)
    error = actual.astype(float) - expected.astype(float)
    assert np.all(np.abs(error) <= .02 + 2e-5 * np.abs(expected)), error
    assert abs(error[:, 2].mean()) <= .05


def test_original_ssa_conservative_range():
    # Original reftra_sw threshold=.9999995 and pure Rayleigh pw=1, pgg=0.
    assert SW._sw_ssa_upper() == BS._ssa_upper() == 1.0
    assert np.float32(SW._sw_ssa_upper()) >= np.float32(.9999995)


def test_default_off_and_legacy_gate(monkeypatch):
    monkeypatch.setenv(KEY, "0")
    assert not SW._sw_reftra_wrf_enabled()
    assert SW._sw_ssa_upper() == BS._ssa_upper() == .999999
    monkeypatch.setenv(KEY, "1")
    monkeypatch.setattr(LW, "_FUSED_TRANSFER", False)
    assert not SW._sw_reftra_wrf_enabled()
    assert SW._sw_ssa_upper() == BS._ssa_upper() == .999999


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_finite_guard_rejects_poison(value):
    with pytest.raises(AssertionError, match="non-finite"):
        finite(np.asarray(value, np.float32))
