"""w42-swdfid: the GPUWRF_SWDDIR_V040_SPLIT opt-out restores the v0.4.0 RRTMG SW direct/diffuse split bitwise (CPU).

ALISIOS (#146 decision 2026-10-19) asked for an opt-out of the v0.4.1 SWDDIR fix (WRF ``ztdbt_nodel`` direct output)
until the cause of the WT1 0420 south-slope difference is clear.  The lane audit found the port's caller path
WRF-faithful, so the fix stays the default; ``GPUWRF_SWDDIR_V040_SPLIT=1`` makes every aer_opt=0 SW producer emit the
v0.4.0 direct beam again (the adding method's delta-scaled transmittance; SWDDIR/SWDDIF history and the slope_rad /
topo_shading diffuse fraction follow it).  Gates:
  K  default Pallas band kernels (constant and Thompson radii, per-band and all-band launch) with ``direct_scaled=True``:
     ALL five outputs bitwise == the frozen v0.4.0 kernel (fixtures/rad_sw_band_sums_v040.py == git blob 98c4be0a of
     tag v0.4.0); the default ``direct_scaled=False`` differs from it in the direct output only, on cloudy columns;
  R  release SW path (GPUWRF_FAST_DEFAULTS=1, CPU interpret) on real WN3 0227 oracle columns: flag ON == the genuine
     v0.4.0-product mutant of w41_swddir (itself bitwise == the v0.4.0 tree on 186 + 107 oracle columns, lane receipt
     OPT01) for the band kernel AND the XLA band-scan branch, with every frozen tier-1 metric identical;
  A  the flag with aer_opt=1/3 (no v0.4.0 counterpart) is refused at trace time;
  E  the flag is a resolved import-time AOT cheap-key constant (E80).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import rad_sw_band_sums as B
from gpuwrf.physics import rrtmg_sw as R

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V040 = HERE / "fixtures" / "rad_sw_band_sums_v040.py"
V040_GIT_BLOB = "98c4be0a751ce1f69f576c4d3f0a39a025e2a9a3"  # git ls-tree v0.4.0 src/gpuwrf/kernels/rad_sw_band_sums.py
RUNNER = ROOT / "tests" / "v025" / "w41_swddir" / "swddir_oracle_runner.py"
NCOL, NLAY, NG, NB, BAND = 8, 6, 16, 14, 5
LIQ_GRID, ICE_GRID = (2.5, 1.0), (5.0, 3.0)  # cldprmc radius grids (start, step)


def _v040_module():
    spec = importlib.util.spec_from_file_location("rad_sw_band_sums_v040", V040)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _inputs(mp_re, all_bands, seed=7):
    """Thin-cloud columns (tau ~ 0.1-2, strongly forward scattering) where the scaled and unscaled beams differ."""

    rng = np.random.default_rng(seed)
    f = lambda lo, hi, shape: rng.uniform(lo, hi, shape).astype(np.float32)  # noqa: E731
    taus = (NCOL, NLAY, NB, NG) if all_bands else (NCOL, NLAY, NG)
    tau_gas, tau_ray = f(1e-4, 0.3, taus), f(1e-4, 0.05, taus)
    cloud = (rng.uniform(size=(NCOL, NLAY, NB, NG)) < 0.25).astype(np.float32)
    cloud[0] = 0.0  # one clear column
    gmask = np.ones((NB, NG), np.float32)
    gmask[BAND, 12:] = 0.0  # a band with fewer g-points: identity lanes
    sflux = f(0.5, 3.0, (NCOL, NB, NG))
    coszen, albedo, scale = f(0.15, 0.95, (NCOL,)), f(0.05, 0.3, (NCOL,)), f(0.97, 1.03, (NCOL,))
    paths = f(0.0, 6.0, (NCOL, NLAY, 3))  # in-cloud liquid / ice / snow paths [g m-2]
    if mp_re:
        coeffs = np.zeros((7, NB, 64), np.float32)  # raw radius tables extliq1 .. fdlice3 (synthetic)
        coeffs[0], coeffs[3] = rng.uniform(0.05, 0.2, (NB, 64)), rng.uniform(0.02, 0.1, (NB, 64))
        coeffs[1], coeffs[4] = rng.uniform(0.99, 0.9999, (NB, 64)), rng.uniform(0.95, 0.999, (NB, 64))
        coeffs[2], coeffs[5] = rng.uniform(0.84, 0.88, (NB, 64)), rng.uniform(0.75, 0.85, (NB, 64))
        coeffs[6] = rng.uniform(0.0, 0.05, (NB, 64))
        idx = rng.integers(0, 40, (NCOL, NLAY, 3))
        radii = np.stack([LIQ_GRID[0] + LIQ_GRID[1] * idx[..., 0], ICE_GRID[0] + ICE_GRID[1] * idx[..., 1],
                          ICE_GRID[0] + ICE_GRID[1] * idx[..., 2]], axis=-1).astype(np.float32)
        incloud = np.concatenate([paths, radii], axis=-1)
    else:
        ext = rng.uniform(0.05, 0.15, (3, NB))
        ssa, asy = rng.uniform(0.99, 0.9999, (3, NB)), rng.uniform(0.84, 0.88, (3, NB))
        fwd = asy * asy
        den = 1.0 - fwd * ssa
        coeffs = np.concatenate([ext, den, ssa * (1 - fwd) / den, (asy - fwd) / (1 - fwd)]).astype(np.float32)
        incloud = paths
    return tuple(jnp.asarray(x) for x in (tau_gas, tau_ray, cloud, incloud, coeffs, gmask, sflux, coszen, albedo, scale))


def _launch(mod, args, all_bands, **kw):
    if all_bands:
        return [np.asarray(x) for x in mod.sw_allband_flux_sums(*args, interpret=True, **kw)]
    return [np.asarray(x) for x in mod.sw_band_flux_sums(*args, jnp.int32(BAND), interpret=True, **kw)]


def _bitwise(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def test_frozen_reference_is_the_v040_kernel_blob():
    data = V040.read_bytes()
    assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == V040_GIT_BLOB


@pytest.mark.parametrize("all_bands", [False, True], ids=["band", "allband"])
@pytest.mark.parametrize("mp_re", [False, True], ids=["const_radii", "thompson_radii"])
def test_optout_band_kernels_bitwise_v040(mp_re, all_bands):
    args = _inputs(mp_re, all_bands)
    v040 = _launch(_v040_module(), args, all_bands)
    optout = _launch(B, args, all_bands, direct_scaled=True)
    default = _launch(B, args, all_bands)
    names = ("down", "up", "direct", "clear_down", "clear_up")
    assert all(np.all(np.isfinite(x)) for x in v040 + optout + default)
    for name, want, got in zip(names, v040, optout):
        assert _bitwise(got, want), (name, float(np.max(np.abs(got.astype(np.float64) - want))))
    for name, want, got in zip(names, v040, default):  # only the direct OUTPUT product differs from v0.4.0
        if name != "direct":
            assert _bitwise(got, want), name
    np.testing.assert_array_equal(default[2][0], v040[2][0])  # clear column: nodel == the scaled beam
    gap = v040[2][1:, 0] - default[2][1:, 0]  # the delta-scaled cloud depth is smaller: v0.4.0 beam too high
    assert np.max(gap) > 0.05 * np.max(v040[2][:, 0]), float(np.max(gap))


def _run(tmp: Path, args, **env) -> tuple[dict, dict]:
    tmp.mkdir(parents=True, exist_ok=True)
    out = tmp / "result.json"
    full = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", GPUWRF_FAST_DEFAULTS="1",
                PYTHONPATH=str(ROOT / "src"))
    full.pop("GPUWRF_SWDDIR_V040_SPLIT", None)
    full.update(env)
    proc = subprocess.run([sys.executable, str(RUNNER), str(out), *args], env=full, capture_output=True, text=True,
                          timeout=3600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    with np.load(str(out) + ".npz") as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    return json.loads(out.read_text()), arrays


@pytest.mark.parametrize("branch", ["band_kernel", "xla_band_scan"])
def test_optout_release_path_equals_v040_product(tmp_path, branch):
    """R: flag ON (port) == the w41 genuine mutant that emits the v0.4.0 product (flag unset), bitwise."""

    band_env = {} if branch == "band_kernel" else {"GPUWRF_RRTMG_SW_BAND_SUMS": "0"}
    mutant = "kernel_scaled_direct" if branch == "band_kernel" else "xla_scaled_direct"
    on, on_arr = _run(tmp_path / "on", ["port:0,6"], GPUWRF_SWDDIR_V040_SPLIT="1", **band_env)
    ref, ref_arr = _run(tmp_path / "ref", [f"{mutant}:0,6"], **band_env)
    assert on["real_entry"] and (on["band_sums"] > 0) == (branch == "band_kernel")
    calls = sum(on["variants"]["port"]["band_kernel_calls"].values())
    assert (calls > 0) == (branch == "band_kernel")  # E114: the path under test ran
    for arm in ("0", "6"):
        for field in ("surface_direct", "surface_down"):
            got, want = on_arr[f"port_{field}_{arm}"], ref_arr[f"{mutant}_{field}_{arm}"]
            assert _bitwise(got.astype(np.float32), want.astype(np.float32)) and np.array_equal(got, want), (arm, field)
        assert on["variants"]["port"]["arms"][arm]["bound"] == ref["variants"][mutant]["arms"][arm]["bound"]
        bound = on["variants"]["port"]["arms"][arm]["bound"]
        assert all(np.isfinite(v) for v in bound.values()), bound  # E200: a Python max would skip a later NaN
        assert bound["swddir"] > 1.0  # the v0.4.0 product misses WRF's SWDDIR (cloudy WN3 columns) ...
        assert max(v for k, v in bound.items() if k not in ("swddir", "swddif")) <= 1.0  # ... and nothing else moves


def _aerosol_runner():
    sys.path.insert(0, str(ROOT / "tests" / "v025" / "w41_aerosol"))
    try:
        import sw_oracle_runner as S
        from tegen_fixture import RE01, SW, load
    finally:
        sys.path.pop(0)
    return S, load(RE01), load(SW)


def test_optout_upfront_xla_path(monkeypatch):
    """R (proof path): the upfront XLA construction (_SW_TAUMOL_CHUNK=False) with the flag == its v0.4.0 product
    (direct depths dropped), and differs from the default on cloudy WN3 columns."""

    S, fx, ae = _aerosol_runner()
    ncol = fx["in_t"].shape[0]
    cols = np.flatnonzero(np.asarray(fx["category"]) != "clear")[:12]
    sub = lambda d, n: {k: (v[cols] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == n else v)  # noqa: E731
                        for k, v in d.items()}
    fx, ae = sub(fx, ncol), sub(ae, ae["coszen"].shape[0])
    monkeypatch.setattr(R, "_SW_TAUMOL_CHUNK", False)
    original = R._sw_band_tile_fluxes
    out = {}
    for name in ("optout", "v040_product", "default"):
        monkeypatch.setattr(R, "_SWDDIR_V040_SPLIT", name == "optout")
        monkeypatch.setattr(R, "_sw_band_tile_fluxes", (lambda *a, **k: original(*a, **{**k, "direct_taus": None}))
                            if name == "v040_product" else original)
        jax.clear_caches()  # E161
        out[name] = S._solve(fx, ae, 0, False)
    jax.clear_caches()
    for key, value in out["optout"].items():
        assert np.array_equal(value, out["v040_product"][key]), key
    assert np.max(np.abs(out["optout"]["swddir"] - out["default"]["swddir"])) > 1.0  # W/m2, cloudy day columns


def test_optout_refuses_aerosol(monkeypatch):
    """A: v0.4.0 had no aer_opt=1/3 path, so the opt-out cannot claim its split there."""

    S, fx, ae = _aerosol_runner()
    arm = int(np.flatnonzero(np.asarray(ae["arm_aer_opt"]) == 1)[0])
    monkeypatch.setattr(R, "_SWDDIR_V040_SPLIT", True)
    jax.clear_caches()  # E161: no trace from another test may be reused
    try:
        with pytest.raises(NotImplementedError, match="GPUWRF_SWDDIR_V040_SPLIT"):
            S._solve(fx, ae, arm, False)
    finally:
        jax.clear_caches()


def test_optout_flag_is_a_resolved_cheap_key_constant(monkeypatch):
    """E: the import-time constant is keyed (resolved value), so ON/OFF executables never share a cache entry."""

    from gpuwrf.runtime import aot_cheap_key as K

    assert ("gpuwrf.physics.rrtmg_sw", "_SWDDIR_V040_SPLIT") in K.IMPORT_TIME_ENV_CONSTANTS
    off = K.module_const_env_hash()
    monkeypatch.setattr(R, "_SWDDIR_V040_SPLIT", not R._SWDDIR_V040_SPLIT)
    assert K.module_const_env_hash() != off
