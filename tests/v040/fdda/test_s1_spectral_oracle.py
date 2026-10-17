"""Spectral nudging vs unmodified WRF+FFTPACK5 on original CPU-WRF fields."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.assimilation import fdda_spectral as sn

jax.config.update("jax_enable_x64", True)
SAVE = Path(__file__).resolve().parents[3] / "proofs/s1_specnudge/savepoints"
MANIFEST = json.loads((SAVE / "MANIFEST.json").read_text())
FIELDS = ("u", "v", "t", "ph", "q")


def replay(name, mutate=None):
    meta = MANIFEST["cases"][name]
    data = dict(np.load(SAVE / name))
    dtype = jnp.float32 if name.endswith("r4.npz") else jnp.float64
    options = sn.SpectralFddaOptions(**meta["options"])
    if mutate:
        options = mutate(options)
    pair = {f"{n}_{s}": jnp.asarray(data[f"frame{i}_{n}"])
            for i, s in ((1, "old"), (2, "new")) for n in FIELDS}
    args = [jnp.asarray(data[f"frame0_{n}"])
            for n in ("u", "v", "th_phy", "ph", "q", "pblh", "ht", "z_at_w")]
    fn = jax.jit(lambda pair, *args: sn.spectral_nudging(options, pair, *args, dtype))
    return data, [fn(pair, *args, float(secs)) for _step, secs in data["calls"]], options


def compare(name, data, results, options):
    # Independent FFT libraries do not share a radix tree. Bound forward + inverse
    # REAL roundoff by 8 eps per log2 stage, plus interpolation/FMA cancellation
    # at the scale of the ORIGINAL WRF expression's operands. No absolute floor.
    nz, ny, nx = data["frame0_t"].shape
    stages = np.ceil(np.log2(nx + 1)) + np.ceil(np.log2(ny + 1))
    gains = (options.guv, options.guv, options.gt, options.gph, options.gq)
    worst = 0.
    for c, got in enumerate(results):
        for n, value, gain in zip(FIELDS, got, gains):
            ref, value = data[f"call{c}_{n}"], np.asarray(value)
            assert value.dtype == ref.dtype
            assert np.isfinite(value).all()
            amp = max(float(np.abs(data[f"frame{i}_{n}"]).max()) for i in (0, 1, 2))
            if n == "t":
                amp += 300.
            # Extrapolated target can reach three times the analysis amplitude.
            tol = 8 * np.finfo(ref.dtype).eps * (stages + 2) * (np.abs(ref) + gain * 3 * amp)
            error = np.abs(value - ref)
            assert np.all(error <= tol), (name, c, n, float(error.max()), float(tol.max()))
            worst = max(worst, float(np.max(error / np.maximum(tol, np.finfo(ref.dtype).tiny))))
            if options.xwavenum < 0 or options.ywavenum < 0 or c == len(results) - 1:
                assert np.count_nonzero(value) == 0, (name, c, n)
    return worst


@pytest.mark.parametrize("name", sorted(n for n, m in MANIFEST["cases"].items()
                                       if m["options"].get("stepfg", 1) == 1))
def test_spectral_tendencies_match_full_pristine_routine(name):
    data, results, options = replay(name)
    assert compare(name, data, results, options) <= 1


def test_reference_assets_and_active_regimes():
    for name, meta in MANIFEST["cases"].items():
        assert hashlib.sha256((SAVE / name).read_bytes()).hexdigest() == meta["sha256"]
        if "negative_cut" not in name:
            assert all(meta["nonzero"][f] > 0 for f in FIELDS), name
    assert len(MANIFEST["cases"]) == 40
    assert "module_fdda_spnudging.F" in MANIFEST["sources_sha256"]
    assert "mrftf1.f90" in MANIFEST["sources_sha256"]


@pytest.mark.parametrize("length", [8, 9, 11, 12])
def test_fft_cutoff_normalization_against_analytic_modes(length):
    x = np.arange(length)
    wave = 2 + .7 * np.cos(2 * np.pi * x / length) + .4 * np.sin(4 * np.pi * x / length)
    got = np.asarray(sn.lowpass_axis(jnp.asarray(wave, jnp.float64), 1, -1))
    np.testing.assert_allclose(got, 2 + .7 * np.cos(2 * np.pi * x / length), rtol=0, atol=2e-15)
    np.testing.assert_allclose(sn.lowpass_axis(jnp.asarray(wave), 0, -1), 2, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(sn.lowpass_axis(jnp.asarray(wave), -1, -1), np.zeros(length))


def test_pbl_deletion_is_killed_by_physical_oracle():
    name = "swiss_8x10_pbl_lid_r4.npz"
    data, results, options = replay(name, lambda o: replace(o, if_no_pbl_nudging_uv=0,
                                                          if_no_pbl_nudging_t=0,
                                                          if_no_pbl_nudging_ph=0,
                                                          if_no_pbl_nudging_q=0,
                                                          if_zfac_uv=0, if_zfac_t=0,
                                                          if_zfac_ph=0, if_zfac_q=0))
    with pytest.raises(AssertionError):
        compare(name, data, results, options)


def driver_replay(name):
    """Full driver/coupling/ownership/finish, compared to literal WRF callers."""
    from types import SimpleNamespace
    from gpuwrf.coupling.fdda_spectral_rk import spectral_step_sources, spectral_finish

    meta = MANIFEST["cases"][name]
    data = dict(np.load(SAVE / name))
    dtype = jnp.float32 if name.endswith("r4.npz") else jnp.float64
    options = sn.SpectralFddaOptions(**meta["options"])
    arrays = {f"{n}_{s}": jnp.asarray(data[f"frame{i}_{n}"])[None]
              for i, s in ((1, "old"), (2, "new")) for n in FIELDS}
    inputs = sn.SpectralFddaInputs(**arrays, options=options, record_steps=(0,))
    carry = sn.initial_spectral_carry(inputs, dtype)
    values = {n: jnp.asarray(data[f"frame0_{n}"]) for n in
              ("u", "v", "th_phy", "ph", "q", "pblh", "ht", "z_at_w", "mu_2", "mub", "muts",
               "c1h", "c2h", "c1f", "c2f")}

    def step(carry, inp, vals, lead, itimestep):
        before = SimpleNamespace(u=vals["u"], v=vals["v"], ph_perturbation=vals["ph"], qv=vals["q"])
        metrics = SimpleNamespace(**{n: vals[n] for n in ("c1h", "c2h", "c1f", "c2f")})
        # Capture UNCOUPLED output from the same driver (before its in-place coupling).
        unc = sn.spectral_driver_tendencies(carry, inp, vals["u"], vals["v"], vals["th_phy"], vals["ph"],
            vals["q"], vals["pblh"], vals["ht"], vals["z_at_w"], lead, itimestep, dtype)
        coupled, owned = spectral_step_sources(
            carry, inp, before, vals["th_phy"], vals["pblh"], vals["ht"], vals["z_at_w"], lead,
            itimestep, vals["mu_2"], vals["mub"], metrics, dtype,
            specified_or_nested=True, periodic_x=False)
        final = spectral_finish(coupled, vals["muts"], metrics, dtype)
        return unc, coupled, owned, final

    fn = jax.jit(step)
    all_results = []
    for itimestep, secs in data["calls"]:
        unc, coupled, owned, carry = fn(carry, inputs, values, float(secs), int(itimestep))
        order = lambda c: (c.rundgdten, c.rvndgdten, c.rthndgdten, c.rphndgdten, c.rqvndgdten)
        own_order = (*owned[:3], owned[4], owned[3])
        all_results.append((order(unc), order(coupled), own_order, order(carry)))
    return data, all_results, fn, (carry, inputs, values, 18., 2)


@pytest.mark.parametrize("name", sorted(MANIFEST["cases"]))
def test_full_spectral_caller_chain_matches_wrf(name):
    data, results, _fn, _args = driver_replay(name)
    for c, stages in enumerate(results):
        for prefix, arrays in zip(("", "cpl_", "own_", "dec_"), stages):
            for n, value in zip(FIELDS, arrays):
                ref, value = data[f"call{c}_{prefix}{n}"], np.asarray(value)
                assert np.isfinite(value).all() and np.isfinite(ref).all()
                assert ref.dtype == value.dtype
                # Normwise FFT forward/backward + FMA bound, including the
                # original subtraction operand scale for theta cancellation.
                amplitude = max(float(np.abs(data[f"frame{i}_{n}"]).max()) for i in (0, 1, 2))
                if n == "t":
                    amplitude += 300.
                gain = MANIFEST["cases"][name]["options"][{"u": "guv", "v": "guv", "t": "gt", "ph": "gph", "q": "gq"}[n]]
                mass = 1 if prefix == "" or n == "ph" else float(np.abs(data["frame0_mu_2"] + data["frame0_mub"]).max())
                scale = max(float(np.abs(ref).max()), 3 * amplitude * gain * mass)
                tol = 128 * np.finfo(ref.dtype).eps * scale
                np.testing.assert_allclose(value, ref, rtol=0, atol=tol, err_msg=f"{name} call{c} {prefix}{n}")
    # SPNUDGING does not overwrite OLD at step1; its nonzero initial forcing
    # distinguishes it from PSU's initial OLD=model overwrite.
    if "negative_cut" not in name:
        assert np.count_nonzero(data["call0_t"]) > 0
