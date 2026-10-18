"""w41-swddir runner: the RRTMG SW path of this process vs pristine WRF V4.7.1 RRTMG_SWRAD on aer_opt=0 arms.

usage: python swddir_oracle_runner.py OUT_JSON [--swiss] VARIANT[:ARMS] ...    e.g.  port:0,2,4,6,8  kernel_scaled_direct:0

Oracles (same caller wiring: Thompson radii has_reqc/i/s = 1, cldovrlp = 2, icloud = 1, o3input = 2, ghg_input = 1):
  default  AE01 (tests/v025/w41_aerosol/fixtures/tegen_sw_0227_v1.npz) on the 186 real RE01 WN3 0227 columns; its
           aer_opt=0 arms 0/2/4/6/8 = the column's own radiation call (arm 0 == RE01 arm A) and julians .3/135/196.6/352.2;
  --swiss  S01 (fixtures/swddir_swiss_s01_v1.npz): real Swiss CPU-WRF 2023-01-15 d01 columns, own call (arm 0).
Release path = GPUWRF_FAST_DEFAULTS=1 in the environment (REAL entry + fused Pallas band kernel, CPU interpret); with
GPUWRF_RRTMG_SW_BAND_SUMS=0 the XLA band-scan branch runs.  Variants: ``port`` and genuine mutants (E202):
  kernel_scaled_direct  the default band kernels emit the delta-scaled adding-method beam (the released v0.4.0 product)
  xla_scaled_direct     the XLA tile solver drops the unscaled direct depths (same product on the XLA branch)
Per variant and arm the JSON holds the worst frozen tier-1 ratio per output (SWDOWN, TOA up, SWDDIR, SWDDIF, clear-sky
surface down / TOA up, surface..model-top profiles, heating) over all columns and over the cloudy day columns, and
the default band-kernel call counts (E114); ``OUT_JSON.npz`` keeps the port surface_direct/surface_down per arm.
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "w41_aerosol"))

import jax  # noqa: E402

from gpuwrf.kernels import rad_sw_band_sums as B  # noqa: E402
from gpuwrf.physics import rrtmg_sw as R  # noqa: E402

import sw_oracle_runner as S  # noqa: E402
from tegen_fixture import RE01, SW, load  # noqa: E402

SWISS = HERE / "fixtures" / "swddir_swiss_s01_v1.npz"
SWISS_SHA = "6e3eea86e877484f41d39ca5f1101c4d7fbb1764ea6937faa583c27db5dc8b2f"  # proofs/w41_swddir/s01_summary.json
CALLS = {"one": 0, "all": 0}


def _load_swiss():
    digest = hashlib.sha256(SWISS.read_bytes()).hexdigest()
    if digest != SWISS_SHA:
        raise ValueError(f"{SWISS}: sha256 {digest} is not the registered S01 fixture")
    with np.load(SWISS, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _spy():
    one, all_ = B.sw_band_flux_sums, B.sw_allband_flux_sums

    def spy_one(*a, **k):
        CALLS["one"] += 1
        return one(*a, **k)

    def spy_all(*a, **k):
        CALLS["all"] += 1
        return all_(*a, **k)
    B.sw_band_flux_sums, B.sw_allband_flux_sums = spy_one, spy_all
    return lambda: (setattr(B, "sw_band_flux_sums", one), setattr(B, "sw_allband_flux_sums", all_))


def _apply(variant):
    """Installs one mutant (returns an undo callable)."""

    if variant == "kernel_scaled_direct":
        source = Path(B.__file__).read_text()
        assert source.count("dn * bn") == 2, "both default kernels (constant and Thompson radii)"
        mod = types.ModuleType("rad_sw_band_sums_mutant")
        exec(compile(source.replace("dn * bn", "dn * b"), "rad_sw_band_sums_mutant.py", "exec"), mod.__dict__)
        saved = (B.sw_band_flux_sums, B.sw_allband_flux_sums)
        B.sw_band_flux_sums, B.sw_allband_flux_sums = mod.sw_band_flux_sums, mod.sw_allband_flux_sums
        return lambda: (setattr(B, "sw_band_flux_sums", saved[0]), setattr(B, "sw_allband_flux_sums", saved[1]))
    if variant == "xla_scaled_direct":
        original = R._sw_band_tile_fluxes
        R._sw_band_tile_fluxes = lambda *a, **k: original(*a, **{**k, "direct_taus": None})
        return lambda: setattr(R, "_sw_band_tile_fluxes", original)
    assert variant == "port", variant
    return lambda: None


def main(path, specs):
    swiss = "--swiss" in specs
    specs = [x for x in specs if x != "--swiss"]
    if swiss:
        fx = ae = _load_swiss()
    else:
        fx, ae = load(RE01), load(SW)
    narm = len(ae["arm_aer_opt"])
    import gpuwrf
    result = {"real_entry": bool(R._real_entry()), "band_sums": int(R._sw_band_sums_mode() or 0),
              "platform": jax.devices()[0].platform, "gpuwrf": gpuwrf.__file__, "oracle": "S01" if swiss else "AE01",
              "variants": {}}
    cloudy_day = None
    saved = {}
    for spec in specs:
        variant, _, arms = spec.partition(":")
        arms = [a for a in range(narm) if ae["arm_aer_opt"][a] == 0] if arms in ("", "all") else \
            [int(x) for x in arms.split(",")]
        assert all(ae["arm_aer_opt"][a] == 0 for a in arms), "aer_opt=0 arms only (aerosol: w41_aerosol)"
        undo = _apply(variant)
        unspy = _spy()
        jax.clear_caches()
        CALLS.update(one=0, all=0)
        try:
            port = {arm: S._solve(fx, ae, arm, False) for arm in arms}
        finally:
            unspy()
            undo()
            jax.clear_caches()
        rows = {}
        for arm in arms:
            port[arm]["swddif"] = port[arm]["swdnb"] - port[arm]["swddir"]
            cloudy_day = (fx["category"] != "clear") & (ae["coszen"][:, arm] > 0.05)
            bound, raw, _ = S._metrics(port[arm], ae, arm, np.ones(cloudy_day.shape, bool))
            rows[str(arm)] = {"bound": bound, "bound_cloudy_day": S._metrics(port[arm], ae, arm, cloudy_day)[0],
                              "n_cloudy_day": int(cloudy_day.sum()),
                              "swddir_max_abs_err": float(np.max(np.abs(port[arm]["swddir"] - ae["swddir"][:, arm])))}
            saved[f"{variant}_surface_direct_{arm}"] = port[arm]["swddir"]
            saved[f"{variant}_surface_down_{arm}"] = port[arm]["swdnb"]
        result["variants"][variant] = {"arms": rows, "band_kernel_calls": dict(CALLS)}
        Path(path).write_text(json.dumps(result, indent=1) + "\n")
        np.savez(str(path) + ".npz", **saved)
    print(json.dumps({v: {a: {k: round(x, 3) for k, x in r["bound"].items()} for a, r in d["arms"].items()}
                      for v, d in result["variants"].items()}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
