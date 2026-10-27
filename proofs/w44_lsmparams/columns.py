"""Retained real WRF land-column workloads plus an explicit snow stress column.

These inputs are not an oracle. Changed-parameter truth comes from pristine
Fortran on the same input, with source and table hashes in its receipt.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def source_columns():
    day = json.loads((ROOT / "proofs/noahmp/savepoints_all.json").read_text())["columns"]
    cold = json.loads((ROOT / "tests/v025/b_noahmp/fixtures/savepoints_energy_frozen.json").read_text())["columns"]
    selected = [copy.deepcopy(day[i]) for i in (0, 2, 5, 7)] + [copy.deepcopy(cold[1])]
    # The retained energy-only cold fixture omitted these caller fields. Use
    # its registered offline geometry and explicit prescribed defaults; these
    # are real-structured full-step workloads, not old full-step WRF outputs.
    selected[-1].update(zsoil=[-.05, -.25, -.7, -1.5], dt=90., dx=1000.,
                        lat_rad=float(np.deg2rad(28.)), shdmax=cold[1]["shdfac"], tbot=285.,
                        scope="real WN3 frozen state/forcing; prescribed missing caller metadata")
    snow = copy.deepcopy(selected[-1])
    snow["name"] = "real_frozen_forest_with_structured_snowpack"
    s = snow["state_in"]
    s.update(isnow=-1, snowh=.3, sneqv=75., sneqvo=75., albold=.75,
             snice=[0., 0., 70.], snliq=[0., 0., 5.])
    s["stc"][2] = 268.
    s["zsnso"] = [0., 0., -.3] + [depth - .3 for depth in snow["zsoil"]]
    snow["scope"] = "real frozen WN3 land column; snowpack added as explicit E95 stress input"
    selected.append(snow)
    return selected


def frozen_soil_columns():
    """Retained real WN3 frozen-soil columns (top soil layer below 273.15 K).

    The cold cohort above has only a frozen SURFACE (soil 273.44-281.84 K), so
    PHASECHANGE never reads BEXP/PSISAT/SMCMAX there (critic F5, E95/E154).
    frozen_grass10 (270.35 K) and frozen_barren16 (265.53 K) exercise soil ice.
    Caller metadata missing from the energy-only fixture is prescribed exactly as
    for the cold cohort; these are real-structured workloads, not WRF outputs.
    """
    cold = json.loads((ROOT / "tests/v025/b_noahmp/fixtures/savepoints_energy_frozen.json").read_text())["columns"]
    selected = [copy.deepcopy(row) for row in cold if row["name"] in ("frozen_grass10", "frozen_barren16")]
    assert [row["name"] for row in selected] == ["frozen_barren16", "frozen_grass10"]
    for row in selected:
        row.update(zsoil=[-.05, -.25, -.7, -1.5], dt=90., dx=1000., lat_rad=float(np.deg2rad(28.)),
                   shdmax=row["shdfac"], tbot=285.,
                   scope="real WN3 frozen-soil state/forcing; prescribed missing caller metadata")
    # Dry frozen columns never move water into soil in one step, so Schaake
    # infiltration (KDT from DKSAT, FRZX from SMCMAX) and ice-impeded
    # conductivity stay idle. Explicit E95 stress: warm rain on frozen ground.
    rain = copy.deepcopy(selected[0])
    rain["name"] = "real_frozen_barren16_with_structured_warm_rain"
    # opt_snf=1 (Jordan): SFCTMP > T0+2.5 K gives all-liquid precipitation; 10 mm/h as a rate.
    rain["forcing"].update(sfctmp=276.15, prcpnonc=10. / 3600.)
    rain["scope"] = "real frozen WN3 soil; warm rain forcing added as explicit E95 stress input"
    # The retained frozen layers hold LESS liquid than the opt_frz=1 supercooled limit (barren16 L1
    # .0674 < .0971), so PHASECHANGE never freezes and its BEXP/PSISAT/SMCMAX limit never binds
    # (critic F5'). Declared wet-frozen stress: top-layer SH2O = SMC = .12, above the limit in every
    # x1.1 arm (<= .1114) and below the one-step energy-limited freeze (~.129).
    wet = copy.deepcopy(selected[0])
    wet["name"] = "real_frozen_barren16_wet_top_layer_stress"
    wet["state_in"]["smc"][0] = wet["state_in"]["sh2o"][0] = .12
    wet["scope"] = "real frozen WN3 column; top-layer liquid .12 added as explicit E95 phase-change stress"
    wet_rain = copy.deepcopy(wet)
    wet_rain["name"] = "real_frozen_barren16_wet_top_layer_with_structured_warm_rain"
    wet_rain["forcing"].update(sfctmp=276.15, prcpnonc=10. / 3600.)
    wet_rain["scope"] = "wet-frozen stress column plus warm rain: ice-impeded infiltration/FRZX stress"
    return selected + [rain, wet, wet_rain]


def build(*, cold=False, options=None, rows=None):
    from gpuwrf.config.paths import wrf_run_dir
    from gpuwrf.physics.noahmp.noahmp_driver import build_energy_params
    from gpuwrf.physics.noahmp.precision import real_tree
    from gpuwrf.physics.noahmp.tables import load_noahmp_parameters
    spec = importlib.util.spec_from_file_location("lsm_energy_inputs", ROOT / "proofs/noahmp/energy_savepoint_gate.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    rows = (source_columns()[4:] if cold else source_columns()[:4]) if rows is None else rows
    items = []
    for row in rows:
        land, forcing, static = fixture.build_state(row), fixture.build_forcing(row), fixture.build_static(row)
        zsoil = np.asarray(row["zsoil"])
        static = static.replace(parameters=load_noahmp_parameters(wrf_run_dir()),
                                lat=jnp.full((1, 1), np.degrees(row["lat_rad"])),
                                shdmax=jnp.full((1, 1), row["shdmax"]),
                                shdfac=jnp.full((1, 1), row["shdfac"]),
                                zsoil=jnp.asarray(zsoil), dzs=jnp.asarray(-np.diff(np.r_[0., zsoil])),
                                tbot=jnp.full((1, 1), row["tbot"]))
        items.append((land, forcing, static))

    def combine(*values):
        first = values[0]
        if hasattr(first, "shape") and first.ndim >= 2 and first.shape[-2:] == (1, 1):
            return jnp.concatenate(values, axis=-1)
        return first

    land, forcing, static = real_tree(jax.tree.map(combine, *items))
    # Columns within a cohort have the same day and vertical geometry. Cold and
    # warm regimes deliberately remain separate, with their original clocks.
    ep, rp = real_tree(build_energy_params(static, land.tv.shape))
    nl = SimpleNamespace(use_noahmp=True, noahmp_static=static, noahmp_land=land,
                         noahmp_energy_params=ep, noahmp_rad_params=rp, noahmp_options=options)
    return land, forcing, static, ep, rp, nl, rows
