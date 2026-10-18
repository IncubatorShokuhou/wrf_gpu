"""`gpuwrf ens prepare --mode stochastic|multiphysics` presets (lane w41-stoch).

Fake-member workflow evidence only: no forecast is run. Seeds are checked against the model's
own WRF rand_seed implementation; admission against the model's own pre-JAX stochastic refusals.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import pytest

from gpuwrf.ens import presets
from gpuwrf.ens.workflows import prepare
from gpuwrf.io.namelist_parser import parse_namelist

REPO = Path(__file__).resolve().parents[1]
T0 = pd.Timestamp("2023-01-15T00:00:00Z")


def _times(ds, hours):
    ds.createDimension("Time", len(hours))
    ds.createDimension("DateStrLen", 19)
    ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array(
        [list((T0+pd.Timedelta(hours=h)).strftime("%Y-%m-%d_%H:%M:%S")) for h in hours], dtype="S1")


def swiss_like_case(tmp_path, name="case", physics=None, stoch="", e_we=43, max_dom=1):
    """The default Swiss namelist layout (43x43, 3 km, MYNN, Noah-MP) with tiny fake inputs."""
    case = tmp_path/name
    case.mkdir()
    for number in range(1, max_dom+1):
        with Dataset(case/f"wrfinput_d{number:02d}", "w") as ds:
            _times(ds, [0])
    with Dataset(case/"wrfbdy_d01", "w") as ds:
        _times(ds, [0, 3, 6])
    phys = dict(mp_physics=8, ra_lw_physics=4, ra_sw_physics=4, radt=10, sf_sfclay_physics=5,
                sf_surface_physics=4, bl_pbl_physics=5, cu_physics=0, num_soil_layers=4)
    phys.update(physics or {})
    rep = lambda v: ", ".join([str(v)]*max_dom)
    text = ("&time_control\n start_year = " + rep(2023) + ",\n start_month = " + rep(1) + ",\n start_day = " + rep(15)
            + ",\n start_hour = " + rep(0) + ",\n start_minute = " + rep(0) + ",\n start_second = " + rep(0)
            + ",\n history_interval = " + rep(60) + ",\n/\n\n&domains\n time_step = 18,\n max_dom = " + str(max_dom)
            + ",\n e_we = " + rep(e_we) + ",\n e_sn = " + rep(e_we) + ",\n e_vert = " + rep(45) + ",\n dx = 3000,\n dy = 3000,\n/\n\n"
            "&physics\n" + "".join(f" {k} = {rep(v)},\n" for k, v in phys.items()) + "/\n\n&dynamics\n gwd_opt = 0,\n/\n" + stoch)
    (case/"namelist.input").write_text(text)
    return case


def configs(plan):
    return [parse_namelist(Path(m["case"])/"namelist.input") for m in plan["members"]]


def test_host_seed_copy_matches_model_rand_seed():
    from gpuwrf.physics.stochastic_rng import seed_array
    rng = random.Random(41)
    cases = [(811, 1, datetime(2023, 1, 15, 0)), (2**31-1, 2146, datetime(2026, 12, 31, 23)),
             (-(2**31), 7, datetime(1999, 2, 28, 6)), (317, 2146, datetime(2023, 7, 25, 18))]
    cases += [(rng.randint(-(2**31), 2**31-1), rng.randint(1, 2146),
               datetime(rng.randint(1980, 2100), rng.randint(1, 12), rng.randint(1, 28), rng.randint(0, 23)))
              for _ in range(400)]
    for iseed, nens, start in cases:
        assert presets.wrf_seed_words(iseed, nens, start) == tuple(int(x) for x in seed_array(iseed, nens, start))


def test_stochastic_members_distinct_nens_unique_seeds_shared_inputs(tmp_path):
    case = swiss_like_case(tmp_path)
    plan = prepare("stochastic", tmp_path/"ens", case=case, hours=6, members=12, schemes=("skebs", "sppt", "spp"),
                   lengthscale_km=15.0)
    assert plan["mode"] == "stochastic" and len(plan["members"]) == 12
    base = parse_namelist(case/"namelist.input")
    words = set()
    for index, (member, config) in enumerate(zip(plan["members"], configs(plan))):
        stoch = config["stoch"]
        assert stoch["nens"] == index+1 and stoch["skebs"] == stoch["sppt"] == stoch["spp"] == 1
        assert stoch["lengthscale_sppt"] == stoch["lengthscale_spp_pbl"] == 15000.
        assert {k: v for k, v in config.items() if k != "stoch"} == base
        dest = Path(member["case"])
        assert (dest/"wrfinput_d01").resolve() == (case/"wrfinput_d01").resolve()
        manifest = json.loads((dest/"member_manifest.json").read_text())["preset"]
        assert manifest["nens"] == index+1
        for key, word in manifest["seed_words"].items():
            assert tuple(word) == presets.wrf_seed_words(dict(zip(presets.SEED_KEYS, presets.SEED_DEFAULTS))[key],
                                                         index+1, datetime(2023, 1, 15))
            words.add(tuple(word))
    assert len(words) == 12*5   # SKEBS + SPPT + three SPP streams per member, all distinct
    assert plan["preset"]["schemes"] == ["skebs", "sppt", "spp"]
    assert len(list((tmp_path/"ens"/"preset_namelists").iterdir())) == 12


def test_default_schemes_and_nens_start(tmp_path):
    case = swiss_like_case(tmp_path)
    plan = prepare("stochastic", tmp_path/"ens", case=case, hours=6, members=3, lengthscale_km=20, nens_start=11)
    stoch = [c["stoch"] for c in configs(plan)]
    assert [s["nens"] for s in stoch] == [11, 12, 13]
    assert all(s["skebs"] == s["sppt"] == 1 and "spp" not in s for s in stoch)


def test_existing_tuning_keys_kept_and_nested_switch_arrays(tmp_path):
    case = swiss_like_case(tmp_path, stoch="\n&stoch\n tot_backscat_psi = 2.0e-5,\n/\n", max_dom=2)
    plan = prepare("stochastic", tmp_path/"ens", case=case, hours=6, members=2, schemes=("skebs",))
    for config in configs(plan):
        assert config["stoch"]["tot_backscat_psi"] == 2.0e-5 and config["stoch"]["skebs"] == [1, 1]


@pytest.mark.parametrize("bad", ["one_member", "unknown_scheme", "sppt_ysu", "spp_ysu", "default_scale",
                                 "explicit_scale", "already_on", "nens_overflow", "rows_in_stochastic"])
def test_stochastic_refusals_write_nothing(tmp_path, bad):
    physics = {"sppt_ysu": dict(bl_pbl_physics=1, sf_sfclay_physics=1),
               "spp_ysu": dict(bl_pbl_physics=1, sf_sfclay_physics=1)}.get(bad)
    stoch = {"already_on": "\n&stoch\n skebs = 1,\n/\n",
             "explicit_scale": "\n&stoch\n lengthscale_sppt = 400000.,\n/\n"}.get(bad, "")
    case = swiss_like_case(tmp_path, physics=physics, stoch=stoch)
    kwargs = dict(case=case, hours=6, members=3, schemes=("skebs", "sppt"), lengthscale_km=15.)
    kwargs.update({"one_member": dict(members=1), "unknown_scheme": dict(schemes=("skebs", "multi_perturb")),
                   "sppt_ysu": dict(schemes=("sppt",)), "spp_ysu": dict(schemes=("spp",)),
                   "default_scale": dict(schemes=("spp",), lengthscale_km=None), "explicit_scale": dict(lengthscale_km=None),
                   "nens_overflow": dict(nens_start=2145), "rows_in_stochastic": dict(physics_rows=("P00", "P01"))}.get(bad, {}))
    with pytest.raises(ValueError):
        prepare("stochastic", tmp_path/"refused", **kwargs)
    assert not (tmp_path/"refused").exists()


def test_default_sppt_scale_resolved_on_swiss_grid_but_spp_pbl_default_refused(tmp_path):
    # The model's own spectral setup: 150 km resolves on 43x43 x 3 km, the 700 km SPP-PBL default does not
    # (original WRF then runs with NaN coefficients and a constant clipped pattern, v-prep c41).
    case = swiss_like_case(tmp_path)
    plan = prepare("stochastic", tmp_path/"ok", case=case, hours=6, members=2, schemes=("skebs", "sppt"))
    assert "lengthscale_sppt" not in configs(plan)[0]["stoch"]
    with pytest.raises(ValueError, match="no finite WRF spectral normalization"):
        prepare("stochastic", tmp_path/"refused", case=case, hours=6, members=2, schemes=("spp",))
    assert not (tmp_path/"refused").exists()


def test_seed_collision_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "wrf_seed_words", lambda iseed, nens, start: (1, 2, 3, 4, 1, 2, 3, 4))
    with pytest.raises(ValueError, match="identical"):
        prepare("stochastic", tmp_path/"refused", case=swiss_like_case(tmp_path), hours=6, members=2,
                schemes=("skebs",))
    assert not (tmp_path/"refused").exists()


def test_multiphysics_rows_only_physics_changes(tmp_path):
    case = swiss_like_case(tmp_path)
    plan = prepare("multiphysics", tmp_path/"mp", case=case, hours=6, members=len(presets.MULTIPHYSICS))
    base = parse_namelist(case/"namelist.input")
    assert [r["row"] for r in plan["preset"]["table"]] == [r[0] for r in presets.MULTIPHYSICS]
    for (row, settings, evidence, e41), member, config in zip(presets.MULTIPHYSICS, plan["members"], configs(plan)):
        assert {k: v for k, v in config.items() if k != "physics"} == {k: v for k, v in base.items() if k != "physics"}
        assert all(config["physics"].get(k) == v for k, v in settings.items()), (row, settings)
        assert {k: v for k, v in config["physics"].items() if k not in settings} == \
               {k: v for k, v in base["physics"].items() if k not in settings}
        info = json.loads((Path(member["case"])/"member_manifest.json").read_text())["preset"]
        assert info["physics_row"] == row and evidence in info["evidence"] and "qualification_note" not in info
    assert len({tuple(sorted(r[1].items())) for r in presets.MULTIPHYSICS}) == len(presets.MULTIPHYSICS)


def test_multiphysics_plus_skebs_and_explicit_rows(tmp_path):
    case = swiss_like_case(tmp_path)
    plan = prepare("multiphysics", tmp_path/"mp", case=case, hours=6, physics_rows=("P07", "P00", "P04"),
                   schemes=("skebs",))
    got = configs(plan)
    assert [c["physics"]["bl_pbl_physics"] for c in got] == [1, 5, 9]
    assert [c["stoch"]["nens"] for c in got] == [1, 2, 3] and all(c["stoch"]["skebs"] == 1 for c in got)


@pytest.mark.parametrize("bad", ["too_many", "unknown_row", "duplicate_row", "count_mismatch", "sppt_with_ysu_row", "one_row"])
def test_multiphysics_refusals(tmp_path, bad):
    case = swiss_like_case(tmp_path)
    kwargs = {"too_many": dict(members=len(presets.MULTIPHYSICS)+1), "unknown_row": dict(physics_rows=("P00", "P99")),
              "duplicate_row": dict(physics_rows=("P00", "P00")), "count_mismatch": dict(physics_rows=("P00", "P01"), members=3),
              "sppt_with_ysu_row": dict(physics_rows=("P00", "P07"), schemes=("sppt",), lengthscale_km=15.),
              "one_row": dict(physics_rows=("P00",))}[bad]
    with pytest.raises(ValueError):
        prepare("multiphysics", tmp_path/"refused", case=case, hours=6, **kwargs)
    assert not (tmp_path/"refused").exists()


def test_non_noahmp_base_is_labelled_unqualified(tmp_path):
    case = swiss_like_case(tmp_path, physics=dict(sf_surface_physics=2))
    plan = prepare("multiphysics", tmp_path/"mp", case=case, hours=6, members=2)
    info = json.loads((Path(plan["members"][0]["case"])/"member_manifest.json").read_text())["preset"]
    assert "UNQUALIFIED" in info["qualification_note"]


def test_preset_options_refused_for_classic_modes(tmp_path):
    case = swiss_like_case(tmp_path)
    with pytest.raises(ValueError, match="preset mode"):
        prepare("physics", tmp_path/"x", case=case, namelists=[case/"namelist.input"]*2, hours=6, members=2)


def test_real_swiss_members_pass_model_admission(tmp_path):
    from gpuwrf.io.namelist_binding import require_namelist_honoured
    from gpuwrf.io.namelist_check import _coerce_config, validate_operational_namelist
    swiss = REPO/"examples/switzerland_d01"
    base_text, base = (swiss/"namelist.input").read_text(), parse_namelist(swiss/"namelist.input")
    built = presets.build_members("stochastic", base_text, base, members=3, schemes=("skebs", "sppt", "spp"),
                                  lengthscale_km=15., start=presets.parse_start(base))
    rows = [r[0] for r in presets.MULTIPHYSICS]
    built += presets.build_members("multiphysics", base_text, base, rows=rows, schemes=None, start=presets.parse_start(base))
    built += presets.build_members("multiphysics", base_text, base, rows=rows[:2], schemes=("skebs",),
                                   start=presets.parse_start(base))
    for index, (text, info) in enumerate(built):
        path = tmp_path/f"m{index}.input"
        path.write_text(text)
        validate_operational_namelist(path)
        require_namelist_honoured(_coerce_config(path), ("d01",), replay=False)


def test_offline_preset_cli_never_imports_jax(tmp_path):
    case = swiss_like_case(tmp_path)
    code = '''import importlib.abc,sys
class RefuseJax(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='jax' or fullname.startswith('jax.'):
            raise AssertionError('Offline ens must not import JAX')
sys.meta_path.insert(0,RefuseJax())
sys.argv=['gpuwrf',*sys.argv[1:]]
from gpuwrf.cli import main
assert main(sys.argv[1:])==0
assert not any(k=='jax' or k.startswith('jax.') for k in sys.modules)
'''
    env = dict(os.environ, PYTHONPATH=str(REPO/"src"), JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
    for args in (["--mode", "stochastic", "--members", "3", "--schemes", "skebs,sppt,spp", "--lengthscale-km", "15"],
                 ["--mode", "multiphysics", "--physics-rows", "P00,P02", "--schemes", "skebs"]):
        out = tmp_path/f"cli_{args[1]}"
        result = subprocess.run([sys.executable, "-c", code, "ens", "prepare", *args, "--case", str(case),
                                 "--hours", "6", "--output-dir", str(out)], cwd=REPO, env=env,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        assert json.loads((out/"plan.json").read_text())["preset"]["mode"] == args[1]
