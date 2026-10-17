"""t-cfix-pblmix F23 / audit B5: QNSE SICE (WRF XICE) at a sea-ice LAND point.

WRF QNSEPBL reads SICE only on land (module_bl_qnsepbl.F:488-499): SNOW>0 .OR. SICE>0.5
scales QFC1 by RLIVWV (sublimation) in the QSFC/QZ0 diagnosis.  Oracle: UNMODIFIED WRF
QNSESFC->QNSEPBL column (proofs/s2_qnse pipeline; the driver reproduces the committed
cases 1-15 byte-for-byte) at XLAND=1, SNOW=0 with XICE=1 and XICE=0
(proofs/t_cfix_pblmix/qnse_sice_fixture.json).  The port kernel carries the branch; the
operational caller passes SICE=0 because the port has no sea-ice surface path at all
(proofs/t_cfix_pblmix/qnse_sice_applicability.json) -- exact on every ice-free domain.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_v040_qnse import FUNCTIONS, kwargs  # noqa: E402
from real4_literal import real4_backend  # noqa: E402  (proofs/s2_qnse, put on sys.path by test_v040_qnse)
from gpuwrf.physics import pbl_qnse, sfclay_qnse, qnse_edmf  # noqa: E402

FX = json.loads((ROOT / "proofs/t_cfix_pblmix/qnse_sice_fixture.json").read_text())


def _port(case, **override):
    inp = kwargs(case, "pbl", jnp.float32)
    inp.update({k: jnp.float32(v) for k, v in override.items()})
    with real4_backend(pbl_qnse, sfclay_qnse, qnse_edmf):
        return FUNCTIONS["pbl"](**inp)


def _byte_mismatches(result, truth):
    bad, compared = [], 0
    for field, value in result.items():
        if field in truth:
            compared += 1
            got = np.asarray(value)
            assert np.isfinite(got).all(), field
            if got.tobytes() != np.asarray(truth[field], got.dtype).tobytes():
                bad.append(field)
    assert compared >= 15
    return bad


def test_fixture_isolates_the_sice_branch():
    a, b = FX["cases"]["fp32"]["sice1"], FX["cases"]["fp32"]["sice0"]
    assert FX["reproduces_committed_cases_1_15"] and a["INPUT"]["xland"] == 1.0 and a["INPUT"]["snow"] == 0.0
    assert a["INPUT"]["sice"] == 1.0 and b["INPUT"]["sice"] == 0.0
    assert [k for k in a["INPUT"] if a["INPUT"][k] != b["INPUT"][k]] == ["sice"]
    assert {"qsfc", "qz0", "RQVBLTEN"} <= {k for k in a["OUTPUT"] if a["OUTPUT"][k] != b["OUTPUT"][k]}


@pytest.mark.parametrize("name", ["sice1", "sice0"])
def test_port_qnsepbl_matches_pristine_real4_bytes(name):
    case = FX["cases"]["fp32"][name]
    assert _byte_mismatches(_port(case), case["OUTPUT"]) == []


def test_dropping_sice_misses_pristine():
    case = FX["cases"]["fp32"]["sice1"]
    assert {"qsfc", "RQVBLTEN"} <= set(_byte_mismatches(_port(case, sice=0.0), case["OUTPUT"]))


# ---------------------------------------------------------------------------------------------
# Admission (rv-t-cfix QS1): the port has no XICE / sea-ice surface path, so QNSE is admitted
# only on sea-ice-free input -- refused before any model/device work in BOTH real loaders.
# ---------------------------------------------------------------------------------------------
class _Run:
    def __init__(self, fields):
        self.fields = fields

    def wrfinput_variables(self, domain):
        if self.fields is None:
            raise FileNotFoundError(f"wrfinput_{domain}")
        return list(self.fields)

    def load_wrfinput(self, domain, var, lazy=True):
        return self.fields[var]


def _ice(value, key="SEAICE"):
    field = np.zeros((1, 4, 5), np.float32)
    field[0, 2, 3] = value
    return {key: field, "IVGTYP": np.full((1, 4, 5), 15, np.int32)}


@pytest.mark.parametrize("fields,bl,refused", [
    (_ice(1.0), 4, True), (_ice(0.3), 4, True), (_ice(1.0, "XICE"), 4, True), (_ice(np.nan), 4, True),
    (_ice(0.0), 4, False),             # inland glaciers (IVGTYP=ISICE) with SEAICE=0 stay admitted
    ({"IVGTYP": _ice(0.0)["IVGTYP"]}, 4, False), (_ice(1.0), 5, False), (_ice(1.0), 1, False),
    (None, 4, False),                  # no wrfinput: nothing to inspect, the loader itself needs the file
])
def test_qnse_sea_ice_admission_predicate(fields, bl, refused):
    from gpuwrf.io.qnse_admission import refuse_qnse_sea_ice
    assert _outcome(lambda: refuse_qnse_sea_ice(_Run(fields), "d01", bl)) == ("refused" if refused else "admitted")


SWISS = ROOT / "examples/switzerland_d01"


class _Reached(Exception):
    pass


def _outcome(call):
    """Explicit outcome so a missing refusal is an assertion failure, never a foreign exception (E202)."""
    try:
        call()
    except _Reached:
        return "reached"
    except ValueError as exc:
        return "refused" if "QNSE" in str(exc) else f"other ValueError: {exc}"
    return "admitted"


def _qnse_swiss_case(tmp_path, monkeypatch, *, sea_ice):
    import re
    from gpuwrf.io.gen2_accessor import Gen2Run
    case = tmp_path / "case"
    case.mkdir()
    for f in SWISS.iterdir():
        if f.is_file() and f.name != "namelist.input":
            (case / f.name).symlink_to(f)
    text = (SWISS / "namelist.input").read_text()
    for key in ("bl_pbl_physics", "sf_sfclay_physics"):
        text = re.sub(rf"^\s*{key}\s*=.*$", f" {key} = 4,", text, flags=re.M | re.I)
    (case / "namelist.input").write_text(text)
    real = Gen2Run.load_wrfinput
    seen = {}

    def load(self, domain, var, lazy=True):
        out = real(self, domain, var, lazy=lazy)
        if var in ("SEAICE", "XICE"):
            out = np.array(out, copy=True)
            xland = np.asarray(real(self, domain, "XLAND", lazy=False))
            ivg = np.asarray(real(self, domain, "IVGTYP", lazy=False))
            seen["glaciers"] = int((ivg == 15).sum())
            if sea_ice:   # one real LAND column becomes a sea-ice point (the critic's counterfactual)
                out[np.unravel_index(int(np.argmax((xland < 1.5) & (ivg != 15))), out.shape)] = 1.0
        return out

    monkeypatch.setattr(Gen2Run, "load_wrfinput", load)
    return case, seen


@pytest.mark.parametrize("pipeline", ["nested", "daily"])
@pytest.mark.parametrize("sea_ice", [True, False], ids=["sea_ice", "ice_free_glaciers"])
def test_real_loaders_refuse_qnse_sea_ice_before_model_work(tmp_path, monkeypatch, pipeline, sea_ice):
    from gpuwrf.integration import daily_pipeline as daily, nested_pipeline as nested
    if not (SWISS / "wrfinput_d01").exists():
        pytest.skip("Swiss example case unavailable")
    case, seen = _qnse_swiss_case(tmp_path, monkeypatch, sea_ice=sea_ice)

    def reached(*a, **kw):
        raise _Reached
    mod = nested if pipeline == "nested" else daily
    monkeypatch.setattr(mod, "build_replay_case", reached)
    if pipeline == "nested":
        call = lambda: nested._load_domains(  # noqa: E731
            nested.NestedPipelineConfig(case, tmp_path / "out", tmp_path / "proof", hours=1, max_dom=1), ("d01",))
    else:
        call = lambda: daily._build_real_case(daily.DailyPipelineConfig(run_id=str(case), domain="d01"))  # noqa: E731
    assert _outcome(call) == ("refused" if sea_ice else "reached")   # refusal precedes build_replay_case
    assert seen["glaciers"] == 22   # the real input's inland glaciers (SEAICE=0) were inspected


# ---------------------------------------------------------------------------------------------
# QSF1 FIX: every present alias is checked (conflicts refused) and the PUBLIC source loaders
# (build_noahmp_land_state, build_replay_case) refuse too, not only the two pipelines.
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seaice,xice,expect", [
    (0.0, 1.0, "refused"), (0.0, np.nan, "refused"), (1.0, 0.0, "refused"),
    (0.0, -1.0, "refused"),          # non-positive but inconsistent aliases: ambiguous XICE
    (0.0, 0.0, "admitted"),
])
def test_every_present_alias_is_checked(seaice, xice, expect):
    from gpuwrf.io.qnse_admission import refuse_qnse_sea_ice
    fields = _ice(seaice)
    fields["XICE"] = _ice(xice, "XICE")["XICE"]
    assert _outcome(lambda: refuse_qnse_sea_ice(_Run(fields), "d02", 4)) == expect


def test_replay_history_source_is_checked(tmp_path):
    import netCDF4
    from gpuwrf.io.qnse_admission import refuse_qnse_sea_ice
    path = tmp_path / "wrfout_d01_2023-01-01_00:00:00"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", 2), ds.createDimension("south_north", 4), ds.createDimension("west_east", 5)
        var = ds.createVariable("SEAICE", "f4", ("Time", "south_north", "west_east"))
        var[:] = 0.0
        var[0, 1, 1] = 1.0
    assert _outcome(lambda: refuse_qnse_sea_ice(_Run(_ice(0.0)), "d01", 4, history_source=path)) == "refused"
    assert _outcome(lambda: refuse_qnse_sea_ice(_Run(_ice(0.0)), "d01", 5, history_source=path)) == "admitted"


@pytest.mark.parametrize("sea_ice", [True, False], ids=["sea_ice", "ice_free_glaciers"])
def test_public_noah_loader_refuses_qnse_sea_ice(tmp_path, monkeypatch, sea_ice):
    from gpuwrf.io.noahmp_land_init import build_noahmp_land_state
    if not (SWISS / "wrfinput_d01").exists():
        pytest.skip("Swiss example case unavailable")
    case, seen = _qnse_swiss_case(tmp_path, monkeypatch, sea_ice=sea_ice)
    assert _outcome(lambda: build_noahmp_land_state(case, "d01")) == ("refused" if sea_ice else "admitted")
    assert seen["glaciers"] == 22


def test_public_noah_loader_checks_field_overrides(tmp_path, monkeypatch):
    """The QS1 counterfactual form: in-memory field_overrides on an unpatched reader."""
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.noahmp_land_init import build_noahmp_land_state
    if not (SWISS / "wrfinput_d01").exists():
        pytest.skip("Swiss example case unavailable")
    case, _ = _qnse_swiss_case(tmp_path, monkeypatch, sea_ice=False)
    run = Gen2Run(case)
    xland = np.asarray(run.load_wrfinput("d01", "XLAND", lazy=False))
    seaice = np.zeros_like(xland)
    seaice[np.unravel_index(int(np.argmax(xland < 1.5)), seaice.shape)] = 1.0
    assert _outcome(lambda: build_noahmp_land_state(case, "d01", field_overrides={"SEAICE": seaice})) == "refused"


@pytest.mark.parametrize("sea_ice", [True, False], ids=["sea_ice", "ice_free_glaciers"])
def test_public_replay_builder_refuses_before_model_init(tmp_path, monkeypatch, sea_ice):
    from gpuwrf.integration.d02_replay import build_replay_case
    from gpuwrf.io.gen2_accessor import Gen2Run
    if not (SWISS / "wrfinput_d01").exists():
        pytest.skip("Swiss example case unavailable")
    case, _ = _qnse_swiss_case(tmp_path, monkeypatch, sea_ice=sea_ice)

    def reached(*a, **kw):
        raise _Reached
    monkeypatch.setattr(Gen2Run, "grid", reached)   # first model-init step after the source boundary
    outcome = _outcome(lambda: build_replay_case(case, domain="d01", standalone=True))
    assert outcome == ("refused" if sea_ice else "reached")


def test_public_replay_builder_checks_the_replay_initial_state_source(tmp_path, monkeypatch):
    from gpuwrf.integration.d02_replay import build_replay_case
    from gpuwrf.io import qnse_admission
    from gpuwrf.io.gen2_accessor import Gen2Run
    if not (SWISS / "wrfinput_d01").exists():
        pytest.skip("Swiss example case unavailable")
    case, _ = _qnse_swiss_case(tmp_path, monkeypatch, sea_ice=False)
    seen = {}

    def spy(run, domain, bl_pbl_physics=None, *, history_source=None):
        seen["source"] = history_source
        raise _Reached
    monkeypatch.setattr(qnse_admission, "refuse_qnse_sea_ice", spy)
    assert _outcome(lambda: build_replay_case(case, domain="d01", standalone=False)) == "reached"
    assert seen["source"] == Gen2Run(case).history_files("d01")[0]
