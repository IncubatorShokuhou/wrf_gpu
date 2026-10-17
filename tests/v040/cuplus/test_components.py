"""KF-CuP components vs exact pristine WRF, not full-driver qualification."""

from functools import partial
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import cumulus_kfcup_support as cup


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "proofs/v040/cuplus/fixtures.json"
DATA = json.loads(FIXTURE.read_text())
JFD = jax.jit(cup.cup_jfd, static_argnames=("num_bins",))
CLOUD = jax.jit(cup.cup_cloud_fraction, static_argnames=("ishall",))
THERMO = jax.jit(lambda t, p, q: {"cp": cup.find_cp(q), "rs": cup.find_rs(t, p), "rsi": cup.find_rsi(t, p)})
RADIATION = jax.jit(cup.cup_radiation_feedback)


def evaluate(case):
    f = lambda x: jnp.asarray(x, jnp.float32)
    if case["routine"] == "cup_jfd":
        sfc, ez, ss, se, tb, rb = map(f, case["scalars"])
        return JFD(sfc, ez, ss, se, case["num_bins"], tb, rb)
    if case["routine"] == "thermodynamics":
        return THERMO(f(case["temperature"]), f(case["pressure"]), f(case["qv"]))
    if case["routine"] == "cup_radiation":
        return RADIATION(*(f(case[k]) for k in ("cldfra","qc","cldfra_cup")),
                         jnp.asarray(case["cldfra1_flag"],jnp.int32),
                         **{k:f(case[k]) for k in ("ishall","cloud_bottom","cloud_top")},
                         icloud=jnp.int32(case["icloud"]), shallowcu_forced_ra=case["shallowcu_forced_ra"])
    return CLOUD(*(f(case[key]) for key in ("qlg", "qig", "qv", "temperature", "height", "pressure")),
                 cloud_bottom=case["cloud_bottom"], cloud_top=case["cloud_top"], ishall=case["ishall"],
                 **{key: f(case[key]) for key in ("wstar", "wparcel", "pblh", "active_frac", "mfup")})


def compare(actual, reference, tolerance):
    assert actual.keys() == reference.keys(), "original branch defined-output contract"
    for name in reference:
        a, r = np.asarray(actual[name]), np.asarray(reference[name], np.float32)
        assert a.dtype == np.float32, f"{name}: WRF REAL carry must stay fp32"
        assert a.shape == r.shape, name
        assert np.isfinite(a).all(), f"nonfinite candidate {name}"
        assert np.isfinite(r).all(), f"nonfinite pristine reference {name}"
        np.testing.assert_allclose(a, r, rtol=tolerance["relative"], atol=tolerance["absolute"], err_msg=name)


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda c: c["id"])
def test_component_pristine(case):
    family = {"cup_jfd": "jfd", "cupCloudFraction": "cloud", "thermodynamics": "thermodynamics",
              "cup_radiation":"radiation"}[case["routine"]]
    compare(evaluate(case), case["output"], DATA["tolerances"][family])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("arm", ["candidate", "reference"])
def test_gate_rejects_nonfinite(bad, arm):
    a, r = {"x": np.float32(1)}, {"x": np.float32(1)}
    (a if arm == "candidate" else r)["x"] = np.float32(bad)
    with pytest.raises(AssertionError, match="nonfinite"):
        compare(a, r, DATA["tolerances"]["cloud"])


def test_fixture_has_active_regimes():
    cases = {c["id"]: c for c in DATA["cases"]}
    assert len(cases) == 45
    for name in ("deep", "shallow", "shallow_tau_cap", "shallow_upper_cap"):
        assert np.count_nonzero(cases["cloud_"+name]["output"]["cldfra"]) > 0
    assert cases["cloud_shallow_tau_cap"]["output"]["taucloud"] == 1800.
    assert max(cases["cloud_shallow_lower_cap"]["output"]["cldfra"]) == np.float32(.01)
    assert max(cases["cloud_shallow_upper_cap"]["output"]["cldfra"]) == 1.
    assert np.count_nonzero(cases["jfd_normal"]["output"]["jfd"]) > 1
    assert "Components only" in DATA["scope"]


def test_does_not_invent_undefined_wrf_outputs():
    for case in DATA["cases"]:
        if case["routine"] == "cupCloudFraction":
            out = evaluate(case)
            assert "cldfratend" not in out
            if case["ishall"] == 2:
                assert set(out) == {"cldfra"}
            elif case["ishall"] == 0:
                assert set(out) == {"cldfra", "tstar"}


def test_batched_columns_match_original():
    cases = [c for c in DATA["cases"] if c["routine"] == "cupCloudFraction" and c["ishall"] == 1]
    arrays = {key: jnp.asarray([c[key] for c in cases], jnp.float32)
              for key in ("qlg", "qig", "qv", "temperature", "height", "pressure", "wstar", "wparcel", "pblh", "active_frac", "mfup")}

    def column(qlg, qig, qv, temperature, height, pressure, wstar, wparcel, pblh, active_frac, mfup, top):
        return cup.cup_cloud_fraction(qlg, qig, qv, temperature, height, pressure,
                                     cloud_bottom=1, cloud_top=top, ishall=1, wstar=wstar,
                                     wparcel=wparcel, pblh=pblh, active_frac=active_frac, mfup=mfup)

    output = jax.jit(jax.vmap(column))(*arrays.values(), jnp.asarray([c["cloud_top"] for c in cases], jnp.int32))
    for i, case in enumerate(cases):
        compare({k: v[i] for k,v in output.items()}, case["output"], DATA["tolerances"]["cloud"])


def test_component_jaxpr_has_no_double():
    # These original procedures contain no DOUBLE islands. Check arithmetic
    # operand/result types as well as the output contract, not just a string
    # search which can include weak Python constants.
    case = next(c for c in DATA["cases"] if c["id"] == "cloud_shallow")
    arrays = [jnp.asarray(case[k], jnp.float32) for k in ("qlg", "qig", "qv", "temperature", "height", "pressure")]
    function = partial(cup.cup_cloud_fraction, cloud_bottom=1, cloud_top=3, ishall=1,
                       **{k: jnp.asarray(case[k], jnp.float32) for k in ("wstar", "wparcel", "pblh", "active_frac", "mfup")})
    hlo = jax.jit(function).lower(*arrays).compiler_ir(dialect="stablehlo")
    assert "f64" not in str(hlo)
    jfd_hlo = JFD.lower(*[jnp.float32(x) for x in (1200, -800, .002, .8)], 9, jnp.float32(.1), jnp.float32(.0001)).compiler_ir(dialect="stablehlo")
    assert "f64" not in str(jfd_hlo)
    a = jnp.ones(6,jnp.float32)
    radiation_hlo = RADIATION.lower(a,a,a,jnp.ones(6,jnp.int32),ishall=jnp.float32(1),cloud_bottom=jnp.float32(1),
                                    cloud_top=jnp.float32(3),icloud=jnp.int32(1),
                                    shallowcu_forced_ra=False).compiler_ir(dialect="stablehlo")
    assert "f64" not in str(radiation_hlo)


def test_source_provenance_and_packaged_asset():
    assert FIXTURE.is_file()
    assert len(DATA["source_commit"]) == 40
    assert set(DATA["source_sha256"]) == {"phys/module_cu_kfcup.F", "share/module_model_constants.F",
                                          "phys/module_radiation_driver.F"}
    assert all(len(v) == 64 for v in DATA["source_sha256"].values())
    # Immutable companion checksum prevents a missing/edited fixture from
    # being silently replaced by a newly generated self-reference.
    expected = FIXTURE.with_suffix(".sha256").read_text().split()[0]
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == expected


def test_invalid_static_configuration_refused():
    with pytest.raises(ValueError, match="num_bins"):
        cup.cup_jfd(1, 2, 3, 4, 0, .1, .001)
    a = jnp.ones(6, jnp.float32)
    with pytest.raises(ValueError, match="ishall"):
        cup.cup_cloud_fraction(a,a,a,a,a,a,cloud_bottom=1,cloud_top=3,ishall=3,
                              wstar=1,wparcel=1,pblh=1,active_frac=.1,mfup=a)


@pytest.mark.parametrize("code", (7,10,11))
def test_full_schemes_disclose_pending_gpu_qualification(code):
    from gpuwrf.io.scheme_catalog import classify_scheme, SupportStatus
    support=classify_scheme("cu_physics",code)
    assert support.status is SupportStatus.IMPLEMENTED
    assert 'CPU-oracle-qualified' in support.reason and 'pending' in support.reason
