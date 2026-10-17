"""Staged single-level nudob parity; this does not qualify obs_nudge_opt=1."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.assimilation import obs_nudging as on

jax.config.update("jax_enable_x64", True)
SAVE = Path(__file__).resolve().parents[2] / "proofs/s1_obsnudge/savepoints"
MANIFEST = json.loads((SAVE / "MANIFEST.json").read_text())


def replay(name):
    meta = MANIFEST["cases"][name]
    with np.load(SAVE / name) as f:
        data = {k: np.asarray(f[k]) for k in f.files}
    options = on.SingleLevelNudgingOptions(**meta["options"])
    obs = on.SingleLevelObservations(*(jnp.asarray(data[k]) for k in on.SingleLevelObservations._fields))
    kwargs = {k: jnp.asarray(data[k]) for k in (
        "mu", "c1h", "c2h", "msfx", "msfy", "pbase", "pp", "kpbl", "vih1", "vih2", "terrain", "z")}

    @jax.jit
    def call(aten, saved, ktau, xtime, tfaci, restart, ktaur):
        return on.single_level_nudob(options, obs, aten, saved, ktau=ktau, xtime=xtime,
                                    tfaci=tfaci, restart=restart, ktaur=ktaur, **kwargs)

    aten, saved = jnp.asarray(data["initial_aten"]), jnp.asarray(data["initial_saved"])
    out = []
    for (ktau, ktaur, rest), (xtime, tfaci) in zip(data["calls"], data["times"]):
        aten, saved = call(aten, saved, ktau, xtime, tfaci, bool(rest), ktaur)
        out.append((np.asarray(aten), np.asarray(saved)))
    return options, data, np.asarray(out)


def compare(name, output=None):
    options, data, result = replay(name) if output is None else output
    expected = np.stack((data["oracle_aten"], data["oracle_saved"]), axis=1)
    assert np.isfinite(expected).all(), "non-finite pristine output"
    assert np.isfinite(result).all(), "non-finite candidate output"
    eps = np.finfo(result.dtype).eps
    innov = np.max(np.abs(data["innovation"][np.abs(data["innovation"]) <= 9e4]))
    nobs = len(data["innovation"])
    if options.ivar < 3:
        mass = data["mu"] / data["msfy" if options.ivar == 1 else "msfx"]
    else:
        mass = data["c1h"][:, None]*data["mu"] + data["c2h"][:, None]
    # Fixed, pre-registered arithmetic envelope: no scale measured from candidate errors.
    bound_aten = 16*eps*(np.abs(data["initial_aten"]) + options.coefficient*np.max(np.abs(mass))*nobs*innov)
    bound_saved = 16*eps*nobs*innov
    err = np.abs(result-expected)
    assert np.all(err[:,0] <= bound_aten), (name, "ATEN", float(err[:,0].max()))
    assert np.all(err[:,1] <= bound_saved), (name, "SAVWT", float(err[:,1].max()))
    return float(err.max())


@pytest.mark.parametrize("name", sorted(MANIFEST["cases"]))
def test_pristine_nudob(name):
    path = SAVE / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MANIFEST["cases"][name]["sha256"]
    compare(name)


def test_oracle_census_is_active():
    for name, meta in MANIFEST["cases"].items():
        c = meta["census"]
        assert c["surface"] > 0 and c["upper"] > 0
        assert c["active_tendency_values"] > 0 if not name.startswith("boundary") else c["active_tendency_values"] == 0


def test_cadence_restart_and_boundaries():
    o, d, result = replay("regime_3_r4.npz")
    np.testing.assert_array_equal(result[1,1], result[0,1])  # ktau=1: held
    assert np.any(result[3,1] != result[2,1])  # ktau=3: restart refresh (ionf=2)
    np.testing.assert_array_equal(result[:,0,:,0], np.broadcast_to(d["initial_aten"][:,0], result[:,0,:,0].shape))
    assert o.ionf == 2
    for ivar in (1,2,3,4):
        _, d, result = replay(f"boundary_{ivar}_r4.npz")
        np.testing.assert_array_equal(result[:,0], np.broadcast_to(d["initial_aten"], result[:,0].shape))
        np.testing.assert_array_equal(result[:,1], np.broadcast_to(d["initial_saved"], result[:,1].shape))


def test_missing_far_and_outside_time_do_not_change_weights():
    _, d, _ = replay("regime_3_r4.npz")
    # Independent exact temporal edges (dt offset included); future/past ramps.
    obtime = jnp.asarray([-2., -.5, 0., .5, .75, 1., 2.], jnp.float32)
    actual = on.time_weight(jnp.float32(1), jnp.float32(1), jnp.float32(1), jnp.float32(1), obtime)
    np.testing.assert_array_equal(actual, [0.,1.,1.,1.,.5,0.,0.])
    assert np.count_nonzero(np.abs(d["innovation"]) > 9e4) == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_parity_gate_rejects_nonfinite(bad):
    packet = replay("regime_1_r4.npz")
    packet[2][0,0,0,0] = bad
    with pytest.raises(AssertionError, match="non-finite candidate"):
        compare("regime_1_r4.npz", packet)


def test_vif_bias_regimes_and_caps():
    pblh = jnp.asarray([20., 1500., 5000., 1000.], jnp.float32)
    vif = jnp.asarray([50.,50.,100.,200.,-5000.,-5100.], jnp.float32)
    low, high = on.regime_vif_heights(vif, jnp.asarray([1.,2.,3.,4.]), pblh)
    np.testing.assert_array_equal(low, [50.,100.,2950.,1000.])
    np.testing.assert_array_equal(high, [100.,300.,3000.,1100.])
    assert np.isnan(on.regime_vif_heights(vif, jnp.asarray([9.]), pblh[:1])[0]).all()


def test_dynamic_ramp():
    x = jnp.asarray([0.,79.,80.,90.,100.,101.], jnp.float32)
    np.testing.assert_array_equal(on.dynamic_init_factor(x, 100, -20), [1.,1.,1.,.5,0.,0.])


@pytest.mark.parametrize("field,value", [("twindo",0), ("dx",np.nan), ("ionf",0), ("sfcfacr",np.inf), ("ivar",5)])
def test_options_refuse_invalid(field, value):
    options = on.SingleLevelNudgingOptions(3,5,11,1000.,10.,.5,.3,3e-4)
    with pytest.raises(ValueError):
        replace(options, **{field: value})


def test_non_wrf_observation_nudging_code_remains_refused():
    from gpuwrf.io.namelist_check import validate_namelist, UnsupportedSchemeError

    with pytest.raises(UnsupportedSchemeError):
        validate_namelist({"fdda": {"obs_nudge_opt": 2}})


@pytest.mark.parametrize("kind", ["r4", "r8"])
def test_pristine_time_and_vif_helpers(kind):
    with np.load(SAVE / f"helpers_{kind}.npz") as data:
        args = jnp.asarray(data["args"])
        actual = jax.jit(on.time_weight)(*(args[:,i] for i in range(5)))
        eps = np.finfo(data["args"].dtype).eps
        assert np.isfinite(actual).all() and np.isfinite(data["time_weight"]).all()
        np.testing.assert_allclose(actual, data["time_weight"], rtol=0, atol=4*eps)
        heights = jax.jit(on.vif_heights)(*(jnp.asarray(data[k]) for k in ("full","ramp","pblh")))
        for actual, key in zip(heights, ("height1","height2")):
            assert np.isfinite(actual).all() and np.isfinite(data[key]).all()
            assert np.all(np.abs(actual-data[key]) <= 4*eps*np.maximum(1,np.abs(data[key])))


@pytest.mark.parametrize("kind", ["r4", "r8"])
def test_raw_decoder_against_pristine_fortran_formats(kind):
    from gpuwrf.io.obs_domain import read_obs_domain

    actual = read_obs_domain(SAVE/"OBS_DOMAIN101")
    expected = json.loads((SAVE/f"decoded_{kind}.json").read_text())
    dtype = np.float32 if kind == "r4" else np.float64
    assert len(actual) == len(expected) == 3
    for report, reference in zip(actual, expected):
        assert report.time.isoformat() == reference["time"]
        for key in ("station_id","name","platform","source","is_sounding","bogus","measurement_count"):
            assert getattr(report,key) == reference[key]
        for key in ("latitude","longitude","elevation"):
            assert dtype(getattr(report,key)) == dtype(reference[key])
        values = [[(v.value,v.qc) for v in row] for row in report.measurements]
        np.testing.assert_array_equal(np.asarray(values,dtype),np.asarray(reference["measurements"],dtype))


@pytest.mark.parametrize("mutation,message", [
    ("truncated", "truncated"), ("nan", "non-finite"), ("inf", "non-finite"),
    ("-inf", "non-finite"), ("date", "date"), ("logical", "logical"),
    ("order", "chronological"), ("count", "measurement count"),
])
def test_raw_decoder_rejects_bad_records(tmp_path, mutation, message):
    from gpuwrf.io.obs_domain import read_obs_domain

    text = (SAVE/"OBS_DOMAIN101").read_text()
    lines = text.splitlines(keepends=True)
    if mutation == "truncated":
        lines = lines[:-1]
    elif mutation in ("nan","inf","-inf"):
        lines[4] = " " + f"{mutation:>11}" + lines[4][12:]
    elif mutation == "date":
        lines[0] = " 202301150000\n"
    elif mutation == "logical":
        lines[3] = lines[3][:48] + "   X" + lines[3][52:]
    elif mutation == "order":
        lines[5] = " 20220115000000\n"
    else:
        lines[3] = lines[3][:60] + "    0\n"
    path = tmp_path/"OBS_DOMAIN101"
    path.write_text("".join(lines))
    with pytest.raises(ValueError, match=message):
        read_obs_domain(path)


def test_fixture_manifest_and_all_files_are_intact():
    for field in ("fixture_id","source","source_commit","scenario","variables","files","license_notes"):
        assert field in MANIFEST
    for name, sha in MANIFEST["files"].items():
        assert hashlib.sha256((SAVE/name).read_bytes()).hexdigest() == sha


def test_raw_decoder_import_stays_backend_dark():
    root = Path(__file__).resolve().parents[2]
    # gpuwrf's package bootstrap can load JAX even for standard-library-only
    # submodules. The decoder must not initialize any backend or allocate arrays.
    script = """
from jax._src import xla_bridge
assert not xla_bridge._backends
def reject(*args, **kwargs):
    raise AssertionError('raw observation decoding initialized a JAX backend')
xla_bridge.get_backend = reject
xla_bridge.backends = reject
xla_bridge._get_backend_uncached = reject
from gpuwrf.io.obs_domain import read_obs_domain
assert len(read_obs_domain('proofs/s1_obsnudge/savepoints/OBS_DOMAIN101')) == 3
assert not xla_bridge._backends
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=root,
                            env={**os.environ, "PYTHONPATH": str(root/"src"), "JAX_PLATFORMS":"cpu"},
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_empty_station_set_and_disabled_switch():
    o, d, _ = replay("regime_1_r4.npz")
    kwargs = {k:jnp.asarray(d[k]) for k in ("mu","c1h","c2h","msfx","msfy","pbase","pp","kpbl","vih1","vih2","terrain","z")}
    aten,saved = jnp.asarray(d["initial_aten"]),jnp.asarray(d["initial_saved"])
    empty = on.SingleLevelObservations(*(jnp.empty((0,),jnp.float32) for _ in on.SingleLevelObservations._fields))
    a,s = jax.jit(lambda a,s:on.single_level_nudob(o,empty,a,s,ktau=0,xtime=jnp.float32(.3),**kwargs))(aten,saved)
    np.testing.assert_array_equal(a,aten)
    np.testing.assert_array_equal(s,np.zeros_like(saved))
    obs = on.SingleLevelObservations(*(jnp.asarray(d[k]) for k in on.SingleLevelObservations._fields))
    for off in (replace(o,enabled=0),replace(o,coefficient=0)):
        a,s = on.single_level_nudob(off,obs,aten,saved,ktau=0,xtime=jnp.float32(.3),**kwargs)
        np.testing.assert_array_equal(a,aten)
        assert np.any(np.asarray(s) != np.asarray(saved))


def test_nonfinite_innovation_is_not_silently_treated_as_missing():
    # WRF skips ABS(ERRF)>9e4, not the complement ABS(ERRF)<=9e4: NaN must
    # remain visible to the finite-output gate on an active station.
    o,d,_ = replay("regime_3_r4.npz")
    kwargs = {k:jnp.asarray(d[k]) for k in ("mu","c1h","c2h","msfx","msfy","pbase","pp","kpbl","vih1","vih2","terrain","z")}
    d["innovation"][0] = np.nan
    obs = on.SingleLevelObservations(*(jnp.asarray(d[k]) for k in on.SingleLevelObservations._fields))
    a,s = on.single_level_nudob(o,obs,jnp.asarray(d["initial_aten"]),jnp.asarray(d["initial_saved"]),
                              ktau=0,xtime=jnp.float32(.3),**kwargs)
    assert not np.isfinite(a).all()
    assert not np.isfinite(s).all()
