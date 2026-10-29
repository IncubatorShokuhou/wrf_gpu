"""Invented geometry ONLY; tests exercise the estimator, never WRF fidelity."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from gpuwrf.optimize import twin

IDENTITY = twin.ForwardIdentity("a"*64, "b"*64, "test-t2:K/wind:m_s-1;instantaneous;v1")
CONFIG = twin.TwinConfig(("cloud", "mixing"), (0.6, 0.6), ((-1., 1.), (-1., 1.)), seed=19)
TRUTH = np.array([0.3, -0.25])


def design():
    # Outage removes all station B rows, the ONLY response to the second knob.
    stations = pd.DataFrame({"station_uid": ["invented-A", "invented-B"],
                             "lat": [28., 28.2], "lon": [-16., -16.4], "elevation_m": [10., 300.]})
    rows = [{"station_uid": station, "valid_time": f"2026-01-01T{hour:02d}:00:00Z", "variable": variable,
             "available": station != "invented-B"}
            for station in stations.station_uid for hour in range(4) for variable in ("t2", "wind")]
    return twin.station_design(stations, pd.DataFrame(rows))


def forward(theta, frame):
    # Uses the passed station coordinates; no implicit row index geometry.
    a = (frame.lat < 28.1).to_numpy(dtype=float)
    b = (frame.lat >= 28.1).to_numpy(dtype=float)
    matrix = np.column_stack([a, b])*3
    values = theta @ matrix.T + np.where(frame.variable.eq("t2"), 280., 5.)
    return twin.PredictionBatch(values, twin.design_keys(frame), IDENTITY)


def run(**kwargs):
    defaults = dict(design=design(), forward=forward, truth_log=TRUTH, config=CONFIG,
                    identity=IDENTITY, sigma={"t2": 0.05, "wind": 0.05}, add_noise=False)
    return twin.run_twin(**(defaults | kwargs))


def test_deterministic_recovery_and_outage_rank_loss():
    report = run()
    assert json.dumps(report, sort_keys=True) == json.dumps(run(), sort_keys=True)
    full, outage = (report["scenarios"][key] for key in ("complete", "outages"))
    assert full["status"] == "RECOVERED_SYNTHETIC_ONLY", full
    assert max(full["abs_log_error"]) < 0.01
    assert full["rank"] == 2 and full["observations"] == 16
    assert outage["rank"] == 1 and outage["observations"] == 8
    assert outage["status"] == "UNIDENTIFIABLE_FROZEN" and outage["estimate_log"] is None
    assert report["forward_calls"] == CONFIG.iterations + 3
    assert "perfect twins prove no real identification" in report["evidence"]
    assert "Undeclared forcing errors" in report["forcing_limit"]


def test_outages_reduce_information_without_destroying_rank():
    frame = design()
    frame["available"] = frame.valid_time.dt.hour.eq(0)
    out = run(design=frame)
    a, b = out["scenarios"]["complete"], out["scenarios"]["outages"]
    assert a["rank"] == b["rank"] == 2
    assert b["status"] == "RECOVERED_SYNTHETIC_ONLY"
    assert np.trace(b["information"]) < np.trace(a["information"])


def test_no_support_and_zero_sensitivity_do_not_fit():
    frame = design()
    frame["available"] = False
    assert run(design=frame)["scenarios"]["outages"]["status"] == "NO_SUPPORT"

    def constant(theta, frame):
        return replace(forward(theta, frame), values=np.zeros((len(theta), len(frame))))

    report = run(forward=constant)
    assert all(r["status"] == "UNIDENTIFIABLE_FROZEN" for r in report["scenarios"].values())
    assert report["forward_calls"] == 2


def test_collinear_parameters_freeze_before_fitting():
    def collinear(theta, frame):
        return forward(np.repeat(theta.sum(axis=1)[:, None], 2, axis=1), frame)

    report = run(forward=collinear)
    for entry in report["scenarios"].values():
        assert entry["rank"] == 1 and entry["status"] == "UNIDENTIFIABLE_FROZEN"
        assert entry["near_collinear_pairs"] == [["cloud", "mixing"]]
    assert report["forward_calls"] == 2


@pytest.mark.parametrize("field,value", [("forcing_sha256", "c"*64),
                                        ("configuration_sha256", "c"*64),
                                        ("observation_schema", "wrong-units-or-schema")])
@pytest.mark.parametrize("adapter", ["truth", "inversion"])
def test_wrong_identity_is_refused_before_eki(field, value, adapter, monkeypatch):
    def wrong(theta, frame):
        return replace(forward(theta, frame), identity=replace(IDENTITY, **{field: value}))

    def forbidden(*args, **kwargs):
        pytest.fail("EKI ran despite an input identity mismatch")

    monkeypatch.setattr(twin.eki, "initial_ensemble", forbidden)
    with pytest.raises(ValueError, match="forcing/configuration/observation schema mismatch"):
        run(**({"truth_forward": wrong} if adapter == "truth" else {"forward": wrong, "truth_forward": forward}))


@pytest.mark.parametrize("kind", ["keys", "shape", "nan", "bare_array"])
def test_wrong_prediction_schema(kind):
    def wrong(theta, frame):
        result = forward(theta, frame)
        if kind == "keys":
            return replace(result, keys=result.keys[::-1])
        if kind == "shape":
            return replace(result, values=result.values[:, :-1])
        if kind == "nan":
            return replace(result, values=result.values*np.nan)
        return result.values

    with pytest.raises(ValueError):
        run(forward=wrong)


def test_incompatible_unlabelled_forcing_leaves_large_residual():
    def wrong_truth(theta, frame):
        batch = forward(theta, frame)
        # Time-dependent forcing error outside the two constant station responses.
        return replace(batch, values=batch.values + np.where(frame.valid_time.dt.hour % 2, 2., -2.))

    report = run(truth_forward=wrong_truth)
    assert report["scenarios"]["complete"]["status"] == "MODEL_MISMATCH"


def test_in_span_unlabelled_forcing_cannot_prove_causal_identification():
    def wrong_truth(theta, frame):
        # Lying adapter: same declared identity; error EXACTLY in knob 1's span.
        return forward(theta + np.array([0.2, 0.]), frame)

    report = run(truth_forward=wrong_truth)
    full = report["scenarios"]["complete"]
    assert full["status"] == "NOT_RECOVERED"
    assert full["normalised_misfit"] < 0.1  # good fit is not a true-parameter proof
    assert full["abs_log_error"][0] > 0.15
    assert "cannot be distinguished from a parameter change" in report["forcing_limit"]


def test_noise_and_registered_correlations():
    frame = design()
    cov = np.eye(len(frame))*0.05**2
    cov[0, 1] = cov[1, 0] = 0.5*0.05**2
    report = run(covariance=cov, add_noise=True)
    assert report == run(covariance=cov, add_noise=True)
    assert report["noise"]["covariance"] == "registered full matrix"
    assert report["scenarios"]["complete"]["status"] == "RECOVERED_SYNTHETIC_ONLY"


@pytest.mark.parametrize("case", ["nonsymmetric", "nonpositive", "diagonal", "shape"])
def test_bad_covariances(case):
    cov = np.eye(len(design()))*0.05**2
    if case == "nonsymmetric":
        cov[0, 1] = 0.001
    elif case == "nonpositive":
        cov[0, 1] = cov[1, 0] = 0.01
    elif case == "diagonal":
        cov[0, 0] = 1.
    else:
        cov = cov[:2, :2]
    with pytest.raises(ValueError):
        run(covariance=cov)


@pytest.mark.parametrize("upper", [4e-17, 9e-13])
def test_critic_tiny_nonsymmetric_covariances_refused_before_forward(upper):
    # Critic F1: scaled ORIGINAL nonsymmetric case, plus correlation=9000.
    cov = np.eye(len(design()))*0.05**2*4e-14
    cov[0, 1] = upper

    def forbidden(*args):
        pytest.fail("invalid covariance reached the forward model")

    with pytest.raises(ValueError, match="symmetric.*correlations"):
        run(covariance=cov, sigma={"t2": 1e-8, "wind": 1e-8}, forward=forbidden)


def test_valid_tiny_spd_covariance_through_real_caller():
    scale = 4e-14
    cov = np.eye(len(design()))*0.05**2
    cov[0, 1] = cov[1, 0] = 0.5*0.05**2

    def scaled_forward(theta, frame):
        batch = forward(theta, frame)
        return replace(batch, values=batch.values*np.sqrt(scale))

    report = run(forward=scaled_forward, covariance=cov*scale,
                 sigma={"t2": 0.05*np.sqrt(scale), "wind": 0.05*np.sqrt(scale)}, add_noise=True)
    assert report["scenarios"]["complete"]["status"] == "RECOVERED_SYNTHETIC_ONLY"
    assert max(report["scenarios"]["complete"]["abs_log_error"]) < CONFIG.tolerance_log


@pytest.mark.parametrize("scale", [4e-14, 1., 4e14])
@pytest.mark.parametrize("case", ["nonsymmetric", "correlation", "indefinite", "singular"])
def test_covariance_admission_is_scale_invariant(scale, case):
    # The indefinite matrix has all |correlations| < 1: a pairwise bound alone
    # must not replace the joint positive-definite check.
    cov = np.eye(3)
    if case == "nonsymmetric":
        cov[0, 1] = 0.4
    elif case == "correlation":
        cov[0, 1] = cov[1, 0] = 9000.
    elif case == "indefinite":
        cov[:] = -0.9
        np.fill_diagonal(cov, 1.)
    else:
        cov[0, 1] = cov[1, 0] = 1.
    with pytest.raises(ValueError):
        twin._observation_covariance(np.full(3, np.sqrt(scale)), cov*scale)


@pytest.mark.parametrize("variances", [[1e-16]*3, [1.]*3, [1e16]*3, [1e-200, 1., 1e200]])
def test_valid_covariance_across_scales_and_mixed_units(variances):
    sd = np.sqrt(variances)
    correlation = np.array([[1., 0.2, -0.1], [0.2, 1., 0.3], [-0.1, 0.3, 1.]])
    cov = (sd[:, None]*correlation)*sd[None, :]
    # Construct exactly symmetric input despite independent scaled multiplies.
    cov = np.tril(cov)+np.tril(cov, -1).T
    original = cov.copy()
    admitted, factor = twin._observation_covariance(sd, cov)
    np.testing.assert_array_equal(cov, original)
    np.testing.assert_array_equal(admitted, original)
    np.testing.assert_allclose((factor@factor.T)/sd[:, None]/sd[None, :], correlation, atol=1e-14)


def test_roundoff_asymmetry_uses_one_triangle_without_mutating_input():
    sd = np.array([1e-8, 1e4])
    cov = np.diag(sd**2)
    cov[0, 1] = (0.2+5e-13)*sd[0]*sd[1]
    cov[1, 0] = 0.2*sd[0]*sd[1]
    original = cov.copy()
    admitted, factor = twin._observation_covariance(sd, cov)
    np.testing.assert_array_equal(cov, original)
    np.testing.assert_array_equal(admitted, admitted.T)
    np.testing.assert_allclose((factor@factor.T)/sd[:, None]/sd[None, :],
                               admitted/sd[:, None]/sd[None, :], atol=1e-14)


@pytest.mark.parametrize("sd", [1e-200, 1e200])
def test_unrepresentable_variances_are_refused(sd):
    with pytest.raises(ValueError, match="representable"):
        twin._observation_covariance(np.array([sd]), None)


@pytest.mark.parametrize("case", ["duplicate", "unknown_station", "naive_time", "truthy", "coordinate", "variable"])
def test_invalid_design(case):
    frame = design()
    stations = frame[["station_uid", "lat", "lon", "elevation_m"]].drop_duplicates()
    if case == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif case == "unknown_station":
        frame.loc[0, "station_uid"] = "missing"
    elif case == "naive_time":
        frame["valid_time"] = frame.valid_time.dt.tz_localize(None)
    elif case == "truthy":
        frame["available"] = "false"
    elif case == "coordinate":
        stations.loc[0, "lat"] = 91
    else:
        frame.loc[0, "variable"] = ""
    with pytest.raises(ValueError):
        twin.station_design(stations, frame)


@pytest.mark.parametrize("change", [{"names": ("a", "b", "c")}, {"prior_sd_log": (0., 0.6)},
                                    {"bounds_log": ((0., 1.), (-1., 1.))}, {"members": 2},
                                    {"rank_rtol": 1.}, {"iterations": 0}])
def test_invalid_config(change):
    with pytest.raises(ValueError):
        replace(CONFIG, **change)


def test_dense_budget_and_truth_bounds():
    with pytest.raises(ValueError, match="observation budget"):
        run(config=replace(CONFIG, max_observations=2))
    with pytest.raises(ValueError, match="truth_log"):
        run(truth_log=[2., 0.])
    with pytest.raises(ValueError, match="boolean"):
        run(add_noise="false")


def test_forward_mutation_cannot_change_design_or_truth():
    frame = design()
    before = frame.copy(deep=True)
    original_truth = TRUTH.copy()

    def mutate(theta, supplied):
        batch = forward(theta, supplied)
        theta[:] = 0
        supplied.loc[:, "lat"] = -70
        return batch

    assert run(forward=mutate)["scenarios"]["complete"]["status"] == "RECOVERED_SYNTHETIC_ONLY"
    pd.testing.assert_frame_equal(frame, before)
    np.testing.assert_array_equal(TRUTH, original_truth)


def test_private_csv_loader_preserves_keys_and_outages(tmp_path):
    frame = design()
    stations = tmp_path/"stations.csv"
    support = tmp_path/"support.csv"
    frame[["station_uid", "lat", "lon", "elevation_m"]].drop_duplicates().to_csv(stations, index=False)
    rows = frame[list(twin.KEYS)+["available"]].copy()
    rows["available"] = rows.available.map({True: "true", False: "false"})
    rows.to_csv(support, index=False)
    loaded, identities = twin.load_design(stations, support)
    assert twin.design_keys(loaded) == twin.design_keys(frame)
    assert loaded.available.tolist() == frame.available.tolist()
    assert len(identities) == 2 and all(len(r["sha256"]) == 64 for r in identities)
    rows["available"] = "FALSE"
    rows.to_csv(support, index=False)
    with pytest.raises(ValueError, match="true/false"):
        twin.load_design(stations, support)


def test_protected_paths_refused_before_payload_read(monkeypatch):
    import gpuwrf.assimilation.letkf.member_driver as driver
    refused = []

    def refuse(operation, paths):
        refused.append((operation, paths))
        raise ValueError("stage through facade")

    monkeypatch.setattr(driver, "refuse_protected_inputs", refuse)
    with pytest.raises(ValueError, match="stage through facade"):
        twin.load_design("/nonexistent/stations.csv", "/nonexistent/support.csv")
    assert len(refused) == 1


def test_actual_wrfout_sampler_wiring_and_units(tmp_path):
    from gpuwrf.io.netcdf_lock import Dataset

    frame = design()
    frame["wind_height_m"] = 20.
    path = tmp_path/"wrfout_test.nc"
    with Dataset(path, "w") as ds:
        ds.GRID_ID, ds.SIMULATION_START_DATE = 1, "2026-01-01_00:00:00"
        ds.MAP_PROJ, ds.POLE_LAT = 6, 90.
        for name, n in (("Time", 4), ("DateStrLen", 19), ("south_north", 2), ("west_east", 2)):
            ds.createDimension(name, n)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[:] = np.array([list(f"2026-01-01_{hour:02d}:00:00") for hour in range(4)], dtype="S1")
        for name, values, units in (("XLAT", [[27., 27.], [29., 29.]], "degrees_north"),
                                    ("XLONG", [[-17., -15.], [-17., -15.]], "degrees_east"),
                                    ("LANDMASK", np.ones((2, 2)), ""),
                                    ("T2", np.full((2, 2), 283.15), "K"),
                                    ("U10", np.full((2, 2), 3.), "m/s"),
                                    ("V10", np.full((2, 2), 4.), "m/s")):
            v = ds.createVariable(name, "f8", ("Time", "south_north", "west_east"))
            v.units, v[:] = units, np.broadcast_to(values, (4, 2, 2))
    value, audit = twin.sample_histories([path], frame, method="nearest", wind_height="neutral-log")
    assert audit["non_10m_stations"] == 2
    assert audit["wind_height"] == "neutral-log"
    np.testing.assert_allclose(value[frame.variable.eq("t2")], 10., atol=1e-10)
    factor = np.log(20./0.03)/np.log(10./0.03)
    np.testing.assert_allclose(value[frame.variable.eq("wind")], 5.*factor)
    frame.loc[0, "valid_time"] = pd.Timestamp("2026-01-02T00:00:00Z")
    with pytest.raises(ValueError, match="complete station design"):
        twin.sample_histories([path], frame, method="nearest")
