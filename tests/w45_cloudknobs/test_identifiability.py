"""Independent analytic sensitivity/rank/nullspace checks; no WRF skill claim."""
import numpy as np
import pytest

from gpuwrf.optimize import continuous, identifiability as ident


def parameters():
    return [{"name": name, "control": "mynn_parameters."+control, "bounds": [.5, 2.], "prior_sd_log": .3}
            for name, control in (("a", "alp1"), ("b", "entrainment"))]


def screen(operator, *, params=None, keys=None):
    params = parameters() if params is None else params
    cfg = ident.configuration()
    branches, columns, _ = ident.design(params, cfg)
    predictions = {name: operator(theta) for name, theta in branches.items()}
    size = len(predictions["prior"])
    keys = [("c1", "s1", "2026-01-01T01:00:00Z", str(i)) for i in range(size)] if keys is None else keys
    return ident.analyse(params, cfg, columns, predictions, np.ones(size), keys)


def test_exact_linear_sensitivity_rank_and_prior():
    result = screen(lambda t: np.array([t[0], 2*t[1], t[0]+t[1]]))
    np.testing.assert_allclose(result["matrix"], [[1, 0], [0, 2], [1, 1]], atol=1.e-14)
    assert result["rank"] == 2 and not result["near_collinear"]
    assert result["frozen_null_basis"] == []
    assert result["parameters"][0]["prior_sd_log"] == .3
    rows = np.array([[.1, -.2], [.3, -.4]])
    np.testing.assert_array_equal(ident.restrict(rows, result, parameters()), rows)


def test_collinear_null_direction_stays_frozen_even_at_unequal_bounds():
    params = parameters()
    params[1]["bounds"] = [.8, 1.2]
    result = screen(lambda t: np.array([t.sum(), 2*t.sum()]), params=params)
    assert result["rank"] == 1
    assert result["near_collinear"][0]["parameters"] == ["a", "b"]
    np.testing.assert_allclose(result["observable_projector"], np.full((2, 2), .5), atol=1.e-14)
    constrained = ident.restrict([[4, 1], [-4, -1]], result, params)
    np.testing.assert_allclose(constrained, [[np.log(1.2)]*2, [np.log(.8)]*2], atol=1.e-14)
    np.testing.assert_allclose(constrained @ np.asarray(result["frozen_null_basis"]).T, 0, atol=1.e-14)


def test_unobservable_parameters_are_frozen_at_prior():
    result = screen(lambda t: np.array([7., 7.]))
    assert result["rank"] == 0 and result["status"] == "UNOBSERVABLE_FROZEN"
    np.testing.assert_array_equal(ident.restrict([[.3, -.2]], result, parameters()), [[0, 0]])


def test_bound_endpoint_reuses_null_branch():
    params = parameters()
    params[0]["bounds"] = [.5, 1.]
    branches, columns, nulls = ident.design(params, ident.configuration())
    assert len(branches) == 4 and columns[0]["plus"] == "prior"
    assert nulls == {"surface_controls": "prior"}


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_response_refuses(value):
    with pytest.raises(ValueError, match="finite"):
        screen(lambda t: np.array([value, 1.]))


def test_correlated_pixels_cannot_be_independent_eki_observations():
    keys = [("c1", "msg3:17", "2026-01-01T01:00:00Z", "cloud_occurrence")]
    with pytest.raises(ValueError, match="correlated MSG pixels"):
        screen(lambda t: np.array([t.sum()]), keys=keys)


@pytest.mark.parametrize("available", [True, False])
def test_backend_capacity_does_not_admit_unfrozen_native_correlation(available, monkeypatch):
    from gpuwrf.surface_params import eki
    monkeypatch.setattr(eki, "ENSEMBLE_SPACE_UPDATE", available)
    with pytest.raises(ValueError, match="admitted superobservations") as caught:
        ident.refuse_native_pixels(["msg3:17"])
    message = str(caught.value)
    assert ("is available" in message) == available
    assert "ALISIOS correlation contract" in message
    ident.refuse_native_pixels(["region:superobservation-1", "station:1"])


def test_more_than_two_active_parameters_refused_before_eki():
    raw = [{"name": f"p{i}", "control": "mynn_parameters."+name} for i, name in enumerate(("alp1", "sqfac", "entrainment"))]
    with pytest.raises(ValueError, match="at most two"):
        continuous._parameters(raw, {"domains": {}})


@pytest.mark.parametrize("raw", [{"step_log": 0}, {"rank_rtol": 1}, {"collinearity_cos": 2},
                                  {"rank_rtol": np.nan}, {"enabled": False}])
def test_sensitivity_policy_is_explicit_and_cannot_be_disabled(raw):
    with pytest.raises(ValueError):
        ident.configuration(raw)
