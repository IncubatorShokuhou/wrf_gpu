"""Finite, equal-weight scalar ensemble statistics; member axis is always zero."""
from __future__ import annotations

import numpy as np


def finite(value, name):
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ValueError(f"{name}: masked/fill values")
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: NaN/Inf are not supported")
    return array


def ensemble(values):
    values = finite(values, "members")
    if values.ndim < 1 or values.shape[0] < 2 or values.size == 0:
        raise ValueError("need at least two members and nonempty samples")
    return values


def levels(values, name, *, probability=False):
    values = finite(values, name)
    if values.ndim != 1 or not values.size or np.any(np.diff(values) <= 0):
        raise ValueError(f"{name}: need nonempty strictly increasing levels")
    if probability and (values[0] < 0 or values[-1] > 1):
        raise ValueError(f"{name}: must be in [0,1]")
    return values


def products(values, *, thresholds=(0.0,), quantiles=(0.1, 0.5, 0.9)):
    """Strict exceedance P(X>threshold); linear quantiles; sample std (ddof=1)."""
    values = ensemble(values)
    thresholds = levels(thresholds, "thresholds")
    quantiles = levels(quantiles, "quantiles", probability=True)
    with np.errstate(over="ignore", invalid="ignore"):
        result = {"mean": values.mean(axis=0), "spread": values.std(axis=0, ddof=1),
                  "quantile": np.quantile(values, quantiles, axis=0, method="linear"),
                  "probability": np.stack([(values > threshold).mean(axis=0) for threshold in thresholds])}
    return {name: finite(value, name) for name, value in result.items()}


def paired(values, observations):
    values, observations = ensemble(values), finite(observations, "observations")
    if values.shape[1:] != observations.shape:
        raise ValueError("observation shape must equal the member sample shape")
    return values, observations


def crps(values, observations, *, fair=False):
    """Energy-form CRPS via sorted pair distances, O(M log M), O(M) storage.

    Empirical: mean|x-y| - sum_ij|x_i-x_j|/(2 M²).
    Fair: replace M² by M(M-1); unbiased only for iid ensemble draws.
    """
    values, observations = paired(values, observations)
    n = values.shape[0]
    ordered = np.sort(values, axis=0)
    # Center before the zero-sum dot product to avoid large-offset cancellation.
    ordered = ordered - ordered[0]
    coefficients = (2*np.arange(n)-n+1).reshape((n,)+(1,)*(values.ndim-1))
    with np.errstate(over="ignore", invalid="ignore"):
        half_pair_sum = (ordered*coefficients).sum(axis=0)
        result = np.abs(values-observations).mean(axis=0)-half_pair_sum/(n*(n-1) if fair else n*n)
    return finite(result, "CRPS")


def rank_histogram(values, observations):
    """Ranks 0..M. Ties divide one count uniformly among admissible ranks.

    This is the expectation of random uniform tie breaking, without RNG noise.
    """
    values, observations = paired(values, observations)
    below = (values < observations).sum(axis=0).ravel()
    tied = (values == observations).sum(axis=0).ravel()
    return np.array([np.sum(((below <= rank) & (rank <= below+tied))/(tied+1))
                     for rank in range(values.shape[0]+1)])


def reliability(probabilities, events, *, edges=tuple(np.linspace(0, 1, 11))):
    """Bin [left,right), last bin includes 1; empty bins use JSON null."""
    probabilities, events = finite(probabilities, "probabilities"), finite(events, "events")
    edges = levels(edges, "probability edges", probability=True)
    if probabilities.shape != events.shape or probabilities.size == 0:
        raise ValueError("reliability needs nonempty equally shaped pairs")
    if edges.size < 2 or edges[0] != 0 or edges[-1] != 1:
        raise ValueError("probability edges must cover [0,1]")
    if np.any((probabilities < 0) | (probabilities > 1)) or not np.isin(events, [0, 1]).all():
        raise ValueError("probabilities must be in [0,1] and events binary")
    indices = np.minimum(np.searchsorted(edges, probabilities, side="right")-1, len(edges)-2)
    rows = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = indices == index
        count = int(mask.sum())
        rows.append({"left": float(left), "right": float(right), "n": count,
                     "forecast_probability": float(probabilities[mask].mean()) if count else None,
                     "observed_frequency": float(events[mask].mean()) if count else None})
    return rows


def spread_skill(values, observations):
    """RMS sample spread versus RMSE of ensemble mean, pooled on equal support."""
    values, observations = paired(values, observations)
    with np.errstate(over="ignore", invalid="ignore"):
        spread = float(np.sqrt(np.mean(values.var(axis=0, ddof=1))))
        rmse = float(np.sqrt(np.mean((values.mean(axis=0)-observations)**2)))
    finite([spread, rmse], "spread/skill")
    ratio = spread/rmse if rmse else None
    if ratio is not None:
        finite(ratio, "spread/RMSE ratio")
    return {"n": int(observations.size), "rms_spread": spread, "mean_rmse": rmse,
            "spread_rmse_ratio": ratio}
