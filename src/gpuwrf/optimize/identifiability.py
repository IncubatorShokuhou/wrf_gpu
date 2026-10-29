"""Small host sensitivity design and observable-subspace restriction before EKI.

These numerical diagnostics do not establish regional skill or change ALISIOS
gate authority. Observations must already have a valid aggregation/covariance
contract; raw correlated satellite pixels are not independent observations.
"""
from __future__ import annotations

import math
import numpy as np


def refuse_native_pixels(location_ids):
    """Backend capacity does not admit an unreviewed pixel-correlation operator."""
    if any(str(location).startswith("msg3:") for location in location_ids):
        from gpuwrf.surface_params import eki
        capability = bool(getattr(eki, "ENSEMBLE_SPACE_UPDATE", False))
        requirement = ("ensemble-space EKI is available, but native vectors need the admitted ALISIOS correlation contract"
                       if capability else "full-native vectors await ensemble-space EKI and the admitted ALISIOS correlation contract")
        raise ValueError("correlated MSG pixels are refused: use admitted superobservations; " + requirement)


def configuration(raw=None):
    raw = {} if raw is None else raw
    defaults = {"step_log": .1, "rank_rtol": 1.e-6, "collinearity_cos": .99}
    if not isinstance(raw, dict) or set(raw) - set(defaults):
        raise ValueError("identifiability takes step_log, rank_rtol, collinearity_cos")
    out = defaults | raw
    for key, value in out.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"identifiability {key} must be finite and positive")
    if out["rank_rtol"] >= 1 or out["collinearity_cos"] > 1:
        raise ValueError("rank_rtol must be <1 and collinearity_cos <=1")
    return out


def design(parameters, config):
    """Prior/control is the null branch for every selected family; ± steps respect bounds.

    An endpoint at factor 1 reuses the control rather than running it twice.
    The family remains enabled at its defaults: this is not an ON/OFF identity
    experiment and adds no extra compiled signature.
    """
    p = len(parameters)
    if not 1 <= p <= 2:
        raise ValueError("identifiability starts with at most two active continuous parameters")
    branches = {"prior": np.zeros(p)}
    columns = []
    for i, param in enumerate(parameters):
        ends = [max(math.log(param["bounds"][0]), -config["step_log"]),
                min(math.log(param["bounds"][1]), config["step_log"])]
        names = []
        for label, delta in zip(("minus", "plus"), ends):
            name = "prior" if delta == 0 else f"sensitivity-{label}-{param['name']}"
            if delta != 0:
                theta = np.zeros(p)
                theta[i] = delta
                branches[name] = theta
            names.append(name)
        if ends[0] == ends[1]:
            raise ValueError(f"{param['name']}: no feasible sensitivity perturbation")
        columns.append({"name": param["name"], "minus": names[0], "plus": names[1],
                        "log_steps": ends, "factors": np.exp(ends).tolist()})
    families = sorted({param.get("runtime", {}).get("family", "surface_controls") for param in parameters})
    return branches, columns, {family: "prior" for family in families}


def analyse(parameters, config, columns, predictions, sigma, keys):
    """Finite-difference columns whitened by the admitted observation errors, with full support."""
    sigma = np.asarray(sigma, np.float64)
    if sigma.ndim != 1 or not len(sigma) or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("sensitivity needs finite positive observation errors")
    if len(keys) != len(sigma):
        raise ValueError("sensitivity support differs from observation errors")
    refuse_native_pixels(key[1] for key in keys)
    values = {name: np.asarray(value, np.float64) for name, value in predictions.items()}
    if any(v.shape != sigma.shape or np.any(~np.isfinite(v)) for v in values.values()):
        raise ValueError("sensitivity predictions require identical finite observation support")
    matrix = np.column_stack([(values[col["plus"]]-values[col["minus"]]) /
                              (col["log_steps"][1]-col["log_steps"][0])/sigma for col in columns])
    if np.any(~np.isfinite(matrix)):
        raise ValueError("nonfinite sensitivity matrix")
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    cutoff = config["rank_rtol"] * (singular[0] if singular.size else 0.)
    rank = int(np.count_nonzero(singular > cutoff))
    observable = vh[:rank]
    projector = observable.T @ observable
    null_values, null_vectors = np.linalg.eigh(np.eye(len(parameters))-projector)
    null_basis = null_vectors[:, null_values > .5].T
    norms = np.linalg.norm(matrix, axis=0)
    near = []
    for i in range(len(parameters)):
        for j in range(i+1, len(parameters)):
            if norms[i] and norms[j]:
                cosine = float(np.clip(np.dot(matrix[:, i], matrix[:, j])/(norms[i]*norms[j]), -1., 1.))
                if abs(cosine) >= config["collinearity_cos"]:
                    near.append({"parameters": [parameters[i]["name"], parameters[j]["name"]], "cosine": cosine})
    return {"schema": "gpuwrf.optimize.identifiability.v1", "settings": config,
            "parameters": [{k: p[k] for k in ("name", "control", "bounds", "prior_sd_log")} for p in parameters],
            "columns": columns, "observation_support": [[str(v) for v in k] for k in keys],
            "sigma": sigma.tolist(), "matrix": matrix.tolist(), "column_norms": norms.tolist(),
            "rank": rank, "singular_values": singular.tolist(), "near_collinear": near,
            "observable_projector": projector.tolist(), "frozen_null_basis": null_basis.tolist(),
            "status": "UNOBSERVABLE_FROZEN" if rank < len(parameters) else "NUMERICALLY_IDENTIFIABLE",
            "scope": "local finite differences at the prior, not proof of global/real-world identification"}


def restrict(theta, report, parameters):
    """Freeze the null subspace at prior=0 while retaining the box bounds.

    Radially shrink a projected row toward the prior if it leaves the box.
    Componentwise clipping would reintroduce unobservable components.
    """
    if report["rank"] == len(parameters):
        return np.asarray(theta, np.float64).copy()
    rows = np.asarray(theta, np.float64) @ np.asarray(report["observable_projector"], np.float64)
    lower = np.log([p["bounds"][0] for p in parameters])
    upper = np.log([p["bounds"][1] for p in parameters])
    for row in rows:
        scale = 1.
        for value, lo, hi in zip(row, lower, upper):
            if value < lo:
                scale = min(scale, lo/value)
            if value > hi:
                scale = min(scale, hi/value)
        row *= scale
    return rows
