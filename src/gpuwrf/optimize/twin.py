"""Host-only station twins for EKI; no forecast, observation service or GPU is opened.

The adapter receives log multipliers and the complete, ordered station/time design.
Its returned identity and keys must match the registered experiment. A twin can
check recovery under those assumptions; it cannot establish real identification.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from gpuwrf.surface_params import eki

KEYS = ("station_uid", "valid_time", "variable")
STATION_FIELDS = ("station_uid", "lat", "lon", "elevation_m", "landmask", "wind_height_m", "roughness_m")
SCHEMA = "gpuwrf.optimize.station-twin.v1"


@dataclass(frozen=True)
class ForwardIdentity:
    """Content digests of forcing and configuration; schema includes units/conventions.

    Adapters must derive these from actual inputs, rather than echoing expectations.
    Configuration includes the model revision, scheme/nest tuple and sampling rule.
    Parameter VALUES are excluded from the configuration digest.
    """

    forcing_sha256: str
    configuration_sha256: str
    observation_schema: str

    def __post_init__(self):
        for value in (self.forcing_sha256, self.configuration_sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("forward identity requires lowercase SHA256 digests")
        if not isinstance(self.observation_schema, str) or not self.observation_schema.strip():
            raise ValueError("observation_schema must name variables, units and sampling conventions")


@dataclass(frozen=True)
class PredictionBatch:
    values: np.ndarray  # (members, complete design rows), never outage-compressed
    keys: tuple
    identity: ForwardIdentity


@dataclass(frozen=True)
class TwinConfig:
    names: tuple[str, ...]
    prior_sd_log: tuple[float, ...]
    bounds_log: tuple[tuple[float, float], ...]
    members: int = 12
    iterations: int = 4
    seed: int = 17
    tolerance_log: float = 0.05
    sensitivity_step_log: float = 0.01
    rank_rtol: float = 1e-6
    max_normalised_misfit: float = 4.0
    max_observations: int = 256

    def __post_init__(self):
        p = len(self.names)
        if not 1 <= p <= 2 or len(set(self.names)) != p or any(not isinstance(n, str) or not n for n in self.names):
            raise ValueError("register one or two distinct parameter names; freeze other directions")
        sd, bounds = np.asarray(self.prior_sd_log), np.asarray(self.bounds_log)
        if sd.shape != (p,) or not np.isfinite(sd).all() or (sd <= 0).any():
            raise ValueError("prior_sd_log must be finite positive, one per parameter")
        if (bounds.shape != (p, 2) or not np.isfinite(bounds).all()
                or (bounds[:, 0] >= 0).any() or (bounds[:, 1] <= 0).any()):
            raise ValueError("bounds_log must strictly contain the table prior (zero)")
        for name, minimum in (("members", p + 2), ("iterations", 1), ("max_observations", 1), ("seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("tolerance_log", "sensitivity_step_log", "rank_rtol", "max_normalised_misfit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite positive")
        if self.rank_rtol >= 1:
            raise ValueError("rank_rtol must be less than one")
        if np.any(bounds[:, 0] > -self.sensitivity_step_log) or np.any(bounds[:, 1] < self.sensitivity_step_log):
            raise ValueError("sensitivity perturbations must lie inside registered bounds")


def station_design(stations: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    """Join station geometry to an EXPLICIT complete sampling/outage grid.

    ``support`` has station_uid, valid_time (offset/UTC), variable, available
    (boolean). Absent rows are not silently interpreted as outages. The caller
    must provide the intended full grid, including unavailable rows. No real
    observation values enter the synthetic truth.
    """
    required = {"station_uid", "lat", "lon", "elevation_m"}
    if not required <= set(stations) or not {*KEYS, "available"} <= set(support):
        raise ValueError("station geometry and explicit boolean availability grid are required")
    stations, support = stations.copy(), support.copy()
    for frame in (stations, support):
        if frame.empty or frame.station_uid.isna().any() or frame.station_uid.astype(str).str.strip().eq("").any():
            raise ValueError("nonempty station IDs and rows are required")
        frame["station_uid"] = frame.station_uid.astype(str)
    if stations.station_uid.duplicated().any():
        raise ValueError("duplicate station geometry")
    xyz = stations[["lat", "lon", "elevation_m"]].to_numpy(dtype=float)
    if not np.isfinite(xyz).all() or np.any(np.abs(xyz[:, 0]) > 90) or np.any(np.abs(xyz[:, 1]) > 180):
        raise ValueError("invalid station geometry")
    stations[["lat", "lon", "elevation_m"]] = xyz
    for column, default in (("landmask", 1), ("wind_height_m", 10.), ("roughness_m", 0.03)):
        if column not in stations:
            stations[column] = default
        stations[column] = pd.to_numeric(stations[column], errors="raise")
    if (not np.isfinite(stations[list(STATION_FIELDS[1:])].to_numpy()).all()
            or not stations.landmask.isin([0, 1]).all()
            or (stations.roughness_m <= 0).any() or (stations.roughness_m >= 10).any()
            or (stations.wind_height_m <= stations.roughness_m).any()):
        raise ValueError("invalid landmask or wind-height/roughness sampling metadata")
    if support.variable.isna().any() or support.variable.astype(str).str.strip().eq("").any():
        raise ValueError("nonempty observation variables are required")
    support["variable"] = support.variable.astype(str)
    times = [pd.Timestamp(t) for t in support.valid_time]
    if any(pd.isna(t) or t.tzinfo is None for t in times):
        raise ValueError("valid_time must explicitly declare UTC/offset")
    support["valid_time"] = pd.to_datetime(times, utc=True)
    if support.duplicated(list(KEYS)).any():
        raise ValueError("duplicate station/time/variable key")
    if not all(isinstance(x, (bool, np.bool_)) for x in support.available):
        raise ValueError("availability must be explicit booleans, never truthy strings/numbers")
    if set(support.station_uid) - set(stations.station_uid):
        raise ValueError("support references a station without geometry")
    # Only these columns cross the forward adapter boundary (never real values).
    return (support[[*KEYS, "available"]].merge(stations[list(STATION_FIELDS)], on="station_uid", validate="many_to_one")
            .sort_values(list(KEYS)).reset_index(drop=True))


def load_design(station_path, support_path):
    """Read private, already staged files; refuse ALISIOS roots before any payload read.

    The support CSV uses exactly the literals true/false in ``available``.
    Returns the design and content identities for a private result manifest.
    """
    from gpuwrf.assimilation.letkf.member_driver import refuse_protected_inputs
    from gpuwrf.verify.observations import read_stations
    from .data import identity

    refuse_protected_inputs("station twin", [station_path, support_path])
    before = [identity(p) for p in (station_path, support_path)]
    stations = read_stations(station_path)
    support = pd.read_csv(support_path, dtype={"station_uid": str, "available": str})
    if "available" not in support or not support.available.isin(["true", "false"]).all():
        raise ValueError("availability CSV must contain only true/false")
    support["available"] = support.available.eq("true")
    design = station_design(stations, support)
    if before != [identity(p) for p in (station_path, support_path)]:
        raise ValueError("station geometry or availability changed while loading")
    return design, before


def design_keys(design):
    return tuple((u, t.isoformat(), v) for u, t, v in design[list(KEYS)].itertuples(index=False, name=None))


def sample_histories(paths, design, **sampling):
    """Real verification extractor, strict full-design alignment, instantaneous T2/wind.

    T2 is returned in degC and scalar wind in m/s (verify's canonical units).
    Private paths must already be facade-staged. This function never runs WRF.
    The returned audit records interpolation/height/elevation assumptions.
    """
    from gpuwrf.assimilation.letkf.member_driver import refuse_protected_inputs
    from gpuwrf.verify.wrfout import extract_wrfout

    paths = list(paths)
    refuse_protected_inputs("station twin histories", paths)
    if set(design.variable) - {"t2", "wind"}:
        raise ValueError("history twin sampler supports instantaneous t2 (degC) and wind (m/s)")
    stations = design[[c for c in STATION_FIELDS if c in design]].drop_duplicates()
    design = station_design(stations, design)
    frame, audit = extract_wrfout(paths, design[list(STATION_FIELDS)].drop_duplicates(), **sampling)
    selected = frame.loc[frame.variable.isin(set(design.variable)), [*KEYS, "value"]]
    aligned = design[list(KEYS)].merge(selected, on=list(KEYS), how="left", validate="one_to_one", indicator=True)
    if not aligned._merge.eq("both").all() or not np.isfinite(aligned.value.to_numpy(dtype=float)).all():
        raise ValueError("history output lacks finite predictions for the complete station design")
    return aligned.value.to_numpy(dtype=float), audit


def _checked(forward, theta, design, identity):
    batch = forward(np.array(theta, dtype=np.float64, copy=True), design.copy(deep=True))
    if not isinstance(batch, PredictionBatch):
        raise ValueError("forward must return a PredictionBatch with actual input identity")
    if batch.identity != identity:
        raise ValueError("forcing/configuration/observation schema mismatch; refuse parameter identification")
    if batch.keys != design_keys(design):
        raise ValueError("forward keys/order do not match the registered station design")
    values = np.asarray(batch.values, dtype=np.float64)
    if values.shape != (len(theta), len(design)) or not np.isfinite(values).all():
        raise ValueError("forward predictions must be finite with shape (members, complete design rows)")
    return values


def _observation_covariance(sd, covariance):
    """Admit in correlation units; return the covariance and its sampling factor.

    Symmetry tolerance is relative to sqrt(d_i*d_j), never an absolute covariance
    unit. Normalize by successive divisions to avoid a tiny/huge scale product.
    One normalized Cholesky proves positive definiteness and supplies the factor.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        variance = sd**2
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("sigma squared must be representable finite positive variances")
    gamma = np.diag(variance) if covariance is None else np.asarray(covariance, dtype=np.float64)
    m = len(sd)
    if gamma.shape != (m, m) or not np.isfinite(gamma).all():
        raise ValueError("covariance must be finite and match the complete design")
    diagonal = np.diag(gamma)
    if np.any(diagonal <= 0) or not np.allclose(diagonal, variance, rtol=1e-10, atol=0):
        raise ValueError("covariance diagonal must equal the registered positive sigma squared")
    scale = np.sqrt(diagonal)
    with np.errstate(over="ignore", invalid="ignore"):
        correlation = (gamma/scale[:, None])/scale[None, :]
    if (not np.isfinite(correlation).all() or np.any(np.abs(correlation) > 1+1e-12)
            or not np.allclose(correlation, correlation.T, rtol=0, atol=1e-12)):
        raise ValueError("covariance must be symmetric with valid correlations relative to its diagonal")
    # For admitted roundoff asymmetry, every later solve must use the SAME
    # lower triangle as Cholesky. Already symmetric inputs stay byte unchanged.
    if not np.array_equal(gamma, gamma.T):
        gamma = gamma.copy()
        lower = np.tril_indices(m, -1)
        upper = (lower[1], lower[0])
        gamma[upper] = gamma[lower]
        correlation[upper] = correlation[lower]
    try:
        chol = np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite; aggregate duplicate information") from exc
    chol *= scale[:, None]
    return gamma, chol


def run_twin(design, forward, truth_log, *, config, identity, sigma, truth_forward=None,
             covariance=None, add_noise=True):
    """Compare complete geometry vs real outages using the SAME noisy synthetic data.

    ``forward(theta, design)`` returns PredictionBatch. ``truth_forward`` may test
    an imperfect twin, but must retain the registered forcing/schema/configuration.
    Declared wrong forcing/schema is refused before EKI; an undeclared forcing
    error in the parameter span is intrinsically confounded (see report caveat).
    Sigma is keyed by variable. A full covariance can represent correlated rows;
    otherwise independence is an explicit experimental assumption. Support must
    be preselected to <= max_observations (dense EKI solve, never implicit thinning).
    """
    if not isinstance(config, TwinConfig) or not isinstance(identity, ForwardIdentity):
        raise ValueError("registered TwinConfig and ForwardIdentity are required")
    if not isinstance(add_noise, bool):
        raise ValueError("add_noise must be boolean")
    # Revalidate caller-built frames, and freeze their canonical order.
    stations = design[[c for c in STATION_FIELDS if c in design]].drop_duplicates()
    design = station_design(stations, design)
    m, p = len(design), len(config.names)
    if m > config.max_observations:
        raise ValueError("design exceeds dense EKI observation budget; preregister blocks/subsampling")
    truth = np.asarray(truth_log, dtype=np.float64)
    bounds = np.asarray(config.bounds_log)
    if (truth.shape != (p,) or not np.isfinite(truth).all()
            or np.any(truth < bounds[:, 0]) or np.any(truth > bounds[:, 1])):
        raise ValueError("truth_log must be finite and inside registered bounds")
    if set(sigma) != set(design.variable):
        raise ValueError("sigma must cover exactly the design variables")
    sd = np.array([sigma[v] for v in design.variable], dtype=float)
    if not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError("sigma must be finite positive")
    gamma, chol = _observation_covariance(sd, covariance)
    calls = 0

    def predict(theta, adapter=forward):
        nonlocal calls
        calls += 1
        return _checked(adapter, theta, design, identity)

    y = predict(truth[None, :], adapter=forward if truth_forward is None else truth_forward)[0]
    if add_noise:
        y = y + chol @ np.random.default_rng(config.seed).standard_normal(m)
    # Null + both perturbations at the table prior, before fitting any data.
    step = config.sensitivity_step_log
    probes = np.vstack([np.zeros(p), np.eye(p)*step, -np.eye(p)*step])
    probed = predict(probes)
    jac = ((probed[1:p+1]-probed[p+1:])/(2*step)).T
    scenarios = {}
    for label, mask in (("complete", np.ones(m, bool)), ("outages", design.available.to_numpy(dtype=bool))):
        n = int(mask.sum())
        entry = {"observations": n, "missing": m-n, "status": "NO_SUPPORT", "estimate_log": None}
        scenarios[label] = entry
        if not n:
            continue
        cov = gamma[np.ix_(mask, mask)]
        whitening = np.linalg.cholesky(cov)
        weighted = np.linalg.solve(whitening, jac[mask]) * np.asarray(config.prior_sd_log)
        _, singular, vt = np.linalg.svd(weighted, full_matrices=True)
        threshold = config.rank_rtol * (singular[0] if len(singular) else 0)
        rank = int(np.sum(singular > threshold))
        information = weighted.T @ weighted
        entry.update(rank=rank, singular_values=singular.tolist(), information=information.tolist(),
                     weak_directions=vt[rank:].tolist(), sensitivity=jac[mask].tolist(),
                     null_predictions=probed[0, mask].tolist(),
                     condition_number=float(singular[0]/singular[-1]) if rank == p else None)
        norm = np.linalg.norm(weighted, axis=0)
        entry["near_collinear_pairs"] = [
            [config.names[i], config.names[j]] for i in range(p) for j in range(i+1, p)
            if norm[i] > 0 and norm[j] > 0
            and abs(float(weighted[:, i] @ weighted[:, j]/(norm[i]*norm[j]))) >= 0.99]
        if rank < p:
            entry["status"] = "UNIDENTIFIABLE_FROZEN"
            continue
        problem = eki.EKIProblem(config.names, np.zeros(p), np.diag(np.asarray(config.prior_sd_log)**2),
                                 y[mask], cov, bounds[:, 0], bounds[:, 1])
        state = eki.initial_ensemble(problem, config.members, seed=config.seed)
        for _ in range(config.iterations):
            state = eki.update(problem, state, predict(state.theta)[:, mask],
                               dt=1.0/config.iterations, perturb=False)
        estimate = state.theta.mean(axis=0)
        residual = y[mask] - predict(estimate[None, :])[0, mask]
        normalised = float(residual @ np.linalg.solve(cov, residual)/n)
        errors = np.abs(estimate-truth)
        status = ("MODEL_MISMATCH" if normalised > config.max_normalised_misfit else
                  "RECOVERED_SYNTHETIC_ONLY" if np.all(errors <= config.tolerance_log) else "NOT_RECOVERED")
        entry.update(status=status, estimate_log=estimate.tolist(), abs_log_error=errors.tolist(),
                     normalised_misfit=normalised, iterations=state.iteration)
    # Digest only: real geometry/outage rows belong in private files, never a Git fixture.
    design_digest = hashlib.sha256(design.to_csv(index=False).encode()).hexdigest()
    report = {"schema": SCHEMA, "design_sha256": design_digest, "parameters": list(config.names),
              "identity": vars(identity), "truth_log": truth.tolist(), "scenarios": scenarios,
              "config": vars(config), "forward_calls": calls,
              "noise": {"added": bool(add_noise), "seed": config.seed,
                        "covariance": "diagonal independence assumption" if covariance is None else "registered full matrix"},
              "evidence": "SYNTHETIC_TWIN_ONLY; perfect twins prove no real identification",
              "forcing_limit": "Declared identity mismatches are refused before EKI. Undeclared forcing errors in "
                               "the parameter-response span cannot be distinguished from a parameter change.",
              "spread_limit": "No posterior uncertainty claim; use held-out real dates and forcing contrasts."}
    json.dumps(report, allow_nan=False)
    return report
