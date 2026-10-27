"""WRF cycling initialization, evaluated once on host arrays.

Reference: pristine WRF 4.7.1 dyn_em/start_em.F:2259-2544 rebalance_cycl.
The native timestep never calls this module. No RH conversion or clipping.
"""
from __future__ import annotations

import numpy as np


def cycling_enabled(namelist):
    value = (namelist.get("time_control") or {}).get("cycling", False)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("&time_control cycling is a single global logical")
        value = value[0]
    if not isinstance(value, (bool, int)) or value not in (False, True):
        raise ValueError("&time_control cycling must be a logical (0/1)")
    return bool(value)


def admit_cycling_input(path):
    """Validate this file's warm-transfer inventory before model allocation.

    Source paths need not still exist: the writer admitted the original source
    and records the actual stored values. No per-case provenance enters JIT aux.
    """
    import hashlib
    import json
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.cycling.atmosphere import MARKER
    from gpuwrf.cycling.boundary import finite_field
    from gpuwrf.cycling.core import _times
    required = {"U", "V", "W", "T", "THM", "P", "PH", "MU", "QVAPOR", "QCLOUD", "QRAIN",
                "QICE", "QSNOW", "QGRAUP", "QNICE", "QNRAIN", "QKE"}
    with Dataset(path) as ds:
        if MARKER not in ds.ncattrs():
            raise ValueError("cycling=.true. requires gpuwrf cycle --atmosphere provenance in every wrfinput")
        report = json.loads(ds.getncattr(MARKER))
        if report.get("schema") != 1 or _times(ds) != [report.get("valid_time")]:
            raise ValueError("cycling input provenance schema/time differs")
        if int(ds.GRID_ID) != report.get("grid_id") or not required <= set(report.get("fields", {})):
            raise ValueError("cycling input domain/field provenance is incomplete")
        for name, record in report["fields"].items():
            data = finite_field(ds, name)
            digest = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
            if list(data.shape) != record.get("shape") or str(data.dtype) != record.get("dtype") or digest != record.get("sha256"):
                raise ValueError(f"cycling {name}: stored data differs from admitted source transport")
    return report


def rebalance_cycling(fields, *, moist_fields, use_theta_m=1, hypsometric_opt=2, libm=None):
    """Original REAL top-down pressure / bottom-up geopotential integration.

    ``fields`` uses WRF names, arrays without Time; T and THM are perturbations
    and PHB/PB/MUB/ALB are the *resolved* start_domain bases. Moisture order is
    Registry moist-array order, excluding number/volume scalars. Output is
    REAL32 P/PH/AL/ALT/P_HYD/PSFC. MU, theta and moisture are read-only.
    """
    if int(use_theta_m) not in (0, 1) or int(hypsometric_opt) != 2:
        raise ValueError("cycling initialization currently requires hypsometric_opt=2, use_theta_m=0/1")
    if not moist_fields or moist_fields[0] != "QVAPOR":
        raise ValueError("cycling moisture must begin with Registry QVAPOR")
    if libm is None:
        # Runtime has already loaded d02_replay. Reuse its original scalar
        # WRF libm array shim; NumPy SIMD pow/log can lose an AL cancellation
        # ULP. A standalone oracle supplies the same source-extracted shim.
        from gpuwrf.io.wrf_init_math import _WRF_INIT_LIBM32
        libm = _WRF_INIT_LIBM32
    f = {}
    for name, value in fields.items():
        a = np.asarray(value, np.float32)
        if not np.isfinite(a).all():
            raise ValueError(f"cycling {name}: non-finite REAL input")
        f[name] = a
    required = ("THM" if int(use_theta_m) else "T", "MU", "MUB", "PB", "PHB", "ALB",
                "C1F", "C2F", "C3F", "C4F", "C3H", "C4H", "RDNW", "RDN", "P_TOP", *moist_fields)
    for name in required:
        if name not in f:
            raise ValueError(f"cycling required field {name} is missing")
    shape = f["PB"].shape
    if len(shape) != 3 or shape[0] < 2:
        raise ValueError("cycling requires at least two mass levels")
    nz, ny, nx = shape
    for name in ("ALB", "T", "THM", *moist_fields):
        if name in f and f[name].shape != shape:
            raise ValueError(f"cycling {name}: incompatible mass-grid shape")
    if f["PHB"].shape != (nz + 1, ny, nx) or any(f[n].shape != (ny, nx) for n in ("MU", "MUB")):
        raise ValueError("cycling MU/PHB shape mismatch")
    for name in ("C1F", "C2F", "C3F", "C4F"):
        if f[name].shape != (nz + 1,):
            raise ValueError(f"cycling {name}: incompatible full-level coefficient")
    for name in ("C3H", "C4H", "RDNW", "RDN"):
        if f[name].shape != (nz,):
            raise ValueError(f"cycling {name}: incompatible half-level coefficient")
    if f["P_TOP"].size != 1:
        raise ValueError("cycling P_TOP must be scalar")
    if any(np.any(f[name] < 0) for name in moist_fields):
        raise ValueError("cycling moisture must be nonnegative")
    real = np.float32
    one, half = real(1), real(.5)
    p1000, rd, g = real(100000), real(287), real(9.81)
    cvpm = -real(2.5) / real(3.5)
    rvovrd = real(461.6) / rd
    theta = f["THM" if int(use_theta_m) else "T"] + real(300)
    mu, mub = f["MU"], f["MUB"]
    p, alt = np.empty(shape, np.float32), np.empty(shape, np.float32)
    for k in range(nz - 1, -1, -1):
        total = np.zeros((ny, nx), np.float32)
        for name in moist_fields:
            total = total + half * (f[name][k] + f[name][k if k == nz - 1 else k + 1])
        qvf2 = one / (one + total)
        qvf1 = total * qvf2
        coupled = f["C1F"][k + 1] * mu + qvf1 * (f["C1F"][k + 1] * mub + f["C2F"][k + 1])
        if k == nz - 1:
            p[k] = -half * coupled / f["RDNW"][k] / qvf2
        else:
            p[k] = p[k + 1] - coupled / qvf2 / f["RDN"][k + 1]
        qvf = one if int(use_theta_m) else one + rvovrd * f["QVAPOR"][k]
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            alt[k] = (rd / p1000) * theta[k] * qvf * libm.pow((p[k] + f["PB"][k]) / p1000, cvpm)
    total_mu = mub + mu
    total_phi = np.empty((nz + 1, ny, nx), np.float32)
    total_phi[0] = f["PHB"][0]
    for k in range(1, nz + 1):
        pfu = f["C3F"][k] * total_mu + f["C4F"][k] + f["P_TOP"]
        pfd = f["C3F"][k - 1] * total_mu + f["C4F"][k - 1] + f["P_TOP"]
        phm = f["C3H"][k - 1] * total_mu + f["C4H"][k - 1] + f["P_TOP"]
        if np.any(pfu <= 0) or np.any(pfd <= pfu) or np.any(phm <= 0):
            raise ValueError("cycling MU: invalid hybrid layer pressures")
        total_phi[k] = total_phi[k - 1] + alt[k - 1] * phm * libm.log(pfd / pfu)
    ph = total_phi - f["PHB"]
    # Re-form ph0 just as WRF does, retaining its storage rounding.
    ph0 = ph + f["PHB"]
    z0 = ph0[0] / g
    z1 = half * (ph0[0] + ph0[1]) / g
    z2 = half * (ph0[1] + ph0[2]) / g
    if np.any(z1 == z2):
        raise ValueError("cycling PH: degenerate surface-pressure interpolation")
    w1 = (z0 - z2) / (z1 - z2)
    w2 = one - w1
    full_p = p + f["PB"]
    psfc = w1 * full_p[0] + w2 * full_p[1]
    result = {"P": p, "PH": ph, "AL": alt - f["ALB"], "ALT": alt,
              "P_HYD": full_p, "PSFC": psfc}
    if any(not np.isfinite(v).all() for v in result.values()) or np.any(full_p <= 0) or np.any(alt <= 0):
        raise ValueError("cycling rebalance produced non-finite/non-positive thermodynamics")
    return result


def start_domain_cycling(run, domain, *, grid, metrics, pb, phb, mub, mu, prognostic_theta,
                         qv, u, v, w, child_input_terrain=None, child_opens_at_start=False):
    """One host initialization arm in WRF start_domain order, never per step.

    After nest base/adjust_tempqv: rebalance_cycl -> press_adj -> possible
    start_domain(parent) repeat -> set_w_surface. No metadata is keyed aux.
    """
    import time
    import jax
    import jax.numpy as jnp
    from gpuwrf.integration.d02_replay import (
        _WRF_INIT_LIBM32, _wrf_use_theta_m, _wrf_set_w_surface,
        _wrf_start_domain_t_init_alb32, _wrf_start_domain_base_scalars32,
    )
    from gpuwrf.cycling.boundary import finite_field
    start = time.perf_counter()
    host = lambda x: np.asarray(jax.device_get(x), np.float32)
    use_theta_m = _wrf_use_theta_m(run, domain)
    with run._read_dataset(run.wrfinput_file(domain)) as ds:
        fields = {name: finite_field(ds, name) for name in (
            "T", "THM", "ALB", "P_TOP", "C1F", "C2F", "C3F", "C4F", "C3H", "C4H", "RDNW", "RDN",
            "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")}
    fields.update(PB=host(pb), PHB=host(phb), MUB=host(mub), MU=host(mu), QVAPOR=host(qv))
    prognostic = (np.asarray(jax.device_get(prognostic_theta), np.float64) - 300.).astype(np.float32)
    fields["THM" if use_theta_m else "T"] = prognostic
    if child_input_terrain is not None:
        _, fields["ALB"] = _wrf_start_domain_t_init_alb32(fields["PB"], _wrf_start_domain_base_scalars32(run, domain))
    moist = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")
    result = rebalance_cycling(fields, moist_fields=moist, use_theta_m=use_theta_m, libm=_WRF_INIT_LIBM32)
    if child_input_terrain is not None:
        # Pristine start_em:878-891, before the child's child re-opens it.
        fields["MU"] = (fields["MU"] + result["AL"][0] / (result["ALT"][0] * fields["ALB"][0])
                         * np.float32(9.81) * (host(grid.terrain_height) - host(child_input_terrain))).astype(np.float32)
        if child_opens_at_start:
            result = rebalance_cycling(fields, moist_fields=moist, use_theta_m=use_theta_m, libm=_WRF_INIT_LIBM32)
    use_w = (run.namelist.get("dynamics") or {}).get("use_input_w", False)
    if isinstance(use_w, (list, tuple)):
        if len(use_w) != 1:
            raise ValueError("use_input_w is one global logical")
        use_w = use_w[0]
    if not isinstance(use_w, (bool, int)) or use_w not in (False, True):
        raise ValueError("use_input_w must be a WRF logical 0/1")
    # start_em:1475-1530: even nonzero supplied W is rebuilt when false.
    w_arg = w if use_w else np.zeros_like(host(w))
    w_new, w_set, _ = _wrf_set_w_surface(w_arg, u, v, grid=grid, metrics=metrics)
    metadata = {"enabled": True, "schema": 1, "wrf_source": "start_em.F:767,878,1475,2259-2544",
                "host_seconds": time.perf_counter()-start, "use_input_w": bool(use_w),
                "w_reinitialized": bool(w_set), "parent_restart_rebalance": bool(child_opens_at_start),
                "libm_provider": _WRF_INIT_LIBM32.provider,
                "diagnostic_buckets": "existing package-gated seed only: PREC_ACC absent at prec_acc_dt=0; RUC ACRUNOFF absent for Noah-MP; Noah-MP runoff initialized on its allocated land carry",
                "xtime_minutes": 0.}
    return (jnp.asarray(result["P"], dtype=pb.dtype), jnp.asarray(result["PH"], dtype=phb.dtype),
            jnp.asarray(fields["MU"], dtype=mu.dtype), jnp.asarray(w_new, dtype=w.dtype), metadata)
