"""Offline Noah-MP twin experiment (OSSE) for the EKI scaffold (CPU-sized, synthetic truth).

The forward model is the operational land model itself: ``noah_mp_step`` with the traced controls applied by
``apply_surface_controls``, scanned over a diurnal forcing and VMAPPED over ensemble members, so one compiled
program serves every member and every EKI iteration (the property Track P exists for). Synthetic stations
observe 2-m temperature and sensible heat flux at the land columns; a known "truth" multiplier set generates
the observations (+ Gaussian noise) and EKI must recover it. This is a twin experiment: it validates the
estimator and the traced plumbing, not real-world identifiability (atmosphere feedback is absent).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TwinConfig:
    groups: tuple[tuple[int, ...], ...] = ((2, 4, 5), (7, 10, 12))  # MODIS forest | shrub/grass/crop
    control: str = "z0mvt"
    truth_log_multipliers: tuple[float, ...] = (np.log(1.6), np.log(0.7))
    hours: float = 12.0
    dt_s: float = 300.0
    obs_every_s: float = 3600.0
    sigma_t2: float = 0.3
    sigma_hfx: float = 15.0
    members: int = 16
    iterations: int = 4
    prior_sd: float = 0.5
    seed: int = 20261017


def _fixture(groups):
    import jax.numpy as jnp

    from gpuwrf.config.paths import wrf_run_dir
    from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic
    from gpuwrf.physics.noahmp.tables import load_noahmp_parameters

    cats = [c for g in groups for c in g]
    veg = np.array(cats * 2, dtype=np.int32).reshape(2, -1)
    shape = veg.shape
    ny, nx = shape
    full = lambda v: jnp.full(shape, v)  # noqa: E731
    z = jnp.zeros(shape)
    land = NoahMPLandState(
        tslb=jnp.broadcast_to(jnp.asarray(293.0), (NSOIL, ny, nx)),
        smois=jnp.broadcast_to(jnp.asarray(0.22), (NSOIL, ny, nx)),
        sh2o=jnp.broadcast_to(jnp.asarray(0.22), (NSOIL, ny, nx)),
        smcwtd=full(0.22), isnow=jnp.zeros(shape, dtype=jnp.int32),
        tsno=jnp.broadcast_to(jnp.asarray(273.0), (NSNOW, ny, nx)),
        snice=jnp.zeros((NSNOW, ny, nx)), snliq=jnp.zeros((NSNOW, ny, nx)),
        zsnso=jnp.broadcast_to(jnp.asarray([0.0, 0.0, 0.0, -0.1, -0.4, -1.0, -2.0]).reshape(NSNOW + NSOIL, 1, 1),
                               (NSNOW + NSOIL, ny, nx)),
        snowh=z, sneqv=z, sneqvo=z, tauss=z, albold=full(0.2), tv=full(290.0), tg=full(290.0), tah=full(290.0),
        eah=full(1400.0), canliq=z, canice=z, fwet=z, lai=full(2.0), sai=full(0.5), cm=full(0.01), ch=full(0.01),
        t_skin=full(290.0), qsfc=full(0.01), znt=full(0.1), emiss=full(0.97), albedo=full(0.2),
        sfcrunoff=z, udrunoff=z,
    )
    static = NoahMPStatic(
        ivgtyp=jnp.asarray(veg), isltyp=jnp.full(shape, 6, dtype=jnp.int32), xland=full(1.0), landmask=full(1.0),
        lakemask=z, lu_index=jnp.asarray(veg), tbot=full(291.0), dzs=jnp.asarray([0.1, 0.3, 0.6, 1.0]),
        zsoil=jnp.asarray([-0.1, -0.4, -1.0, -2.0]), lat=full(28.3), dx_m=1000.0,
        parameters=load_noahmp_parameters(wrf_run_dir()),
        shdmax=jnp.asarray(np.where(np.isin(veg, groups[0]), 0.85, 0.45)), shdfac=full(0.5),
    )
    return land, static, shape


def _forcing(shape, step, dt_s):
    """Clear-sky diurnal forcing starting 06 LT (mid-June, 28 N), wind 4 m/s, dry season."""
    import jax.numpy as jnp

    from gpuwrf.physics.noahmp.types import NoahMPForcing

    hour = 6.0 + step * dt_s / 3600.0
    cosz = jnp.maximum(jnp.cos((hour - 12.0) * jnp.pi / 12.0) * 0.95, 0.0)
    full = lambda v: jnp.broadcast_to(jnp.asarray(v, jnp.float64), shape)  # noqa: E731
    sfctmp = 291.0 + 5.0 * jnp.sin((hour - 9.0) * jnp.pi / 12.0)
    zero = full(0.0)
    return NoahMPForcing(sfctmp=full(sfctmp), sfcprs=full(95000.0), psfc=full(95600.0), uu=full(4.0),
                         vv=full(1.0), qair=full(0.007), qc=zero, soldn=full(1000.0 * cosz),
                         lwdn=full(330.0 + 0.5 * (sfctmp - 291.0)), prcpconv=zero, prcpnonc=zero, prcpsnow=zero,
                         prcpgrpl=zero, prcphail=zero, cosz=full(cosz), zlvl=full(30.0),
                         julian=jnp.asarray(168.0), yearlen=jnp.asarray(365.0))


def build_forward(cfg: TwinConfig):
    """(forward(theta (N, p)) -> predictions (N, m), info). One jit of the vmapped member program."""
    import types

    import jax
    import jax.numpy as jnp

    from gpuwrf.physics.noahmp.noahmp_driver import build_energy_params, noah_mp_step
    from gpuwrf.surface_params import controls as sc

    from gpuwrf.physics.noahmp.precision import real_tree

    land0, static, shape = _fixture(cfg.groups)
    land0 = real_tree(land0)  # the scan carry keeps Noah-MP's own REAL/legacy precision
    ep, rp = build_energy_params(static, shape)
    nl = types.SimpleNamespace(use_noahmp=True, noahmp_static=static, noahmp_energy_params=ep,
                               noahmp_rad_params=rp, noahmp_options=None)
    base = sc.table_base(cfg.control, nl)                      # float64 table value per cell
    dtype = sc.target_dtype(cfg.control, nl)
    vt = np.asarray(static.ivgtyp)
    masks = jnp.asarray(np.stack([np.isin(vt, g) for g in cfg.groups]).astype(np.float64))
    n_steps = int(round(cfg.hours * 3600.0 / cfg.dt_s))
    every = int(round(cfg.obs_every_s / cfg.dt_s))

    def member(log_mult):
        # host expansion done in-graph here for the twin only: field = round_once(table x exp(theta_group))
        mult = jnp.exp(jnp.tensordot(log_mult, masks, axes=1))  # exp(0) = 1 outside the groups
        controls = {cfg.control: (jnp.asarray(base) * mult).astype(dtype)}
        s2, ep2, rp2 = sc.apply_surface_controls(static, ep, rp, controls)

        def body(land, k):
            land, flux = noah_mp_step(land, _forcing(shape, k, cfg.dt_s), s2, cfg.dt_s,
                                      energy_params=ep2, rad_params=rp2)
            return land, (flux.t2, flux.hfx)

        _, (t2, hfx) = jax.lax.scan(body, land0, jnp.arange(n_steps))
        sel = slice(every - 1, None, every)
        return jnp.concatenate([t2[sel].reshape(-1), hfx[sel].reshape(-1)])

    program = jax.jit(jax.vmap(member))
    calls = {"n": 0, "seconds": 0.0}

    def forward(theta):
        t0 = time.time()
        out = np.asarray(program(jnp.asarray(theta, jnp.float64)), dtype=np.float64)
        calls["n"] += 1
        calls["seconds"] += time.time() - t0
        return out

    n_obs_t = (n_steps // every) * int(np.prod(shape))
    sigma = np.concatenate([np.full(n_obs_t, cfg.sigma_t2), np.full(n_obs_t, cfg.sigma_hfx)])
    return forward, {"program": program, "calls": calls, "sigma": sigma, "shape": shape, "steps": n_steps}


def run_twin(cfg: TwinConfig = TwinConfig()) -> dict:
    from gpuwrf.surface_params import eki

    forward, info = build_forward(cfg)
    rng = np.random.default_rng(cfg.seed)
    truth = np.asarray(cfg.truth_log_multipliers, np.float64)
    clean = forward(np.tile(truth, (cfg.members, 1)))[0]  # members-shaped: the SAME compiled program
    sigma = info["sigma"]
    obs = clean + rng.normal(size=clean.shape) * sigma
    p = truth.size
    problem = eki.EKIProblem(names=tuple(f"{cfg.control}:group{i}" for i in range(p)), prior_mean=np.zeros(p),
                             prior_cov=np.eye(p) * cfg.prior_sd ** 2, obs=obs, obs_cov=sigma ** 2,
                             lower=np.full(p, np.log(0.2)), upper=np.full(p, np.log(5.0)), tikhonov=True)
    state = eki.run(problem, forward, n_members=cfg.members, iterations=cfg.iterations, seed=cfg.seed)
    est = state.theta.mean(axis=0)
    return {"truth_multipliers": np.exp(truth).tolist(), "estimate_multipliers": np.exp(est).tolist(),
            "estimate_sd_log": state.theta.std(axis=0, ddof=1).tolist(),
            "misfit_history": [h["normalised_misfit"] for h in state.history],
            "forward_calls": info["calls"]["n"], "forward_seconds": round(info["calls"]["seconds"], 2),
            "jit_compiles": int(info["program"]._cache_size()), "members": cfg.members,
            "iterations": cfg.iterations, "observations": int(obs.size), "grid": list(info["shape"]),
            "steps_per_window": info["steps"]}
