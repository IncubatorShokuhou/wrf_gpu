"""Shared real-table Noah-MP fixture for the Track P tests (2x3 land grid, five MODIS categories)."""
from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.config.paths import wrf_run_dir  # noqa: E402
from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic  # noqa: E402
from gpuwrf.physics.noahmp.noahmp_driver import build_energy_params  # noqa: E402
from gpuwrf.physics.noahmp.tables import load_noahmp_parameters  # noqa: E402

TABLE_DIR = wrf_run_dir()
pytestmark = pytest.mark.skipif(not (TABLE_DIR / "MPTABLE.TBL").exists(), reason="pristine WRF MPTABLE missing")
VEG = np.array([[2, 5, 10], [12, 7, 10]])  # MODIS: EBF, mixed forest, grassland, cropland, open shrubland
SHAPE = VEG.shape


class _State:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def replace(self, **u):
        return _State(**{**self.__dict__, **u})


def _fixture(julian=172.0, smois=0.25):
    ny, nx = SHAPE
    z = jnp.zeros(SHAPE)
    full = lambda v: jnp.full(SHAPE, v)  # noqa: E731
    p0, rcp = 1.0e5, 287.0 / 1004.0
    t = np.array([[296.0, 297.0, 299.0], [300.0, 298.0, 301.0]])
    p = np.full(SHAPE, 95000.0)
    lead = lambda a: jnp.asarray(np.asarray(a, float).reshape(1, ny, nx))  # noqa: E731
    state = _State(
        u=jnp.full((1, ny, nx + 1), 5.0), v=jnp.full((1, ny + 1, nx), 1.5),
        theta=lead(t * (p0 / p) ** rcp), qv=lead(np.full(SHAPE, 0.009)), qc=jnp.zeros((1, ny, nx)),
        p=lead(p), dz=jnp.full((1, ny, nx), 60.0),
        ph=jnp.asarray(np.stack([np.zeros(SHAPE), np.full(SHAPE, 9.80665 * 60.0)])),
        t_skin=full(302.0), xland=full(1.0), roughness_m=full(0.1), soil_moisture=full(0.25),
        mavail=full(0.6), ustar=full(0.2), lakemask=z, sfc_wspd=z,
    )
    land = NoahMPLandState(
        tslb=jnp.broadcast_to(jnp.asarray(296.0), (NSOIL, ny, nx)),
        smois=jnp.broadcast_to(jnp.asarray(smois), (NSOIL, ny, nx)),
        sh2o=jnp.broadcast_to(jnp.asarray(smois), (NSOIL, ny, nx)),
        smcwtd=full(smois), isnow=jnp.zeros(SHAPE, dtype=jnp.int32),
        tsno=jnp.broadcast_to(jnp.asarray(273.0), (NSNOW, ny, nx)),
        snice=jnp.zeros((NSNOW, ny, nx)), snliq=jnp.zeros((NSNOW, ny, nx)),
        zsnso=jnp.broadcast_to(jnp.asarray([0.0, 0.0, 0.0, -0.1, -0.4, -1.0, -2.0]).reshape(NSNOW + NSOIL, 1, 1),
                               (NSNOW + NSOIL, ny, nx)),
        snowh=z, sneqv=z, sneqvo=z, tauss=z, albold=full(0.2), tv=full(299.0), tg=full(300.0), tah=full(298.0),
        eah=full(1800.0), canliq=z, canice=z, fwet=z, lai=full(2.0), sai=full(0.5), cm=full(0.01), ch=full(0.01),
        t_skin=full(302.0), qsfc=full(0.012), znt=full(0.1), emiss=full(0.97), albedo=full(0.2),
        sfcrunoff=z, udrunoff=z,
    )
    static = NoahMPStatic(
        ivgtyp=jnp.asarray(VEG, dtype=jnp.int32), isltyp=jnp.full(SHAPE, 6, dtype=jnp.int32),
        xland=full(1.0), landmask=full(1.0), lakemask=z, lu_index=jnp.asarray(VEG, dtype=jnp.int32),
        tbot=full(292.0), dzs=jnp.asarray([0.1, 0.3, 0.6, 1.0]), zsoil=jnp.asarray([-0.1, -0.4, -1.0, -2.0]),
        lat=full(28.3), dx_m=1000.0, parameters=load_noahmp_parameters(TABLE_DIR),
        shdmax=jnp.asarray([[0.9, 0.8, 0.5], [0.7, 0.3, 0.45]]), shdfac=full(0.4),
    )

    class _Rad:
        soldn = full(750.0)
        lwdn = full(360.0)
        cosz = full(0.8)

    class _Clock:
        yearlen = 365.0

    _Clock.julian = julian
    from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
    from gpuwrf.physics.noahmp_coupler import assemble_noahmp_forcing
    forcing = assemble_noahmp_forcing(_build_column_view(state), static, _Rad(), _Clock(), 60.0)
    return land, forcing, static


def _namelist(static, options=None):
    ep, rp = build_energy_params(static, SHAPE)
    return types.SimpleNamespace(use_noahmp=True, noahmp_static=static, noahmp_energy_params=ep,
                                 noahmp_rad_params=rp, noahmp_options=options)
