"""Operational coupling of the single-layer urban canopy model (sf_urban_physics=1).

WRF couples SLUCM to Noah-MP in three places, all reproduced here:

1. ``noahmplsm`` runs every ISURBAN/LCZ cell as ``VEGTYP = NATURAL_TABLE`` with
   ``FVGMAX = 0.96`` (module_sf_noahmpdrv.F:916-928), so ``parameters%URBAN_FLAG`` is
   false (no urban soil hard-codes) -- :func:`urban_noahmp_static` remaps the
   Noah-MP static once at load time.
2. ``noahmp_urban`` (module_sf_noahmpdrv.F:3373-3545) runs ``urban`` on those cells
   and area-blends TSK/HFX/QFX/LH/GRDFLX/ALBEDO/QSFC/UST with FRC_URB2D --
   :func:`apply_urban_slucm`, called by ``noahmp_coupler.noahmp_surface_adapter``
   right after ``noah_mp_step`` (same forcing Noah-MP received).
3. ``surface_driver`` overwrites T2/Q2/U10/V10 on urban cells (:3485-3500), before
   the post-LSM Q2 cap -- :func:`urban_history_overwrite`.

The prognostic urban state rides ``OperationalCarry.urban_slucm`` (an
:class:`UrbanCarry`); ``None`` keeps every released configuration unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics import urban_slucm as us

#: WRF FVGMAX for urban cells under sf_urban_physics>0 (module_sf_noahmpdrv.F:926).
URBAN_FVGMAX = 0.96


class UrbanCarry(NamedTuple):
    """Prognostic SLUCM carry: WRF ``*_URB2D/3D`` + ``C*_SFCDIF`` state plus the
    per-cell FRC_URB2D / UTYPE_URB2D set by ``urban_var_init`` (constant in time)."""

    state: us.UrbanState
    frc_urb: Any
    utype: Any
    inputs: Any = None  # opt-in LCZ morphology / AHE; required by bundle when configured (E78)


@dataclass(frozen=True, eq=False)
class UrbanBundle:
    """Static SLUCM configuration (URBPARM.TBL after ``urban_param_init``)."""

    table: us.UrbanTable
    isurban: int
    natural: int
    use_wudapt_lcz: bool = False
    ahe_opt: int = 0
    start: Any = None
    rad_interval_s: float = 1800.0
    solar_every_step: bool = False
    requires_inputs: bool = False
    dt_s: float = 1.0
    has_morphology: bool = False

    def __post_init__(self):
        if self.ahe_opt not in (0,1,2):
            raise ValueError("distributed_ahe_opt must be 0, 1 or 2")
        if self.table.ahoption or self.table.alhoption or self.ahe_opt or self.has_morphology:
            object.__setattr__(self,"requires_inputs",True)


def mptable_natural(table_dir, dataset: str = "MODIFIED_IGBP_MODIS_NOAH") -> int:
    """``NATURAL`` category of the MPTABLE land-use block (MODIS 14, USGS 5)."""
    from gpuwrf.physics.noahmp.tables import _parse_mptable_block

    block = {"MODIFIED_IGBP_MODIS_NOAH": "noahmp_modis_parameters", "USGS": "noahmp_usgs_parameters"}[dataset]
    text = (Path(table_dir) / "MPTABLE.TBL").read_text()
    return int(_parse_mptable_block(text, block)["NATURAL"][0])


def load_urban_bundle(table_dir, *, isurban: int, dataset: str = "MODIFIED_IGBP_MODIS_NOAH",
                      dtype=np.float32, use_wudapt_lcz=False, **options) -> UrbanBundle:
    """Read URBPARM.TBL (``urban_param_init``) and refuse unported switches.

    LCZ selects URBPARM_LCZ.TBL. Diurnal table heat needs the dated production
    clock and longitude supplied by the loader in UrbanCarry.inputs.
    """
    table = us.load_urban_table(table_dir, dtype=dtype, use_wudapt_lcz=use_wudapt_lcz)
    if table.icate != (11 if use_wudapt_lcz else 3):
        raise ValueError("urban table category count does not match use_wudapt_lcz")
    return UrbanBundle(table=table, isurban=int(isurban), natural=mptable_natural(table_dir, dataset),
                       use_wudapt_lcz=bool(use_wudapt_lcz), **options)


#: Relative margin on 0.5*DZ8W(kts) for the load-time WRF FATAL check: WRF re-checks ``ZDC+Z0C+2 >= ZA``
#: every step and aborts; the scan cannot abort, so the first-level height must clear the canopy by this
#: margin at load (covers the diurnal/mass drift of the lowest-layer thickness, a few percent).
ZA_DRIFT_MARGIN = 0.15


def check_first_level_above_canopy(bundle: UrbanBundle, *, ivgtyp, dz1, domain: str = "",
                                   margin: float = ZA_DRIFT_MARGIN, morphology=None) -> None:
    """WRF ``urban`` FATAL (module_sf_urban.F:826) as a loud load-time refusal.

    WRF stops with "ZDC + Z0C + 2m is larger than the 1st WRF level - Stop in subroutine urban - change ZDC
    and Z0C" whenever ``ZDC+Z0C+2 >= ZA = 0.5*DZ8W(kts)`` on an urban cell (urban_var_init gives every
    ISURBAN cell UTYPE 2).  Below that height the canopy formulas leave their domain (U10 ~ 177 m/s at
    dz8w1 = 12 m, NaN below 10 m, rv-urban RVUR09), so refuse instead of running.
    """
    v = np.asarray(ivgtyp)
    mask = (v == bundle.isurban) | ((v >= 51) & (v <= 61))
    if not mask.any():
        return
    t = bundle.table
    idx = np.where(v == bundle.isurban, 4 if bundle.use_wudapt_lcz else 1,
                   np.clip(v - 51, 0, t.icate - 1))
    limit = (t.zdc[idx] + t.z0c[idx] + np.dtype(t.dtype).type(2.0))[mask]
    if morphology is not None:
        limit = (morphology.zdc+morphology.z0c+np.dtype(t.dtype).type(2.0))[mask]
    za = 0.5 * np.asarray(dz1, np.float64)[mask]
    bad = limit >= za * (1.0 - margin)
    if bad.any():
        raise ValueError(
            f"{domain}: sf_urban_physics=1 refused: {int(bad.sum())} urban cell(s) have a first model level "
            f"0.5*DZ8W(kts) = {za[bad].min():.2f} m that does not clear the urban canopy ZDC+Z0C+2 = "
            f"{limit[bad].max():.2f} m by the {margin:.0%} drift margin. WRF stops here (module_sf_urban.F:826 FATAL "
            "'ZDC + Z0C + 2m is larger than the 1st WRF level - Stop in subroutine urban - change ZDC and Z0C'); "
            "use a thicker lowest layer or adjust ZR/ROOF_WIDTH/ROAD_WIDTH in URBPARM.TBL.")


def urban_cells(ivgtyp, isurban: int):
    """ISURBAN cells and WRF4.7.1 LCZ_1..11 = 51..61 (flag checked at load)."""
    v = jnp.asarray(ivgtyp, jnp.int32)
    return (v == int(isurban)) | ((v >= 51) & (v <= 61))


def urban_noahmp_static(static, bundle: UrbanBundle):
    """Noah-MP static as WRF's noahmplsm sees it with sf_urban_physics>0."""
    m = urban_cells(static.ivgtyp, bundle.isurban)
    ivg = jnp.where(m, jnp.int32(bundle.natural), jnp.asarray(static.ivgtyp, jnp.int32))
    upd = {"ivgtyp": ivg}
    if static.shdmax is not None:
        upd["shdmax"] = jnp.where(m, jnp.asarray(URBAN_FVGMAX, static.shdmax.dtype), static.shdmax)
    return static.replace(**upd)


def initial_urban_carry(bundle: UrbanBundle, *, ivgtyp, tsk, tslb, frc_urb2d, dtype=jnp.float32,
                        inputs=None) -> UrbanCarry:
    """``urban_var_init`` (non-restart) from wrfinput TSK / TSLB(NSOIL,ny,nx) / FRC_URB2D."""
    m = urban_cells(ivgtyp, bundle.isurban)
    tslb_last = jnp.moveaxis(jnp.asarray(tslb, dtype), 0, -1)
    state, frc, utype = us.urban_var_init(bundle.table, urban_mask=m, tsk=jnp.asarray(tsk, dtype),
                                          tslb=tslb_last, frc_urb_in=jnp.asarray(frc_urb2d, dtype),
                                          dtype=dtype, **({"ivgtyp": ivgtyp, "isurban": bundle.isurban,
                                              "use_wudapt_lcz": True} if bundle.use_wudapt_lcz else {}))
    return UrbanCarry(state=state, frc_urb=frc, utype=utype, inputs=inputs)


class UrbanStepDiagnostics(NamedTuple):
    diag: us.UrbanDiagnostics
    mask: Any
    frc_urb: Any


def apply_urban_slucm(bundle: UrbanBundle, carry: UrbanCarry, *, forcing, nm, ust, chs, chs2, cqs2,
                      qv_mixing_ratio, dt, lead_seconds=None):
    """``noahmp_urban`` on the Noah-MP outputs of this step.

    ``forcing`` is the NoahMPForcing Noah-MP consumed (T3D, QV, U/V, SWDOWN, GLW,
    P_ML, 0.5*DZ8W and the previous-step precipitation rates); ``nm`` its
    NoahMPFluxes; ``ust``/``chs``/``chs2``/``cqs2`` the surface-layer values.
    Returns ``(nm', ust', carry', UrbanStepDiagnostics)``.
    """
    dt_c = carry.state.tr.dtype
    c = lambda x: jnp.asarray(x, dt_c)  # noqa: E731
    m = carry.utype > 0
    extra = {}
    if bundle.requires_inputs and carry.inputs is None:
        raise ValueError("urban LCZ/AHE/morphology restart is missing required urban inputs (E78)")
    if (bundle.ahe_opt or bundle.table.ahoption or bundle.table.alhoption) and bundle.start is None and carry.inputs.clock is None:
        raise ValueError("urban heat requires dated production clock anchors (E78)")
    if bundle.has_morphology and carry.inputs.morphology is None:
        raise ValueError("urban morphology selected but geometry inputs are missing (E78)")
    if bundle.ahe_opt and lead_seconds is None:
        raise ValueError("distributed AHE requires the production step clock")
    if bundle.ahe_opt and carry.inputs.ahe is None and carry.inputs.hourly is None:
        raise ValueError("distributed AHE selected but forcing inputs are missing (E78)")
    if bundle.table.ahoption or bundle.table.alhoption:
        if ((bundle.start is None and carry.inputs.clock is None)
                or carry.inputs.longitude is None or lead_seconds is None):
            raise ValueError("diurnal urban heat requires a dated production clock and longitude")
    if carry.inputs is not None:
        if carry.inputs.morphology is not None:
            extra["morphology"] = carry.inputs.morphology
        if bundle.table.ahoption or bundle.table.alhoption:
            if lead_seconds is None:
                raise ValueError("diurnal urban heat requires the production step clock")
            from gpuwrf.coupling.urban_inputs import urban_clock
            extra["omg"], extra["jmonth"] = urban_clock(bundle, carry.inputs, lead_seconds)
    # RAINBL = MAX(RAINCV+RAINNCV, 0) of this step (module_surface_driver.F:1568-1570);
    # Noah-MP's previous-step rates are the same amounts / DT.
    rainbl = jnp.maximum((c(forcing.prcpconv) + c(forcing.prcpnonc)) * c(dt), c(0.0))
    sfcprs = c(forcing.sfcprs)
    lsm = us.UrbanLSMFields(tsk=c(nm.tsk), hfx=c(nm.hfx), qfx=c(nm.qfx), lh=c(nm.lh), grdflx=c(nm.grdflx),
                            albedo=c(nm.albedo), qsfc=c(nm.qsfc), ust=c(ust))
    lsm_out, state_out, diag = us.noahmp_urban(
        bundle.table, urban_mask=m, frc_urb=carry.frc_urb, utype=carry.utype,
        t3d1=c(forcing.sfctmp), qv1=c(qv_mixing_ratio), u1=c(forcing.uu), v1=c(forcing.vv),
        swdown=c(forcing.soldn), glw=c(forcing.lwdn),
        # P_ML = (P8W(kts+1)+P8W(kts))*0.5 is Noah-MP's sfcprs; (x+x)*0.5 == x exactly.
        p8w1=sfcprs, p8w2=sfcprs,
        rainbl=rainbl, dz8w1=c(2.0) * c(forcing.zlvl), dt=float(dt),
        chs=c(chs), chs2=c(chs2), cqs2=c(cqs2), lsm=lsm, state=carry.state, **extra)

    def back(new, old):
        # Only urban cells take the blend; every other cell keeps its value bit-for-bit (no REAL round trip
        # of the rural fields when Noah-MP runs in float64).
        old = jnp.asarray(old)
        return jnp.where(m, jnp.asarray(new, old.dtype), old)

    nm_out = nm._replace(tsk=back(lsm_out.tsk, nm.tsk), hfx=back(lsm_out.hfx, nm.hfx),
                         qfx=back(lsm_out.qfx, nm.qfx), lh=back(lsm_out.lh, nm.lh),
                         grdflx=back(lsm_out.grdflx, nm.grdflx), albedo=back(lsm_out.albedo, nm.albedo),
                         qsfc=back(lsm_out.qsfc, nm.qsfc))
    if getattr(nm, "history", None) is not None and "GRDFLX" in nm.history:
        hist = dict(nm.history)
        hist["GRDFLX"] = back(lsm_out.grdflx, hist["GRDFLX"])
        nm_out = nm_out._replace(history=hist)
    ust_out = back(lsm_out.ust, ust)
    return nm_out, ust_out, carry._replace(state=state_out), UrbanStepDiagnostics(diag, m, carry.frc_urb)


def urban_history_overwrite(t2, q2, u10, v10, step: UrbanStepDiagnostics, psfc):
    """Surface-driver urban T2/Q2/U10/V10 (module_surface_driver.F:3485-3500).

    ``t2``/``q2`` are the rural Noah-MP land values (FVEG*T2MV+(1-FVEG)*T2MB with the
    NATURAL vegetation type); the result goes through the post-LSM Q2 cap after.
    """
    d = step.diag
    dt_c = d.th2_urb.dtype
    c = lambda x: jnp.asarray(x, dt_c)  # noqa: E731
    one = c(1.0)
    frc = step.frc_urb
    rcp = np.dtype(dt_c).type(287.0) / (np.dtype(dt_c).type(7.0) * np.dtype(dt_c).type(287.0)
                                        / np.dtype(dt_c).type(2.0))
    q2_u = c(q2) * (one - frc) + d.q2_urb * frc
    t2_u = c(t2) * (one - frc) + (d.th2_urb / ((c(1.0e5) / c(psfc)) ** rcp)) * frc
    m = step.mask

    def sel(new, old):
        return jnp.where(m, jnp.asarray(new, jnp.asarray(old).dtype), old)

    return sel(t2_u, t2), sel(q2_u, q2), sel(d.u10_urb, u10), sel(d.v10_urb, v10)


__all__ = [
    "URBAN_FVGMAX", "UrbanCarry", "UrbanBundle", "UrbanStepDiagnostics", "mptable_natural",
    "load_urban_bundle", "urban_cells", "urban_noahmp_static", "initial_urban_carry", "ZA_DRIFT_MARGIN",
    "check_first_level_above_canopy",
    "apply_urban_slucm", "urban_history_overwrite",
]
