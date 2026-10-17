"""Single-layer urban canopy model (SLUCM, ``sf_urban_physics=1``).

Faithful JAX port of WRF ``phys/module_sf_urban.F`` (Kusaka & Kimura SLUCM) as it
is coupled to Noah-MP for ``sf_urban_physics=1``:

* ``urban_param_init`` (URBPARM.TBL reader + derived canyon geometry)   :2043-2551
* ``urban_var_init``   (state initialization, non-restart)              :2557-3052
* ``urban``            (one urban column step)                          :305-1692
* ``SFCDIF_URB``, ``multi_layer``, ``force_restore``                    :3116, :1861, :3058
* the per-cell wrapper ``noahmp_urban`` of ``module_sf_noahmpdrv.F``      :3373-3545
  (forcing derivation, CHS/CHS2/CQS2 limits, FRC_URB2D area blend)
* the ``module_surface_driver.F`` urban 2-m / 10-m overwrite             :3485-3507

Arithmetic follows WRF REAL literally (left-to-right Fortran evaluation, typed
literals, same branch structure; E133/E15).  Every routine is dtype-generic:
float64 inputs reproduce the ``-fdefault-real-8`` oracle build, float32 inputs the
WRF REAL build.  Newton loops are fixed-trip ``fori_loop``s with a per-cell
``done`` mask that freezes the iterate and the last-iteration flux terms exactly
like the Fortran ``EXIT``.

Supported URBPARM.TBL switches (anything else is refused by
:func:`check_supported_options` before tracing): CH_SCHEME=2, TS_SCHEME 1|2,
IMP_SCHEME 1|2, AHOPTION 0|1, ALHOPTION 0|1, BOUNDR/BOUNDB/BOUNDG 1|2, GROPTION=0,
IRI_SCHEME=0. LCZ and zero-initialized-caller NUDAPT geometry are prepared by
coupling.urban_inputs. Distributed drag remains refused. See docs/urban_lcz.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

URBAN_SOURCE = "phys/module_sf_urban.F"
NOAHMP_URBAN_SOURCE = "phys/module_sf_noahmpdrv.F:noahmp_urban"

# Number of URBAN NSOIL-like layers: WRF sets num_roof/wall/road_layers =
# num_soil_layers (urban_param_init :2120); urban_var_init hard-codes 4.
NUM_URBAN_LAYERS = 4


# --------------------------------------------------------------------------------------
# URBPARM.TBL
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class UrbanTable:
    """URBPARM.TBL after ``urban_param_init`` (per-category arrays, REAL rounded).

    Arrays are numpy in the requested ``dtype`` (float32 = WRF REAL build,
    float64 = ``-fdefault-real-8`` build); index ``utype - 1``.
    """

    dtype: np.dtype
    icate: int
    zr: np.ndarray
    sigma_zed: np.ndarray
    z0c: np.ndarray
    z0hc: np.ndarray
    zdc: np.ndarray
    svf: np.ndarray
    r: np.ndarray
    rw: np.ndarray
    hgt: np.ndarray
    ah: np.ndarray
    alh: np.ndarray
    betr: np.ndarray
    betb: np.ndarray
    betg: np.ndarray
    frc_urb: np.ndarray
    capr: np.ndarray
    capb: np.ndarray
    capg: np.ndarray
    aksr: np.ndarray
    aksb: np.ndarray
    aksg: np.ndarray
    albr: np.ndarray
    albb: np.ndarray
    albg: np.ndarray
    epsr: np.ndarray
    epsb: np.ndarray
    epsg: np.ndarray
    z0r: np.ndarray
    z0b: np.ndarray
    z0g: np.ndarray
    z0hb: np.ndarray
    z0hg: np.ndarray
    trlend: np.ndarray
    tblend: np.ndarray
    tglend: np.ndarray
    akanda_urban: np.ndarray
    dzr: np.ndarray  # [cm], len NUM_URBAN_LAYERS
    dzb: np.ndarray
    dzg: np.ndarray
    boundr: int
    boundb: int
    boundg: int
    ch_scheme: int
    ts_scheme: int
    imp_scheme: int
    iri_scheme: int
    ahoption: int
    alhoption: int
    groption: int
    ahdiuprf: np.ndarray  # (24,)
    alhdiuprf: np.ndarray  # (48,)
    alhseason: np.ndarray  # (4,)
    porimp: np.ndarray  # (3,)
    dengimp: np.ndarray  # (3,)
    fgr: float = 0.0
    source: str = ""
    raw_keys: tuple = field(default_factory=tuple)
    street_width: object = None
    building_width: object = None


_CATEGORY_KEYS = (
    "ZR", "SIGMA_ZED", "ROOF_WIDTH", "ROAD_WIDTH", "AH", "ALH", "FRC_URB", "CAPR", "CAPB", "CAPG",
    "AKSR", "AKSB", "AKSG", "ALBR", "ALBB", "ALBG", "EPSR", "EPSB", "EPSG", "AKANDA_URBAN",
    "Z0B", "Z0G", "Z0R", "TRLEND", "TBLEND", "TGLEND",
)
# BEP/BEM-only and multi-layer keys urban_param_init also accepts (read, unused by SLUCM).
_IGNORED_KEYS = (
    "COP", "BLDAC_FRC", "COOLED_FRC", "PWIN", "BETA", "SW_COND", "TIME_ON", "TIME_OFF", "TARGTEMP",
    "GAPTEMP", "TARGHUM", "GAPHUM", "PERFLO", "HSEQUIP", "HSEQUIP_SCALE_FACTOR", "IRHO",
    "PV_FRAC_ROOF", "GR_FRAC_ROOF", "GR_FLAG", "GR_TYPE", "OASIS", "DZGR",
)


def _fortran_reals(text: str) -> list[float]:
    """List-directed REAL read of a value list (commas/blanks; trailing comma ok)."""
    out = []
    for tok in text.replace(",", " ").split():
        out.append(float(tok.replace("d", "e").replace("D", "E")))
    return out


def parse_urbparm(path: str | os.PathLike, dtype=np.float32, sf_urban_physics: int = 1) -> UrbanTable:
    """``urban_param_init`` for SLUCM: read URBPARM.TBL and derive canyon geometry.

    Values are rounded to ``dtype`` at read time and every derived quantity is
    evaluated op-by-op in ``dtype`` (Fortran REAL), mirroring :2190-2545.
    """
    dt = np.dtype(dtype)
    T = dt.type
    path = Path(path)
    lines = path.read_text().splitlines()
    icate = None
    vals: dict[str, np.ndarray] = {}
    scalars: dict[str, int] = {}
    dz: dict[str, np.ndarray] = {}
    seq: dict[str, np.ndarray] = {}
    keys = []
    streets = {}
    it = iter(lines)
    for string in it:
        if string[:1] == "#" or string.strip() == "":
            continue
        indx = string.find(":")
        if indx < 0:
            continue
        name = string[:indx].strip()
        rest = string[indx + 1:]
        keys.append(name)
        if name == "Number of urban categories":
            icate = int(rest.split()[0])
        elif name in _CATEGORY_KEYS:
            v = _fortran_reals(rest)[:icate]
            if len(v) < icate:
                raise ValueError(f"URBPARM.TBL {name}: {len(v)} values < ICATE={icate}")
            vals[name] = np.asarray(v, dtype=dt)
        elif name in ("DDZR", "DDZB", "DDZG"):
            v = _fortran_reals(rest)[:NUM_URBAN_LAYERS]
            # dzr = dzr * 100.0  (m -> cm, REAL)
            dz[name] = np.asarray(v, dtype=dt) * T(100.0)
        elif name in ("BOUNDR", "BOUNDB", "BOUNDG", "CH_SCHEME", "TS_SCHEME", "AHOPTION",
                      "ALHOPTION", "IMP_SCHEME", "IRI_SCHEME", "GROPTION"):
            scalars[name] = int(rest.split()[0].rstrip(","))
        elif name == "FGR":
            scalars[name] = float(_fortran_reals(rest)[0])
        elif name == "AHDIUPRF":
            seq[name] = np.asarray(_fortran_reals(rest)[:24], dtype=dt)
        elif name == "ALHDIUPRF":
            seq[name] = np.asarray(_fortran_reals(rest)[:48], dtype=dt)
        elif name == "ALHSEASON":
            seq[name] = np.asarray(_fortran_reals(rest)[:4], dtype=dt)
        elif name == "PORIMP":
            seq[name] = np.asarray(_fortran_reals(rest)[:3], dtype=dt)
        elif name == "DENGIMP":
            seq[name] = np.asarray(_fortran_reals(rest)[:3], dtype=dt)
        elif name == "STREET PARAMETERS":
            for s in it:
                if s.strip() == "END STREET PARAMETERS":
                    break
                if s.strip() and not s.lstrip().startswith("#"):
                    row = _fortran_reals(s)
                    if len(row) == 4:
                        streets.setdefault(int(row[0]), (row[2],row[3]))
        elif name == "BUILDING HEIGHTS":
            for s in it:
                if s.strip() == "END BUILDING HEIGHTS":
                    break
        elif name in _IGNORED_KEYS:
            continue
        else:
            raise ValueError(f"URBPARM.TBL: Unrecognized NAME = \"{name}\" in Subr URBAN_PARAM_INIT")
    if icate is None:
        raise ValueError("URBPARM.TBL: missing 'Number of urban categories'")

    # unit conversions done at read time (:2289-2310), op-by-op in REAL
    conv_cap = T(1.0) / T(4.1868)
    for k in ("CAPR", "CAPB", "CAPG"):
        vals[k] = (vals[k] * conv_cap) * T(1.0e-6)
    for k in ("AKSR", "AKSB", "AKSG"):
        vals[k] = (vals[k] * conv_cap) * T(1.0e-2)

    zr = vals["ZR"]
    road = vals["ROAD_WIDTH"]
    roof = vals["ROOF_WIDTH"]
    z0b = vals["Z0B"]
    z0g = vals["Z0G"]
    z0hb = T(0.1) * z0b
    z0hg = T(0.1) * z0g
    n = icate
    hgt = np.empty(n, dt); r = np.empty(n, dt); rw = np.empty(n, dt)
    zdc = np.empty(n, dt); z0c = np.empty(n, dt); z0hc = np.empty(n, dt)
    svf = np.empty(n, dt); z0r = vals["Z0R"].copy()
    vonk = T(0.4)
    cd = T(1.2)
    alpha_macd = T(4.43)
    beta_macd = T(1.0)
    with np.errstate(all="ignore"):
        for lc in range(n):
            hgt[lc] = zr[lc] / (road[lc] + roof[lc])
            r[lc] = roof[lc] / (road[lc] + roof[lc])
            rw[lc] = T(1.0) - r[lc]
            lam_p = r[lc]
            lam_f = hgt[lc]
            zdc[lc] = zr[lc] * (T(1.0) + (alpha_macd ** (-lam_p)) * (lam_p - T(1.0)))
            fac = T(1.0) - zdc[lc] / zr[lc]
            z0c[lc] = zr[lc] * fac * np.exp(
                -(T(0.5) * beta_macd * cd / (vonk ** 2) * fac * lam_f) ** T(-0.5))
            if sf_urban_physics == 1:
                lam_fr = vals["SIGMA_ZED"][lc] / (road[lc] + roof[lc])
                z0r[lc] = zr[lc] * fac * np.exp(
                    -(T(0.5) * beta_macd * cd / (vonk ** 2) * fac * lam_fr) ** T(-0.5))
            z0hc[lc] = T(0.1) * z0c[lc]
            dhgt = hgt[lc] / T(100.0)
            h = hgt[lc] - dhgt / T(2.0)
            vfws = T(0.0)
            for _ in range(99):
                h = h - dhgt
                vfws = vfws + T(0.25) * (T(1.0) - h / np.sqrt(h ** T(2.0) + rw[lc] ** T(2.0)))
            vfws = vfws / T(99.0)
            vfws = vfws * T(2.0)
            vfgs = T(1.0) - T(2.0) * vfws * hgt[lc] / rw[lc]
            svf[lc] = vfgs

    zeros = np.zeros(n, dt)
    need = {"DDZR", "DDZB", "DDZG"} - set(dz)
    if need:
        raise ValueError(f"URBPARM.TBL lacks {sorted(need)}")
    return UrbanTable(
        dtype=dt, icate=n, zr=zr, sigma_zed=vals["SIGMA_ZED"], z0c=z0c, z0hc=z0hc, zdc=zdc, svf=svf,
        street_width=np.asarray([streets[i+1][0] for i in range(n)],dt),
        building_width=np.asarray([streets[i+1][1] for i in range(n)],dt),
        r=r, rw=rw, hgt=hgt, ah=vals["AH"], alh=vals["ALH"], betr=zeros, betb=zeros, betg=zeros,
        frc_urb=vals["FRC_URB"], capr=vals["CAPR"], capb=vals["CAPB"], capg=vals["CAPG"],
        aksr=vals["AKSR"], aksb=vals["AKSB"], aksg=vals["AKSG"], albr=vals["ALBR"],
        albb=vals["ALBB"], albg=vals["ALBG"], epsr=vals["EPSR"], epsb=vals["EPSB"], epsg=vals["EPSG"],
        z0r=z0r, z0b=z0b, z0g=z0g, z0hb=z0hb, z0hg=z0hg, trlend=vals["TRLEND"],
        tblend=vals["TBLEND"], tglend=vals["TGLEND"], akanda_urban=vals["AKANDA_URBAN"],
        dzr=dz["DDZR"], dzb=dz["DDZB"], dzg=dz["DDZG"],
        boundr=scalars.get("BOUNDR", 1), boundb=scalars.get("BOUNDB", 1),
        boundg=scalars.get("BOUNDG", 1), ch_scheme=scalars.get("CH_SCHEME", 2),
        ts_scheme=scalars.get("TS_SCHEME", 1), imp_scheme=scalars.get("IMP_SCHEME", 1),
        iri_scheme=scalars.get("IRI_SCHEME", 0), ahoption=scalars.get("AHOPTION", 0),
        alhoption=scalars.get("ALHOPTION", 0), groption=scalars.get("GROPTION", 0),
        ahdiuprf=seq.get("AHDIUPRF", np.ones(24, dt)), alhdiuprf=seq.get("ALHDIUPRF", np.ones(48, dt)),
        alhseason=seq.get("ALHSEASON", np.ones(4, dt)), porimp=seq.get("PORIMP", np.full(3, 0.45, dt)),
        dengimp=seq.get("DENGIMP", np.full(3, 0.001, dt)), fgr=float(scalars.get("FGR", 0.0)),
        source=str(path), raw_keys=tuple(keys),
    )


def check_supported_options(table: UrbanTable) -> None:
    """Refuse URBPARM.TBL switches this port does not implement (fail closed)."""
    bad = []
    if table.ch_scheme != 2:
        bad.append(f"CH_SCHEME={table.ch_scheme} (only 2: empirical wall/road exchange)")
    if table.ts_scheme not in (1, 2):
        bad.append(f"TS_SCHEME={table.ts_scheme}")
    if table.imp_scheme not in (1, 2):
        bad.append(f"IMP_SCHEME={table.imp_scheme}")
    if table.ahoption not in (0, 1):
        bad.append(f"AHOPTION={table.ahoption}")
    if table.alhoption not in (0, 1):
        bad.append(f"ALHOPTION={table.alhoption}")
    if table.groption != 0:
        bad.append(f"GROPTION={table.groption} (green roof not ported)")
    if table.iri_scheme != 0:
        bad.append(f"IRI_SCHEME={table.iri_scheme} (urban irrigation not ported)")
    for nm, b in (("BOUNDR", table.boundr), ("BOUNDB", table.boundb), ("BOUNDG", table.boundg)):
        if b not in (1, 2):
            bad.append(f"{nm}={b}")
    if bad:
        raise NotImplementedError(
            "sf_urban_physics=1 SLUCM: unsupported URBPARM.TBL option(s): " + "; ".join(bad)
            + ". Supported: WRF defaults (CH_SCHEME=2, TS_SCHEME=1|2, IMP_SCHEME=1|2, "
            "AHOPTION/ALHOPTION 0|1, GROPTION=0, IRI_SCHEME=0).")


def load_urban_table(table_dir=None, dtype=np.float32, *, use_wudapt_lcz=False) -> UrbanTable:
    """Read ``URBPARM.TBL`` from the WRF run directory (like MPTABLE.TBL)."""
    if table_dir is None:
        from gpuwrf.config.paths import wrf_run_dir

        table_dir = wrf_run_dir()
    name = "URBPARM_LCZ.TBL" if use_wudapt_lcz else "URBPARM.TBL"
    table = parse_urbparm(Path(table_dir) / name, dtype=dtype, sf_urban_physics=1)
    check_supported_options(table)
    return table


# --------------------------------------------------------------------------------------
# State / IO containers
# --------------------------------------------------------------------------------------


class UrbanState(NamedTuple):
    """SLUCM prognostic state (WRF Registry ``*_URB2D/3D`` + ``C*_SFCDIF``).

    2-D leaves are ``(...)``; layer leaves are ``(..., NUM_URBAN_LAYERS)``.
    """

    tr: jax.Array
    tb: jax.Array
    tg: jax.Array
    tc: jax.Array
    qc: jax.Array
    uc: jax.Array
    trl: jax.Array
    tbl: jax.Array
    tgl: jax.Array
    xxxr: jax.Array
    xxxb: jax.Array
    xxxg: jax.Array
    xxxc: jax.Array
    cmr: jax.Array
    chr: jax.Array
    cmc: jax.Array
    chc: jax.Array
    drelr: jax.Array
    drelb: jax.Array
    drelg: jax.Array
    flxhumr: jax.Array
    flxhumb: jax.Array
    flxhumg: jax.Array


class UrbanForcing(NamedTuple):
    """Inputs of one ``urban`` call (already derived as in ``noahmp_urban``)."""

    ta: jax.Array      # T3D(kts) [K]
    qa: jax.Array      # specific humidity QV/(1+QV)
    ua: jax.Array      # max(|V|, 1)
    u1: jax.Array
    v1: jax.Array
    ssg: jax.Array     # SWDOWN
    llg: jax.Array     # GLW
    rain: jax.Array    # [mm/h]
    rhoo: jax.Array
    za: jax.Array      # 0.5*DZ8W(kts)
    chs: jax.Array     # max(CHS, 0.01)
    chs2: jax.Array    # max(CQS2, 0.01)
    omg: jax.Array     # solar hour angle [rad] (AHOPTION/ALHOPTION only)


class UrbanOutputs(NamedTuple):
    ts: jax.Array
    qs: jax.Array
    sh: jax.Array
    lh: jax.Array
    lh_kinematic: jax.Array
    sw: jax.Array
    alb: jax.Array
    lw: jax.Array
    g: jax.Array
    rn: jax.Array
    psim: jax.Array
    psih: jax.Array
    gz1oz0: jax.Array
    u10: jax.Array
    v10: jax.Array
    th2: jax.Array
    q2: jax.Array
    ust: jax.Array
    za_violation: jax.Array  # WRF FATAL: ZDC+Z0C+2 >= ZA


class UrbanCellParams(NamedTuple):
    zr: jax.Array
    z0c: jax.Array
    z0hc: jax.Array
    zdc: jax.Array
    svf: jax.Array
    r: jax.Array
    rw: jax.Array
    hgt: jax.Array
    ah: jax.Array
    alh: jax.Array
    betr: jax.Array
    betb: jax.Array
    betg: jax.Array
    capr: jax.Array
    capb: jax.Array
    capg: jax.Array
    aksr: jax.Array
    aksb: jax.Array
    aksg: jax.Array
    albr: jax.Array
    albb: jax.Array
    albg: jax.Array
    epsr: jax.Array
    epsb: jax.Array
    epsg: jax.Array
    z0r: jax.Array
    trlend: jax.Array
    tblend: jax.Array
    tglend: jax.Array
    akanda: jax.Array


def cell_params(table: UrbanTable, utype) -> UrbanCellParams:
    """``read_param``: gather the per-category table by UTYPE (1-based)."""
    idx = jnp.maximum(jnp.asarray(utype, jnp.int32), 1) - 1
    dt = table.dtype

    def g(arr):
        return jnp.take(jnp.asarray(arr, dtype=dt), idx, mode="clip")

    return UrbanCellParams(
        zr=g(table.zr), z0c=g(table.z0c), z0hc=g(table.z0hc), zdc=g(table.zdc), svf=g(table.svf),
        r=g(table.r), rw=g(table.rw), hgt=g(table.hgt), ah=g(table.ah), alh=g(table.alh),
        betr=g(table.betr), betb=g(table.betb), betg=g(table.betg), capr=g(table.capr),
        capb=g(table.capb), capg=g(table.capg), aksr=g(table.aksr), aksb=g(table.aksb),
        aksg=g(table.aksg), albr=g(table.albr), albb=g(table.albb), albg=g(table.albg),
        epsr=g(table.epsr), epsb=g(table.epsb), epsg=g(table.epsg), z0r=g(table.z0r),
        trlend=g(table.trlend), tblend=g(table.tblend), tglend=g(table.tglend),
        akanda=g(table.akanda_urban),
    )


# --------------------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------------------


def _c(dt):
    """Typed literal factory (Fortran REAL constant in the working precision)."""
    t = np.dtype(dt).type
    return lambda v: t(v)


def sfcdif_urb(zlm, z0, thz0, thlm, sfcspd, akanda, akms, akhs):
    """``SFCDIF_URB`` (:3116-3324) without ZT_OUT/VEGFRAC; returns (akms, akhs, rlmo, cd)."""
    dt = jnp.result_type(thz0)
    c = _c(dt)
    wwst = c(1.2)
    wwst2 = wwst * wwst
    g = c(9.8)
    vkrm = c(0.40)
    excm = c(0.001)
    beta = c(1.0) / c(270.0)
    btg = beta * g
    elfc = vkrm * btg
    wold = c(0.15)
    wnew = c(1.0) - wold
    pihf = c(3.14159265) / c(2.0)
    epsu2 = c(1.0e-4)
    epsust = c(0.07)
    ztmin = c(-5.0)
    ztmax = c(1.0)
    hpbl = c(1000.0)
    sqvisc = c(258.2)
    sqvisc2 = sqvisc * sqvisc
    two_thirds = c(2.0) / c(3.0)

    def pspmu(xx):
        return (-(c(2.0) * jnp.log((xx + c(1.0)) * c(0.5))) - jnp.log((xx * xx + c(1.0)) * c(0.5))
                + c(2.0) * jnp.arctan(xx)) - pihf

    def pspms(yy):
        return c(5.0) * yy

    def psphu(xx):
        return -(c(2.0) * jnp.log((xx * xx + c(1.0)) * c(0.5)))

    def wstar2_of(akhs_):
        arg = btgh * akhs_ * dthv
        return jnp.where(arg != c(0.0), wwst2 * jnp.abs(arg) ** two_thirds, c(0.0))

    def zt_of(ustar_):
        return jnp.exp(c(2.0) - akanda * (sqvisc2 * ustar_ * z0) ** c(0.25)) * z0

    zu = z0
    rdz = c(1.0) / zlm
    cxch = excm * rdz
    dthv = thlm - thz0
    du2 = jnp.maximum(sfcspd * sfcspd, epsu2)
    btgh = btg * hpbl
    wstar2 = wstar2_of(akhs)
    ustar = jnp.maximum(jnp.sqrt(akms * jnp.sqrt(du2 + wstar2)), epsust)
    zt = zt_of(ustar)
    zslu = zlm + zu
    zslt = zlm + zt
    rlogu = jnp.log(zslu / zu)
    rlogt = jnp.log(zslt / zt)
    rlmo = elfc * akhs * dthv / (ustar * ustar * ustar)

    def body(_, carry):
        rlmo, zt, zslt, rlogt, akms, akhs, wstar2, ustar = carry
        zetalt = jnp.maximum(zslt * rlmo, ztmin)
        rlmo = zetalt / zslt
        zetalu = zslu * rlmo
        zetau = zu * rlmo
        zetat = zt * rlmo
        unstable = rlmo < c(0.0)
        # unstable branch (Paulson)
        xlu = jnp.sqrt(jnp.sqrt(c(1.0) - c(16.0) * zetalu))
        xlt = jnp.sqrt(jnp.sqrt(c(1.0) - c(16.0) * zetalt))
        xu = jnp.sqrt(jnp.sqrt(c(1.0) - c(16.0) * zetau))
        xt = jnp.sqrt(jnp.sqrt(c(1.0) - c(16.0) * zetat))
        psmz_u = pspmu(xu)
        simm_u = pspmu(xlu) - psmz_u + rlogu
        pshz_u = psphu(xt)
        simh_u = psphu(xlt) - pshz_u + rlogt
        # stable branch
        zetalu_s = jnp.minimum(zetalu, ztmax)
        zetalt_s = jnp.minimum(zetalt, ztmax)
        psmz_s = pspms(zetau)
        simm_s = pspms(zetalu_s) - psmz_s + rlogu
        pshz_s = pspms(zetat)
        simh_s = pspms(zetalt_s) - pshz_s + rlogt
        simm = jnp.where(unstable, simm_u, simm_s)
        simh = jnp.where(unstable, simh_u, simh_s)
        ustar = jnp.maximum(jnp.sqrt(akms * jnp.sqrt(du2 + wstar2)), epsust)
        zt = zt_of(ustar)
        zslt = zlm + zt
        rlogt = jnp.log(zslt / zt)
        ustark = ustar * vkrm
        akms = jnp.maximum(ustark / simm, cxch)
        akhs = jnp.maximum(ustark / simh, cxch)
        wstar2 = wstar2_of(akhs)
        rlmn = elfc * akhs * dthv / (ustar * ustar * ustar)
        rlma = rlmo * wold + rlmn * wnew
        return (rlma, zt, zslt, rlogt, akms, akhs, wstar2, ustar)

    carry = lax.fori_loop(0, 5, body, (rlmo, zt, zslt, rlogt, akms, akhs, wstar2, ustar))
    rlmo, zt, zslt, rlogt, akms, akhs, wstar2, ustar = carry
    cd = ustar * ustar / (sfcspd * sfcspd)
    return akms, akhs, rlmo, cd


def multi_layer(bound: int, g0, cap, aks, tsl, dz, delt, tslend):
    """``multi_layer`` (:1861-1936): implicit layer heat conduction, Thomas solve.

    ``tsl`` is ``(..., km)``; ``dz`` a length-``km`` sequence (cm), ``bound`` static.
    """
    dt = jnp.result_type(tsl)
    c = _c(dt)
    km = tsl.shape[-1]
    dzs = [c(float(v)) for v in np.asarray(dz, dtype=dt)]
    dzend = dzs[km - 1]
    two = c(2.0)
    a = [None] * km
    b = [None] * km
    cc = [None] * km
    d = [None] * km
    a[0] = c(0.0)
    b[0] = cap * dzs[0] / delt + two * aks / (dzs[0] + dzs[1])
    cc[0] = -(two * aks / (dzs[0] + dzs[1]))
    d[0] = cap * dzs[0] / delt * tsl[..., 0] + g0
    for k in range(1, km - 1):
        a[k] = -(two * aks / (dzs[k - 1] + dzs[k]))
        b[k] = cap * dzs[k] / delt + two * aks / (dzs[k - 1] + dzs[k]) + two * aks / (dzs[k] + dzs[k + 1])
        cc[k] = -(two * aks / (dzs[k] + dzs[k + 1]))
        d[k] = cap * dzs[k] / delt * tsl[..., k]
    k = km - 1
    a[k] = -(two * aks / (dzs[k - 1] + dzs[k]))
    if bound == 1:
        b[k] = cap * dzs[k] / delt + two * aks / (dzs[k - 1] + dzs[k])
        cc[k] = c(0.0)
        d[k] = cap * dzs[k] / delt * tsl[..., k]
    else:
        b[k] = cap * dzs[k] / delt + two * aks / (dzs[k - 1] + dzs[k]) + two * aks / (dzs[k] + dzend)
        cc[k] = c(0.0)
        d[k] = cap * dzs[k] / delt * tsl[..., k] + two * aks * tslend / (dzs[k] + dzend)
    p = [None] * km
    q = [None] * km
    p[0] = -(cc[0] / b[0])
    q[0] = d[0] / b[0]
    for k in range(1, km):
        den = a[k] * p[k - 1] + b[k]
        p[k] = -(cc[k] / den)
        q[k] = (-(a[k] * q[k - 1]) + d[k]) / den
    x = [None] * km
    x[km - 1] = q[km - 1]
    for k in range(km - 2, -1, -1):
        x[k] = p[k] * x[k + 1] + q[k]
    return jnp.stack([jnp.broadcast_to(v, tsl.shape[:-1]) for v in x], axis=-1)


def force_restore(cap, aks, delt, s, r, h, le, tslend, tsp):
    """``force_restore`` (:3058-3069)."""
    dt = jnp.result_type(tsp)
    c = _c(dt)
    c2 = c(24.0) * c(3600.0) / c(2.0) / c(3.14159)
    c1 = jnp.sqrt(c(0.5) * c2 * cap * aks)
    return tsp + delt * ((s + r - h - le) / c1 - (tsp - tslend) / c2)


def _hour_index(omg, pi, nper, c):
    """``tloc``/``tloc2`` (:600-610): Fortran INT (trunc) + MOD (sign of dividend)."""
    if nper == 24:
        x = omg / pi * c(180.0) / c(15.0) + c(12.0) + c(0.5)
    else:
        x = (omg / pi * c(180.0) / c(15.0) + c(12.0)) * c(2.0) + c(0.5)
    t = lax.rem(jnp.trunc(x).astype(jnp.int32), jnp.int32(nper))
    t = jnp.where(t < 0, t + nper, t)
    return jnp.where(t == 0, nper, t)


def urban(table: UrbanTable, p: UrbanCellParams, f: UrbanForcing, s: UrbanState, delt,
          jmonth=None) -> tuple[UrbanOutputs, UrbanState]:
    """One SLUCM step for every cell (``SUBROUTINE urban``, default WRF branches)."""
    dt = jnp.result_type(f.ta)
    c = _c(dt)
    cp = c(0.24)
    el = c(583.0)
    sig = c(8.17e-11)
    pi = c(3.14159)
    sratio = c(0.75)
    cpp = c(1004.5)
    ell = c(2.442e06)
    albv = c(0.20)  # green roof albedo (SGR1 only; GROPTION=0 never uses it)
    del albv
    delt = c(delt) if not hasattr(delt, "dtype") else jnp.asarray(delt, dt)
    hundred = c(100.0)
    e_c1 = (c(2.5) * (c(10.0) ** c(6.0))) / c(461.51)

    ah = p.ah
    alh = p.alh
    if table.ahoption == 1:
        tloc = _hour_index(f.omg, pi, 24, c)
        ah = ah * jnp.take(jnp.asarray(table.ahdiuprf, dt), tloc - 1)
    if table.alhoption == 1:
        tloc2 = _hour_index(f.omg, pi, 48, c)
        jm = jnp.asarray(jmonth, jnp.int32)
        kalh = jnp.where((jm == 3) | (jm == 4) | (jm == 5), 1,
               jnp.where((jm == 6) | (jm == 7) | (jm == 8), 2,
               jnp.where((jm == 9) | (jm == 10) | (jm == 11), 3, 4)))
        alh = alh * jnp.take(jnp.asarray(table.alhdiuprf, dt), tloc2 - 1) \
            * jnp.take(jnp.asarray(table.alhseason, dt), kalh - 1)

    zr, zdc, z0c, r, rw, hgt, svf = p.zr, p.zdc, p.z0c, p.r, p.rw, p.hgt, p.svf
    ta, qa, ua, za = f.ta, f.qa, f.ua, f.za
    za_violation = (zdc + z0c + c(2.0)) >= za

    ssg = f.ssg
    ssgd = sratio * ssg
    ssgq = ssg - ssgd

    w = c(2.0) * c(1.0) * hgt
    vfgs = svf
    vfgw = c(1.0) - svf
    vfwg = (c(1.0) - svf) * (c(1.0) - r) / w
    vfws = vfwg
    vfww = c(1.0) - c(2.0) * vfwg

    sx = (ssgd + ssgq) / c(697.7) / c(60.0)
    rx = f.llg / c(697.7) / c(60.0)
    rho = f.rhoo * c(0.001)

    trp, tbp, tgp, tcp, qcp = s.tr, s.tb, s.tg, s.tc, s.qc
    flxhumrp, flxhumbp, flxhumgp = s.flxhumr, s.flxhumb, s.flxhumg
    drelrp, drelbp, drelgp = s.drelr, s.drelb, s.drelg

    tav = ta * (c(1.0) + c(0.61) * qa)
    ps = f.rhoo * c(287.0) * tav / c(100.0)

    # ---- canopy wind
    tall = zr + c(2.0) < za
    ur = ua * jnp.log((zr - zdc) / z0c) / jnp.log((za - zdc) / z0c)
    zc_t = c(0.7) * zr
    xlb = c(0.4) * (zr - zdc)
    bb = c(0.4) * zr / (xlb * jnp.log((zr - zdc) / z0c))
    uc_t = ur * jnp.exp(-bb * (c(1.0) - zc_t / zr))
    uc = jnp.where(tall, uc_t, ua / c(2.0))

    # ---- net short wave (no shadow model)
    albr, albb, albg = p.albr, p.albb, p.albg
    sr1 = sx * (c(1.0) - albr)
    sg1 = sx * vfgs * (c(1.0) - albg)
    sb1 = sx * vfws * (c(1.0) - albb)
    sg2 = sb1 * albb / (c(1.0) - albb) * vfgw * (c(1.0) - albg)
    sb2 = sg1 * albg / (c(1.0) - albg) * vfwg * (c(1.0) - albb) + sb1 * albb * vfww
    sun = ssg > c(0.0)
    zero = jnp.zeros_like(ta)
    sr = jnp.where(sun, sr1, zero)
    sg = jnp.where(sun, sg1 + sg2, zero)
    sb = jnp.where(sun, sb1 + sb2, zero)
    snet = jnp.where(sun, r * sr1 + w * (sb1 + sb2) + rw * (sg1 + sg2), zero)

    # ---- roof exchange (SFCDIF_URB, MOST)
    t1vr = trp * (c(1.0) + c(0.61) * qa)
    th2v = (ta + (c(0.0098) * za)) * (c(1.0) + c(0.61) * qa)
    cmr, chr_urb, _, cdr = sfcdif_urb(za, p.z0r, t1vr, th2v, ua, p.akanda, s.cmr, s.chr)
    alphar = rho * cp * chr_urb
    chr_ = alphar / rho / cp / ua

    rain = f.rain
    rain1 = rain * c(0.001) / c(3600.0)
    betr = p.betr
    drelr = drelrp
    if table.imp_scheme == 1:
        betr = jnp.where(rain > c(1.0), c(0.7), betr)
    else:
        porr, denr = c(float(table.porimp[0])), c(float(table.dengimp[0]))
        flxhumrp = jnp.where(flxhumrp <= c(0.0), c(0.0), flxhumrp)
        drelr = drelrp + (rain1 - flxhumrp * f.rhoo / c(1000.0)) * delt / porr
        drelr = jnp.where((rain > c(0.0)) & (drelr < drelrp), drelrp, drelr)
        betr = jnp.where(drelr <= c(0.0), c(0.0), jnp.where(drelr <= denr, drelr / denr * porr, porr))
        drelr = jnp.where(drelr <= c(0.0), c(0.0), jnp.where(drelr <= denr, drelr, denr))
        betr = jnp.where(betr < c(1.0e-5), c(0.0), betr)

    aksr, capr = p.aksr, p.capr
    dzr = [c(float(v)) for v in np.asarray(table.dzr, dt)]
    trl1 = s.trl[..., 0]

    def es_of(t):
        return c(6.11) * jnp.exp(e_c1 * (t - c(273.15)) / (c(273.15) * t))

    def t4(t):
        return t ** c(4.0)

    def t3(t):
        return t ** c(3.0)

    if table.ts_scheme == 1:
        def roof_body(_, carry):
            trp_, done, rr, hr, eler, g0r = carry
            es = es_of(trp_)
            desdt = e_c1 * es / (trp_ * trp_)
            qs0r = c(0.622) * es / (ps - c(0.378) * es)
            dqs0rdtr = desdt * c(0.622) * ps / ((ps - c(0.378) * es) * (ps - c(0.378) * es))
            rr_n = p.epsr * (rx - sig * t4(trp_) / c(60.0))
            hr_n = rho * cp * chr_ * ua * (trp_ - ta) * hundred
            eler_n = rho * el * chr_ * ua * betr * (qs0r - qa) * hundred
            g0r_n = aksr * (trp_ - trl1) / (dzr[0] / c(2.0))
            fv = sr + rr_n - hr_n - eler_n - g0r_n
            drrdtr = (-(c(4.0) * p.epsr * sig * t3(trp_))) / c(60.0)
            dhrdtr = rho * cp * chr_ * ua * hundred
            delerdtr = rho * el * chr_ * ua * betr * dqs0rdtr * hundred
            dg0rdtr = c(2.0) * aksr / dzr[0]
            dfdt = drrdtr - dhrdtr - delerdtr - dg0rdtr
            dtr = fv / dfdt
            tr_n = trp_ - dtr
            conv = (jnp.abs(fv) < c(0.000001)) & (jnp.abs(dtr) < c(0.000001))
            keep = done
            return (jnp.where(keep, trp_, tr_n), done | conv,
                    jnp.where(keep, rr, rr_n), jnp.where(keep, hr, hr_n),
                    jnp.where(keep, eler, eler_n), jnp.where(keep, g0r, g0r_n))

        init = (trp, jnp.zeros(trp.shape, bool), zero, zero, zero, zero)
        tr, _, rr, hr, eler, g0r = lax.fori_loop(0, 20, roof_body, init)
        trl = multi_layer(table.boundr, g0r, capr, aksr, s.trl, table.dzr, delt, p.trlend)
    else:
        es = es_of(trp)
        qs0r = c(0.622) * es / (ps - c(0.378) * es)
        rr = p.epsr * (rx - sig * t4(trp) / c(60.0))
        hr = rho * cp * chr_ * ua * (trp - ta) * hundred
        eler = rho * el * chr_ * ua * betr * (qs0r - qa) * hundred
        g0r = sr + rr - hr - eler
        tr = force_restore(capr, aksr, delt, sr, rr, hr, eler, p.trlend, trp)
        trl = s.trl

    flxthr = hr / rho / cp / hundred
    flxhumr = eler / rho / el / hundred

    # ---- canyon exchange
    t1vc = tcp * (c(1.0) + c(0.61) * qa)
    cmc, chc_urb, _, cdc = sfcdif_urb(za, z0c, t1vc, th2v, ua, p.akanda, s.cmc, s.chc)
    alphac = rho * cp * chc_urb
    # CH_SCHEME == 2 (empirical, Jurges)
    alphab_lo = rho * cp * (c(6.15) + c(4.18) * uc) / c(1200.0)
    alphab_hi = rho * cp * (c(7.51) * uc ** c(0.78)) / c(1200.0)
    alphab = jnp.where(uc > c(5.0), alphab_hi, alphab_lo)
    alphag = alphab
    chc = alphac / rho / cp / ua
    chb = alphab / rho / cp / uc
    chg = alphag / rho / cp / uc

    betb = p.betb
    betg = p.betg
    drelb, drelg = drelbp, drelgp
    if table.imp_scheme == 1:
        betb = jnp.zeros_like(betb)
        betg = jnp.where(rain > c(1.0), c(0.7), betg)
    else:
        porb, denb = c(float(table.porimp[1])), c(float(table.dengimp[1]))
        porg, deng = c(float(table.porimp[2])), c(float(table.dengimp[2]))
        flxhumbp = jnp.where(flxhumbp <= c(0.0), c(0.0), flxhumbp)
        flxhumgp = jnp.where(flxhumgp <= c(0.0), c(0.0), flxhumgp)
        drelb = drelbp + (rain1 - flxhumbp * f.rhoo / c(1000.0)) * delt / porb
        drelb = jnp.where((rain > c(0.0)) & (drelb < drelbp), drelbp, drelb)
        drelg = drelgp + (rain1 - flxhumgp * f.rhoo / c(1000.0)) * delt / porg
        drelg = jnp.where((rain > c(0.0)) & (drelg < drelgp), drelgp, drelg)
        betb = jnp.where(drelb <= c(0.0), c(0.0), jnp.where(drelb <= denb, drelb / denb * porb, porb))
        drelb = jnp.where(drelb <= c(0.0), c(0.0), jnp.where(drelb <= denb, drelb, denb))
        betg = jnp.where(drelg <= c(0.0), c(0.0), jnp.where(drelg <= deng, drelg / deng * porg, porg))
        drelg = jnp.where(drelg <= c(0.0), c(0.0), jnp.where(drelg <= deng, drelg, deng))
        betg = jnp.where(betg < c(1.0e-5), c(0.0), betg)
        betb = jnp.where(betb < c(1.0e-5), c(0.0), betb)

    epsb, epsg = p.epsb, p.epsg
    aksb, aksg, capb, capg = p.aksb, p.aksg, p.capb, p.capg
    dzb = [c(float(v)) for v in np.asarray(table.dzb, dt)]
    dzg = [c(float(v)) for v in np.asarray(table.dzg, dt)]
    tbl1 = s.tbl[..., 0]
    tgl1 = s.tgl[..., 0]
    one = c(1.0)

    def rad_bg(tbp_, tgp_):
        rg1 = epsg * (rx * vfgs + epsb * vfgw * sig * t4(tbp_) / c(60.0) - sig * t4(tgp_) / c(60.0))
        rb1 = epsb * (rx * vfws + epsg * vfwg * sig * t4(tgp_) / c(60.0)
                      + epsb * vfww * sig * t4(tbp_) / c(60.0) - sig * t4(tbp_) / c(60.0))
        rg2 = epsg * ((one - epsb) * (one - svf) * vfws * rx
                      + (one - epsb) * (one - svf) * vfwg * epsg * sig * t4(tgp_) / c(60.0)
                      + epsb * (one - epsb) * (one - svf) * (one - c(2.0) * vfws) * sig * t4(tbp_) / c(60.0))
        rb2 = epsb * ((one - epsg) * vfwg * vfgs * rx
                      + (one - epsg) * epsb * vfgw * vfwg * sig * t4(tbp_) / c(60.0)
                      + (one - epsb) * vfws * (one - c(2.0) * vfws) * rx
                      + (one - epsb) * vfwg * (one - c(2.0) * vfws) * sig * epsg * t4(tgp_) / c(60.0)
                      + epsb * (one - epsb) * (one - c(2.0) * vfws) * (one - c(2.0) * vfws) * sig
                      * t4(tbp_) / c(60.0))
        return rg1 + rg2, rb1 + rb2

    if table.ts_scheme == 1:
        def wall_body(_, carry):
            (tbp_, tgp_, tcp_, qcp_, done, rb, rg, hb, hg, eleb, eleg, g0b, g0g) = carry
            es = es_of(tbp_)
            desdt = e_c1 * es / (tbp_ * tbp_)
            qs0b = c(0.622) * es / (ps - c(0.378) * es)
            dqs0bdtb = desdt * c(0.622) * ps / ((ps - c(0.378) * es) * (ps - c(0.378) * es))
            es = es_of(tgp_)
            desdt = e_c1 * es / (tgp_ * tgp_)
            qs0g = c(0.622) * es / (ps - c(0.378) * es)
            dqs0gdtg = desdt * c(0.622) * ps / ((ps - c(0.378) * es) * (ps - c(0.378) * es))
            rg_n, rb_n = rad_bg(tbp_, tgp_)
            tb3 = t3(tbp_)
            tg3 = t3(tgp_)
            drbdtb1 = epsb * (c(4.0) * epsb * sig * tb3 * vfww - c(4.0) * sig * tb3) / c(60.0)
            drbdtg1 = epsb * (c(4.0) * epsg * sig * tg3 * vfwg) / c(60.0)
            drbdtb2 = epsb * (c(4.0) * (one - epsg) * epsb * sig * tb3 * vfgw * vfwg
                              + c(4.0) * epsb * (one - epsb) * sig * tb3 * vfww * vfww) / c(60.0)
            drbdtg2 = epsb * (c(4.0) * (one - epsb) * epsg * sig * tg3 * vfwg * vfww) / c(60.0)
            drgdtb1 = epsg * (c(4.0) * epsb * sig * tb3 * vfgw) / c(60.0)
            drgdtg1 = epsg * (-(c(4.0) * sig * tg3)) / c(60.0)
            drgdtb2 = epsg * (c(4.0) * epsb * (one - epsb) * sig * tb3 * vfww * vfgw) / c(60.0)
            drgdtg2 = epsg * (c(4.0) * (one - epsb) * epsg * sig * tg3 * vfwg * vfgw) / c(60.0)
            drbdtb = drbdtb1 + drbdtb2
            drbdtg = drbdtg1 + drbdtg2
            drgdtb = drgdtb1 + drgdtb2
            drgdtg = drgdtg1 + drgdtg2
            hb_n = rho * cp * chb * uc * (tbp_ - tcp_) * hundred
            hg_n = rho * cp * chg * uc * (tgp_ - tcp_) * hundred
            den_t = rw * alphac + rw * alphag + w * alphab
            dtcdtb = w * alphab / den_t
            dtcdtg = rw * alphag / den_t
            dhbdtb = rho * cp * chb * uc * (one - dtcdtb) * hundred
            dhbdtg = rho * cp * chb * uc * (c(0.0) - dtcdtg) * hundred
            dhgdtg = rho * cp * chg * uc * (one - dtcdtg) * hundred
            dhgdtb = rho * cp * chg * uc * (c(0.0) - dtcdtb) * hundred
            eleb_n = rho * el * chb * uc * betb * (qs0b - qcp_) * hundred
            eleg_n = rho * el * chg * uc * betg * (qs0g - qcp_) * hundred
            den_q = rw * alphac + rw * alphag * betg + w * alphab * betb
            dqcdtb = w * alphab * betb * dqs0bdtb / den_q
            dqcdtg = rw * alphag * betg * dqs0gdtg / den_q
            delebdtb = rho * el * chb * uc * betb * (dqs0bdtb - dqcdtb) * hundred
            delebdtg = rho * el * chb * uc * betb * (c(0.0) - dqcdtg) * hundred
            delegdtg = rho * el * chg * uc * betg * (dqs0gdtg - dqcdtg) * hundred
            delegdtb = rho * el * chg * uc * betg * (c(0.0) - dqcdtb) * hundred
            g0b_n = aksb * (tbp_ - tbl1) / (dzb[0] / c(2.0))
            g0g_n = aksg * (tgp_ - tgl1) / (dzg[0] / c(2.0))
            dg0bdtb = c(2.0) * aksb / dzb[0]
            dg0bdtg = c(0.0)
            dg0gdtg = c(2.0) * aksg / dzg[0]
            dg0gdtb = c(0.0)
            fb = sb + rb_n - hb_n - eleb_n - g0b_n
            fx = drbdtb - dhbdtb - delebdtb - dg0bdtb
            fy = drbdtg - dhbdtg - delebdtg - dg0bdtg
            gf = sg + rg_n - hg_n - eleg_n - g0g_n
            gx = drgdtb - dhgdtb - delegdtb - dg0gdtb
            gy = drgdtg - dhgdtg - delegdtg - dg0gdtg
            dtb = (gf * fy - fb * gy) / (fx * gy - gx * fy)
            dtg = -(gf + gx * dtb) / gy
            tb_n = tbp_ + dtb
            tg_n = tgp_ + dtg
            tc1 = rw * alphac + rw * alphag + w * alphab
            tc2 = rw * alphac * ta + rw * alphag * tg_n + w * alphab * tb_n
            tc_n = tc2 / tc1
            qc1 = rw * alphac + rw * alphag * betg + w * alphab * betb
            qc2 = rw * alphac * qa + rw * alphag * betg * qs0g + w * alphab * betb * qs0b
            qc_n = qc2 / qc1
            dtc = tcp_ - tc_n
            conv = ((jnp.abs(fb) < c(0.000001)) & (jnp.abs(dtb) < c(0.000001))
                    & (jnp.abs(gf) < c(0.000001)) & (jnp.abs(dtg) < c(0.000001))
                    & (jnp.abs(dtc) < c(0.000001)))
            k = done

            def sel(old, new):
                return jnp.where(k, old, new)

            return (sel(tbp_, tb_n), sel(tgp_, tg_n), sel(tcp_, tc_n), sel(qcp_, qc_n), done | conv,
                    sel(rb, rb_n), sel(rg, rg_n), sel(hb, hb_n), sel(hg, hg_n),
                    sel(eleb, eleb_n), sel(eleg, eleg_n), sel(g0b, g0b_n), sel(g0g, g0g_n))

        init = (tbp, tgp, tcp, qcp, jnp.zeros(tbp.shape, bool),
                zero, zero, zero, zero, zero, zero, zero, zero)
        tb, tg, tc, qc, _, rb, rg, hb, hg, eleb, eleg, g0b, g0g = lax.fori_loop(0, 20, wall_body, init)
        tbl = multi_layer(table.boundb, g0b, capb, aksb, s.tbl, table.dzb, delt, p.tblend)
        tgl = multi_layer(table.boundg, g0g, capg, aksg, s.tgl, table.dzg, delt, p.tglend)
    else:
        es = es_of(tbp)
        qs0b = c(0.622) * es / (ps - c(0.378) * es)
        es = es_of(tgp)
        qs0g = c(0.622) * es / (ps - c(0.378) * es)
        rg, rb = rad_bg(tbp, tgp)
        hb = rho * cp * chb * uc * (tbp - tcp) * hundred
        eleb = rho * el * chb * uc * betb * (qs0b - qcp) * hundred
        g0b = sb + rb - hb - eleb
        hg = rho * cp * chg * uc * (tgp - tcp) * hundred
        eleg = rho * el * chg * uc * betg * (qs0g - qcp) * hundred
        g0g = sg + rg - hg - eleg
        tb = force_restore(capb, aksb, delt, sb, rb, hb, eleb, p.tblend, tbp)
        tg = force_restore(capg, aksg, delt, sg, rg, hg, eleg, p.tglend, tgp)
        tc1 = rw * alphac + rw * alphag + w * alphab
        tc2 = rw * alphac * ta + rw * alphag * tg + w * alphab * tb
        tc = tc2 / tc1
        qc1 = rw * alphac + rw * alphag * betg + w * alphab * betb
        qc2 = rw * alphac * qa + rw * alphag * betg * qs0g + w * alphab * betb * qs0b
        qc = qc2 / qc1
        tbl, tgl = s.tbl, s.tgl

    flxthb = hb / rho / cp / hundred
    flxhumb = eleb / rho / el / hundred
    flxthg = hg / rho / cp / hundred
    flxhumg = eleg / rho / el / hundred

    # ---- totals (GROPTION=0)
    flxth = r * flxthr + w * flxthb + rw * flxthg
    if table.ahoption == 1:
        flxth = flxth + ah / f.rhoo / cpp
    flxhum = r * flxhumr + w * flxhumb + rw * flxhumg
    if table.alhoption == 1:
        flxhum = flxhum + alh / f.rhoo / ell
    flxuv = (r * cdr + rw * cdc) * ua * ua
    flxg = r * g0r + w * g0b + rw * g0g
    lnet = r * rr + w * rb + rw * rg

    sh = flxth * f.rhoo * cpp
    lh = flxhum * f.rhoo * ell
    lh_kin = flxhum * f.rhoo
    lw = f.llg - (lnet * c(697.7) * c(60.0))
    sw = ssg - (snet * c(697.7) * c(60.0))
    alb = jnp.where(jnp.abs(ssg) > c(0.0001), sw / ssg, c(0.0))
    g = -(flxg * c(697.7) * c(60.0))
    rn = (snet + lnet) * c(697.7) * c(60.0)
    ust = jnp.sqrt(flxuv)
    tst = -(flxth / ust)
    z0 = z0c
    z0h = p.z0hc
    z = za - zdc

    def clamp(x):
        x = jnp.where(x >= c(1.0), c(1.0), x)
        return jnp.where(x <= c(-5.0), c(-5.0), x)

    def psi(x, last):
        xx = (c(1.0) - c(16.0) * x) ** c(0.25)
        psim_u = (c(2.0) * jnp.log((c(1.0) + xx) / c(2.0)) + jnp.log((c(1.0) + xx * xx) / c(2.0))
                  - c(2.0) * jnp.arctan(xx) + last)
        psih_u = c(2.0) * jnp.log((c(1.0) + xx * xx) / c(2.0))
        st = x > c(0.0)
        return (jnp.where(st, c(-5.0) * x, psim_u), jnp.where(st, c(-5.0) * x, psih_u))

    xxx = clamp(c(0.4) * c(9.81) * z * tst / ta / ust / ust)
    psim, psih = psi(xxx, pi / c(2.0))
    gz1oz0 = jnp.log(z / z0)
    ts = ta + flxth / f.chs
    qs = qa + flxhum / f.chs
    xxx2 = clamp((c(2.0) / z) * xxx)
    psim2, psih2 = psi(xxx2, c(2.0) * jnp.arctan(c(1.0)))
    xxx10 = clamp((c(10.0) / z) * xxx)
    psim10, _ = psi(xxx10, c(2.0) * jnp.arctan(c(1.0)))
    psix = jnp.log(z / z0) - psim
    psit = jnp.log(z / z0h) - psih
    psit2 = jnp.log(c(2.0) / z0h) - psih2
    psix10 = jnp.log(c(10.0) / z0) - psim10
    u10 = f.u1 * (psix10 / psix)
    v10 = f.v1 * (psix10 / psix)
    th2 = ts + (ta - ts) * (f.chs / f.chs2)
    q2 = qs + (qa - qs) * (psit2 / psit)

    out = UrbanOutputs(ts=ts, qs=qs, sh=sh, lh=lh, lh_kinematic=lh_kin, sw=sw, alb=alb, lw=lw, g=g,
                       rn=rn, psim=psim, psih=psih, gz1oz0=gz1oz0, u10=u10, v10=v10, th2=th2,
                       q2=q2, ust=ust, za_violation=za_violation)
    new = UrbanState(tr=tr, tb=tb, tg=tg, tc=tc, qc=qc, uc=uc, trl=trl, tbl=tbl, tgl=tgl,
                     xxxr=s.xxxr, xxxb=s.xxxb, xxxg=s.xxxg, xxxc=s.xxxc, cmr=cmr, chr=chr_urb,
                     cmc=cmc, chc=chc_urb, drelr=drelr, drelb=drelb, drelg=drelg,
                     flxhumr=flxhumr, flxhumb=flxhumb, flxhumg=flxhumg)
    return out, new


# --------------------------------------------------------------------------------------
# noahmp_urban wrapper + surface-driver diagnostics
# --------------------------------------------------------------------------------------


class UrbanLSMFields(NamedTuple):
    """Noah-MP (rural) surface fields the urban blend reads and overwrites."""

    tsk: jax.Array
    hfx: jax.Array
    qfx: jax.Array
    lh: jax.Array
    grdflx: jax.Array
    albedo: jax.Array
    qsfc: jax.Array
    ust: jax.Array


class UrbanDiagnostics(NamedTuple):
    """``*_URB2D`` outputs of ``noahmp_urban`` (history/surface-driver inputs)."""

    ts_urb: jax.Array
    sh_urb: jax.Array
    lh_urb: jax.Array
    g_urb: jax.Array
    rn_urb: jax.Array
    psim_urb: jax.Array
    psih_urb: jax.Array
    gz1oz0_urb: jax.Array
    u10_urb: jax.Array
    v10_urb: jax.Array
    th2_urb: jax.Array
    q2_urb: jax.Array
    ust_urb: jax.Array
    akms_urb: jax.Array
    chs: jax.Array       # CHS after the urban lower limit
    chs2: jax.Array      # CHS2 := max(CQS2, 0.01) on urban cells
    cqs2: jax.Array
    za_violation: jax.Array


def noahmp_urban(table: UrbanTable, *, urban_mask, frc_urb, utype, t3d1, qv1, u1, v1,
                 swdown, glw, p8w1, p8w2, rainbl, dz8w1, dt, chs, chs2, cqs2,
                 lsm: UrbanLSMFields, state: UrbanState, omg=None, jmonth=None, morphology=None):
    """``noahmp_urban`` for ``sf_urban_physics == 1`` (module_sf_noahmpdrv.F:3373-3545).

    Every argument is a 2-D (or any common-shape) field; ``urban_mask`` selects the
    ISURBAN cells.  Returns ``(lsm', state', diag)`` where non-urban cells keep
    their input values bit-for-bit.
    """
    dt_ = jnp.result_type(t3d1)
    c = _c(dt_)
    karman = c(0.4)
    ta = t3d1
    qa = qv1 / (c(1.0) + qv1)
    ua = jnp.sqrt(u1 * u1 + v1 * v1)
    ua = jnp.where(ua < c(1.0), c(1.0), ua)
    delt = c(float(dt))
    rain = rainbl / delt * c(3600.0)
    rhoo = (p8w2 + p8w1) * c(0.5) / (c(287.04) * ta * (c(1.0) + c(0.61) * qa))
    za = c(0.5) * dz8w1
    chs_l = jnp.where(chs < c(1.0e-02), c(1.0e-02), chs)
    chs2_l = jnp.where(chs2 < c(1.0e-02), c(1.0e-02), chs2)
    cqs2_l = jnp.where(cqs2 < c(1.0e-02), c(1.0e-02), cqs2)
    omg_ = jnp.zeros_like(ta) if omg is None else omg
    forc = UrbanForcing(ta=ta, qa=qa, ua=ua, u1=u1, v1=v1, ssg=swdown, llg=glw, rain=rain,
                        rhoo=rhoo, za=za, chs=chs_l, chs2=cqs2_l, omg=omg_)
    params = cell_params(table, utype)
    if morphology is not None:
        params = params._replace(**morphology._asdict())
    out, new = urban(table, params, forc, state, delt, jmonth=jmonth)

    frc = frc_urb
    rural = c(1.0) - frc
    m = urban_mask

    def blend(urb, rur):
        return jnp.where(m, frc * urb + rural * rur, rur)

    lsm_out = UrbanLSMFields(
        tsk=blend(out.ts, lsm.tsk),
        hfx=blend(out.sh, lsm.hfx),
        qfx=blend(out.lh_kinematic, lsm.qfx),
        lh=blend(out.lh, lsm.lh),
        grdflx=blend(out.g * c(-1.0), lsm.grdflx),
        albedo=blend(out.alb, lsm.albedo),
        qsfc=blend(out.qs, lsm.qsfc),
        ust=blend(out.ust, lsm.ust),
    )

    def keep(new_leaf, old_leaf):
        mm = m if new_leaf.ndim == m.ndim else m[..., None]
        return jnp.where(mm, new_leaf, old_leaf)

    state_out = UrbanState(*[keep(n_, o_) for n_, o_ in zip(new, state)])
    zero = jnp.zeros_like(ta)

    def um(x):
        return jnp.where(m, x, zero)

    diag = UrbanDiagnostics(
        ts_urb=um(out.ts), sh_urb=um(out.sh), lh_urb=um(out.lh), g_urb=um(out.g), rn_urb=um(out.rn),
        psim_urb=um(out.psim), psih_urb=um(out.psih), gz1oz0_urb=um(out.gz1oz0),
        u10_urb=um(out.u10), v10_urb=um(out.v10), th2_urb=um(out.th2), q2_urb=um(out.q2),
        ust_urb=um(out.ust),
        akms_urb=um(karman * out.ust / (out.gz1oz0 - out.psim)),
        chs=jnp.where(m, chs_l, chs), chs2=jnp.where(m, cqs2_l, chs2), cqs2=jnp.where(m, cqs2_l, cqs2),
        za_violation=m & out.za_violation,
    )
    return lsm_out, state_out, diag


def urban_surface_diagnostics(*, urban_mask, frc_urb, fveg, t2mv, t2mb, q2mv, q2mb, psfc, diag,
                              t2, q2, u10, v10, rcp):
    """Surface-driver urban 2-m / 10-m overwrite (module_surface_driver.F:3485-3500)."""
    dt_ = jnp.result_type(t2mv)
    c = _c(dt_)
    one = c(1.0)
    frc = frc_urb
    q2_u = (fveg * q2mv + (one - fveg) * q2mb) * (one - frc) + diag.q2_urb * frc
    t2_u = ((fveg * t2mv + (one - fveg) * t2mb) * (one - frc)
            + (diag.th2_urb / ((c(1.0e5) / psfc) ** rcp)) * frc)
    m = urban_mask
    return (jnp.where(m, t2_u, t2), jnp.where(m, q2_u, q2),
            jnp.where(m, diag.u10_urb, u10), jnp.where(m, diag.v10_urb, v10))


def urban_var_init(table: UrbanTable, *, urban_mask, tsk, tslb, frc_urb_in, dtype=None,
                   ivgtyp=None, isurban=13, use_wudapt_lcz=False):
    """``urban_var_init`` (non-restart, sf_urban_physics=1, no NUDAPT/LCZ).

    ``tslb`` is ``(..., nsoil)`` (nsoil >= 3).  Returns ``(state, frc_urb, utype)``:
    UTYPE=2 (high-intensity) on ISURBAN cells, FRC_URB2D from wrfinput when in
    ``(0, 1]`` else ``FRC_URB_TBL(UTYPE)``; non-urban cells get FRC=0, UTYPE=0.
    """
    dt_ = jnp.result_type(tsk) if dtype is None else dtype
    c = _c(dt_)
    tsk = jnp.asarray(tsk, dt_)
    tslb = jnp.asarray(tslb, dt_)
    m = jnp.asarray(urban_mask, bool)
    utype = jnp.where(m, 2, 0).astype(jnp.int32)
    if use_wudapt_lcz:
        v = jnp.asarray(ivgtyp, jnp.int32)
        utype = jnp.where(v == isurban, 5, jnp.where((v >= 51) & (v <= 61), v - 50, 0))
    frc_in = jnp.asarray(frc_urb_in, dt_)
    ok = (frc_in > c(0.0)) & (frc_in <= c(1.0))
    frc_tbl = jnp.asarray(table.frc_urb, dt_)[1]
    if use_wudapt_lcz:
        frc_tbl = jnp.take(jnp.asarray(table.frc_urb, dt_), jnp.maximum(utype, 1) - 1)
    frc = jnp.where(m, jnp.where(ok, frc_in, frc_tbl), c(0.0))
    t0 = tslb[..., 0]
    t1 = tslb[..., 1]
    t2_ = tslb[..., 2]
    layered = jnp.stack([t0 + c(0.0), c(0.5) * (t0 + t1), t1 + c(0.0), t1 + (t2_ - t1) * c(0.29)], axis=-1)
    tgl = tslb[..., :NUM_URBAN_LAYERS] + c(0.0)
    z = jnp.zeros_like(tsk)
    ts0 = tsk + c(0.0)
    state = UrbanState(
        tr=ts0, tb=ts0, tg=ts0, tc=ts0, qc=z + c(0.01), uc=z,
        trl=layered, tbl=layered, tgl=tgl,
        xxxr=z, xxxb=z, xxxg=z, xxxc=z,
        cmr=z, chr=z, cmc=z, chc=z,
        drelr=z, drelb=z, drelg=z, flxhumr=z, flxhumb=z, flxhumg=z,
    )
    return state, frc, utype


__all__ = [
    "NUM_URBAN_LAYERS", "UrbanTable", "UrbanState", "UrbanForcing", "UrbanOutputs",
    "UrbanCellParams", "UrbanLSMFields", "UrbanDiagnostics", "parse_urbparm",
    "check_supported_options", "load_urban_table", "cell_params", "sfcdif_urb", "multi_layer",
    "force_restore", "urban", "noahmp_urban", "urban_surface_diagnostics", "urban_var_init",
]
