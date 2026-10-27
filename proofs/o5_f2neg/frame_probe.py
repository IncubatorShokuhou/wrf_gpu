"""Light probe (F2N02): port final-RK scalar-chain FUNCTIONS vs the verbatim pristine chain
on real states of our own GPU Swiss release-default run (v-run wrfout, hourly).

Operands are built once per frame and fed identically to both implementations:
  q_old = q_cur = history species (PD stage with field = field_old, as w43-conv CA01),
  muts1 = mu1 = mut3 = mu2 = MU + MUB (full), xkhh from the port Smagorinsky chain
  (km_opt=4, CK01-qualified vs pristine), ru/rv = calc_mu_uv-coupled U/V, rom from
  pristine calc_ww_cp, physics sc_tend = 0 (not in history), QV boundary = 0.
Port side: explicit_diffusion.horizontal_diffusion_coord_scalar_tendency (as
_diffopt1_scalar_horizontal_diffusion calls it), wrf_sixth_order_scalar_tendf (dt/3),
operational_mode._rk_update_scalar_pd, kernels.dyn_pd_fp32.advect_scalar_pd_fp32 (interpret),
operational_mode._root_scalar_stage_tendencies, _apply_moisture_large_step, flow_dep_bdy.
CPU only, < 1 GB.  Not the port's in-run operands (capture.py) and not a GPU claim.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests/v025/b_diff"), str(HERE)]
assert os.environ.get("JAX_PLATFORMS") == "cpu"
import gpuwrf  # noqa: E402,F401  (release defaults)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import netCDF4  # noqa: E402

netCDF4.set_chunk_cache(4 * 2**20, 521, 0.75)  # light-memory reads of 375-variable wrfout (d02 OOM at 1 G otherwise)
import numpy as np  # noqa: E402

import pristine_chain as PC  # noqa: E402
import pristine_dyn_real as DR  # noqa: E402
import replay_lib as RL  # noqa: E402
from gpuwrf.coupling.boundary_apply import BoundaryConfig, flow_dep_bdy  # noqa: E402
from gpuwrf.dynamics import explicit_diffusion as ED  # noqa: E402
from gpuwrf.kernels.dyn_pd_fp32 import advect_scalar_pd_fp32  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

SPECIES = {"qv": "QVAPOR", "qc": "QCLOUD", "qr": "QRAIN", "qi": "QICE", "qs": "QSNOW", "qg": "QGRAUP",
           "Ni": "QNICE", "Nr": "QNRAIN"}
BUILD = Path("<USER_HOME>/wrf_gpu2_lanes/o5-f2neg/build")
MUTANT = os.environ.get("O5_MUTANT", "")  # sensitivity check only: dt6 | nohdiff
RING_MIN = int(os.environ.get("O5_RING_MIN", "0"))  # nested children: classify ring >= spec_bdy_width only
CROP = os.environ.get("O5_CROP", "")  # "j0,j1,i0,i1": light-memory window; use RING_MIN >= 7 so crop edges cannot matter


class Stub(SimpleNamespace):
    def replace(self, **kw):
        d = dict(vars(self))
        d.update(kw)
        return Stub(**d)


def frame(path, libs, dr_lib):
    z = netCDF4.Dataset(path)
    win = None
    if CROP.startswith("auto"):  # N x N window with the largest column condensate (qc+qr+qi+qs+qg)
        size = int(CROP[4:])
        cond = sum(np.asarray(z[v][0], np.float64).sum(axis=0) for v in ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"))
        cs = np.pad(cond.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        box = cs[size:, size:] - cs[:-size, size:] - cs[size:, :-size] + cs[:-size, :-size]
        j0, i0 = np.unravel_index(int(np.argmax(box)), box.shape)
        win = (int(j0), int(j0) + size, int(i0), int(i0) + size)
    elif CROP:
        win = tuple(int(x) for x in CROP.split(","))

    def g(v):
        a = np.asarray(z[v][0], np.float32)
        if win is None or a.ndim < 2:
            return a
        j0, j1, i0, i1 = win
        sy = slice(j0, j1 + (1 if a.shape[-2] == z.dimensions["south_north_stag"].size else 0))
        sx = slice(i0, i1 + (1 if a.shape[-1] == z.dimensions["west_east_stag"].size else 0))
        return a[..., sy, sx]
    U, V, MU, MUB = g("U"), g("V"), g("MU"), g("MUB")
    nz, ny, nx = g("T").shape
    dx, dy, dt = float(z.DX), float(z.DY), float(z.DT)
    spec_zone = 1
    c1h, c2h = g("C1H"), g("C2H")
    maps = {k: g(v) for k, v in (("msftx", "MAPFAC_MX"), ("msfty", "MAPFAC_MY"), ("msfux", "MAPFAC_UX"),
                                  ("msfuy", "MAPFAC_UY"), ("msfvx", "MAPFAC_VX"), ("msfvy", "MAPFAC_VY"))}
    mut = MU + MUB
    mp = np.pad(mut, 1, mode="edge")
    muu = np.float32(0.5) * (mp[1:-1, :-1] + mp[1:-1, 1:])
    muv = np.float32(0.5) * (mp[:-1, 1:-1] + mp[1:, 1:-1])
    ru = (c1h[:, None, None] * muu + c2h[:, None, None]) * U / maps["msfuy"][None]
    rv = (c1h[:, None, None] * muv + c2h[:, None, None]) * V / maps["msfvx"][None]
    o = DR.Oracle(dr_lib, (nz, ny, nx))
    for n, v in (("u", U), ("v", V), ("mu", MU), ("mub", MUB), ("mut", mut), ("msftx", maps["msftx"]),
                 ("msfty", maps["msfty"]), ("msfux", maps["msfux"]), ("msfuy", maps["msfuy"]), ("msfvx", maps["msfvx"]),
                 ("msfvx_inv", np.float32(1) / maps["msfvx"]), ("msfvy", maps["msfvy"]), ("c1h", c1h), ("c2h", c2h),
                 ("dnw", g("DNW")), ("rdx", np.float32(1.0 / dx)), ("rdy", np.float32(1.0 / dy))):
        o.set(n, v)
    o.run("calc_ww_cp")
    rom = o.field("ww", (nz + 1, ny, nx))
    J = lambda a: jnp.asarray(np.asarray(a, np.float32))  # noqa: E731
    zx, zy, rdzw = ED.wrf_nonperiodic_diffusion_metrics(J(g("PH") + g("PHB")), dx_m=dx, dy_m=dy)
    d11, d22, d12 = ED.horizontal_deformation_2d(
        J(U), J(V), dx_m=dx, dy_m=dy, msftx=J(maps["msftx"]), msfty=J(maps["msfty"]), msfux=J(maps["msfux"]),
        msfuy=J(maps["msfuy"]), msfvx=J(maps["msfvx"]), msfvy=J(maps["msfvy"]), zx=zx, zy=zy, rdzw=rdzw, fnm=J(g("FNM")),
        fnp=J(g("FNP")), cf1=J(z["CF1"][0]), cf2=J(z["CF2"][0]), cf3=J(z["CF3"][0]), dn=J(g("DN")), dnw=J(g("DNW")))
    _, xkhh = ED.smag2d_horizontal_km(d11, d22, d12, dx_m=dx, dy_m=dy, c_s=0.25, msftx=J(maps["msftx"]),
                                      msfty=J(maps["msfty"]))
    xkhh = np.array(xkhh, np.float32)
    xkhh[:, 0, :] = xkhh[:, -1, :] = xkhh[:, :, 0] = xkhh[:, :, -1] = 0.0  # smag2d_km owns ids+1:ide-2 (CK01)
    spec = SPECIES
    if os.environ.get("O5_SPECIES"):  # e.g. WDM6: qv:QVAPOR,...,Nn:QNCCN,Nc:QNCLOUD,Nr:QNRAIN
        spec = dict(x.split(":") for x in os.environ["O5_SPECIES"].split(","))
    q = {n: g(v) for n, v in spec.items()}
    names = tuple(q)
    rdx, rdy = 1.0 / dx, 1.0 / dy
    mass = J(c1h)[:, None, None] * J(mut)[None] + J(c2h)[:, None, None]

    @jax.jit
    def port_chain(qs, xk, mu):
        hd = {n: ED.horizontal_diffusion_coord_scalar_tendency(
            qs[n], xk, mass, dx_m=dx, dy_m=dy, msftx=J(maps["msftx"]), msfty=J(maps["msfty"]), msfux=J(maps["msfux"]),
            msfuy=J(maps["msfuy"]), msfvx=J(maps["msfvx"]), msfvy=J(maps["msfvy"]), nonperiodic_owned=True) for n in names}
        if MUTANT == "nohdiff":  # deletion mutant: port without the diff_opt=1 scalar mixing
            hd = {n: jnp.zeros_like(hd[n]) for n in names}
        d6 = {n: ED.wrf_sixth_order_scalar_tendf(qs[n], mu, c1=J(c1h), c2=J(c2h), msftx=J(maps["msftx"]),
                                                 msfty=J(maps["msfty"]), dt=dt if MUTANT == "dt6" else dt / 3.0,
                                                 diff_6th_factor=0.12,
                                                 monotonic=True, specified_or_nested=True) for n in names}
        other = {n: hd[n].astype(d6[n].dtype) + d6[n] for n in names}
        metrics = Stub(c1h=J(c1h), c2h=J(c2h))
        origin = Stub(mu_total=mu, **qs)
        pre = om._rk_update_scalar_pd(origin, names, (None, other), dt, metrics)
        tend = advect_scalar_pd_fp32(
            jnp.stack([qs[n] for n in names]), jnp.stack([getattr(pre, n) for n in names]), J(ru), J(rv), J(rom),
            mu, mu, J(c1h), J(c2h), J(maps["msftx"]), J(maps["msfty"]), J(g("RDNW")), J(g("FNM")), J(g("FNP")),
            rdx=rdx, rdy=rdy, dt=dt, interpret=True)
        adv = tuple(tend[i] for i in range(len(names)))
        state = Stub(mu_total=mu, qv_bdy=jnp.zeros(1), **{f"{n}_bdy": None for n in names if n != "qv"})
        qt = om._root_scalar_stage_tendencies(adv, names, state, BoundaryConfig(), J(maps["msfty"]), bounded=True,
                                              sc_tendencies={})
        upd = om._apply_moisture_large_step(state, pre, q_tendencies=qt, dt_rk=dt, metrics=metrics, species=names)
        fin = {n: (getattr(upd, n) if n == "qv" else flow_dep_bdy(getattr(upd, n), J(ru), J(rv), BoundaryConfig()))
               for n in names}
        return hd, d6, other, {n: getattr(pre, n) for n in names}, adv, qt, {n: getattr(upd, n) for n in names}, fin

    t = time.monotonic()
    hd, d6, other, pre, adv, qt, upd, fin = jax.block_until_ready(port_chain({n: J(q[n]) for n in names}, J(xkhh), J(mut)))
    port_s = time.monotonic() - t
    rows = {}
    for i, n in enumerate(names):
        args = dict(nx=nx, ny=ny, nz=nz, spec_zone=spec_zone, flow=n != "qv", ownqv=0, opt=2, qold=q[n], qcur=q[n],
                    xkhh=xkhh, phys=np.zeros_like(q[n]), bdy=np.zeros_like(q[n]), ru=ru, rv=rv, ww=rom, muts1=mut,
                    mu1=mut, mut3=mut, mu2=mut, mx=maps["msftx"], my=maps["msfty"], ux=maps["msfux"], uy=maps["msfuy"],
                    vx=maps["msfvx"], vy=maps["msfvy"], fnm=g("FNM"), fnp=g("FNP"), rdnw=g("RDNW"), c1=c1h, c2=c2h,
                    rdx=rdx, rdy=rdy, dt=dt, dt1=dt / 3.0, factor=0.12)
        full = {c: libs[c](**args) for c in libs}
        full_ftz = libs["off"](**args, ftz=True)  # E117 diagnostic: pristine with x86 flush-to-zero
        iso_in = dict(sc_in=np.asarray(other[n]), pre_in=np.asarray(pre[n]), adv_in=np.asarray(adv[i]),
                      upd_in=np.asarray(upd[n]))
        iso = {c: libs[c](**args, **iso_in) for c in libs}
        scale = RL.footprint_max(q[n], np.asarray(pre[n]))
        port = {"hdiff": hd[n], "diff6": d6[n], "sc": other[n], "pre": pre[n], "adv": adv[i], "upd": upd[n], "fin": fin[n]}
        rec = {"isolated": {k: RL.compare_field(np.asarray(port[k]), iso["off"][k], iso["fast"][k]) for k in port},
               "chain": {k: RL.compare_field(np.asarray(port[k]), full["off"][k], full["fast"][k]) for k in port},
               "classify_fin": RL.classify(np.asarray(fin[n]), full["off"]["fin"], full["fast"]["fin"], scale, RING_MIN),
               "classify_pre": RL.classify(np.asarray(pre[n]), full["off"]["pre"], full["fast"]["pre"], scale, RING_MIN),
               "classify_fin_vs_pristine_ftz": RL.classify(np.asarray(fin[n]), full_ftz["fin"], full["fast"]["fin"], scale,
                                                           RING_MIN),
               "diff6_vs_pristine_ftz": RL.compare_field(np.asarray(d6[n]), full_ftz["diff6"]),
               "birth": RL.birth(q[n], np.asarray(pre[n]), np.asarray(fin[n]), np.asarray(mass), dt,
                                 {"hdiff": np.asarray(hd[n]), "diff6": np.asarray(d6[n])}),
               "q_max": float(q[n].max())}
        rows[n] = rec
    return {"file": str(path), "window": win, "port_jit_s": port_s, "species": rows}


def main():
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    paths = [a for a in sys.argv[1:] if "wrfout" in a]
    libs = {c: PC.Chain(BUILD / f"chain_{c}.so") for c in ("off", "fast")}
    dr_lib = DR.build(BUILD / "dynreal")
    res = {"scope": __doc__, "mutant": MUTANT, "ring_min": RING_MIN, "crop": CROP, "frames": []}
    for p in paths:
        r = frame(p, libs, dr_lib)
        res["frames"].append(r)
        print(Path(p).name, {n: (v["classify_fin"]["verdict"], v["classify_fin"]["port_negative"],
                                 v["classify_fin"]["wrf_negative"], v["classify_fin"]["port_min"],
                                 v["classify_fin"]["wrf_min"]) for n, v in r["species"].items()}, flush=True)
        out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
