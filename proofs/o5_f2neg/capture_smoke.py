"""Light smoke test of capture.py taps -> npz -> replay.py (no model step, < 1 GB, CPU).

A stub RK chain calls the REAL tapped port functions in _rk_scan_step's order on one real
Swiss GPU frame (sixth dt/3, _rk_update_scalar_pd, native advect_scalar_pd_fp32 through its
module attribute, _root_scalar_stage_tendencies, _apply_moisture_large_step x3, _root_flow_dep x3);
hdiff and the QV boundary are fakes (their taps are exercised, values zero/simple).  Then
analyse_step/build_arrays/np.savez and replay.step_eval run on the result.  Validates plumbing
only; no scientific claim.
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "tests/v025/b_diff")]
import capture as CAPM  # noqa: E402  (installs cpu_safe, imports gpuwrf release defaults)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402

import pristine_chain as PC  # noqa: E402
import pristine_dyn_real as DR  # noqa: E402
import replay as RP  # noqa: E402
from gpuwrf.coupling.boundary_apply import BoundaryConfig  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402
from gpuwrf.kernels import dyn_pd_fp32  # noqa: E402

netCDF4.set_chunk_cache(4 * 2**20, 521, 0.75)
OUT = Path(sys.argv[sys.argv.index("--out") + 1])
OUT.mkdir(parents=True, exist_ok=True)
FRAME = sys.argv[1]
SPECIES = {"qv": "QVAPOR", "qc": "QCLOUD", "qr": "QRAIN", "qi": "QICE", "qs": "QSNOW", "qg": "QGRAUP",
           "Ni": "QNICE", "Nr": "QNRAIN"}


class Stub(SimpleNamespace):
    def replace(self, **kw):
        d = dict(vars(self))
        d.update(kw)
        return Stub(**d)


om._diffopt1_scalar_horizontal_diffusion = lambda ref, xkhh, muts, nl: {
    n: jnp.zeros_like(getattr(ref, n)) for n in SPECIES}
om.root_scalar_boundary_tendencies = lambda ref, lead, metrics, dt, cfg, relaxed: {"qv": jnp.zeros_like(ref.qv)}
CAPM.install()

z = netCDF4.Dataset(FRAME)
g = lambda v: np.asarray(z[v][0], np.float32)  # noqa: E731
U, V, MU, MUB = g("U"), g("V"), g("MU"), g("MUB")
nz, ny, nx = g("T").shape
dx, dt = float(z.DX), float(z.DT)
c1h, c2h = g("C1H"), g("C2H")
maps = {k: g(v) for k, v in (("msftx", "MAPFAC_MX"), ("msfty", "MAPFAC_MY"), ("msfux", "MAPFAC_UX"),
                              ("msfuy", "MAPFAC_UY"), ("msfvx", "MAPFAC_VX"), ("msfvy", "MAPFAC_VY"))}
mut = MU + MUB
mp = np.pad(mut, 1, mode="edge")
muu = np.float32(0.5) * (mp[1:-1, :-1] + mp[1:-1, 1:])
muv = np.float32(0.5) * (mp[:-1, 1:-1] + mp[1:, 1:-1])
ru = (c1h[:, None, None] * muu + c2h[:, None, None]) * U / maps["msfuy"][None]
rv = (c1h[:, None, None] * muv + c2h[:, None, None]) * V / maps["msfvx"][None]
o = DR.Oracle(DR.build(Path("<USER_HOME>/wrf_gpu2_lanes/o5-f2neg/build/dynreal")), (nz, ny, nx))
for n_, v in (("u", U), ("v", V), ("mu", MU), ("mub", MUB), ("mut", mut), ("msftx", maps["msftx"]), ("msfty", maps["msfty"]),
              ("msfux", maps["msfux"]), ("msfuy", maps["msfuy"]), ("msfvx", maps["msfvx"]),
              ("msfvx_inv", np.float32(1) / maps["msfvx"]), ("msfvy", maps["msfvy"]), ("c1h", c1h), ("c2h", c2h),
              ("dnw", g("DNW")), ("rdx", np.float32(1.0 / dx)), ("rdy", np.float32(1.0 / dx))):
    o.set(n_, v)
o.run("calc_ww_cp")
rom = o.field("ww", (nz + 1, ny, nx))
J = lambda a: jnp.asarray(np.asarray(a, np.float32))  # noqa: E731
names = tuple(SPECIES)
metrics = Stub(c1h=J(c1h), c2h=J(c2h))
# slightly different masses per stage so the wiring cross-checks are discriminating
mu_stage = (J(mut) * 1.0001, J(mut) * 1.0002, J(mut) * 1.0003)


@jax.jit
def stub_step(qs):
    origin = Stub(mu_total=J(mut), qv_bdy=jnp.zeros(1), **{f"{n}_bdy": None for n in names if n != "qv"}, **qs)
    hd = om._diffopt1_scalar_horizontal_diffusion(origin, J(np.ones((nz, ny, nx))), mu_stage[0], None)
    om.wrf_sixth_order_scalar_tendf(origin.qv - 300.0, J(mut), c1=J(c1h), c2=J(c2h), msftx=J(maps["msftx"]),
                                    msfty=J(maps["msfty"]), dt=dt, diff_6th_factor=0.12, monotonic=True,
                                    specified_or_nested=True)
    d6 = {n: om.wrf_sixth_order_scalar_tendf(qs[n], mu_stage[0], c1=J(c1h), c2=J(c2h), msftx=J(maps["msftx"]),
                                             msfty=J(maps["msfty"]), dt=dt / 3.0, diff_6th_factor=0.12,
                                             monotonic=True, specified_or_nested=True) for n in names}
    bdy = om.root_scalar_boundary_tendencies(origin, None, None, dt, None, ())
    other = {n: hd[n] + d6[n] for n in names}
    other["qv"] = other["qv"] + bdy["qv"]
    state = origin
    for stage, dt_rk in enumerate((dt / 3.0, dt / 2.0, dt)):
        cur = state.replace(mu_total=mu_stage[stage])
        step_origin = origin
        if stage == 2:
            step_origin = om._rk_update_scalar_pd(origin, names, (None, other), dt_rk, metrics)
        tend = dyn_pd_fp32.advect_scalar_pd_fp32(
            jnp.stack([getattr(state, n) for n in names]), jnp.stack([getattr(step_origin, n) for n in names]),
            J(ru), J(rv), J(rom), mu_stage[stage], J(mut), J(c1h), J(c2h), J(maps["msftx"]), J(maps["msfty"]),
            J(g("RDNW")), J(g("FNM")), J(g("FNP")), rdx=1.0 / dx, rdy=1.0 / dx, dt=dt, interpret=True)
        adv = tuple(tend[i] for i in range(len(names)))
        qt = om._root_scalar_stage_tendencies(adv, names, cur, BoundaryConfig(), J(maps["msfty"]), bounded=True,
                                              sc_tendencies={})
        upd = om._apply_moisture_large_step(cur, step_origin, q_tendencies=qt, dt_rk=dt_rk, metrics=metrics,
                                            species=names)
        state = om._root_flow_dep(upd, tuple(n for n in names if n != "qv"), J(ru), J(rv), BoundaryConfig())
    return {n: getattr(state, n) for n in names}


q = {n: J(g(v)) for n, v in SPECIES.items()}
CAPM.CAP["buf"].clear()
jax.block_until_ready(stub_step(q))
row, stats, by, final_upd, final_flow = CAPM.analyse_step(list(CAPM.CAP["buf"]), dt, 1, 0.0)
print("taps", row["taps"])
assert row["taps"] == {"hdiff": 1, "diff6": 9, "bdy": 1, "upd": 3, "flow": 3, "pd": 1, "pdk": 3, "stage": 3}, row["taps"]
arrays = CAPM.build_arrays(by, final_upd, final_flow, row)
np.savez_compressed(OUT / "step00001.npz", **arrays)
np.savez(OUT / "constants.npz", c1h=c1h, c2h=c2h, msftx=maps["msftx"], msfty=maps["msfty"], msfux=maps["msfux"],
         msfuy=maps["msfuy"], msfvx=maps["msfvx"], msfvy=maps["msfvy"], fnm=g("FNM"), fnp=g("FNP"), rdnw=g("RDNW"))
work = {"dt_s": dt, "rk_order": 3, "spec_zone": 1, "diff_6th_opt": 2, "diff_6th_factor": 0.12}
libs = {k: PC.Chain(RP.BUILD / f"chain_{k}.so") for k in ("off", "fast")}
r = RP.step_eval(OUT / "step00001.npz", np.load(OUT / "constants.npz"), work, libs, set(row["flow_names"]))
print(json.dumps(r["checks"]))
for n, v in r["species"].items():
    print(n, v["operand_checks"], v["classify_fin_chain"]["verdict"], v["classify_fin_chain"]["port_negative"],
          v["classify_fin_chain"]["port_min"], v["classify_fin_chain"]["wrf_min"])
(OUT / "smoke.json").write_text(json.dumps(r, indent=1) + "\n")
