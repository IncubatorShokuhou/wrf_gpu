"""w42-icloud0 Z02: this process's RRTMG path fed the port's icloud views vs the pristine radiation_driver (Z01 receipt).

Release path = GPUWRF_FAST_DEFAULTS=1 in the environment (REAL entry, fused Pallas band kernels in CPU interpret,
maximum-random McICA); JAX_PLATFORMS=cpu.  Per Z01 column and arm the port gets exactly the RRTMG operands its coupler
builds for that icloud (physics_couplers._rrtmg_column_inputs_impl):
  icloud = 0  cloud_fraction = 0, qc = qi = qs = qg = 0, no microphysics radii (WRF: IF (ICLOUD .ne. 0) skipped)
  icloud = 1  the driver's own CLDFRA (cal_cldfra1) + the fixture's radiation condensate and Thompson radii (control)
Frozen RE01 tier-1 bounds: flux 1 W/m2 + 5 %, heating 1e-4 K/s + 5 % (rrtmg_mp_re_fixture).  Variants:
  port              the product kernels
  scaled_direct     the w41-swddir genuine mutant (band kernels emit the delta-scaled beam = v0.4.0, the SWDDIR fix
                    reverted): measures what the SWDDIR fix changes under icloud = 0 vs icloud = 1
usage: python port_vs_driver.py Z01_JSON OUT_JSON [VARIANT ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests/v025/fid_q2"), str(ROOT / "tests/v025/w41_swddir"), str(ROOT / "tests/v025/w41_aerosol")]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from gpuwrf.physics import rrtmg_lw as L  # noqa: E402
from gpuwrf.physics import rrtmg_sw as R  # noqa: E402
from gpuwrf.kernels import rad_mcica  # noqa: E402
from rrtmg_mp_re_fixture import FIXTURE, FLUX_BOUND, HEATING_BOUND, bound_ratio, load  # noqa: E402
import swddir_oracle_runner as SWD  # noqa: E402

RRSW_SCON = 1368.22
KEYS = {"glw": "glw", "olr": "olr", "lwupb": "lwupb", "lwdnbc": "lwdnbc", "lwuptc": "lwuptc", "swdnb": "swdown",
        "swupt": "swupt", "swddir": "swddir", "swddif": "swddif", "swdnbc": "swdnbc", "swuptc": "swuptc",
        "hlw": "rthratenlw", "hsw": "rthratensw"}


def solve(fx, idx, rows, icloud):
    a = lambda k: jnp.asarray(np.asarray(fx[k][idx], np.float64))  # noqa: E731
    nz = fx["in_t"].shape[1]
    scal = np.array([[r["scalars"][n] for n in ("coszen", "solconx")] for r in rows])
    common = dict(T=a("in_t"), p=a("in_p_hyd"), qv=a("in_qv"), dz=a("in_dz8w")[:, :nz], rho=a("in_rho"),
                  pressure_interfaces=a("in_p_hyd_w"), temperature_interfaces=a("in_t8w"))
    if icloud == 0:
        zero = jnp.zeros_like(common["T"])
        common.update(qc=zero, qi=zero, qs=zero, qg=zero, cloud_fraction=zero)
    else:
        common.update(qc=a("in_qc_rad"), qi=a("in_qi_rad"), qs=a("in_qs"), qg=a("in_qg"),
                      cloud_fraction=jnp.asarray(np.array([r["profiles"]["cldfra"] for r in rows], np.float64)),
                      re_cloud=a("col_re_cloud"), re_ice=a("col_re_ice"), re_snow=a("col_re_snow"), xland=a("in_xland"))
    lw = L.solve_rrtmg_lw_column(L.RRTMGLWColumnState(**common, surface_temperature=a("in_tsk"),
                                 surface_emissivity=a("in_emiss"), top_pressure_pa=float(fx["in_p_top"])),
                                 with_clear_sky=True)
    sw = R.solve_rrtmg_sw_column(R.RRTMGSWColumnState(**common, surface_albedo=a("in_albedo"),
                                 coszen=jnp.asarray(scal[:, 0]), solar_source_scale=jnp.asarray(scal[:, 1] / RRSW_SCON)),
                                 with_clear_sky=True)
    pi = np.asarray(fx["in_pi"][idx], np.float64)
    out = {"glw": lw.surface_down, "olr": lw.toa_up, "lwupb": lw.surface_up, "lwdnbc": lw.clear_flux_down[:, 0],
           "lwuptc": lw.clear_flux_up[:, -1], "swdnb": sw.surface_down, "swupt": sw.toa_up,
           "swddir": sw.surface_direct, "swdnbc": sw.clear_flux_down[:, 0], "swuptc": sw.clear_flux_up[:, -1],
           "hlw": np.asarray(lw.heating_rate) / pi, "hsw": np.asarray(sw.heating_rate) / pi}
    out = {k: np.asarray(v, np.float64) for k, v in out.items()}
    out["swddif"] = out["swdnb"] - out["swddir"]
    return out


def main(z01_path, out_path, variants):
    z01 = json.loads(Path(z01_path).read_text())
    fx = load(FIXTURE)
    assert z01["real_operand_fixture_sha256"] == __import__("hashlib").sha256(FIXTURE.read_bytes()).hexdigest()
    import gpuwrf
    result = dict(platform=jax.devices()[0].platform, gpuwrf=gpuwrf.__file__, real_entry=bool(R._real_entry()),
                  maxrand=bool(rad_mcica._MAXRAND), band_sums=int(R._sw_band_sums_mode() or 0), z01=str(z01_path),
                  columns=z01["input_columns"], categories=z01["categories"], variants={})
    for variant in variants or ["port"]:
        undo = SWD._apply({"scaled_direct": "kernel_scaled_direct"}.get(variant, variant))
        unspy = SWD._spy()
        jax.clear_caches()
        SWD.CALLS.update(one=0, all=0)
        arms = {}
        try:
            for icloud in (0, 1):
                rows = [r for r in z01["rows"] if r["icloud"] == icloud]
                assert [r["column"] for r in rows] == z01["input_columns"]
                port = solve(fx, np.array(z01["input_columns"]), rows, icloud)
                ref = {k: np.array([r["scalars"][v] for r in rows] if k not in ("hlw", "hsw")
                                   else [r["profiles"][v] for r in rows], np.float64) for k, v in KEYS.items()}
                ratio = {k: bound_ratio(ref[k], port[k], HEATING_BOUND if k in ("hlw", "hsw") else FLUX_BOUND)
                         for k in KEYS}
                ratio = {k: np.where(np.isfinite(port[k]), v, np.inf) for k, v in ratio.items()}  # E200
                arms[str(icloud)] = dict(
                    bound={k: float(np.max(v)) for k, v in ratio.items()},
                    max_abs={k: float(np.max(np.abs(port[k] - ref[k]))) for k in KEYS},
                    port={k: port[k].tolist() for k in ("glw", "swdnb", "swddir", "swddif", "olr", "swupt")},
                    port_clear_minus_all={"lw_sfc_down": float(np.max(np.abs(port["lwdnbc"] - port["glw"]))),
                                          "sw_sfc_down": float(np.max(np.abs(port["swdnbc"] - port["swdnb"])))})
        finally:
            unspy()
            undo()
            jax.clear_caches()
        result["variants"][variant] = dict(arms=arms, band_kernel_calls=dict(SWD.CALLS))
        Path(out_path).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({v: {a: round(max(r["bound"].values()), 4) for a, r in d["arms"].items()}
                      for v, d in result["variants"].items()}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
