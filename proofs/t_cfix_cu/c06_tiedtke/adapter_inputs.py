"""Port tiedtke_adapter inputs vs the exact WRF phy_prep inputs (p1/phyprep.npz), and the port KERNEL
(tiedtke_column_jax) run on the exact WRF inputs vs pristine module_cu_tiedtke.F.

usage: JAX_PLATFORMS=cpu python adapter_inputs.py RUN_DIR P1_DIR OUT_JSON
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

assert os.environ.get("JAX_PLATFORMS") == "cpu"
RUN, P1, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
INPUT = Path("<USER_HOME>/wrf_gpu2_lanes/v-prep/c06/input")


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    s = np.sqrt(np.mean(b * b))
    return {"rel_rms": float(np.sqrt(np.mean((a - b) ** 2)) / s) if s > 0 else None,
            "max_abs": float(np.max(np.abs(a - b))), "ref_absmax": float(np.max(np.abs(b)))}


def main():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from gpuwrf.contracts import state as state_module
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig
    from gpuwrf.validation.wrf_restart_hydration import load_restart_templates, read_wrfrst
    from gpuwrf.coupling import scan_adapters as sa
    from gpuwrf.physics.cumulus_tiedtke_jax import tiedtke_column_jax

    ref = dict(np.load(P1 / "phyprep.npz"))
    snap = read_wrfrst(RUN / "wrfrst_d01_2023-01-15_15:00:00")
    work = OUT.parent / "work"
    config = NestedPipelineConfig(input_dir=INPUT, output_dir=work / "unused_history", proof_dir=work / "proof",
                                  hours=24, max_dom=1, feedback=False, emit_initial_history=False)
    original = state_module._gpu_device
    state_module._gpu_device = lambda: jax.devices("cpu")[0]
    try:
        _, bundles, _, _, _, cold = load_restart_templates(config, {"d01": snap})
    finally:
        state_module._gpu_device = original
    namelist = bundles["d01"].namelist
    grid = namelist.grid
    tmpl = cold["d01"].state
    v = snap.values

    def put(old, raw):
        raw = np.asarray(raw, np.float64)
        assert raw.shape == tuple(old.shape), (raw.shape, old.shape)
        return jnp.asarray(raw.astype(old.dtype))
    upd = {a: put(getattr(tmpl, a), v[n]) for a, n in (("u", "U_2"), ("v", "V_2"), ("w", "W_2"), ("qv", "QVAPOR"),
           ("qc", "QCLOUD"), ("qr", "QRAIN"), ("qi", "QICE"), ("qs", "QSNOW"), ("qg", "QGRAUP"))
           if getattr(tmpl, a, None) is not None}
    upd["theta"] = put(tmpl.theta, v["THM_2"].astype(np.float64) + 300.0)
    for stem, pert, base in (("p", "P", "PB"), ("ph", "PH_2", "PHB"), ("mu", "MU_2", "MUB")):
        upd[stem + "_total"] = put(getattr(tmpl, stem + "_total"), v[pert].astype(np.float64) + v[base])
        upd[stem + "_perturbation"] = put(getattr(tmpl, stem + "_perturbation"), v[pert])
    upd["rhosfc"] = put(tmpl.rhosfc, v["RHO"][0])
    state = tmpl.replace(_cast=False, **upd)
    g = lambda x: np.asarray(jax.device_get(x), np.float64)  # noqa: E731

    # --- the adapter's own input assembly (copied call-for-call from scan_adapters.tiedtke_adapter) ---
    nz = state.theta.shape[0]
    metrics = getattr(grid, "metrics", None)
    rho = (sa._wrf_phy_prep_rho_from_state(state, metrics, output_dtype=state.theta.dtype)
           if metrics is not None else sa._rho_from_state(state) * (1.0 + state.qv))
    T = sa._temperature_from_theta(sa._dry_theta_view(state), state.p)
    pii = sa._exner_columns(state.p)
    prep = sa._wrf_cumulus_prep(state, grid)
    eta = getattr(grid, "eta_levels", None)
    znu = (0.5 * (np.asarray(eta)[:-1] + np.asarray(eta)[1:])) if eta is not None and len(eta) == nz + 1 else None
    w_int = g(state.w)
    res = {"metrics_present": metrics is not None, "eta_present": znu is not None,
           "inputs": {"T3D": rel(g(T), ref["t_phy"]), "PI3D": rel(g(pii), ref["pi_phy"]), "RHO3D": rel(g(rho), ref["rho"]),
                      "PCPS": rel(g(prep["p"]), ref["p_hyd"]), "P8W": rel(g(prep["p8w"]), ref["p_hyd_w"]),
                      "DZ8W": rel(g(prep["dz"]), ref["dz8w"][:nz]), "W": rel(w_int, ref["w"]),
                      "U3D": rel(g(sa._u_mass(state)), ref["u_phy"]), "V3D": rel(g(sa._v_mass(state)), ref["v_phy"]),
                      "ZNU": rel(znu, ref["znu"]) if znu is not None else "absent"}}

    # --- port KERNEL on the exact WRF inputs (adapter bypassed) vs pristine Fortran on the same inputs ---
    ny, nx = ref["qfx"].shape
    def cols(a):
        return jnp.asarray(np.moveaxis(np.asarray(a, np.float64), 0, -1).reshape(ny * nx, -1))
    def one(*a):
        out = tiedtke_column_jax(*a, 18.0, stepcu=1)
        return out["RTHCUTEN"], out["RQVCUTEN"], out["RQCCUTEN"], out["RQICUTEN"], out["RUCUTEN"], out["RVCUTEN"], out["RAINCV"]
    args = (cols(ref["t_phy"]), cols(ref["qv"]), cols(ref["qc"]), cols(ref["qi"]), cols(ref["p_hyd"]),
            cols(ref["p_hyd_w"]), cols(ref["dz8w"][:nz]), cols(ref["rho"]), cols(ref["pi_phy"]), cols(ref["u_phy"]),
            cols(ref["v_phy"]), cols(ref["w"]), cols(ref["qvften"]), cols(ref["qvpblten"]),
            jnp.asarray(ref["qfx"].reshape(-1), jnp.float64), jnp.asarray(ref["xland"].reshape(-1), jnp.float64),
            cols(np.broadcast_to(ref["znu"][:, None, None], (nz, ny, nx))))
    outs = jax.vmap(one)(*args)
    back = lambda c: np.moveaxis(g(c).reshape(ny, nx, nz), -1, 0)  # noqa: E731
    names = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN", "RUCUTEN", "RVCUTEN")
    kern = {n: back(o) for n, o in zip(names, outs[:6])}
    kr = g(outs[6]).reshape(ny, nx)
    res["kernel_on_wrf_inputs_vs_pristine_fortran"] = {n: rel(kern[n], ref["ref_" + n]) for n in names}
    res["kernel_on_wrf_inputs_vs_pristine_fortran"]["RAINCV"] = rel(kr, ref["ref_RAINCV"])
    res["kernel_raincv"] = {"n_pos": int((kr > 0).sum()), "sum": float(kr.sum()),
                            "pristine_n_pos": int((ref["ref_RAINCV"] > 0).sum()), "pristine_sum": float(ref["ref_RAINCV"].sum())}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
