"""c06 real-data cu6 oracle: pristine WRF's own Tiedtke inputs -> the port's tiedtke caller+kernel.

Reference = private pristine serial wrf.exe (c06 original inputs/namelist; restart at 15:00, auxhist2
every step 15:00:00..15:03:00).  For the Tiedtke call of the step starting at 15:00:00 WRF reads
(first_rk_step_part1 / cumulus_driver.F:1389-1412): the time-n state (= wrfrst 15:00), the held
RQVFTEN (aux frame 15:00:00 == wrfrst), this step's PBL RQVBLTEN and surface QFX (aux frame 15:00:18,
held after the step), and writes R*CUTEN / RAINCV (aux frame 15:00:18).

usage: JAX_PLATFORMS=cpu python tiedtke_oracle.py RUN_DIR OUT_JSON
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

assert os.environ.get("JAX_PLATFORMS") == "cpu"
if Path("/tmp/wrf_gpu2_quiet").exists():
    raise SystemExit(3)

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
INPUT = Path("<USER_HOME>/wrf_gpu2_lanes/v-prep/c06/input")
STAMP = "2023-01-15_15:00:00"


def aux_frames():
    from netCDF4 import Dataset, chartostring
    with Dataset(RUN / f"aux2_d01_{STAMP}") as ds:
        times = [str(t) for t in chartostring(ds["Times"][:])]
        out = {name: np.asarray(ds[name][:], dtype=np.float64) for name in ds.variables if name != "Times"}
    return times, out


def stats(port, wrf, ring=0):
    port = np.asarray(port, np.float64)
    wrf = np.asarray(wrf, np.float64)
    if ring:
        sl = (Ellipsis, slice(ring, -ring), slice(ring, -ring))
        port, wrf = port[sl], wrf[sl]
    assert np.isfinite(port).all() and np.isfinite(wrf).all()
    err = port - wrf
    scale = float(np.sqrt(np.mean(wrf * wrf)))
    return {"rms_wrf": scale, "rms_err": float(np.sqrt(np.mean(err * err))),
            "rel_rms": float(np.sqrt(np.mean(err * err)) / scale) if scale > 0 else None,
            "max_abs_err": float(np.max(np.abs(err))), "max_abs_wrf": float(np.max(np.abs(wrf)))}


def main():
    t0 = time.monotonic()
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from gpuwrf.contracts import state as state_module
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig
    from gpuwrf.validation.wrf_restart_hydration import load_restart_templates, read_wrfrst
    from gpuwrf.coupling.scan_adapters import tiedtke_adapter
    assert jax.devices()[0].platform == "cpu"

    times, aux = aux_frames()
    assert times[0] == STAMP and times[1] == "2023-01-15_15:00:18", times[:2]
    snap = read_wrfrst(RUN / f"wrfrst_d01_{STAMP}")
    work = OUT.parent / "work"
    work.mkdir(parents=True, exist_ok=True)
    config = NestedPipelineConfig(input_dir=INPUT, output_dir=work / "unused_history", proof_dir=work / "proof",
                                  hours=24, max_dom=1, feedback=False, emit_initial_history=False)
    original = state_module._gpu_device
    state_module._gpu_device = lambda: jax.devices("cpu")[0]
    try:
        _, bundles, _, epoch, dt, cold = load_restart_templates(config, {"d01": snap})
    finally:
        state_module._gpu_device = original
    # Minimal hydration of exactly the leaves tiedtke_adapter reads (hydrate_wrfrst requires every
    # active template leaf, e.g. QNICE, which WSM6 restarts do not carry): same mapping as hydrate_wrfrst.
    namelist = bundles["d01"].namelist
    assert int(namelist.cu_physics) == 6 and int(namelist.bl_pbl_physics) == 8
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
    hydrated = sorted(upd)
    dt_s = float(namelist.dt_s)
    assert dt_s == 18.0

    # Held RQVFTEN: restart == aux frame 0 (both phy_prep_part2 output of the previous step).
    rqvften_rst = np.asarray(snap.values["RQVFTEN"], np.float64)
    assert np.array_equal(rqvften_rst.astype(np.float32), aux["RQVFTEN"][0].astype(np.float32))
    held_port = rqvften_rst  # the carry contract: phy_prep_part2 output, copied (wrf_restart_hydration._hydrate_cu_forcing)
    rqvblten = aux["RQVBLTEN"][1]
    qfx = aux["QFX"][1]
    rhosfc = np.asarray(jax.device_get(state.rhosfc), np.float64)
    st = state.replace(qv_flux=jnp.asarray(qfx / rhosfc, dtype=state.qv_flux.dtype))

    def run(qvften, qvpblten, s=st):
        rates, raincv = tiedtke_adapter(s, dt_s, namelist.grid, qvften=jnp.asarray(qvften),
                                        qvpblten=jnp.asarray(qvpblten), mp_physics=int(namelist.mp_physics),
                                        return_rates=True)
        return {k: (None if v is None else np.asarray(jax.device_get(v), np.float64)) for k, v in rates.items()}, \
            np.asarray(jax.device_get(raincv), np.float64)

    rates, raincv = run(held_port, rqvblten)
    wrf_raincv = aux["RAINCV"][1]
    fields = {"theta": "RTHCUTEN", "qv": "RQVCUTEN", "qc": "RQCCUTEN", "qi": "RQICUTEN", "u": "RUCUTEN", "v": "RVCUTEN"}
    result = {"run_dir": str(RUN), "restart_sha256": snap.sha256, "stamp": STAMP,
              "hydrated_leaves": hydrated, "xland_unique": sorted(set(np.unique(np.asarray(jax.device_get(state.xland))).tolist())),
              "raincv": {"wrf_n_pos": int((wrf_raincv > 0).sum()), "port_n_pos": int((raincv > 0).sum()),
                         "wrf_sum": float(wrf_raincv.sum()), "port_sum": float(raincv.sum()),
                         "both_pos": int(((wrf_raincv > 0) & (raincv > 0)).sum()),
                         "stats": stats(raincv, wrf_raincv)},
              "rates": {}}
    for key, name in fields.items():
        if rates.get(key) is None:
            result["rates"][name] = "port None"
            continue
        result["rates"][name] = {"all": stats(rates[key], aux[name][1]), "interior5": stats(rates[key], aux[name][1], 5)}
    # Controls on the same inputs (input sensitivity of the trigger, not mutants of the port).
    controls = {}
    for label, qf, qb in (("zero_RQVFTEN", np.zeros_like(held_port), rqvblten),
                          ("zero_RQVBLTEN", held_port, np.zeros_like(rqvblten))):
        _, rc = run(qf, qb)
        controls[label] = {"port_n_pos": int((rc > 0).sum()), "port_sum": float(rc.sum()), "stats": stats(rc, wrf_raincv)}
    result["controls"] = controls
    # Steps 2..N: same comparison for later steps when the aux stream carries the state too.
    result["wall_s"] = time.monotonic() - t0
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
