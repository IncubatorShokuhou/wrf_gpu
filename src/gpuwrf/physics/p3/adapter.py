"""WRF phy_prep/microphysics/finish boundary for the four P3 wrappers."""
import jax.numpy as jnp
from functools import partial

from gpuwrf.contracts.ice_diagnostics import ice_diagnostic_mapping
from .column import step
from .registry import SCHEMES, input_variables


def output_mapping(mp):
    mapping = {"qv": "qv", "qc": "qc", "qr": "qr", "qi": "qi", "ni": "Ni",
               "nr": "Nr", "qir": "qir", "bir": "qib", "th_old": "mp_th_old",
               "qv_old": "mp_qv_old", "re_cloud": "re_cloud", "re_ice": "re_ice"}
    if mp != 50:
        mapping["nc"] = "Nc"
    if mp == 52:
        mapping.update(qi2="qi2", ni2="Ni2", qir2="qir2", bir2="qib2")
    if mp == 53:
        mapping["qzi"] = "qzi"
    mapping.update(ice_diagnostic_mapping(mp))
    return mapping


def p3_adapter(state, dt, grid=None, *, mp=50, timestep=1, return_precipitation=False,
               return_radar=False, return_sr=False):
    if mp not in SCHEMES:
        raise ValueError(f"unsupported P3 variant {mp}")
    required = input_variables(mp)
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError(f"P3 requires initialized particle/history fields: {missing}")
    from gpuwrf.coupling.physics_couplers import (
        _to_columns, _from_columns, _dry_theta_view,
        _column_dz_from_state, _theta_m_from_microphysics,
    )
    f = lambda value: jnp.asarray(value, jnp.float32)
    real = state.replace(theta=f(state.theta), qv=f(state.qv), _cast=False)
    old_th, old_qv = _dry_theta_view(real), f(state.qv)
    pressure = f(state.p)
    fields = {name: _to_columns(f(getattr(state, source))) for name, source in {
        "qv": "qv", "qc": "qc", "qr": "qr", "qi": "qi", "ni": "Ni",
        "nr": "Nr", "nc": "Nc", "qir": "qir", "bir": "qib",
        "th_old": "mp_th_old", "qv_old": "mp_qv_old",
    }.items()}
    fields.update(th=_to_columns(old_th), p=_to_columns(pressure),
                  dz=_column_dz_from_state(real, grid, dtype=jnp.float32))
    if mp == 52:
        fields.update({name: _to_columns(f(getattr(state, source))) for name, source in
                       {"qi2": "qi2", "ni2": "Ni2", "qir2": "qir2", "bir2": "qib2"}.items()})
    if mp == 53:
        fields["qzi"] = _to_columns(f(state.qzi))
    mapping = output_mapping(mp)
    from gpuwrf.backends import pallas as backend
    if backend.cuda_native():
        from gpuwrf.kernels.phys_p3 import full_column
        out = full_column(fields, f(dt), timestep, mp=mp,
                          output_names=(*mapping,"th","rainncv","snowncv",*(("ze",) if return_radar else ()),
                                        *(("sr",) if return_sr else ())))
    else:
        out = step(fields, f(dt), timestep, mp=mp)
    updates = {target: f(_from_columns(out[source])) for source, target in mapping.items()}
    updates["theta"] = _theta_m_from_microphysics(
        old_th, old_qv, _from_columns(out["th"]), updates["qv"], jnp.float32)
    # P3 RAINNCV includes snow. Model accumulators separate liquid and solid;
    # adding RAINNCV to rain_acc would count the ice precipitation twice.
    liquid, solid = out["rainncv"] - out["snowncv"], out["snowncv"]
    updates.update(rain_acc=state.rain_acc + liquid, snow_acc=state.snow_acc + solid)
    updated = state.replace(**updates)
    precip = {"rain": liquid, "snow": solid, "graupel": jnp.zeros_like(solid), "ice": jnp.zeros_like(solid)}
    if return_sr:  # P3 SR=SR: solid/(liquid+solid+1e-12) of the column surface fluxes
        precip["sr"] = jnp.reshape(out["sr"], jnp.shape(liquid))
    if return_radar:
        # P3_MAIN diag_ze is already dBZ at the final diagnostics time level.
        # The WRF wrapper passes it straight to REFL_10CM; never re-diagnose it.
        return updated, precip, f(_from_columns(out["ze"]))
    return (updated, precip) if return_precipitation else updated


p3_50_adapter = partial(p3_adapter, mp=50)
p3_51_adapter = partial(p3_adapter, mp=51)
p3_52_adapter = partial(p3_adapter, mp=52)
p3_53_adapter = partial(p3_adapter, mp=53)
