"""P3-specific radiation_driver and RRTMG caller views, without State writes."""
import jax.numpy as jnp

from .registry import SCHEMES
from gpuwrf.physics.rrtmg_mp_re import mp_re_active as legacy_active, mp_re_config


def mp_re_active(namelist=None, **selection):
    config = mp_re_config(namelist)
    config.update(selection)
    if config['mp_physics'] not in (*SCHEMES,55):
        return legacy_active(namelist, **selection)
    # P3/Jensen carry cloud/ice effective radii; has_reqs=0.
    return (config['use_mp_re'] == 1 and config['ra_lw_physics'] == 4
            and config['ra_sw_physics'] == 4)


def optical_view(state, ice_columns):
    """RRTMG_SW:10851-10863 / LW:12250-12262, flags (1,1,0).

    P3 ice is transferred to the snow optics channel, using its own effective
    radius; the ice optics channel is cleared. The solver applies snow's 10um
    minimum/130um mass correction. No snow radius is carried in State.
    """
    from gpuwrf.coupling.physics_couplers import _to_columns
    if state.re_cloud is None or state.re_ice is None:
        raise ValueError('P3 radiation needs held cloud/ice effective radii')
    zero = jnp.zeros_like(ice_columns)
    radii = dict(re_cloud=_to_columns(state.re_cloud),
                 re_ice=jnp.full_like(ice_columns, 10.e-6),
                 re_snow=_to_columns(state.re_ice), xland=state.xland)
    return zero, ice_columns, radii
