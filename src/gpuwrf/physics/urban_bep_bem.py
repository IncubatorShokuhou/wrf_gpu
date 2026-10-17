"""BEP/BEM public column entrypoints and historical Registry inventory.

WRF references:

* ``phys/module_sf_bep.F:BEP`` (multi-layer Building Effect Parameterization)
* ``phys/module_sf_bem.F:BEM`` (building-energy model used by BEP+BEM)
* ``Registry/Registry.EM_COMMON`` packages ``bepscheme`` and
  ``bep_bemscheme`` for the required urban carry fields

Full column implementations are qualified in proofs/v040/urban_bep_bem.
Operational Noah-MP/PBL coupling uses coupling.urban_multilayer_hook.
"""

from __future__ import annotations

from typing import NamedTuple

BEP_SOURCE = "<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_bep.F"
BEM_SOURCE = "<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_bem.F"
REGISTRY_SOURCE = "<USER_HOME>/src/wrf_pristine/WRF/Registry/Registry.EM_COMMON"

BEP_REGISTRY_STATE = (
    "a_u_bep",
    "a_v_bep",
    "a_t_bep",
    "a_q_bep",
    "a_e_bep",
    "b_u_bep",
    "b_v_bep",
    "b_t_bep",
    "b_q_bep",
    "b_e_bep",
    "dlg_bep",
    "dl_u_bep",
    "sf_bep",
    "vl_bep",
    "trb_urb4d",
    "tw1_urb4d",
    "tw2_urb4d",
    "tgb_urb4d",
    "sfw1_urb3d",
    "sfw2_urb3d",
    "sfr_urb3d",
    "sfg_urb3d",
    "hi_urb2d",
    "lp_urb2d",
    "hgt_urb2d",
    "lb_urb2d",
    "trl_urb3d",
    "tgl_urb3d",
    "tbl_urb3d",
    "tsk_rural",
)

BEM_EXTRA_REGISTRY_STATE = (
    "tlev_urb3d",
    "qlev_urb3d",
    "tw1lev_urb3d",
    "tw2lev_urb3d",
    "tglev_urb3d",
    "tflev_urb3d",
    "sf_ac_urb3d",
    "lf_ac_urb3d",
    "cm_ac_urb3d",
    "sfvent_urb3d",
    "lfvent_urb3d",
    "sfwin1_urb3d",
    "sfwin2_urb3d",
    "ep_pv_urb3d",
    "t_pv_urb3d",
    "trv_urb4d",
    "qr_urb4d",
    "qgr_urb3d",
    "tgr_urb3d",
    "drain_urb4d",
    "draingr_urb3d",
    "sfrv_urb3d",
    "lfrv_urb3d",
    "dgr_urb3d",
    "dg_urb3d",
    "lfr_urb3d",
    "lfg_urb3d",
)

BEP_BEM_REGISTRY_STATE = BEP_REGISTRY_STATE + BEM_EXTRA_REGISTRY_STATE


class UrbanCanopyState(NamedTuple):
    """Historical inventory handle retained for source-audit tooling.

    Operational state uses UrbanMultilayerCarry with BepState or BemState.
    """

    registry_members: tuple[str, ...]
    static_tables: object = None
    carry: object = None


def bep_step(state, forcing, geometry):
    """Advance a BEP column with explicit material state, forcing and geometry."""
    from gpuwrf.physics.urban_bep_model import bep_column_step
    return bep_column_step(state, forcing, geometry)


def bep_bem_step(state, forcing, geometry, **kwargs):
    """Advance BEP+BEM, including indoor energy, moisture and HVAC."""
    from gpuwrf.physics.urban_bem_model import bem_column_step
    return bem_column_step(state, forcing, geometry, **kwargs)


__all__ = [
    "BEP_SOURCE",
    "BEM_SOURCE",
    "REGISTRY_SOURCE",
    "BEP_REGISTRY_STATE",
    "BEM_EXTRA_REGISTRY_STATE",
    "BEP_BEM_REGISTRY_STATE",
    "UrbanCanopyState",
    "bep_step",
    "bep_bem_step",
]
