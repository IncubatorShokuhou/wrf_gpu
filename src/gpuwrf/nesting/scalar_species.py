"""Extra State species with WRF Registry EM_COMMON ``d``/``f`` nest flags.

These tuples describe boundary storage, not forecast qualification. Admission
of mp=18/24/26/28/40 on a live nest remains refused until the two-domain oracle
gate passes. Names use the State interface (NSSL QNDROP and QNCLOUD both map to
Nc in their mutually exclusive scheme families).
"""
from collections.abc import Mapping

from gpuwrf.contracts.precision import NTU_SCALAR_LEAVES

_EXTRA_BOUNDARY_SPECIES = {
    56: ("qh", "Nc", "Ns", "Ng", "Nh", *NTU_SCALAR_LEAVES),
    9: ("qh", "Nc", "Ns", "Ng", "Nh"),
    5: ("qt",),
    7: ("qh",),
    15: ("qrimef",),
    10: ("Ns", "Ng"),
    11: ("Nc", "Ns"),
    14: ("Nc", "Nn"),
    16: ("Nc", "Nn"),
    18: ("qh", "Nc", "Ns", "Ng", "Nh", "Nn", "qvolg", "qvolh"),
    24: ("qh",),
    26: ("qh", "Nc", "Nn"),
    28: ("Nc", "nwfa", "nifa"),
    38: ("Ng", "Nc", "qvolg", "nwfa", "nifa"),
    40: ("Nc", "Ns", "Ng"),
}


def epsilon_pbl_scalars(bl_pbl_physics: int) -> tuple[str, ...]:
    """Registry scalar package, independent of whether the PBL reads it back.

    KEPS bl_keps_adv=0 disables physics feedback, not WRF's scalar-array loop.
    """
    return {16:("pek_adv","pep_adv"),17:("tke_adv","diss_adv","tpe_adv")}.get(int(bl_pbl_physics),())


def extra_boundary_species(mp_physics: int, *, bl_pbl_physics: int = 0) -> tuple[str, ...]:
    """Static extra moist/scalar members represented by the selected scheme."""
    return _EXTRA_BOUNDARY_SPECIES.get(int(mp_physics), ()) + epsilon_pbl_scalars(bl_pbl_physics)


def required_extra_boundary_records(namelist) -> tuple[str, ...]:
    """E78: a live child's selected records cannot silently cold-start on resume."""
    from gpuwrf.io.restart_schema import physics_option

    mp = physics_option(namelist, "mp_physics", 8)
    if isinstance(namelist, Mapping):
        boundary = namelist.get("boundary_config", {})
    else:
        boundary = getattr(namelist, "boundary_config", None)
    nested = (boundary.get("nested_frozen_wrf_boundary_bundle", False) if isinstance(boundary, Mapping)
              else getattr(boundary, "nested_frozen_wrf_boundary_bundle", False))
    bl = physics_option(namelist,"bl_pbl_physics",0)
    return tuple(f"{name}_bdy" for name in extra_boundary_species(mp,bl_pbl_physics=bl)) if nested else ()
