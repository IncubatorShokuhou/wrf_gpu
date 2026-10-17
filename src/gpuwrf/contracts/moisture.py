"""WRF moist-array mass members, excluding scalar numbers and volumes."""

from .physics_registry import MOIST_SPECIES, MOIST_SPECIES_ALL, MOIST_WRFOUT_NAME


# Registry order: the frozen six-class core, hail, then additional ice classes.
# Optional State leaves are None when their Registry package is inactive.
_OPTIONAL_MOIST = tuple(
    name for name in (*MOIST_SPECIES_ALL, *MOIST_WRFOUT_NAME)
    if name not in MOIST_SPECIES
)
_OPTIONAL_MOIST = tuple(dict.fromkeys(_OPTIONAL_MOIST))


def active_moist_arrays(state):
    """Return the represented WRF moist members in accumulation order.

    The six core slots are always allocated (inactive slots are zero). Extra
    mass classes are selected statically by presence, never by their values.
    Number, volume, rime-partition and aerosol scalars must not load the air.
    """
    return tuple(getattr(state, name) for name in MOIST_SPECIES) + tuple(
        value for name in _OPTIONAL_MOIST
        if (value := getattr(state, name, None)) is not None
    )


def total_moist_mass(state, *, core_dtype=None):
    """Sum mass mixing ratios while preserving the core's existing rounding.

    Acoustic glue casts the completed core sum, then adds converted optional
    members. History keeps the State precision. Both retain no-hail arithmetic.
    """
    moist = active_moist_arrays(state)
    core_size = len(MOIST_SPECIES)
    total = sum(moist[1:core_size], moist[0])
    if core_dtype is not None:
        total = total.astype(core_dtype)
    for value in moist[core_size:]:
        total = total + (value if core_dtype is None else value.astype(core_dtype))
    return total
