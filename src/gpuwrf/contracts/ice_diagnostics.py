"""Held WRF P3/Jensen history diagnostics, never advected or recomputed.

Registry.EM_COMMON:1599–1610. The schemes reset these outputs on each
microphysics call; zero at initialization is the Registry value, not a
substitute for the producer. Only the selected categories are allocated.
"""

ICE_DIAGNOSTIC_LEAVES = tuple(
    name + suffix
    for suffix in ("", "2", "3")
    for name in ("d_ice", "rho_ice", "v_ice", "phi_ice")
)


def ice_diagnostic_mapping(mp):
    """Column output name -> State leaf at the original WRF time level."""
    mp = int(mp)
    if mp in (50, 51, 52, 53):
        return {
            source + suffix: target + suffix
            for suffix in (("", "2") if mp == 52 else ("",))
            for source, target in (("di", "d_ice"), ("rhoi", "rho_ice"), ("vm", "v_ice"))
        }
    if mp == 55:
        return {
            source + str(cat): target + (str(cat) if cat > 1 else "")
            for cat in (1, 2, 3)
            for source, target in (("di", "d_ice"), ("rho", "rho_ice"),
                                   ("vm", "v_ice"), ("phi", "phi_ice"))
        }
    return {}


def ice_diagnostic_leaves(mp):
    return tuple(ice_diagnostic_mapping(mp).values())
