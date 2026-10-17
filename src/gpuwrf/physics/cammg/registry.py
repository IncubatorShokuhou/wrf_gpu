"""CAM-MG Registry state, separate from transported moist/scalar members.

These fields are WRF REAL, held between driver calls, and allocated only for
mp=11. Interface diagnostics are the actual CAM-UW producer outputs. QNDROP
is a driver work argument; mp11 activates scalar QNCLOUD, not QNDROP history.
"""
from gpuwrf.contracts.cammg_fields import MASS_FIELDS, INTERFACE_FIELDS, SURFACE_FIELDS, STATE_FIELDS, HISTORY_NAMES


def warn_configuration(cu, shcu):
    """The original driver warns and zeros unsupported convection operands."""
    import warnings
    if int(cu) != 7:
        warnings.warn('CAMMGMP: cu_physics != 7; deep-convection inputs are zeroed as in WRF',
                      RuntimeWarning, stacklevel=2)
    if int(shcu) != 2:
        warnings.warn('CAMMGMP: shcu_physics != 2; shallow-convection inputs are zeroed as in WRF',
                      RuntimeWarning, stacklevel=2)
