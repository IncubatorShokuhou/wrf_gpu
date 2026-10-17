"""Actual bound NTU providers, including every CCNTY/kind bank, for AOT keys."""
import numpy as np


def resolved_bindings():
    from gpuwrf.kernels import phys_ntu
    from . import column, initialize
    return {(name, kind, ccnty): module.selected_tables(dtype, ccnty)
            for name, module in (('native', phys_ntu), ('portable', column), ('init', initialize))
            for kind, dtype in (('r4', np.float32), ('r8', np.float64))
            for ccnty in (1, 2, 3, 4)}
