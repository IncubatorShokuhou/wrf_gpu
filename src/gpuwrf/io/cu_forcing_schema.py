"""Fail-closed restart schema of the held WRF RTHFTEN/RQVFTEN cumulus forcing (E78).

``OperationalCarry.cu_forcing`` (coupling.cu_forcing) is required exactly for the
cumulus schemes that read it; a checkpoint written before the group existed, or
for another scheme, is refused instead of silently resuming with zero forcing.
"""
from __future__ import annotations

import numpy as np

from gpuwrf.io.restart_schema import physics_option

SCHEMA_VERSION = 1


def validate(value, namelist, shape, version=None):
    from gpuwrf.coupling.cu_forcing import cu_forcing_required

    required = cu_forcing_required(physics_option(namelist, 'cu_physics', 0))
    if value is None:
        if required:
            raise ValueError('cumulus RTHFTEN/RQVFTEN restart carry missing (E78)')
        if version is not None:
            raise ValueError('inactive cumulus forcing restart has schema metadata (E78)')
        return
    if not required:
        raise ValueError('cumulus forcing carry exists with an inactive cumulus selector (E78)')
    if version != SCHEMA_VERSION:
        raise ValueError('cumulus forcing restart schema version mismatch (E78)')
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError('cumulus forcing restart carry must be the (RTHFTEN, RQVFTEN) pair (E78)')
    for name, array in zip(('RTHFTEN', 'RQVFTEN'), value):
        a = np.asarray(array)
        if a.shape != tuple(map(int, shape)) or a.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(f'cumulus forcing restart {name} shape/dtype mismatch (E78)')
        if not np.isfinite(a).all():
            raise ValueError(f'cumulus forcing restart {name} contains nonfinite values (E78)')
