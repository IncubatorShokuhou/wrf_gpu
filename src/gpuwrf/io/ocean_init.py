"""Backend-dark OML binding; NetCDF reads use the process-wide lock."""
from dataclasses import replace
from pathlib import Path

import numpy as np


def bind_ocean_model(namelist, run_dir, domain, *, state):
    if int(namelist.sf_ocean_physics) != 1:
        return namelist
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.physics.ocean_mixed_layer import initialize
    path = Path(run_dir)/f'wrfinput_{domain}'
    shape = tuple(state.t_skin.shape)
    values = {}
    if namelist.oml_hml0 <= 0:
        required = ('h0ml',) if namelist.oml_hml0 == 0 else ('h0ml','huml','hvml','tmoml')
        with Dataset(path) as ds:
            for name in required:
                if name.upper() not in ds.variables:
                    raise ValueError(f'{path}: oml_hml0={namelist.oml_hml0} requires {name.upper()}')
                array = np.asarray(ds[name.upper()][0],dtype=np.float32)
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(f'{path}: invalid {name.upper()} shape/nonfinite values')
                values[name] = array
    initial = initialize(state.t_skin,namelist.oml_hml0,**values)
    if any(not np.isfinite(np.asarray(x)).all() for x in initial) or np.any(np.asarray(initial.h0ml)<0):
        raise ValueError('OML initial fields must be finite with nonnegative H0ML')
    return replace(namelist,ocean_initial=initial)
