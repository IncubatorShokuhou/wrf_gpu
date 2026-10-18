"""Copy a WRF NetCDF file keeping only the first N records of its Time dimension.

usage: python truncate_records.py SRC DST N
Every variable with a leading Time dimension is sliced; dims, attributes, dtypes
and the remaining variables are copied unchanged (classic/NetCDF4 format kept).
"""
import sys

import numpy as np
from netCDF4 import Dataset

src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with Dataset(src) as a, Dataset(dst, "w", format=a.data_model) as b:
    b.setncatts({k: a.getncattr(k) for k in a.ncattrs()})
    for name, dim in a.dimensions.items():
        b.createDimension(name, None if dim.isunlimited() else len(dim))
    for name, var in a.variables.items():
        fill = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
        out = b.createVariable(name, var.dtype, var.dimensions, fill_value=fill)
        out.setncatts({k: var.getncattr(k) for k in var.ncattrs() if k != "_FillValue"})
        var.set_auto_maskandscale(False)
        out.set_auto_maskandscale(False)
        data = var[:]
        if var.dimensions and var.dimensions[0] == "Time":
            data = data[:n]
        out[:] = np.asarray(data)
print(dst, n)
