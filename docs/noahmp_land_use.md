# Noah-MP land-use tables

WRF's `MMINLU` input attribute identifies the land-use dataset. A numeric
category is meaningful only within that dataset: MODIS water is 17, whereas
USGS water is 16 in the audited MPTABLE. Changing only the dataset label does
not convert the land-use fields.

## v0.4.4 admission correction

The merged v0.4.4 development loader requires each domain's wrfinput to declare
`MMINLU='MODIFIED_IGBP_MODIS_NOAH'`. It refuses USGS, unknown or missing `MMINLU`
before land allocations. This is a gpuwrf support limit, not a WRF restriction.
Original WRF can select USGS tables; full USGS support is not implemented
throughout this GPU land path. The old loader silently selected MODIS even
for a USGS declaration. Do not relabel USGS as MODIS to pass the check;
prepare a consistent supported case through WPS/real.exe instead.

The shipped Swiss and audited staged release inputs declare
`MODIFIED_IGBP_MODIS_NOAH` with 21 categories. Existing MODIS/LCZ support is not
broadened by this guard, and a category count alone cannot select a dataset.
MODIS LCZ classes 51–61 remain admitted under the existing urban-option
requirements; they do not need a different MMINLU declaration.
Parameter tuning should use the matching [surface basis](surface_params.md).

## Qualification and cost

CPU-source and admission tests reproduce the wrong selection and confirm the
refusal. All 118 initialized MODIS land/static leaves are byte-identical to the
old loader in the recorded test. This is not a USGS forecast or GPU fidelity
qualification. Unchanged-program checks for the default Swiss and two-nest
Canary Islands cases pass; the complete release still needs its full checks.

For the shipped case, you can inspect the actual input declaration on CPU:

```python
from netCDF4 import Dataset
with Dataset("examples/switzerland_d01/wrfinput_d01") as ds:
    print(ds.getncattr("MMINLU"), ds.getncattr("NUM_LAND_CAT"))
```

Run Python file inspections with `JAX_PLATFORMS=cpu`. Expected Swiss metadata
is `MODIFIED_IGBP_MODIS_NOAH`, 21. The metadata inspection does not run Noah-MP
or qualify a different dataset.

The guard reads a string from an already-open input once per domain. It adds
no timestep arithmetic, state leaf or device allocation by source inspection;
host elapsed overhead and GPU speed/VRAM deltas are unmeasured [U]. The
details are in the merged [LUTAB report](../proofs/w44_lutab/LUTAB.md) and
[receipt](../proofs/w44_lutab/RECEIPT.json). See the
[Noah-MP option contract](noahmp-options.md) for the existing LCZ requirements.

## CPU-WRF comparison history

Keep `MMINLU`, `NUM_LAND_CAT`, category fields and MPTABLE with both inputs.
If absent from CPU-WRF history, request:

```text
+:h:0:LU_INDEX,IVGTYP,ISLTYP,LANDMASK,XLAND
```

These are CPU-WRF `iofields_filename` selections, not a way to change its
table choice. Dataset attributes and tables must still be checked directly.

Return to [Tune WRF for your region](regional_optimization.md).
