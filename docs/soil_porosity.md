# Check soil water against an explicit porosity cap

The merged v0.4.4 host helper checks already-loaded SMOIS and optional SH2O
against an effective cap. It is a gpuwrf Python validation API, not a WRF
namelist parameter or a new soil-control scheme. It changes no arrays, reads
no tables and introduces no tolerance. It is CPU-qualified; forecast/control
fidelity and GPU performance are outside this helper's evidence.

## Python example

Run Python with `JAX_PLATFORMS=cpu`. This small example demonstrates admission,
not a real forecast or a soil calibration:

```python
import numpy as np
from gpuwrf.io.wrfinput_consistency import validate_soil_porosity

smois = np.full((4, 2, 2), 0.2, dtype=np.float32)
sh2o = np.full_like(smois, 0.1)
landmask = np.ones((2, 2), dtype=np.float32)
lu_index = np.ones((2, 2), dtype=np.int32)
ok = validate_soil_porosity(
    smois, 0.4, landmask, lu_index, sh2o=sh2o, surface_physics=4,
)
assert ok
```

For real data, supply the **effective consumer cap**, including urban
overrides, in the Noah-MP consumer's precision. Caps may be scalar or arrays
broadcastable over SMOIS. For a one-dimensional WRF class table, pass
`isltyp=soil_classes`; indexing is one-based, with unused slot zero. Optional
`seaice` distinguishes the glacier exception.

## Bounds, status and cost

The return is a Python boolean. Land requires `0 <= SMOIS <= cap`; supplied
SH2O also requires `0 <= SH2O <= SMOIS` and `SH2O <= cap`. Noah-MP glacier
category 15 with SEAICE <= 0 retains its existing cap=1 rule. Invalid,
nonfinite/masked or incompatible-shape inputs return `False`; the helper
never clips or repairs them. The table-backed
[override checker](initial_overrides.md#physical-checks-and-explicit-waivers)
delegates to this predicate.

The [porosity receipt](../proofs/w44_override/SOIL_POROSITY_GATE.json) records
177 passing CPU tests, including 43 helper/delegation checks, and genuine
deletion controls for porosity and liquid bounds. Three staged soil grids with
an example explicit cap take 0.927 ms total [M], excluding loading; process
peak RSS is about 196 MiB including imports and arrays. This is host admission
cost, not a forecast speed, GPU VRAM or new parameter-fidelity measurement.
There are no timestep kernels or persistent device arrays in this host API.

## CPU-WRF comparison data

Retain the actual SOILPARM.TBL, ISLTYP, land/ice categories and any consumer
overrides when comparing an effective cap with original WRF. If absent, use
CPU-WRF `iofields_filename` selections:

```text
+:h:0:SMOIS,SH2O,ISLTYP,LU_INDEX,LANDMASK,SEAICE
```

History fields alone do not reconstruct a changed cap; keep its provenance.
Return to [Tune WRF for your region](regional_optimization.md).
