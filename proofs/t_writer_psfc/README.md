# Direct writer PSFC: active moist mass

Branch `lane/t-writer-psfc-hail`, based on approved `9d1cad351` (the requested
`b9852764f` source plus its proof-only successor). Product commit `e39edfcf5`.

The full writer's hydrostatic PSFC fallback omitted hail and extra ice classes.
The PSFC-only writer instead extrapolated total pressure despite resident hybrid
metrics. Both now call one host fallback using the shared `total_moist_mass`
selector. Registry history aliases also drive selected-stream batched source
materialization, including rain, hail and extra ice. Particle numbers, volumes,
rime partitions and aerosols do not load the air. No new prognostic field or
microphysics calculation is introduced.

The fallback retains the original writer's species precision and Kahan/plain
vertical reduction; missing hybrid metrics retain the existing extrapolation.
Producer-supplied PSFC still wins. All other product source is byte-identical to
the approved parent. Its measured Swiss/PROD program IDs are reused through this
source proof; they were not traced again for this host-only change.

| Gate | Measured CPU result |
|---|---|
| Focused writer, subset, authority, backend and history consumers | 92 PASS, 3 unavailable-reference SKIP, 0 errors; 20.43 s |
| New direct-writer tests | 26 PASS, including pristine REAL4 `phy_prep` BYTE comparisons, aliases, batching and hail NetCDF publication |
| Original writer before fix | 16 genuine assertions, 8 working controls, 0 setup errors |
| Delete only hail from actual writer helper | 4 genuine numerical assertions, 4 working controls, 0 setup errors |
| No-hail full output | All 55 fields BYTE-identical to parent |

Host timing at 44×117×267, seven samples per arm: full writer median
13.453 → 11.449 ms, traced NumPy/Python peak 80.738 → 61.180 MB. Corrected
PSFC-only integration costs 5.881 ms versus the incorrect extrapolation's
1.894 ms, and 55.681 versus 34.371 MB peak. These are controlled output-time
measurements, not forecast/GPU speed or trajectory qualification. The selected
fallback requires the active mass volumes at the existing batched output
boundary; no timestep kernel, carry, restart source or device computation changes.

`QUALIFICATION.json` seals receipts and raw logs; `SOURCE_REUSE.json` binds
unchanged arithmetic, the pristine reference and parent program receipts.
`bench.py` reproduces the host comparison; `mutant.py` must exit 1 with four
assertions and four controls. Use the official RAM slot, CPU platform, immutable
snapshot, QUIET guard and timeout. The initial benchmark import setup failure
and mutant CPU-guard refusal are retained and receive no numerical credit.

All four snapshots and pytest directories were removed; persistent cache was
disabled. Upstream hail GPU/D6/E41 qualification remains separately pending.
