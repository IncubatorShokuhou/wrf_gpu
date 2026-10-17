# O2 restart schema qualification

Baseline: `6a7590e4d` (released v0.3.4 model source). CPU only, core 24,
nice 19, `JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0`; immutable snapshots.
This is serialization and schema evidence, not new physics qualification.

## State classification and original WRF evidence

| Port field | Introduced by | Original WRF semantics | Restart handling |
|---|---|---|---|
| `camuw_pbl` | `0d3c88fc2`, o1-camuw | PBL driver:2049 maps KVM3D/KVH3D to EXCH_M/H (`Registry.EM_COMMON`:1057–1058, `r`); residual stresses `registry.cam`:2–3, `r`; TKE_PBL `Registry.EM_COMMON`:1230, `rh` | Persist all five REAL fields in full pickle, NetCDF late pytree, and authenticated store; E78 marker, no active cold start |
| `sfc_wspd` | `ef133e231`, BP90 | `Registry.EM_COMMON`:1041, work field (`-`); held by the port for the MYNN surface handoff | Preserve populated values; optional when off, refuse missing when on |
| `re_cloud/re_ice/re_snow` | `3463b7c5f`, b-column, before O1 | `Registry.EM_COMMON`:497–499, `r`; Morrison-aero package:3047 includes all three | Preserve all three; required for active Thompson mediation or mp=40, independent of the Thompson flag |

None of these fields is added to the additive zero-initialization inventory.
Inactive old State-only checkpoints retain absent optional values. Old full
pickle checkpoints may omit CAM-UW only when inactive. Old NetCDF files with
unknown PBL selection require `read_wrfrst_carry(path, namelist=resolved_namelist)`;
unknown or active missing carry is refused. A supplied active Morrison selection
also requires the stored radii.

## Evidence

Receipts live under `<USER_HOME>/wrf_gpu2_lanes/o2-rstschema/`:

- `RS00-source-audit.json`: original WRF line excerpts and source SHA256s.
- `RS01-tests.log`: 43 passed in 27.58 s at `fcf037645`; closes the four baseline
  red tests recorded by o2-km2 KM06B. Exact dtype, shape and bytes for every
  carry leaf, including CAM-UW and mp=40, in both formats and RestartStore.
- `RS02-release-identity.json`: real Swiss CLI loader and location-stripped
  production step jaxpr, **2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a**
  (13,405,317 characters), identical to the frozen v0.3.3/v0.3.4 program.
- `RS03-mutants.json`: 3/3 deletion controls killed by assertions, zero setup
  errors: inventory (1), State-held guard (4), CAM-UW guard (5). An initial
  imported-alias mutation attempt is recorded as invalid, not a kill.
- `RS04-consumers.log/xml`: **312 passed, 4 skipped, 0 failed, 858.82 s** at
  `fcf037645`. All six required shared consumer files pass. The four skips are
  native-carry ozone tests disabled by `GPUWRF_FAST_DEFAULTS=0`, not missing
  reference files. The subsequent change only affects legacy NetCDF selection;
  unchanged model/shared callers reuse this run, with affected IO rerun below.
- `RS05-final-tests.log`: final affected IO gates, including explicit inactive,
  unknown and active legacy selections: **43 passed, 0 skipped, 11.50 s** on
  `7c2b792d3`, rebased on main `85ed676a5`.
- `RS06-legacy-mutant.log`: deleting the unknown-selection refusal gives a
  genuine DID NOT RAISE assertion (6.02 s), zero setup errors. **4/4 total
  deletion controls killed.** Final source tree and receipts: `gate.json`.

Both immutable snapshots and pytest scratch were removed after their last
dependent arms; caches were disabled. Retained small receipts total 396 KiB.

## Efficiency and limits

No timestep, physics, State allocation, or compiled numerical source changed.
The measured release trace is identical; its numerical qualification is reused.
Validation runs at restart boundaries and adds no device operations to the step.
CAM-UW NetCDF writes now serialize required existing carry instead of refusing.
Its payload is `4*ny*nx*(3*(nz+1)+2)` bytes (966,672 bytes for Swiss 44×42×42,
inferred from REAL shapes), plus file metadata and the existing host IO copies.
No new forecast, GPU memory, IO throughput or cached-readiness timing is claimed.

The v0.3.4 failures were loud: unsupported populated CAM-UW NetCDF writes and
non-additive-field rejection in the State-only reader. The CLI's generic
authenticated checkpoint store already preserved full carry. See KNOWN_ISSUES.
