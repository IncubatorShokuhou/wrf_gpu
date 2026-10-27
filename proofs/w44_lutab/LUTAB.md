# Noah-MP land-use table audit — BUG

Lane `w44-lutab`, base `748191e1eb`, CPU-only audit, 2026-10-23.

The operational Noah-MP input loader silently used MODIS parameters for an
input declaring USGS. The correction explicitly refuses non-MODIS inputs at
the land initialization boundary, before field reads or land allocations.
Full USGS support is not claimed. The direct table parser's existing USGS
capability remains available for reference/operator work.

## Proof and scope

Pristine WRF `phys/module_sf_noahmpdrv.F:1999` calls
`read_mp_veg_parameters(trim(MMINLU))`. The unmodified table reader in
`phys/module_sf_noahmplsm.F:11531` selects `noahmp_usgs_*` for USGS and
`noahmp_modis_*` otherwise, then transfers the selected class indices.
The port's `io/noahmp_land_init.py` previously called
`load_noahmp_parameters(tdir)` without forwarding or checking MMINLU.

`audit.py` reproduced this on immutable `ad28802304`: a private copy of the
shipped Swiss wrfinput, with only MMINLU=USGS and NUM_LAND_CAT=24 changed,
was admitted and returned MODIS indices. This is an admission/table-selection
reproduction using real fields, **not a USGS forecast or fidelity test**.
No native USGS input was present among the checked staged examples.

| MPTABLE dataset | urban | water | barren | ice |
|---|---:|---:|---:|---:|
| MODIS (wrongly selected) | 13 | 17 | 16 | 15 |
| USGS (pristine table) | 1 | 16 | 19 | 24 |

`LU01_baseline.json` records both parsed tables and the actual initialized
parameters [M CPU]. `LU02_candidate.json` records the explicit refusal.
The diagnostic names the input path, declared MMINLU and NUM_LAND_CAT.
Unknown and missing MMINLU also refuse. MMINLU is the selection authority;
category count alone is not a safe dataset identifier. Existing MODIS/LCZ
handling is not broadened or changed.

## Category audit

| Location | Finding |
|---|---|
| `physics/noahmp/tables.py:359–367,417–419` | Parses either MODIS or USGS when explicitly requested, including table sentinels; the production caller omitted the dataset. |
| `physics/noahmp/phenology.py:66–80,203–206,238–242,270–295` | Both monthly and input-LAI branches pin MODIS no-vegetation/FVEG sentinels and the 20-class TMIN vector; clipping can hide USGS out-of-range indices. |
| `physics/noahmp/noahmp_driver.py:272,479` | Energy and urban soil use parameter isurban, falling back to MODIS13; LCZ indices51–61 are explicit. |
| `physics/noahmp/water_hydro.py:860,884` | Optional drainage/MMF paths hard-code urban13 and LCZ51–61. |
| `io/noahmp_land_init.py` | Glacier/soil initialization uses parameter isice; input-LAI initialization uses parameter sentinels. Wrong table selection therefore affects initialization too. |
| `physics/noahmp/groundwater.py:317–319` | Uses parameter sentinels plus LCZ51–61. |
| `io/land_history.py:76,102` | Default parameter load is MODIS; seasonal LANDUSE.TBL selection independently uses input MMINLU, allowing mixed conventions before the fix. Production builds land first. |
| `coupling/urban_slucm_hook.py:68–87` | Natural-category table helper defaults to MODIS (14 vs USGS5); urban mask otherwise uses the supplied isurban. |
| `io/noahmp_crop.py`, `io/noahmp_irrigation.py` | Crop explicitly requires MODIS; irrigation has USGS/MODIS mappings but cannot establish whole-path dataset support. |
| `physics/surface_layer.py:684–685,1133–1147`; `coupling/sfclay_selected.py` | MYNN land/water classification uses XLAND and carried roughness, not categorical MODIS indices. |
| `io/lsm_static_extract.py:59–62,420–425,517–523` | Separate generic land/surface extraction embeds MODIS LANDUSE/VEGPARM and water17 fallbacks. Not changed by this Noah-MP admission fix. |
| `io/wrfout_writer.py:2615–2630` | Generic fallback metadata uses MODIS water17/ice15/urban13/MMINLU/count21; source_global_attrs override input metadata afterward. No writer change. |

The complete bounded source search is retained in
`<USER_HOME>/wrf_gpu2_lanes/w44-lutab/source_category_audit.txt`.

Changing only the MPTABLE dataset argument would leave these coupled
assumptions inconsistent. A faithful extension needs dataset-keyed phenology,
TMIN and optional urban/hydrology/caller auditing, original WRF USGS column
oracles, and affected composed forecast gates. Estimate: 1–2 worker days plus
scheduled validation [I]; no empirical support-duration claim.

## Staged input metadata

Read only MMINLU and NUM_LAND_CAT on the shipped Swiss example and all
`<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/*/wrfinput_d0*` staged inputs:
19 files, all MODIFIED_IGBP_MODIS_NOAH/21 [M]. Includes WN3 0227/0502/0614
three-domain inputs and the staged 0227_48h/0227_icloud0/0502_icloud0 variants.
No ALISIOS source paths or observations were read.

## Verification and costs

`LU02_identity.json`: all 118 initialized MODIS land/static leaves match the
baseline byte-for-byte (dtype, shape and SHA256), and initialization metadata
matches [M CPU]. Numerical calculations and parameter arrays are unchanged.
The final diagnostic formatting edit affects only the rejected-input error.

The sole production change reads an attribute from the already-open input
handle and compares a host string once per domain initialization. No new file
open in the public scoped path, model state leaf, timestep operation, device
transfer, kernel, or device allocation is introduced [source proof]. The
host overhead and actual GPU timing/VRAM impact are unmeasured [U]; no speedup
is claimed. Existing CPU init times are observational and not a matched timing
benchmark. Release identities and consumer/mutant results are recorded in the
final receipt beside this file.

Final gates [M CPU/source]: 221 tests passed, 30 skipped (159.98 s); all seven
new admission tests passed. Deleting the guard causes the real-input test to
fail with `DID NOT RAISE ValueError` (one assertion failure, zero setup errors).
The first mutant launcher hit a CPU-guard import-order setup error; that is
retained as an unsuccessful harness attempt, not counted as a kill.

Frozen production identities on immutable `8cf72850ef`:

| Case | Exact location-stripped production jaxpr SHA256 |
|---|---|
| Swiss | `b977712fe325c953f3ebbf8f83ea1ca0b9260833ef70981e48bd8a49410e142a` |
| PROD | `321aa651067f1d074b25ebe164536bf27fea5dca151a80c2861dcbab8b58f96b` |

The changed host loader is absent from the actual 519-file AOT source import
closure; all keyed source files remain unchanged. It therefore adds no AOT
source-fingerprint invalidation [M source], without a cached-readiness timing
claim. Production/test sources remain exact to `05c1c5cb9b` throughout the
consumer cohort and final probes; later commits touch only proof tooling/docs.

Status: **READY-FOR-CRITIC**. Probe scripts and artifact checksums are beside
`RECEIPT.json`; the main EXPERIMENTS rows LU01–LU08 record each arm. No numerical repeat
is needed for the byte-identical MODIS program.

CPU-only lane handoff does not assert completion of manager-owned RW1–RW4 or
a release gate. No GPU command or GPU-lock waiter was started.
