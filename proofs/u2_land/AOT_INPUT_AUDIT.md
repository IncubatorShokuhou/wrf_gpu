# Bound land-table identity

Rebased on main's `76a72bdd1` keyed-provider mechanism. No new NPZ or binary
asset is read by the operational land modules; the 20 JSONs here are independent
test references. The new external readers resolve values during initialization:

| Reader | Data | Key disposition |
|---|---|---|
| `io/noahmp_options.py` | MPTABLE BATS, RSURF_SNOW, FSATMX, TIMEAN, DRAIN_LAYER_OPT | Frozen NoahMPOptions auxiliary; explicitly registered bound-value provider in `aot_baked_tables` |
| `io/noahmp_soil.py` | SOILCL/SOILCOMP and optional pedotransfer coefficients | SelectedSoilParameters inside existing NoahMPStatic content-keyed static holder; maps become actual selected coefficient arrays |
| `io/noahmp_hydrology.py` | SOILPARM runoff, MPTABLE tile geometry, TD_FRACTION | SelectedHydrologyParameters inside the same static holder, including derived equivalent depth and actual fraction map |
| `io/noahmp_carbon.py` | Actual MODIS carbon rows and cold pools | CarbonStatic traced namelist child; values participate in function identity when a fused closure bakes the child |
| `io/noahmp_crop.py` | Crop scalar/stage arrays, category fractions, calendars, season GDD | CropStaticParameters in that traced child; crop photosynthesis overrides also ride the existing energy-parameter static holder |
| `io/noahmp_irrigation.py` | Irrigation constants, calendar/fraction/root/soil maps | IrrigationStatic traced child; content-keyed when baked into a fused closure |

The provider hashes **resolved values**, without rereading a table on key
construction. Ordinary carbon/crop/irrigation parameters remain operands,
preserving cross-case executable reuse. Closed namelist children use main's
content-aware function identity. Inactive land options add no provider record.

`tests/test_u2_aot_keys.py` checks actual full-key/HLO/output separation for all
five newly resolved MPTABLE controls, plus dynamic-versus-closed child identity.
These are cache regression checks, not WRF fidelity claims. The existing main
key tests cover every Noah-MP energy-parameter field and static-holder bytes.
Soil/runoff/tile-holder collision checks are retained alongside these controls.
