# NSSL cumulus qualification for v0.4.0

| Selection | Support | Qualification |
|---|---|---|
| `mp_physics=18`, `cu_physics=0/1/2/3/5/6/16/93`, single domain | Experimental opt-in | CPU-oracle-qualified; GPU/coupled-forecast qualification pending |
| `mp_physics=18`, nested domains | Refused | Parent-to-child forcing of NSSL numbers and volumes remains unwired |
| `cu_physics=4` with any microphysics | Refused | Pristine WRF rejects this SAS option for ARW |
| `mp_physics=40`, `aercu_opt=0`, RRTMG | Experimental opt-in, default radiation radii | Matches pristine WRF's `has_reqc/i/s=0` |
| `mp_physics=40`, `aercu_opt>0` | Refused | Requires the unported aerosol–MSKF (`cu_physics=11`) caller and input substrate |

The NSSL column now reproduces `calcnfromcuten` immediately after the first-step
`calcnfromq` call and before sedimentation. Its inputs are `dtp*Q*CUTEN` in kg/kg.
KF supplies the held rates after `advance_ppt` clears the last cloud step and
`phy_prep_part2` decouples them using the final dry mass. The update adds rain and
snow numbers to the density-scaled internal stack (#/m³), with WRF DOUBLE locals
only where the pristine routine declares them. No new restart field is needed.

The pristine cloud/ice conditions test **number** slots in `ancuten`, while the
caller fills only **mass** slots and initializes every other slot to zero.
Consequently cloud/ice-only detrainment is a no-op in this WRF caller. The other
supported cumulus schemes supply no rain/snow mass tendencies. Their previous
numerics are retained.

The new oracle stages unmodified `module_mp_nssl_2mom.F` in a private lane build.
Fourteen fixtures (seven columns, both REAL builds) include active rain/snow
sources, negative/subthreshold rates, cloud/ice-only forcing, and a first step
with mass but no numbers and nonzero activated CCN. Print-only stage dumps must
produce byte-identical final output to the pristine arm. The manifest freezes
tolerances and source/file hashes; the tests include a killed helper-deletion
mutant. Full-driver gates retain the existing NSSL tolerance policy.

## Morrison radiation rule correction

Allocating `re_cloud/re_ice/re_snow` in Registry does not activate their use.
`module_physics_init.F:985–1025` excludes `MORR_TM_AERO` from its `has_req*`
selection. A compiled, literal extraction of this block gives zero flags for
mp=40 at both `use_mp_re=0` and `1`. `module_radiation_driver.F:1339–1394` enables
the blended Morrison/MSKF radii inside **`aercu_opt>0`**. The currently supported
`aercu_opt=0` point therefore correctly uses RRTMG's default radii. Enabling
Morrison radii there would change the pristine WRF program. This corrects the
previously disclosed radius gap; the aerosol–MSKF implementation remains future
work. Evidence: `proofs/v040/nssl_cumulus/radius_audit.json`.

The same audit identified a separate NSSL radiation gap: WRF activates
mp=18 radii with `use_mp_re=1`. The subsequent [microphysics radius fix](S1_MP_FIXES.md)
wires and CPU-qualifies the held radii and RRTMG caller handoff. The cumulus
qualification here remains separate; GPU/coupled-forecast qualification is pending.

The Thompson release path is unchanged. CPU component allocation and timing
are diagnostic only; GPU launch, memory and forecast performance are unmeasured.

Final CPU evidence is indexed in `proofs/v040/nssl_cumulus/STATUS.json`: 14 new
pristine/E26 fixture pairs, 29 oracle tests, 20 caller/selection checks, all 249
required shared-consumer checks, and three killed full-driver mutants. The real
Swiss NSSL+KF run completed three CPU steps with zero non-finite values and zero
counted guard events. The release jaxpr remains `2bf12cb8`.

With dynamic resident cumulus rates, a 40-level CPU column uses 99,104 bytes of
compiler temporary storage in both control and candidate. The isolated candidate
adds two 160-byte input vectors; the forecast already carries those KF rates.
Median CPU calls were 1.124/1.044 ms and compile-plus-first-call 20.977/20.535 s;
these noisy component measurements establish no speedup or GPU cost claim.

After rebasing onto the merged station-verification CLI, 447 original source
files remained byte-identical; only the CLI and new `verify/` package differed.
All 173 affected CLI/namelist consumer checks passed. The separate
`FINAL_COMPOSITION.json` seal binds this composition to the qualification packet.
