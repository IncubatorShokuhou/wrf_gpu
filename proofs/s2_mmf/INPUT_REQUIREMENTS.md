# Noah-MP MMF geography and qualification

The input blocker is closed. NCAR's official groundwater archive downloaded
successfully (677,741,073 bytes, below the authorized 5 GB download limit).
The original onboarding WPS/real path produced the private Swiss lowland case
`<DATA_ROOT>/wrf_gpu2/scratch/s2-mmf/MMF34_lowland`: 47×41×44, 3 km, DT=15 s,
with no glacier cells. All four geographic fields are finite; FDEPTH is active
at all 1927 cells. Pristine WRF opt_run=5 completed 12 steps with four lateral
calls (WTDDT=0.75 minutes). Reference outputs are in `MMF35_cpu`.
The earlier `MMF05_swiss` debugging case has two glacier cells and is outside
the admitted ice-free scope. Its component-oracle inputs remain valid, but its
earlier coupled smoke is not the final caller qualification.

Source: [NCAR WPS geographic downloads](https://www2.mmm.ucar.edu/wrf/users/download/get_sources_wps_geog.html),
[groundwater.tar.bz2](https://www2.mmm.ucar.edu/wrf/src/wps_files/groundwater.tar.bz2).
SHA256: `ac204fd5bc7efe667fc9ba47cb5539dfa41f784745b7d1aedeac48891b853fa6`.
Expanded payload is 7,464,961,116 bytes (2601 archive members); this differs
from the compressed download size. Check available disk space before extraction.
Acquisition, preparation and reference receipts are MMF04, MMF34–MMF35 in the lane
artifact directory. Original case inputs and reference output are preserved.

The implementation is experimental: original-WRF component oracles, complete
carry/restart/history, opt-in binding and CPU coupled checks qualify its
development scope. GPU timing, GPU robustness/E41 and full forecast D6 remain
pending. Initializer and soil native compiler bounds are recorded separately
from literal REAL4 parity. No GPU forecast qualification is implied.

The initial MMF00 census found `FDEPTH`, `EQZWT`, `RECHCLIM`, and `RIVERBED`
entirely zero in the Swiss, PROD, and WN3 d01 inputs. A variable name in
`wrfinput` therefore does not establish MMF initialization. `FDEPTH=0` makes
`KCELL=0` in the pristine lateral-flow calculation. Simply selecting MMF on
these inputs cannot exercise active lateral groundwater transport.

`audit_inputs.py` records file hashes, finite/masked/nonzero counts, active land
coverage, and local geography-directory availability. A populated file receives
only `CANDIDATE_REQUIRES_PROVENANCE_AND_PRISTINE_ORACLE`; this tool never grants
scientific qualification. It does not change the CLI's admission policy.

The sealed [qualification packet](QUALIFICATION.json) links source hashes,
the original trajectory, final coupled/restart checks, both caller bindings,
cache regressions, default identities and CPU artifact accounting. Rebuild
the unedited original routines with `python proofs/s2_mmf/build_oracles.py
<pristine-WRF-root> <new-owned-output-directory>` under the lane's CPU,
QUIET and timeout wrapper. The three tracked Fortran callers document their
binary input protocols; `original_wrf_real4.npz` retains the corresponding
`grid/<regime>/input`, `init/input` and `column/<opt_inf>/<regime>/input`
packets. `ORACLE_REBUILD.json` records fresh original-output byte identity.

## Prepare a real MMF case

1. Use the domain's original WPS configuration and forcing, in a private lane
   work directory. Select `geogrid/GEOGRID.TBL.ARW.noahmp` as `GEOGRID.TBL`.
   WRF 4.7.1 `run/README.namelist:1317-1319` explicitly requires this table for
   `opt_run=5`. Preserve existing case directories and references.
2. Supply the actual MMF geographic tiles under `geog_data_path`. The installed
   WPS v4.6 table defines these mappings at lines 1042-1076:

   | WRF field | Relative geography directory | Meaning at input |
   | --- | --- | --- |
   | `FDEPTH` | `groundwater/f/` | Transmissivity e-folding depth, m |
   | `EQZWT` | `groundwater/wtd/` | Equilibrium water-table depth, m |
   | `RECHCLIM` | `groundwater/recharge/` | Recharge climatology, mm in Registry |
   | `RIVERBED` | `groundwater/riverbed/` | River-bed elevation before initialization, m |

   Each directory needs its WPS `index` and tiles covering the domain. The
   entries are `optional=yes`; successful geogrid exit alone does not prove
   these data were read. Record the dataset provenance and check the resulting
   `geo_em` fields and geogrid log. Table defaults/fill values alone are not
   evidence of real geography. Download NCAR's archive linked above, extract
   into the private geography root, and retain the archive SHA in provenance.
3. Run geogrid, metgrid and `real.exe` in the private case. Verify all four fields
   survive into every `met_em`/`wrfinput` domain, with finite valid land values
   and active `FDEPTH`. Do not populate them with arbitrary constants or
   overwrite the original reference inputs. WRF's Registry declares them input
   streams 0/1/2 in `Registry/registry.noahmp:94-97`.
4. Set `&physics sf_surface_physics=4, wtddt=<minutes>` and `&noah_mp opt_run=5`
   in the pristine CPU-WRF oracle case. `WTDDT` defaults to 30 minutes
   (`Registry.EM_COMMON:2705`). Build/run only a lane-owned pristine copy, with
   the dump path redirected as required by FINDINGS E104. Preserve the original
   Fortran routine bodies. Apply the wave's compiler initialization policy if
   needed; inspect sensitivity rather than manufacturing MMF inputs.
5. Capture the initialized arrays and a real coupled trajectory spanning multiple
   lateral-flow calls, including cold-start and resumed cadence. Census active
   lateral/baseflow/recharge/spring regimes before accepting numerical coverage.
   Only then can the MMF port be qualified against CPU-WRF.

The three installed geography roots under
`<DATA_ROOT>/canairy_meteo/artifacts/wps_geog/` (`WPS_GEOG`,
`WPS_GEOG_CANAIRY_SURFACE_V21`, `WPS_GEOG_LOW_RES`) have no `groundwater`
subdirectory at this audit. This is a local inventory, not a claim that these
datasets are unavailable elsewhere.

## Required implementation scope after inputs are supplied

- The pristine grid model is
  `phys/noahmp/src/module_sf_noahmp_groundwater.F`, 614 lines:
  `WTABLE_mmf_noahmp:12-201`, `LATERALFLOW:204-299`, `UPDATEWTD:302-610`.
  `LATERALFLOW` is an eight-neighbor stencil with explicit outer-ring exclusion
  and literal sequential accumulation order. Preserve these on device.
- `NOAHMP_INIT` computes `AREA=DX*DY/(MSFTX*MSFTY)` at driver line 2151 and
  `STEPWTD=max(NINT(WTDDT*60/DT),1)` at 2324-2325. A separate cold-start call
  from `dyn_em/start_em.F:1277-1293` invokes `GROUNDWATER_INIT` at
  `module_sf_noahmpdrv.F:2442-2755`: topographic water-table adjustment (up to
  500 iterations), river conductance, equilibrium soil/deep moisture and flux
  initialization. `AREA`, `SMOISEQ`, `SMCWTD`, `RIVERCOND`, and `PEXP` being absent
  from raw `wrfinput` is expected; they are derived, not missing geographic data.
- Every Noah-MP column step needs the MMF soil/deep-recharge coupling, including
  `SHALLOWWATERTABLE:9774-9909` and the `OPT_RUN=5` SRT/SSTEP/soil branches in
  `module_sf_noahmplsm.F`. A column-only replacement omits spatial transport.
- `module_surface_driver.F:3345-3361` calls the grid model after Noah-MP when
  `MOD(itimestep,STEPWTD)==0`. This is distinct from radiation/LSM cadence.
  `module_first_rk_step_part1.F:541-547` exchanges WTD at the actual caller's
  location before the surface driver; `Registry.EM_COMMON:3492-3493` declares
  steady-state and initialization halos. Oracle wiring must reproduce that
  ordering, physical-edge exclusion and tile bounds.
- The groundwater Registry group at `registry.noahmp:78-97` includes the
  equilibrium soil profile, recharge/deep recharge, geometry, river and spring
  fluxes and sums, conductance/exponent, and persisted `STEPWTD`. Retain WRF
  REAL4 and source units/lifetimes, with complete restart/input/history mapping.
  Do not silently replace checkpoint-required fields with zeros on resume.

## Performance and memory

The optional MMF bundle adds 136012 bytes on 40×40 [M]: 85 bytes per cell
including eligibility, plus 12 bytes of controls. SMCWTD already belongs to
the unchanged Noah-MP land carry. ACRECH/ACQSPRING are invariant zero under
the admitted `noahmp_acc_dt=0`; restart retains that explicit scalar metadata,
and history broadcasts zero only at output. Two redundant device planes are
therefore unnecessary. Positive `noahmp_acc_dt` is explicitly refused.
The final 47×41 lowland coupled smoke measures 163807 bytes for this bundle,
zero nonfinite values and zero guard events over three steps. All four restart
formats resume across the third-step lateral call byte-identically. These
restart comparisons establish reproducibility, separately from WRF fidelity.

The MMF soil recurrences execute within one Pallas column call. Spatial
exchange uses an eight-neighbor full-domain device stencil at STEPWTD cadence,
with original outer-ring exclusion. No host transfer occurs inside a step.
The up-to-500 spatial cold-init iterations run only at setup. CPU artifact
timing and intermediate memory are separate from GPU launch/traffic/peak
measurements, which remain unknown until an authorized GPU window.

Registry names, units and values are preserved as written in WRF. In particular,
DEEPRECH and instantaneous QLAT use metres internally although their Registry
unit strings say `mm`; QSLAT/QRFS/QSPRINGS and accumulated RECH multiply the
corresponding depth increments by 1000. The port adds no invented conversion.
