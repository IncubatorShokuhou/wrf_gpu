# Nested option qualification for v0.4 development

The released Thompson nested program remains the baseline. An independently
qualified column or a single-domain smoke does not qualify a new nested option.

| Option | Development admission | Evidence and missing gate |
|---|---|---|
| Extra microphysics species (`mp=18/24/26/28/40`) | Single domain only; admission depends on s1-mpfix | Ten optional REAL boundary records, coupled parent force-down, RK1-frozen tendencies, final specified-zone pinning and restart storage are staged. Pristine `bdy_interp1`/`SINTB` and scalar boundary routines exercise every extra species with signed, nonzero operands. Raw nest-start `SINT`, initialized-parent aerosol inheritance and input-stream precedence are implemented. The reviewed number lane is composed, including mp26 initialization and removal of duplicate WDM Nr transport. The real two-domain mp40 gate fails on a root Nc defect; default radius mediation also blocks 18/24/26/28. No new nested code is admitted. |
| RUC (`sf_surface_physics=3`) | Single domain only | The column/init/diagnostic oracles and single-domain smoke exist. RUC land initialization, held diagnostics and evolution through child subcycles lack the two-domain CPU-WRF comparison. Refused before model import in the CLI and before case building in the nested API. |
| SLUCM (`sf_urban_physics=1`) | Single domain only | Column/coupling oracles and single-domain urbanized Swiss smoke exist. Child canopy initialization/carry and surface fluxes lack the two-domain CPU-WRF comparison. Same early refusal. |
| LES TKE (`diff_opt=2/km_opt=2`) | Single domain only | Single-domain CPU-oracle-qualified. Child-boundary/halo advection and diffusion lack the two-domain CPU-WRF comparison. Registry `tke` is restart-only, without `d`/`f` flags; it must not be treated as another moist/scalar force-down species. Same early refusal. |
| Grid FDDA (`grid_fdda=1`) | Existing CPU-qualified path retained | WRF declares per-domain controls. The loader selects each domain's `wrffdda_d0N`, coefficients, grid shape, time interval and cursor. Omitted entries take Registry defaults rather than copying the parent's settings. This is per-domain binding evidence plus the existing pristine FDDA operator oracle; whole nested forecast/GPU qualification remains pending. |
| DFI (`dfi_opt=3`) | Single domain only | WRF supports simultaneous-start nests: `main/module_wrf_top.F:436` and the recursive initialization/filter routines in `share/dfi.F`. The port lacks recursive filter phases and nest clocks. The refusal is a port limitation, not a WRF prohibition. |

The original `module_physics_init.F` enables `has_reqc/has_reqi/has_reqs=1`
for mp18/24/26/28 with the default `use_mp_re=1` and the RRTMG pair. The
port's radius predicate currently accepts mp8 only. Its WSM7 and aerosol-aware
Thompson adapters do not retain those diagnosed radii, and the NSSL adapter
discards its optional radius diagnostics. The unchanged original predicate
was compiled for all six schemes with `use_mp_re=0/1`; all 12 rows match this
disposition. Morrison mp40 with the supported `aercu_opt=0` correctly retains
WRF's default-radii behavior. A dry coupled case cannot qualify the missing
active-cloud path. The extraction is [radius_admission_audit.py](../../proofs/s1_nestspec/radius_admission_audit.py).

The completed real WN3 mp40 comparison advances two parent and six child steps
over 108 seconds. Scored core D6 regions pass and every sampled carry is finite, with
zero guard events and zero policy water changes. Cloud number fails in both
child regions: the port has 1,374,516 nonzero cells while original WRF has zero
through step three and 220 at steps four through six. The final boundary error
reaches 2.892441344e9 per kg. The original scheme skips its constant-droplet
assignment in dry empty cells; the port currently performs it unconditionally.
Eight negative Nc visits also appear before masks; their amplitudes were not
retained in that arm and remain unknown. These findings are assigned to
`s1-mpfix`, along with the radius mediation above. The references and failed
gate are retained; no tolerance or admission guard is relaxed.

The nest-start helper follows `med_nest_initial`: interpolate raw Registry
fields with `interp_fcn_sint/SINT`, then apply fields present in the child's
main input stream when `input_from_file` permits it. Omitted per-domain input
flags use the Fortran default, rather than repeating the parent's value.
Reading an overriding input first avoids materializing an interpolation WRF
immediately overwrites. This is initialization work; it adds no timestep
allocation or transfer. CPU initialization timing and the two-domain receipts
are separate from GPU readiness and forecast qualification.

The extra-record initializer allocates only the selected scheme's records.
Absent records add no array leaves, device operations or persistent bytes to
the default configuration. For each additional field, two time records require
`2 × 4 sides × bdy_width × nz × max(nx+1,ny+1) × 4 bytes`. The compiled producer
is still one executable and the existing boundary kernels do the arithmetic;
enabled fields add work and temporary storage. CPU changed-artifact compile,
latency and executable memory are reported separately. GPU launch counts,
timing and peak VRAM are unmeasured in this CPU-only wave.

Measured CPU component costs use the same signed 3-level, 22×19 child operands
with absent records and with ten materialized records. Warm values are medians
of seven calls on shared core 24 at nice 19. Temporary bytes are compiler memory
reports, not measured process RSS or GPU peaks.

| Component | Compile seconds, baseline → ten records | Warm milliseconds | Temporary bytes |
|---|---:|---:|---:|
| Force-down | 21.221 → 41.735 | 7.979 → 27.488 | 4,728,120 → 6,069,944 |
| RK1 boundary tendency | 6.064 → 11.586 | 0.929 → 4.596 | 94,312 → 211,432 |
| Final boundary pin | 1.570 → 2.090 | 0.775 → 1.438 | 41,040 → 110,480 |

State ownership follows the existing mass-grid boundary layout. Force-down
updates only boundary records; the child's prognostic arrays retain their
identity. Final pinning preserves the interior byte-for-byte, with the WRF
specified/relaxation widths and Y-side corner ownership. No layout or halo
exchange is added. Record construction and consumers use JAX arrays; the added
code contains no explicit host transfer or host callback inside stepping.
The GPU transfer audit remains pending. Active records survive pickle and
NetCDF restart exactly; missing required records are refused.

The [CPU evidence packet](../../proofs/s1_nestspec/CPU_QUALIFICATION.json) binds
source hashes, test coverage, the genuine deletion mutant and the unchanged
Swiss release program hash. The [oracle manifest](../../proofs/s1_nestspec/ORACLE_MANIFEST.yaml)
records source provenance, units, shapes and tolerances.
