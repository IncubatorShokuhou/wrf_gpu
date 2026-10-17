# Full BEP/BEM and urban PBL CPU qualification

The port implements the WRF Noah-MP BEP (`sf_urban_physics=2`) and BEP+BEM
(`3`) caller chain, including urban sources consumed by YSU and BouLac.
Supported configurations and refusals: [urban guide](../../../docs/URBAN_BEP_BEM.md).
CPU-oracle-qualified; GPU/coupled-forecast qualification pending.

The final packet records **10/10 oracle arms with zero violations**, **318
unique consumer tests passing**, **five full default-program hashes unchanged**,
and **three genuine numerical mutant failures with zero setup errors**. Both
full-domain coupled smokes have three steps, 31 urban cells, zero nonfinite
values and zero guards. Their final composed production graphs and all 457
initial dynamic leaves match BYTE; completed executions are reused by this proof.

`urban_bep_model.py` implements canyon view factors/radiation/shadowing,
roof/wall/ground diffusion, exchange and urban-to-mesoscale sources/lengths.
`urban_bem_building.py` and `urban_bem_model.py` add indoor radiation, windows,
floors, material/room-air state, occupants/equipment, ventilation, HVAC and
moisture. `urban_bep_blend.py` follows the literal Noah-MP blend. Urban
YSU/BouLac consumers, actual revised-MM5 and runtime forcing live in
`urban_bep_pbl.py` and `coupling/urban_*`. Raw PBL rates enter RK once;
material/indoor/water, sources and held surface/radiation state survive restart.
Ordinary MYJ cannot substitute for WRF's missing `module_bl_myjurb`.

The full oracle stages unmodified `module_sf_urban.F`, `module_sf_bep.F`,
`module_sf_bep_bem.F`, `module_sf_bem.F`, constants and helpers, and calls the
verbatim `noahmp_urban` extract. It retains WRF Registry sizes (15 building
types, 18 urban levels), `num_urban_hi=15`, and check_a_mundo's `nurbm=3`.
Surface-driver diagnostic overwrites are reproduced explicitly. Both
REAL4/REAL8 outputs are tracked; all inputs are recorded exactly. Five WN3-derived
lowest-level forcings have **constructed** eight-level extensions and fixed
rural LSM/surface inputs. A sixth multi-layer stress column exercises sources
above the first level. Each arm has six dt=6 s steps; BEM has AC on/off.
These are full caller comparisons, not real vertical-column forecast replays.
Separate fixtures invoke original YSU/BouLac urban branches for six regimes
in both precisions. The 48-regime helper oracle covers calm, stable/unstable,
heat caps, wind branches, one/two street directions and heating/cooling/comfort,
including the mutated HVAC latent demand INOUT.

Original REAL8, state and geometry gates remain fixed. The manager approved
the pre-registered [ROUND01](evidence/ROUND01.json) WRF-operand roundoff budget
only for conditioned REAL4 fluxes/tendencies. Its generator
[`roundoff_budget.py`](roundoff_budget.py) read no candidate results: calibrated
wall/flat operands, analytic Louis/saturation sensitivity, positive
urban_meso/Noah-MP weights, original PBL update/gradient rounding intervals.
All 87 original arrays remain BYTE with the WRITE-only observer; all
wall/flat/wet-latent values join the pristine outputs BYTE. All 24 gradient
interval checks pass. The fixed eight-epsilon packet and approval/checksums
are published in `evidence/ROUND01_APPROVAL.json`; comparisons reject altered
packets and retain raw failure counts. No D6, state, station, drift or project
tolerance changes are made. The original proposal is preserved in
`CONDITIONING_REQUEST.md` with the subsequent approval identified separately.

Full canopy comparisons: `bep_driver_parity.py`, `bem_driver_parity.py`;
original urban PBL comparison: `pbl_parity.py`. Example:

```bash
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=src taskset -c 28 nice -n 19 timeout 600 python proofs/v040/urban_bep_bem/bem_driver_parity.py --save proofs/v040/urban_bep_bem/savepoints --budget proofs/v040/urban_bep_bem/evidence/ROUND01.npz --out <USER_HOME>/wrf_gpu2_lanes/s1-bepbem/bem.json
```

Rebuild references with `build_oracle.py` and `pbl_oracle_build.py`, using
lane-local build/save paths and the same QUIET, CPU pin and timeout rules.
The parser rejects missing/duplicate records and incomplete step inventories:
Fortran STOP-zero is not completion. To reproduce ROUND01, pass
`--operands evidence/WRF_flux_operands.json`, `--wall-bounds evidence/BEP26.npz`,
`--save savepoints`, pristine `--table .../WRF/run/URBPARM.TBL`, and a lane-local
`--out` to the generator.

Deletion controls remove actual AC exhaust or urban drag/moisture sources.
WRF-only controls inject one material ULP within the budget and reverse heat
sign outside it. They require numerical failures with zero setup errors.
Tests cover active initialization, exact restart roundtrips, incomplete-state
refusal, actual SF1 frozen oracle gates, admission/pairing refusals and archive
assets. Historical partial receipts are separate from the final
[qualification](evidence/qualification.json).

Authentic Madrid smoke uses 42×42×44 points, 31 urban cells, dt=50 s and three
own-steps in one production chunk. CPU Pallas interpretation and REAL instruction
emulation establish finite/zero-guard coupling only. Default identity receipts
compare the full Swiss release program and YSU/BouLac with urban0/1; this is
regression evidence, never CPU-WRF truth.
After the portability merge, the local probe holds the native CUDA Pallas
callable while interpreting/emulating on actual CPU devices, matching the
original smoke. This does not qualify the portable CPU backend. Initial carry
hashes bind every dynamic input, in addition to the complete production graph.

State is compact over active urban columns, building types and levels; six
held SF1 arrays cover the domain. Independent BEM floors are batched:
58/58 first-step fields BYTE versus the scalar version; CPU temporary storage
701505→686529 B; shared-core seven-call warm medians 0.925/0.777 ms.
Compile 38.77/37.96 s is noise, not a proven improvement. These are component
CPU measurements. GPU E41, launches, peak VRAM, speed and long forecast
fidelity remain unmeasured.
