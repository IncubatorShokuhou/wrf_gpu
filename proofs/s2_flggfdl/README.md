# FLG/GFDL experimental CPU-only port evidence

This branch implements paired FLG7/7 and GFDL99/99 as **EXPERIMENTAL CPU-ONLY**
operational options. GPU selection is refused before JAX initialization: native
GPU column kernels are a v0.4.x follow-up. See [the caller scope and disclosures](../../docs/flg-gfdl-radiation.md).

The final receipts include 48 full FLG caller cases byte-exact (FG62), 40 GFDL
whole calls ×17 outputs byte-exact through the product boundary (FG52/61),
28 native CO2 initialization fields byte-exact (FG59), source SOLARD/ETARA gates,
three coupled Swiss CPU steps per option (FG66/69), full-carry restart/history
checks (FG68/70), and actual composed CPU artifact precision/resources (FG71/72).
FLG's one tiny boundary QC event has the explicit manager F2 disposition in FG66;
the matched RRTMG control and GFDL had zero guard events. No guard/tolerance changed.
Compiled numerical excesses remain disclosed and pending D6. No GPU, forecast
speed or full coupled-forecast fidelity claim follows from these receipts.

The old v018 history-only savepoints omit inputs. The new native oracle compiles
unchanged WRF 4.7.1 `module_ra_flg.F`, `module_ra_gfdleta.F`, `module_mp_etanew.F`
and `module_model_constants.F` in a lane directory. Stubs implement serial WRF
infrastructure and the configuration type's sole consumed member `co2tf`; they do
not replace physical routines. Source hashes, compiler flags and units are in
`manifest.json`. Native source and compiled binaries never write in pristine WRF.

Inputs reuse the 40-column CAM01 compact fixture's CPU-WRF thermodynamic fields,
never its CAM outputs. Missing interface T, eta grid, vegetation and convective
accumulators are **constructed routine-boundary operands**, explicitly described
in the manifest. This is not the original forecast radiation-call replay. The
Fortran driver follows FLG's one combined call and GFDL's LW call followed by the
skip-solve SW call. Native full calls have finite outputs on 40 columns, with
21 daylight SW-active columns. Bounds/FPE and signaling-NaN initialization pass;
FLG O0/O2 outputs are byte-identical, GFDL has small optimization-rounding differences.

Leaf references cover all 121 NGAS=0 gas quadrature points, all 18 continuum and
precipitation/Rayleigh bands, and all five prescribed ozone profiles. Real columns
have no graupel; the leaf oracle adds explicit contents `[0, 1e-6, 1e-5, 1e-4, .01]`
g/m3 to test inactive, exact-threshold and active branches. This does not establish
whole-column active-graupel qualification. Frozen leaf gate: finite float32 arrays,
16 reference ulps plus 1e-12 absolute. Uncontracted REAL polynomial intermediates
are required: the initial JIT port failed two band-15 elements at 19 ulps; the
value-identical `nextafter(x,x)` rounding boundaries pass without gate changes.

Build and regenerate (CPU affinity, JAX CPU env, same-line QUIET guard and timeout
must be supplied by the invoking lane):

```text
python fixture.py data/fixtures/cam01-compact-v1.npz LANE/inputs
bash build.sh LANE/build
cd LANE/build
./oracle 7 LANE/inputs/input.bin LANE/ra7.bin
./oracle 99 LANE/inputs/input.bin LANE/ra99.bin
./tables tables.bin
./leaves LANE/inputs/input.bin leaves.bin
python REPO/proofs/s2_flggfdl/parse.py LANE/build LANE/parsed --input LANE/inputs/input.bin
```

`flg_tables_v1.npz` is a 41-KiB runtime asset with WRF DATA values, loaded as host
NumPy only before conversion at use. The 848-KiB committed test fixture includes
operands and independent native leaf outputs. Ordinary release dispatch is unchanged.
The rounding helper adds elementwise bit operations; compiled cost and memory must
be assessed again in the final composed column kernel, where fusion/layout can differ.

Critic repair: `receipts/CRITIC_REPAIR_HANDOFF.json` supersedes the rejected
`FINAL_CPU_HANDOFF.json` for the operational caller. Both solvers now consume
the shared WRF SGS condensate view and existing PBL policy. Eight seam cases
and nine genuine mutant failures establish deletion/corruption sensitivity.
The exact 17 known strict compiled transport excesses remain band-specific
expected failures pending D6; separate hard regression checks cannot hide
zero/nonfinite compiled outputs. That baseline is JAX regression only.

The accepted repair verdict is preserved in
`receipts/FG93_VERDICT_REPAIR_dc5139f7e.md` (MERGE-WITH-FIXES, disclosure only).
`receipts/FG93_flg_compiler_disposition.json` and the exact independent
`receipts/FG93_flg_native_bound.json` distinguish seven source-bounded strict
deviations from ten residual bands (285 values) outside the native envelope.
Those ten are experimental/D6-pending deviations only. The compiled-zero hard
FAIL is an intentional mutant kill; the clean candidate has 89PASS/17XFAIL/0FAIL.
The raw mutant and clean candidate XML reports are tracked alongside the verdict.
