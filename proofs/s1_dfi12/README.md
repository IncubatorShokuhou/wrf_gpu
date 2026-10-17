# DFL/DDFI CPU qualification

Experimental single-domain native `dfi_opt=1/2`, extending the retained o2-dfi TDFI driver.
No GPU qualification or GPU performance claim. All numerical truth is original CPU-WRF,
not another port version. CPU JAX executes release kernels with Pallas interpretation and
the existing o1-nlbind CPU tridiagonal/inline-PTX shims.

Pristine reference: clean RD11 WRF 4.7.1 executable copied into
`<USER_HOME>/wrf_gpu2_lanes/s1-dfi12/wrf.exe`, SHA/provenance in `oracle_manifest.json`.
Swiss 42×42×44, dt=18 s, original RD11 input and boundary, Thompson/MYNN/Noah-MP/RRTMG.
Primary oracles: `oracle/opt1` (forward 18 minutes, forecast midpoint +9 minutes,
`diff_opt_dfi=1`, `km_opt_dfi=4`) and `oracle/opt2` (backward 18 minutes,
forward from −18 to +9 minutes, unfiltered backward endpoint). Both pristine runs
finish successfully and write `wrfinput_initialized_d01`.

`run_e2e.py SOURCE TAG ORACLE` runs the corresponding port prepass on an immutable
source snapshot. `compare_filtered.py` inventories every numeric oracle variable,
rejects nonfinite values before scoring, and records every field's error and exactness.
Primary gates were frozen before readout: residual RMSE ≤10% of the WRF DFI increment,
Q2 ≤25%, with an 8-epsilon REAL storage floor. These are prepass comparison bounds,
not a whole-forecast or machine-precision claim. The original o2-dfi filter-coefficient
machine-precision proof (64/64 bitwise configurations) is reused with byte-identical source.

| Primary field | DFL E2 residual | DDFI E3 residual |
|---|---:|---:|
| U / V | 0.43% / 0.23% | 0.68% / 0.52% |
| W | 1.22% | 1.39% |
| PH / MU | 1.46% / 1.32% | 1.09% / 2.29% |
| THM / QVAPOR / QCLOUD | 0.41% / 0.27% / 0.95% | 0.64% / 0.34% / 1.14% |
| P / AL | 2.43% / 0.35% | 5.28% / 0.59% |
| TSK / Q2 | 8.63% / 17.27% | 6.60% / 16.98% |

Both full windows pass 26/26 primary checks, all 320 numeric fields are finite,
none are missing; 213/214 numeric fields are exact. PB/PHB/MUB are exact in both.
Tiny rain/number increments have residuals up to 4.09% (DFL), 8.31% (DDFI).
Complete reports: `DF12-E2/compare.json`, `DF12-E3/compare.json` in the lane artifact root.

Known nonprimary differences are retained and disclosed as in o2-dfi: `T` is the dry
theta corresponding to filtered THM, whereas WRF's initialized-file `T` retains a
forward-stage diagnostic (RMSE 0.287/0.403 K); QKE is cold-started (WRF reinitializes
MYNN at the first forecast step); SNOW RMSE 0.000604/0.000763 kg m−2. Forward-stage
AC*/runoff sums and Noah-MP TAUSS are not transferred through the wrfinput adapter.
These receipts do not claim all variables are byte-identical or a qualified coupled forecast.

DFL clock details follow `share/set_timekeeping.F`: forecast launch at the integer
midpoint, original requested end retained, ceil steps until that end is reached/passed.
Positive run_days/hours/minutes/seconds make WRF retain its original simulation start
when resolving the forecast end, so filtered W is preserved; an end-date-only run
keeps midpoint simulation start and resets W. Independent pristine profile witness:
`DF12-WRF-CLOCK-WITNESS.json` (end-only W=Wsurface×ZNW² within 5.88e−8 relative;
positive-run-length filtered W differs by 0.817 relative). The native loader preserves
filtered W also when surface W is zero. DDFI never filters/floors its backward endpoint.
DFL consumes diff_opt_dfi/km_opt_dfi directly (Registry defaults 0/1); DDFI/TDFI save
and restore original diffusion options in their backward stage.

`build_short_oracle.py` reproduces the three shortest phase witnesses using the lane
copy of pristine WRF: DFL 2 forward steps with Registry diffusion defaults; DFL 2
forward steps with an end date only; DDFI 2 backward +4 forward steps.
`run_short_suite.py SOURCE` runs them sequentially in one CPU JAX process with unchanged
flags. Final short receipts and consumer/mutation evidence are listed in `evidence.json`.

Default release jaxpr is exactly
`2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a`
(`DF12-IDENTITY.json`, final composed snapshot 588dcb089). Existing TDFI functions,
filter implementation and shared numerical helpers remain source-exact; final source
carry proof binds the later metadata/clock changes to the measured primary cases.

Final composed source `fabb53569` (main `407c6268b`, source tree `34fcdb077`) also
matches all three active programs and all 641 carry/namelist/clock array records.
This is regression/source binding; the original CPU-WRF comparisons remain the truth.
The five cases total 130/130 primary passes and 1600 finite numeric field inventories.
Final compatibility checks: 220 passed, including NetCDF process locking and the four
shared consumer files changed by intervening main merges; clean archive: 12 passed.
Earlier owned/assets checks: 82 passed; eight required shared consumers: 250 assertions
passed. That shared test body completed, then the child exited naturally after long
cleanup; its original watchdog controller returned 124 after resuming a pending timer.
No pytest exit-zero claim is made for that arm. Both numerical endpoint/launch mutants
fail genuine assertions with zero setup errors. The final actual native loader retains
filtered fp32 W bytes at zero surface W; deleting its marker fails the assertion
(maximum W change 0.98376 m s−1). Hashed receipts and limitations: `evidence.json`.

Performance/memory: opt-in prepass only; no new State/carry fields or forecast kernels.
DFL removes the backward stage; DDFI omits backward accumulation/reset. The retained
forward-filter helper has the same accumulator launches, temporary fields and host
step scheduling as TDFI. DFL re-anchors coupled boundary strips onto gcd(interval,
midpoint) knots while retaining every original interval change. This can expand boundary
storage; the measured Swiss 1 h request uses six records at 540 s. A 4096-record cap
refuses extreme expansions. GPU memory/transfer/compile/E41 costs remain unqualified.
Full cold CPU interpreted prepass walls were 608.6 s (DFL) and 711.7 s (DDFI, peak CPU
RSS 8.804 GiB) on shared core12/nice19; these are not GPU or contention-free timings.
The sequential three-case CPU short suite reached 11.86 GiB peak RSS, including
multiple retained CPU executables. This is a CPU process measurement, not GPU VRAM.

Critic F1 correction: DFL forward and forecast writers now restore `diff_opt` and
`km_opt` in `&dynamics` when forecast keys are omitted. The shared TDFI writer and
DDFI route are retained. Defaults 0/1 and explicit 1/4 pass eight real writer/native
binding cases; deleting the local writer gives four genuine assertions and four
passing controls, with no setup errors. Six measured-case stage namelists remain
byte-identical. The post-`u2-tslist` main rebase retains all three programs and all
641 seed-array records, so original CPU-WRF numerical receipts are reused.
The correction adds only host text editing. [F1.json](F1.json) records the fixed
source, tests, source reuse, costs and invalid/retried harness arms; `evidence.json`
remains the immutable prior qualification packet.
