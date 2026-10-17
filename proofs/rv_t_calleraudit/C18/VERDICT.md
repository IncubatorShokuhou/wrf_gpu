# MERGE — native MYNN passive-number lowering fix

Reviewed `db78f9bd8c..2fe76a358c857f24815589af05e140a416222066`, product commits `9153f96074` and `6aefb36fd6`.
Scope: native MYNN passive-number dispatch plus full traceback reporting at the two existing nested-AOT lowering failure sites. No product edits or GPU execution by the critic. The lowering blocker is closed in the reviewed CPU/native-frontend scope; GPU compilation and scientific qualification remain pending.

## Independent evidence [M CPU/source]

- **28 PASS**, 0 failures/errors, 155.28 s: native-entry pristine DMP_mf oracle, both traceback checks, and the 25 F8 number-mixing/caller consumers (`RP01-gates.xml`). Passive fluxes meet the existing per-column 5% QNI gate on the frozen 16 real c19 columns; each species equals a separate native QNI-lane call bitwise. All plain-call outputs remain bitwise unchanged on this compiled CPU fixture. Mass-flux activity in the reference is 3/16 columns, disclosed rather than represented as all-active.
- **Four exact-input native CUDA frontend PASSes**, on the immutable final candidate, using the sealed v-prep composite inputs and a captured sm_120 target. Physical devices remain CPU-only; the actual traced trees have **zero interpreted Pallas calls**; the backend is a CompileOnly client; no executable compile or forecast is attempted. Unlike portable CPU interpretation, these arms lower the native Pallas and product tridiagonal paths to CUDA MLIR.

| case | relevant path | native lower | traced Pallas calls | frontend peak RSS KiB |
|---|---|---:|---:|---:|
| c18 / MP26 | Nc; base QNI inactive | PASS | 164 | 2,729,392 |
| c19 / MP28 | Nc + nwfa + nifa | PASS | 181 | 3,233,772 |
| c22 / MP51 | P3 + Nc | PASS | 163 | 2,659,064 |
| c88 / MP56 | NTU + Nc | PASS | 201 | 2,671,212 |

The frontend durations were 45.69/46.73/46.94/54.70 s, respectively; these are diagnostic setup timings, not forecast-speed claims. Receipts (`RP02-*/receipt.json`) bind source commit, input hashes, target hash, native switches, program kwargs, CPU-only devices and cgroup. Private caches were removed after each arm.

- Independent final release identities match the accepted **db78** baselines exactly: Swiss `7379c7d77e94517f07e7bf37763cc1409d1287a13e50684ed9d624caafaec55e`; PROD d01 `6487f5b039a074f02f63d140f1e9212b7ede832d04bf5450e44ad2b3f2c34e73` (`RP03-swiss.json`, `RP03-prod.json`). These supersede the older f672/6fc baseline for this review.
- **Two genuine assertion kills**, 0 collection/setup errors: keeping original QNI values in every species lane fails the pristine/byte gate (2.92 s); suppressing traceback printing fails the unchanged error-text/traceback check (.05 s). Mutations are process-local and production bytes stay clean (`RP04-*-mutant.json/xml/log`). The initial mutation harness imported JAX before the new strict CPU guard and was refused before collection. That invalid attempt is preserved in `RP04-setup-refusal.log`, counted as zero kills; the corrected harness imports cpu_guard first.
- The two reviewed product files are **byte-identical** to producer-measured `710ea6e12c`. Other source changes to the final composed candidate are confined to Tiedtke/scan_adapters; the eight affected composites use CU0 (`RP00-source-proof.json`). This supports reuse of the producer's eight portable whole-root lowerings and relevant paid 914-consumer evidence. Four representative native CUDA lowerings were independently acquired above; no claim is made that the critic reran all eight native CUDA cases.

## Correctness and error handling

The prior wrapper converted every keyword through jnp.asarray, including the passive dictionary. The fix binds that dictionary separately, keeps the original argument conversions and plain evaluation, then exposes each species' QNI-lane flux under the existing F8 key. WRF `module_bl_mynnedmf.F:6099–6103`, `:6172–6176` and `:6411–6420` use matching initial weighting, entrainment and flux accumulation for QNI/QNC/QNWFA/QNIFA; passive numbers do not feed back into plume thermodynamics. The existing limiter is retained. No new clipping, tolerance, species admission, State leaf, carry/restart schema, static table or feature flag is introduced.

The traceback helper prints the actual exception stack to stderr and returns the original one-line `type: message` status. Both lowering sites call it. Success paths and the existing fail-open fallback policy are unchanged.

The producer's mutation packet includes expected TypeError/KeyError negative controls as well as assertion failures. Those runtime controls are useful diagnostics, but the two independent kills counted here are genuine assertions under E202.

## Performance, memory and remaining gates

No-passive MP8 executes the original native evaluation once; both complete default identities remain exact. For Nc-carrying MPs the fix adds **one extra full native plume-sums evaluation**, or **three** for MP28/38. The wrapper reconstructs the setup for each call; XLA common-subexpression elimination of that setup is an inference, not measured evidence. The opaque plume calls themselves cannot be assumed to disappear. Visible additional species fluxes are fp32 `(columns, levels+1)` arrays, plus each opaque call's existing outputs/workspace; actual GPU peak allocation, dispatch overhead, compile/readiness and elapsed time are **[U]**. The CPU frontend RSS table is not GPU VRAM evidence.

This is a correctness/lowering enabler, not a maximum-speed verdict. Use the already planned affected GPU reruns to record wall/peak and investigate a material overhead; a combined passive-lane kernel is a possible later optimization at unchanged gates. No extra standalone GPU experiment is requested by this review.

Manager may merge the reviewed fix and return **c18/c19/c20/c22/c23/c24/c65/c88** to v-prep's queue. Keep its native CPU preflight on the final integrated source before GPU allocation, and retain actual executable/E41, original CPU-WRF D6, finite/census and resource gates. Frontend PASS does not establish GPU numerical fidelity, stack limits or performance.

Evidence: `<USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/C18/`; durable copy `proofs/rv_t_calleraudit/C18/`. All heavy probes ran under official 6G/no-swap RAM slots, JAX CPU, cores 8/9/12 nice19, explicit QUIET guards and timeouts. No persistent cache or executable artifacts were retained. Snapshot and pytest basetemp removed after the final dependent arm. Tracked log/XML copies trim trailing whitespace; original raw receipts remain unchanged externally.
