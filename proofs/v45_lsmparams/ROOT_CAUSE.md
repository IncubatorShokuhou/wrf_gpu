# W2 default-value diagnosis

These are CPU regression/transport measurements, not CPU-WRF or GPU fidelity.
The nine controls remain publicly refused. The earlier w44 full-root and GPU
failures remain failures; no new result supersedes their scientific gates yet.

The LS71 experiment already refuted the proposed canopy Newton exit mechanism:
TV and TAH were byte-identical, and capturing the same control values as
constants removed the flux differences. That intervention also introduced a
separate 1–2 ULP frozen-soil class, so it did not establish whole-root identity.

Current-source BISECT01 isolates the warm-column joint difference to SMCMAX,
BEXP and PSISAT. Each has four unequal outputs when selected alone; the joint
nine have six. HVT/HVB/Z0SNO/SNOW_EMIS/DKSAT/RSMIN selected alone are byte-exact
on those warm columns. This does not establish their whole-root/GPU class.

FOLD01 shows all five inspected parameter-only expressions have the same REAL
bits as constants, dynamic operands and NumPy host calculations. FOLD03 then
finds the first PSISAT state-expression difference at `potential * GRAV`, while
`potential` itself has identical bits. Its downstream relative-humidity change
is 5.96e-8 on the retained real inputs. Both use the actual ENERGY constants.
The incorrect downstream constants in the earlier FOLD02 probe were discarded.

The retained intervention binds `-PSISAT * GRAV` once on the host and carries
it through the existing clock argument. ENERGY combines that coefficient with
the state-dependent power. COEF01 restores all 48 isolated PSISAT output bytes,
and reduces the joint six unequal leaves to four. All 48 OFF hashes match the
unmodified BISECT01 baseline. This establishes a causal association mechanism
for this PSISAT column class, not for every remaining coupled difference.

Direct DRSURF prebinding did not improve the SMCMAX/BEXP class and was removed.
COEF02 tests its reciprocal because a baked denominator permits reciprocal
multiplication at the state-expression boundary. It halves the largest warm
Q2 difference (9.31e-10 to 4.66e-10); four unequal leaves remain. This is a
provisional class enabler, not a byte-identity fix. Further claims require the
full-root, pristine-WRF and native-resource gates.

The two new coefficients are internal derived clock operands, not accepted
public control names or checkpoint state. Original controls remain the source
of their values and authentication. Values are absent from static namelist
metadata/compile keys. The pack is stripped before the column-kernel argument
bundle, avoiding unused column operands. OFF and legacy maps return unchanged.
Host binding under JIT is refused explicitly; callers must pass an already
bound clock rather than capture control values in a content-blind static aux.

Before the full root, its bound remains the already registered 4e-6 per-field
scale, including failures at zero reference scale. Real-column changed-value
physics still uses pristine WRF at its existing tolerances and closure floors.
No tolerance is selected from the new full-root output. Actual GPU/compiled
resource qualification follows [GPU_PLAN.md](GPU_PLAN.md); failure keeps the
family refused and triggers measured localization.

The final full-PROD CPU arm (`4235f06373`, explicit Jarvis2/all nine) failed
the frozen bound: 112/355 default carry leaves differed. Parent pressure
perturbation reached 3.78e-4 of field scale; child W reached 8.17e-3. This is
not established as the column-local class. All three arms were finite,
advanced parent/child 1/3 own-steps, and CHANGED altered 130 leaves with both
domain keys and concrete two-entry CPU chunk cache reused. The run took
1774.3 s with peak RSS 15,125,928 KiB under the approved 16 GiB cap.

Manager 2026-10-28 01:11Z authorized a causal A/B extending exactly the three
condition lines of the existing v05 V-PGF output fence to W2 control names.
OFF and legacy Track P must stay exact. This compiler-topology intervention
is a hypothesis until the measured A/B; the existing CFL fence's evidence is
not proof for W2. Compiler/component prescreens precede one newly approved
16 GiB full-root arm. The new pass condition is default **BYTE identity**;
the earlier 4e-6 table remains diagnostic, not an alternative pass. If this
intervention fails, bisect the coupled program rather than add guessed fences.

The authorized fence A/B (`cf4ce90dcd`) is refuted. The 112 unequal-leaf
error records and every named-field error table are exactly the same as the
unfenced run. OFF/ON/CHANGED each have 355 finite leaves, 1/3 own-steps and a
two-entry chunk cache; CHANGED still changes 130 leaves with unchanged keys.
The run took 2231.5 s, peak RSS 14,850,576 KiB, and failed strict BYTE identity.
The three experimental condition lines were discarded. Native/compiler limits
were satisfied, but that did not predict numerical identity. Further work is
coupled-program bisection, beginning with actual namelist round-trip, clock
operands and prior values rather than another guessed fusion fence.

Bisection after the refuted fence locates the discrepancy before PBL/RK: the
actual surface handoff has 15 unequal leaves. Splitting controls gives 15 for
SMCMAX+BEXP and 7 for the other seven. A cut immediately before the actual
canopy solve then finds only DF_TOP (182 cells), RHSUR (14 cells), and RSURF
(1 cell) in the first group; the other group differs only in RHSUR (20 cells).
Parameter and other canopy inputs match. These are separately compiled prefixes,
so they localize the error without proving full-program transparency.

The saved 8,400-cell operands reproduce all inspected production expressions
byte for byte when state arguments remain dynamic. This exposes two soil folds:
SMC/SMCMAX becomes a reciprocal multiplication for baked parameters, and the
parameter-only dry conductivity has different FMA behavior when traced. The
previous PSISAT*GRAV prebinding is also refuted for this heterogeneous grid:
OFF retains two multiplications; combining them creates the 20 RHSUR errors.
The earlier 48/48 four-column result remains a measurement of that narrower
compiler context and no longer supports keeping that intervention.

Causal expression probes bind inverse SMCMAX and dry conductivity in REAL,
retain reciprocal DRSURF, and restore the original PSISAT expression. Both host
coefficients match the original folds; all five thermoprop outputs and both
humidity/resistance outputs become byte-identical on all 8,400 cells. The
candidate carries those three derived operands; the pack remains outside the
canopy column arguments. This is not yet a full-root result. The obsolete
PSISAT consumer test is replaced by recorded heterogeneous PROD soil operands
and live changed-value/cache evidence, alongside the existing clock callers.
Fresh original-WRF checks now pass 666 field comparisons and 74 water/energy
cases at unchanged limits and oracle/floor SHA. All five cohorts reuse one ON
key with two traces; warm Jarvis RSMIN changes 21 leaves, while cold RSMIN is
inactive. Warm/cold column defaults are 43/48 byte-equal leaves, so these
column checks are not a general BYTE-identity claim.

The composed candidate's actual parent surface prefix is 199/199 BYTE and
complete physics prefix 208/208 BYTE. The OFF program passes 20 measured
full-constant/native/census comparisons across parent, force-down and child.
Twelve updated component tests pass; isolated clock/dry/inverse source
deletions cause seven genuine assertions, with zero setup errors or skips.
The authorized reduced archive collects 21,191 tests with zero errors after
57 exact, collection-proved support files are added and blob-verified.

Native parent compilation still fails: a U-staggered boundary/RK/flux-advection
kInput fusion has 574 optimized HLO instructions (limit 500), while maximum
STACK is 240 B (limit 256 B). Manager assigned this shared seam to v45-seam
and explicitly held child/root work. A new composed full-root BYTE check and
actual GPU qualification remain open. No threshold is relaxed; all nine
controls remain publicly refused.

The shared-seam review subsequently identifies that native fusion as introduced
by the W2 candidate: matched OFF 406, ON 574, scoped completed U-output fence
407 HLO instructions. Main `cb48138349` merges the reviewed fence. Rebasing the
kept lane changes produces `c1782be2dc`, whose full `src` tree equals reviewed
merge tree `7923244a`; the Noah source and checked dependencies remain byte-exact
to `21b4178016`. The original-WRF column evidence therefore carries. Twelve
rebased component tests pass, and a fresh parent E41 prescreen confirms
407/240 B/163 native calls/zero census errors.

The newly approved strict-BYTE full nested-root test starts at 04:34:25Z with
72.412 GiB available, 16 GiB MemoryMax, no swap, CPU12/28 nice19 and a 2700 s
model timeout. It passes all 355 default carry leaves byte for byte, with all
three arms finite and parent/child own-steps 1/3. CHANGED affects 130 leaves;
both domain keys and the concrete two-entry chunk cache are reused. Wall is
1283.774 s, peak RSS 15,200,332 KiB; RAMRUN ends rc0 at 04:55:49Z. This measures
the composed corrected coefficient and output-fence program; it does not
independently attribute every old numerical difference to the fence alone.
Public admission remains refused until actual GPU qualification passes.
