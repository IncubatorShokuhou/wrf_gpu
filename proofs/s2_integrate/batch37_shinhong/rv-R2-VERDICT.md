# MERGE — lane/t-shinhong 2903f695d

Final same-critic verdict for `2903f695d1f2ffcfbef9492655b5934deab6f87f`.
This closes the tests-only conditions in the original rv-t-shinhong review of
`44b7b226b55f991d8a5aa26da207ff049dfb5be9`. The original decision is preserved.

The revised compiled gate rejects nonfinite candidate/reference/bounds, compares
when either value is normal, bounds differences for pairs where both values are
subnormal/zero, and checks mode0 driving rates against proven identical-input
mode1-cold bounds while holding TKE/EL. The reciprocal zero-reference/normal-
candidate and NaN sentinels are present. The standalone check_envelope helper
uses the same finite/OR/subnormal rule.

Independent RV08 on an immutable detached target: **4 PASS** (compiled gate and
three sentinels), plus **one intentional genuine mutant AssertionError**,
`outside envelope row 0 exch_h k=1`, **zero setup errors**, 4.603 s. The mutant
retained TKE and erased all compiled u/v/theta/qv/qc/qi/EL/EXCH_H outputs, exactly
the previously surviving class. No source files were changed by this probe.

The additional disclosed row3 v-level9 entry is the previously unchecked mode0
sibling of rows4/5: identical profiles, surface inputs and original WRF driving
rates are asserted before reusing bounds. The original per-cell solved-wind
quantum cap is unchanged. This is a coverage expansion, not a new tolerance.

Whole `src`, `data`, `scripts`, `pyproject.toml` and `docs` trees are BYTE-identical
to reviewed44b7b226b (RESULT.json). Therefore prior independent original-WRF
parity, physical mutant, default IDs Swiss f672a573/PROD6fc715a2, caller/RK seam,
and measured CPU cost evidence carry forward. These tests/proofs-only edits add
no product launches, arrays, transfers, synchronization or compile/readiness cost.
No old numerical/default/component timing runs were repeated.

**c07 full24h GPU D6 PASS is accepted on the identical product44b7b226b**:
25 frames, zero bounded misses, zero nonfinite, asserted GPU, rc0. Independently
verified all four retained receipt hashes from C07_SHINHONG_FULL24.json and
inspected the source/work qualification. E41 remains **FAIL, disclosed for this
opt-in scheme**, and F2 remains **OPEN**. These are unchanged limits, not a
blanket release or roofline PASS. The cold shared-work wall993.766s and sampled
VRAM5380MiB are qualification disclosures, not warm S1/R1 measurements.

No outstanding fix remains for this lane review. Manager may integrate2903f695d.
No new GPU run is needed for this tests-only successor. The YSU sibling is a
separate lane. RV08/reuse artifacts are under this R2 directory; snapshot and
pytest scratch were removed, persistent JAX cache disabled, no GPU or agents.
