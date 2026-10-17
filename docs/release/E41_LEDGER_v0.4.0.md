# v0.4.0 E41 opt-in ledger inventory (release candidate)

## Source-bound opt-in E41 disclosure (00:26Z final)

This table is generated from v-prep's final `FINAL_E41_DISCLOSURES.json`
at `2026-10-17T00:30:45.643194+00:00`: **32 FINAL resource FAILs**, with
33 PASS and c89 UNKNOWN counted separately. The [portable exact-content
projection](../../proofs/r_docs040/FINAL_E41_DISCLOSURES.json) retains every actual STACK site, kInput fusion,
non-leaf/recompute callee, ELF/JIT/AOT origin and receipt SHA. All rows below
are measured registered FINAL sources (F); the three historical-only failures
in the earlier provisional inventory do not enter the final counts. Raw
E41 FAIL stays FAIL despite scientific D6 PASS. Missing scan data is UNKNOWN,
never zero. STACK is bytes; kInput is optimized HLO; callees are SASS. Callee
columns give site count and independent maxima; exact sites remain in JSON.
Options are MP/CU/PBL/SFCLAY/LSM/LW/SW.

| Case / source scope | Options | Raw D6 / scientific gate | Max STACK B | kInput >500 (HLO) | Non-leaf/recompute sites (max SASS / max nested calls) |
| --- | --- | --- | --- | --- | --- |
| c02 / `04cb09f816` F | 1/1/1/1/4/1/1 | PASS / PASS [E41 disclosed] | 16 | 549, 570, 586, 632, 659, 661 | — |
| c03 / `04cb09f816` F | 2/2/2/2/4/4/2 | PASS / PASS [E41 disclosed] | 2456 | 540, 598 | — |
| c05 / `04cb09f816` F | 4/5/7/1/4/4/4 | PASS / PASS [E41 disclosed] | 856 | 785, 954 | 1 site; 14418 / 72 |
| c06 / `db78f9bd8c` F | 6/6/8/1/4/4/4 | PASS / PASS [E41 disclosed] | 40 | 536, 598, 722 | — |
| c07 / `04cb09f816` F | 9/16/11/1/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 538, 615, 622, 643, 679 | — |
| c08 / `04cb09f816` F | 10/14/12/1/4/4/4 | PASS / PASS [E41 disclosed] | 408 | 714, 904, 973, 1320, 1600 | — |
| c09 / `04cb09f816` F | 13/94/99/1/4/4/4 | PASS / PASS [E41 disclosed] | 424 | 626, 1383 | — |
| c10 / `04cb09f816` F | 14/95/16/1/4/4/4 | PASS / PASS [E41 disclosed] | 536 | 1463 | — |
| c11 / `b823c8ca87` F | 16/96/17/1/4/4/4 | PASS / PASS [E41 disclosed] | 512 | 1353 | — |
| c12 / `04cb09f816` F | 17/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 1336 | 562, 660, 724, 835, 1041, 1064, 1128, 1172 | — |
| c13 / `04cb09f816` F | 18/5/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 681, 973, 1405, 2096 | — |
| c14 / `04cb09f816` F | 19/99/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 900, 1034, 1096 | — |
| c15 / `04cb09f816` F | 21/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 208 | 660, 1235, 1287 | — |
| c16 / `04cb09f816` F | 22/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 640, 641, 649, 856, 944, 1060 | — |
| c18 / `5c15881ede` F | 26/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 904 | 618, 1211 | — |
| c19 / `5c15881ede` F | 28/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 741, 1439 | — |
| c20 / `5c15881ede` F | 40/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 820, 832, 2227 | — |
| c26 / `2ed8c04c26` F | 8/0/4/4/4/4/4 | PASS / PASS [E41 disclosed] | 7232 | — | — |
| c27 / `2ed8c04c26` F | 8/0/10/10/4/4/4 | PASS / PASS [E41 disclosed] | 32 | 529, 531, 543 | — |
| c40 / `36fcfaa64d` F | 8/0/1/1/8/4/1 | PASS / PASS [E41 disclosed] | 384 | 1369 | — |
| c42 / `5c15881ede` F | 8/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 572 | — |
| c46 / `5c15881ede` F | 8/1/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 8 | 502, 569 | — |
| c55 / `5c15881ede` F | 8/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 80 | 503, 645, 820, 956 | — |
| c56 / `04cb09f816` F | 8/0/1/1/4/4/4 | PASS / PASS [E41 disclosed] | 968 | 516, 544, 566, 586, 687, 759, 937, 1015, 1179, 2477 | — |
| c57 / `2ed8c04c26` F | 8/0/8/1/4/4/4 | PASS / PASS [E41 disclosed] | 968 | 509, 578, 643, 708, 712, 759, 797, 830, 992, 1015, 1253, 1707, 2566 | 15 sites; 9612 / 2 |
| c61 / `04cb09f816` F | 97/0/0/91/4/4/4 | PASS / PASS [E41 disclosed] | 328 | — | — |
| c65 / `5c15881ede` F | 38/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 496 | 638, 644, 1089, 1134, 1314 | — |
| c83 / `5c15881ede` F | 8/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 16 | 887 | — |
| c85 / `04cb09f816` F | 8/0/1/1/4/4/4 | PASS / PASS [E41 disclosed] | 384 | — | — |
| c87 / `04cb09f816` F | 55/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 5520 | — | — |
| c88 / `5c15881ede` F | 56/0/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 3624 | 894 | — |
| c90 / `5c15881ede` F | 8/1/5/5/4/4/4 | PASS / PASS [E41 disclosed] | 8 | 502, 569 | — |

For c03, STACK2456B is pre-existing since the f4fd predecessor; final c03 also
has kInput540/598 HLO. For c05, STACK856B, kInput954/785 HLO and a14,418-SASS
non-leaf callee with72 nested calls are a recomputation signature. This does
not measure its runtime share; c05 runtime attribution is a v0.4.1 follow-up.
Earlier c05 raw D6/resource failures remain preserved.

The producer separately lists c89 as resource UNKNOWN despite D6/inventory/NF0 PASS under
its attempt-selection scope. Missing flags are UNKNOWN, never zero; historical
reuse/successor receipts in the older ledger retain their exact-source validity.
Neither list approves an unmeasured final executable. CAM-MG's separate offline
native STACK1240B scan must not be replaced by an XLA glue-only PASS.

The JSON retains the historical five-case F20 packet at sourcee7e37390cc:
bounded D6 PASS but raw E41 FAIL. That old failure remains failed. The frozen
C5′ successor now closes current default D6/E413/3 and transfer gates; its
identity/default qualification is in the release checkpoint. The old
`default_F20` field is not a current default blocker. F2 remains separate and
these profiling/component results make no S1 claim.

## Historical ledger (2026-10-15 15:27Z)

Provisional producer ledger: `2026-10-15T15:27:31.954738+00:00`. This historical table lists every E41 FAIL at that checkpoint, including partial/failed forecasts. Current measured sources are in the table above; it does not waive default gates.

The 2026-10-16 04:36Z decision keeps E41 release-blocking for default WN3/PROD/Swiss F20 programs. Opt-in composites require strict original-CPU D6, complete inventory and NF0; passing that gate with resource failures is labelled **E41 FAIL (perf/robustness), D6 PASS**. The raw E41 verdict stays FAIL. Incomplete D6 remains incomplete, and F2 justification remains separate. This applies to resolved composites, not every combination using the same scheme code.

Options are MP/CU/PBL/SFCLAY/LSM/LW/SW. Raw D6, full inventory and E41 are separate. PASS-with-disclosure does not turn raw INCOMPLETE into fullD6PASS. F2 remainsOPEN. Runtime spill cost is UNKNOWN without matched profiles.

All source/producer E41/PIN locators and SHA bindings are in the [portable snapshot](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json). The [older attribution supplement](E41_OPT_IN_v0.4.md) preserves exact historical kernel details; its counts do not represent the entire current ledger.

| Case | Options | Paid source | Raw D6 / frames | E41 |
| --- | --- | --- | --- | --- |
| c02 | 1/1/1/1/4/1/1 | `5c0e7c11ed` | PASS / 25 | FAIL [M] |
| c03 | 2/2/2/2/4/4/2 | `90af92cfc1` | PASS / 25 | FAIL [M] |
| c04 | 3/3/3/3/4/4/1 | `061d53020e` | PASS / 25 | FAIL [M] |
| c05 | 4/5/7/1/4/4/4 | `de374a3c00` | FAIL / 25 | FAIL [M] |
| c06 | 6/6/8/1/4/4/4 | `de374a3c00` | PASS / 25 | FAIL [M] |
| c07 | 9/16/11/1/4/4/4 | `44b7b226b5` | PASS / 25 | FAIL [M] |
| c08 | 10/14/12/1/4/4/4 | `49900e8841` | PASS / 25 | FAIL [M] |
| c09 | 13/94/99/1/4/4/4 | `90af92cfc1` | PASS / 25 | FAIL [M] |
| c10 | 14/95/16/1/4/4/4 | `de374a3c00` | PASS / 25 | FAIL [M] |
| c11 | 16/96/17/1/4/4/4 | `de374a3c00` | PASS / 25 | FAIL [M] |
| c12 | 17/0/5/5/4/4/4 | `20ca350508` | PASS / 25 | FAIL [M] |
| c13 | 18/5/5/5/4/4/4 | `627a033284` | PASS / 25 | FAIL [M] |
| c14 | 19/99/5/5/4/4/4 | `20ca350508` | INCOMPLETE / 25 | FAIL [M] |
| c15 | 21/0/5/5/4/4/4 | `afcdb5072c` | INCOMPLETE / 25 | FAIL [M] |
| c16 | 22/0/5/5/4/4/4 | `7468605609` | PASS / 25 | FAIL [M] |
| c17 | 24/93/5/5/4/4/4 | `20ca350508` | PASS / 25 | FAIL [M] |
| c19 | 28/0/5/5/4/4/4 | `49900e8841` | INCOMPLETE / 25 | FAIL [M] |
| c20 | 40/0/5/5/4/4/4 | `49900e8841` | INCOMPLETE / 25 | FAIL [M] |
| c26 | 8/0/4/4/4/4/4 | `4dba72ce4c` | PASS / 25 | FAIL [M] |
| c27 | 8/0/10/10/4/4/4 | `8a491bf8de` | PASS / 25 | FAIL [M] |
| c29 | 8/7/9/5/4/4/4 | `7468605609` | PASS / 25 | FAIL [M] |
| c37 | 8/0/0/5/4/4/4 | `dd619e4809` | PASS / 25 | FAIL [M] |
| c38 | 8/0/0/5/4/4/4 | `dd619e4809` | PASS / 25 | FAIL [M] |
| c40 | 8/0/1/1/8/4/1 | `7468605609` | PASS / 25 | FAIL [M] |
| c46 | 8/1/5/5/4/4/4 | `dd619e4809` | PASS / 25 | FAIL [M] |
| c55 | 8/0/5/5/4/4/4 | `20ca350508` | PASS / 25 | FAIL [M] |
| c56 | 8/0/1/1/4/4/4 | `49900e8841` | PASS / 25 | FAIL [M] |
| c57 | 8/0/8/1/4/4/4 | `49900e8841` | INCOMPLETE / 25 | FAIL [M] |
| c60 | 8/0/5/5/4/4/4 | `f5d90ca0d3` | PASS / 25 | FAIL [M] |
| c61 | 97/0/0/91/4/4/4 | `d2782882bd` | PASS / 25 | FAIL [M] |
| c65 | 38/0/5/5/4/4/4 | `584d396dd4` | INCOMPLETE / 25 | FAIL [M] |
| c83 | 8/0/5/5/4/4/4 | `49900e8841` | PASS / 25 | FAIL [M] |
| c85 | 8/0/1/1/4/4/4 | `f5d90ca0d3` | PASS / 25 | FAIL [M] |
| c88 | 56/0/5/5/4/4/4 | `584d396dd4` | INCOMPLETE / 1 | FAIL [M] |
| c90 | 8/1/5/5/4/4/4 | `dd619e4809` | PASS / 25 | FAIL [M] |

STACK is bytes; nonleaf size is SASS; kInput size is optimized HLO. Missing kernel sizes are UNKNOWN, never zero. This inventory does not replace actual JIT+AOT scan coverage or transfer an old source/kernel resource measurement to a new executable.
