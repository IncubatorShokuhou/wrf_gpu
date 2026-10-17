# Continuous GPU qualification — October15

Updated 2026-10-15T02:22:36.137841+00:00. WN3 full24h passes D6 for75frames/three domains and actual E41. B01/B02/B03 have ended; B04 is active. Original references:68/69 complete, CAM-MG11/c89 running. All scientific results remain bound to their measured source; F2 and final-RC identity are separate.

Manager02:20: hold c07/c12/c13/c14/c17/c18/c64/c88 until the qh-mass fix merges. Their existing results are provisional. Original CPU-WRF references remain valid. Re-run affected full24h D6/E41 once on final changed source. NEW v-run producer reuse is conditional on exact input/work/program proof and own-CPU scoring.

| Batch | Cases | Status |
|---|---|---|
| B01 | c03, c04, c26, c39 | DONE c03/c04/c39 D6PASS25; c26cap124/lead0 compiler readinessBLOCKED |
| B02 | c05, c06, c09, c10 | DONE c05/c06 RAINNCFAIL10/11; c09D6PASS; c10cap24/25frames |
| B03 | c07, c11, c62, c40 | DONE all4rc0/full25; c11/c62/c40D6PASS, c07pending/provisionalqh |
| B04 | c16, c29, c30, c31 | ACTUALACTIVE c16/c29/c30/c31 nonhail, acquired~02:20Z; cap<=02:50Z |
| B05 | c45, c89, c37, c38 | Afternewrefscomplete; Goddardrad5/NTU56/CAMMG11/diffusion |
| B06 | c43, c46, c33, c36 | NOT_RUN; bind final composed main and preflight first |
| B07 | c71, c72, c68, c69 | NOT_RUN; bind final composed main and preflight first |
| B08 | c70, c86, c77, c78 | NOT_RUN; bind final composed main and preflight first |
| B09 | c79, c55, c49, c50 | NOT_RUN; bind final composed main and preflight first |
| B10 | c51, c73, c83, c47 | NOT_RUN; bind final composed main and preflight first |
| B11 | c02, c56, c57, c27 | NOT_RUN; bind final composed main and preflight first |

Every ordinary cohort is3–4 processes, total cap30min with1770s children. Four-way admission requires actualfreeVRAM>=27696MiB and a fresh free-g check. GPU15 source746860 has src/data/scripts identical to main2af43322b/ad8b0c729; take a fresh immutable snapshot for any affected new model merge.

Standalone30min batches: c42, c28, c65, c60, c84, c85. Nested/c42, CAM/c28, hail/c65 and Atlantic/c60/c84/c85 run alone. No unchanged retry for bootstrap-only compiler blockers. Preserve any incomplete result; never silently extend a cap.

B01: c03/c04/c39 fullD6PASS; c26healthyhost fullcap/lead0/no steppingELF, compiler-readiness blocker. B02: c05/c06 fail onlyRAINNC10/11frames; c09D6PASS; c10partial24/25frames with0 boundedmisses/rawNF0. B03: c07/c11/c62/c40 allD6PASS25each; c62E41PASS, otherthreeE41flags disclosed; c07provisional qh.

c64 correctedowner output reused: full25D6/E41PASS but nowprovisional qh. c08 correctedowner output reused:25/0bounded/NF0, incomplete onlyEXCH_TKE/E41FAIL; requires writer-only follow-up when model unchanged. Both reuse checks prove actualOWNinputs and CPUreceipt/manifestBYTE, with10nativePINs freshly SHAverified. Oldfailedattempts are archived.

c13 newNSSL18-cu5 producer output reused with ownCPU D6PASS25/full372fields; E41/F2open and nowprovisionalqh. MYNN original-repeat check c00/c17 passed18075fieldframes+50canonicalfileSHA; separate P3zero9a67 refs c21–24 preserved. No raw original reference is replaced.

The machine-readable execution_groups partitions all69 active cases exactly once;0unassigned/duplicates. Remaining compiler holds c23/c26/c87, P3diagnostic program/schema work c21/c22/c24, numerical failures c05/c06, and affected qh group stay explicit. API-only LSM1/2/7, ownerMP15/km3, crop/tile/map branches remain outside the fullGPU gate.

GPU allocation ends19:15Z, plannedfinalrelease19:10Z. Continuous ordinary cohorts until18:45Z; no later start unless its cap ends by19:10Z. Official lease/lock and fresh timeout for every acquisition. Four globalRAMslots; aggregate15GiB cap lifted only until19:15Z. GPU cohort aggregate40GiB/noSwap; CPU singletons6–12GiB/noSwap via officialRAMslot. CPUwork0/1/16nice19/JAX_PLATFORMS=cpu; QUIET guard beforeheavycommands. New caches/snapshots underDATA scratch with>=120GiB free.

Pipeline CPU E41/full-original-field D6 while the nextGPU cohort runs. Keep actualJIT+AOT nativePINs and census/extrema receipts; clean exactcache/tmp paths immediately after finalconsumers. WN3CENSUS0 is not F2pre-maskclosure; reported negativephasevisits are not unique cells or an automatic scientific justification.

[I]Remaining ready cap budget about480GPUmin afterB03, plus60min for two qh cohorts and any compiler fixes/foreignqueues; inferred range400–540 excludes those additions. These are scheduling caps, not warmS1 performance claims. Recompute as cohorts finish.
