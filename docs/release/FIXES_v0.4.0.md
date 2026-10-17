# v0.4.0 caller-audit and validation fixes

Batches38–65 integrate the caller-audit class-A repairs and F20 default repair
in the admitted scopes below. The [original audit](../../proofs/t_calleraudit/CALLER_AUDIT.md)
is a dated defect inventory. Integration/source/CPU proof is separate from the
[affected GPU composite gate](FINAL_COMPOSITES_v0.4.0.md). No failed historical
trajectory or intermediate candidate is relabelled. Some class-B limits remain
disclosed in [known issues](../../KNOWN_ISSUES.md).

| Audit / area | Resulting behavior | Integration receipts |
| --- | --- | --- |
| F1 | Public microphysics precipitation reaches land for every admitted MP. | [batch39](../../proofs/s2_integrate/batch39/QUALIFICATION.json) |
| F2 + F24 | Convective RAINCV reaches land; API-only slab/Noah/PX receive their WRF precipitation, radiation and humidity operands. | [batch44](../../proofs/s2_integrate/batch44/QUALIFICATION.json) |
| F3 + F12/F13/F19 | One selected surface solve feeds land/PBL/history; MYJ memory and ELFLX/CHKLOWQ, GFS REAL/DOUBLE forcing and UST/WSPD writeback are retained. | [batch53](../../proofs/s2_integrate/batch53/QUALIFICATION.json) |
| F4/F5; F15/F16 | Legacy PBL/cumulus receive WRF dry thermodynamics, hydrostatic geometry/pressure and density; tendencies convert back to moist theta. QNSE/TEMF/EEPS/KEPS/Grell staging follows. | [batch41](../../proofs/s2_integrate/batch41/QUALIFICATION.json), [batch49](../../proofs/s2_integrate/batch49/QUALIFICATION.json) |
| F6 | ACM2, GFS, MRF, MYJ and YSU cloud/ice mixing and tendencies; species and PBLFLG gates. | [batch38](../../proofs/s2_integrate/batch38/QUALIFICATION.json), [batch40](../../proofs/s2_integrate/batch40/QUALIFICATION.json), [batch41](../../proofs/s2_integrate/batch41/QUALIFICATION.json) |
| F7/F9/F10/F11; F14/F18 | Held WRF closure forcing, RK-coupled tendencies, WRF shallow default, condensate species flags and diagnosed KPBL/radiation cadence. | [batch56](../../proofs/s2_integrate/batch56/QUALIFICATION.json) |
| F8 | MYNN cloud/aerosol-number mixing plus the native passive-number lowering entry; no missing-signature fallback. | [batch51](../../proofs/s2_integrate/batch51/QUALIFICATION.json), [batch61](../../proofs/s2_integrate/batch61/QUALIFICATION.json) |
| F20 default | Bottom-face vertical velocity and consumer-local HFX/QFX limits; device-only branch-free C5′ removes the failed intermediate layout/host-predicate route. | [batch50](../../proofs/s2_integrate/batch50/QUALIFICATION.json), [batch60](../../proofs/s2_integrate/batch60/QUALIFICATION.json), [batch62](../../proofs/s2_integrate/batch62/QUALIFICATION.json) |
| F17/F21/F23 | GBM radiation lifetime/PBLH, sf0 early return, Noah-MP opt_sfc2 coefficient ownership and qualified QNSE admission/refusals. | [batch51](../../proofs/s2_integrate/batch51/QUALIFICATION.json), [batch53](../../proofs/s2_integrate/batch53/QUALIFICATION.json), [batch54](../../proofs/s2_integrate/batch54/QUALIFICATION.json) |
| Microphysics preparation | Ferrier/Goddard/NSSL use WRF dry microphysics density and layer thickness. | [batch43](../../proofs/s2_integrate/batch43/QUALIFICATION.json) |
| Class-D and history | Initialized/held urban/aerosol fields, Morrison/NSSL radar, Jensen every-call radar, NTU QHAIL and inactive-shallow DLF2. Active unsupported branches remain refused. | [batch40](../../proofs/s2_integrate/batch40/QUALIFICATION.json), [batch41](../../proofs/s2_integrate/batch41/QUALIFICATION.json), [batch48](../../proofs/s2_integrate/batch48/QUALIFICATION.json), [batch49](../../proofs/s2_integrate/batch49/QUALIFICATION.json) |
| CAM-MG compiler | Stock Triton3.6 native route replaces the bundled compiler miscompile; offline native STACK failure remains disclosed. | [batch45](../../proofs/s2_integrate/batch45/QUALIFICATION.json) |
| NTU compiler/performance | Typed column memory, corrected closure and measured byte-preserving iteration skips; final c88 full24h D6/inventory/NF0 now passes with E41FAIL; c11 now passes full24h after the compiler repair. | [batch42](../../proofs/s2_integrate/batch42/QUALIFICATION.json), [batch55](../../proofs/s2_integrate/batch55/QUALIFICATION.json), [batch57](../../proofs/s2_integrate/batch57/QUALIFICATION.json) |
| c06 Tiedtke ROOT1–3 | Reset convection eligibility and cloud-base buoyancy; preserve WRF row/tile ICUM return semantics; stop organized detrainment at the carried IHMIN. | [batch58](../../proofs/s2_integrate/batch58/QUALIFICATION.json) |
| c40 SSiB SWDOWN | Use current downward shortwave in the SSiB/YSU surface boundary. | [batch59](../../proofs/s2_integrate/batch59/QUALIFICATION.json) |
| Hardware, packaging and tooling | Ampere/sm_80 preflight, pinned lazy assets/package data and remote tooling; no numerical qualification inferred from packaging. | [batch46](../../proofs/s2_integrate/batch46/QUALIFICATION.json), [batch47](../../proofs/s2_integrate/batch47/QUALIFICATION.json), [batch52](../../proofs/s2_integrate/batch52/QUALIFICATION.json), [batch55](../../proofs/s2_integrate/batch55/QUALIFICATION.json) |

Batch65 additionally repairs the WDM6 pathological compiler fusion with a
BYTE-preserving split. Its actual sourceb823c8ca87 c11 full24h now passes
original-CPU D6/inventory/NF0. Raw E41STACK512B/kInput flags remain disclosed;
old60-minute compile failure is not relabelled. [Final c11 source/receipt](../../proofs/r_docs040/RELEASE_CHECKPOINT.json).

The c06 fix reproduces the serial reference's x-row tile decision; an original
MPI/OpenMP split of that row can itself select a different ICUM return. This
scope is retained, with pristine-column and row oracle evidence in the
[Tiedtke root-cause packet](../../proofs/t_cfix_cu/c06_tiedtke/README.md).

Default C5′ qualification is closed on the frozen source: Swiss25frames,
PROD50 and WN3-0227 75, bounded D6/NF0 and actual E41 PASS3/3. Host
copy/byte/sync/callback counts equal F20. Manager accepted matched nsys root
58.05 versus58.89ms (−1.4%); ABAB has zero valid samples and remains incomplete.
This is not whole-run S1 or a general performance/memory claim.
[Final default gate and identities](../../proofs/s2_integrate/batch62/gate_WN3_closed/GATE_CLOSURE.json).
