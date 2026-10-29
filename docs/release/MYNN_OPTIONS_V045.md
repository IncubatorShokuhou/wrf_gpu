# MYNN option qualification for v0.4.5

CPU qualification ready for independent review; final composition and native release gates pending.

| WRF selection | Implementation / CPU evidence | Native forecast status |
| --- | --- | --- |
| icloud_bl=0 | Existing radiation-only merge switch; literal pristine-WRF oracle, active liquid/ice, binding and deletion checks. [Audit](../../proofs/v45_mynnopts/ICLOUD_BL.md). | Prior combined short run; isolated V045 PROD day/night and cost pending. |
| bl_mynn_edmf=0 | Existing native master plume switch; real WN3 day/night two-call pristine-WRF oracle, binding/refusals/restart. [Audit](../../proofs/v45_mynnopts/EDMF_OFF.md). | Composed GPU qualification pending. |
| bl_mynn_cloudpdf=0/1 | New optional native/legacy formulas, previous-call moments, cold initialization, RK/restart wiring. Fresh original-WRF 384-column recapture; 56 tests and 21,235-node full collection pass. [Guide](../mynn_cloudpdf.md), [evidence](../../proofs/v45_mynnopts/REPORT.md). | Exact-call component CPU smoke passes. GPU not booked or qualified; actual E41, cost and coupled RW gates pending. |

Default program identity passes the constant-aware native-first E222/E224 probe,
with the narrowly documented final source carry. These CPU results do not
substitute for the final composed release gate.
