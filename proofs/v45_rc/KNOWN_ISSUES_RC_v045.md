# v0.4.5 RC known issues and open gates

Model pin `254c4823745275e3e9d397ae4d1db0fd35ec361d`, src `827470e10f7e29cad1613e18d0a0c84f8f95693f`.

RW1/RW2/RW4 and Swiss 24 h PASS. N72 covers 3 domains × 73 frames;
regular frozen A3 PASS with five FLOOR-LIMITED checks and zero REAL FAIL.
Raw D6 remains FAIL; output integrity separately flags d03 QGRAUP at t72.
The graupel fix is active in the actual loaded GPU executable, and pristine WRF
on tested reconstructed RC t72 columns also gives zero. The 71–72 h process and
trajectory cause remain OPEN (v45-graupel report 340d8d8b56), proposed v0.4.6.
This does not establish a coupled “trajectory trace, not a port defect” conclusion
or apply an automatic integrity waiver. Manager07:20 explicitly accepts the
disclosed trace disposition; decision retained in docs/release/evidence/v045/.

R6 freshN2/O2/N3/O3 all4counted+strict-shadowPASS; speedN/O0.9965,
S0.0181/upper1.0146<1.10, whole1.0072; ownPIDVRAM0.9735/RSS0.9995 PASS.
Old density-rejected arms stay rejected; rules unchanged. O72 absent disclosed.
Compile proxy 489.9 vs 473.420 s (+3.48%); differing autotune
starting state disclosed. Diffusion/c_s and nine LSM controls remain REFUSED.
MYNN cloudpdf 0/1 coupled GPU qualification remains pending; default 2 unchanged.

E41 CPU full-shape prescreens pass all five programs; only the Thompson family
changes for the default path. Actual-cache hard-resource checks PASS; compact
optimized-HLO metadata UNKNOWN is disclosed, not waived or claimed PASS.
Reader-registry metadata and resume-journal error handling are recorded freeze
exceptions; no extra model/source change is introduced by release preparation.

Timing plan3195a4d7 and manager R5 gatec0df6b10 are critic READY but activation
waits for ALISIOS and resources. Current timing production admission requires
ROOT ≥24.25 GiB and DATA ≥412.5 GiB; manager gate rounds DATA to413 (423 if
near14Z). No DATA writes by this lane until the stop is explicitly released.
All release drafts/collection artifacts stay on ROOT.

Evidence: existing `COMPOSITION.json`, `CPU_SUITE.json`,
`RESUME_JOURNAL_INCLUSION.json`, `COMPILEONLY_IDENTITIES.json`; fresh release
prep manifest binds original follow-up receipts. Package gates and release
decision remain open. Historical nine-domain, icloud=0 twin and calibration
limits remain as recorded in the public known issues.
