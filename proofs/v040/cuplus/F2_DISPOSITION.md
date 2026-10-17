# CuPlus CPU smoke: pre-mask event disposition

The primary CPU smoke checks every floating carry leaf for nonfinite values,
all 66 main guard counters, CAM's original source events, and checkpoint
continuation. Full F2 negative extrema are measured separately over all three
steps, before the microphysics call. No tendency or state repair was added.

| Pairing | Post-RK interior visits | Minimum in the census summary |
|---|---|---|
| KF-CuP / MYNN | QC 1; Nr 16 | QC −1.59192599e−25 kg/kg; Nr −2.38074308e−22 kg⁻¹ |
| MSKF / MYNN | QC 1; Nr 10 | QC −1.59192599e−25 kg/kg; Nr −1.06655635e−22 kg⁻¹ |
| CAM-ZM / MYJ | QC 1 | QC −5.08200317e−25 kg/kg |
| CAM-ZM / CAM-UW | QC 1 | QC −1.01719444e−27 kg/kg |

These are **option-only** events: the matched CU0/MYNN control has zero
negative visits. They are not classified as pre-existing by the manager's
equal-field/count/location rule. Counts can vary between separately compiled
CPU artifacts: the earlier one-step CuP calls report 15 Nr visits rather than
the three-step artifact's 16. Both raw receipts remain preserved.

The measured location is exclusively post-RK, before original WRF-order
microphysics. Entry and both pre-guard phases have zero negative counts;
nonfinite, out-of-range and repair counters are zero. The unchanged scalar
PD/RK arithmetic subtracts outgoing flux and source terms near zero
(`module_advect_em.F:7733–7779`). All 53 dynamics/native-kernel source blobs
are byte identical to approved main. Independent original-CPS RK seam gates
check the newly inserted thermal, moisture, momentum and number sources,
including both nonzero momentum axes. Thus the evidence is consistent with
compiler cancellation at zero, rather than a new negative physical source.
That classification is an inference, not a claim that original CPU-WRF emitted
the same signed samples. The small events are explicitly disclosed under the
standing compiler-deviation policy; they do not establish release F2 or D6.

This CPU facade carries wide prognostic arrays. Census extrema are rounded
to REAL32, so `exact_real32_extremum` is false and no exact wide extremum is
claimed. A conservative magnitude bound for every visit is the absolute value
of the next REAL32 number below the recorded minimum. The companion receipt
records that bound and the full visit coverage, source hashes and raw hashes.
GPU, REAL-carry coupled forecasts and the final D6 gate remain pending.

Evidence: lane artifacts `F201-cup.json`, `F202-mskf.json`, `F203-myj.json`,
`CC12-camuw-fixed.json`, `CC11-control-mynn.json`, `SA01-source-audit.json`; original arithmetic and
literal/compiled qualifications in `UPSTREAM_WRF_BUGS.md` and `full_callers/`.
