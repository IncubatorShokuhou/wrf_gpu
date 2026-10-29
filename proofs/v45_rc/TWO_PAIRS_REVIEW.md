# Four fresh timing arms: review packet

Manager ruling 2026-10-29T04:30:22Z requires at least two counted arms of each kind. The proposed cohort is N2, O2, N3, O3, each 24 h, in that order. N2 repeats a density-rejected observation; old R4 observations are not pooled into the new cohort. Two accepted fresh pairs suffice for the unchanged frozen rule. No N72 or fidelity rerun is proposed.

Plan: `<USER_HOME>/wrf_gpu2_lanes/v45-rc/PLAN_TIMING_TWO_PAIRS_V2.json`, SHA256 `3195a4d771b09a026cfc9171a3d407e11e0a97012a798571c1d0ad5216403bc8`. Entry: `tools_timing_pairs_v2/entry_rw3.sh`. Tools sums SHA256 `72cd6f8352d9000946b1b465787e815786afa822e7c0e1172b249b6e6742149f`. Source copy is committed in this directory.

The host-only changes include mandatory slot refusals exiting nonzero (4486586d1a, 12 controls) and absolute-monotonic sampling deadlines (cb72983473). Actual CPU production-period sampling gave 12 samples/12 s, mean interval 1.015 s, max 1.07 s. All process tick arithmetic, acceptance thresholds and verdict code are unchanged. Old density rejects remain rejected. Physics source 254c482374/SRC827470e10f, both paid bundles and the closed R4 tools remain unchanged.

Exact manager `night_seq_v9.sh` CPU SEQ_SMOKE ran 05:17:35–05:22:47Z, 313.206 s; actual step/session/post rc 0, all four exact CLI dry runs rc 0, SDK schema dry runs start/end, compiled_new 0, one supervisor setsid, all children reaped, no owned GPU lock FDs, per-arm temporary layers cleaned. No native GPU model or new scientific validation was executed. Both CPU stand-ins explicitly use the old read-only bundle. The scientific verdict remains INCONCLUSIVE, as required. See `receipts/TWO_PAIRS_SEQ_SMOKE.json`, manager summary/log, and ROOT steady samples.

Request 65 min: 60 entry = 36 model caps + 12 worst quiet gates + 12 checks/teardown/post margin; 2 sequencer + 3 admission/margin. Typical 18–20 min is inferred from prior measured arms, not a new timing result. All four arms collect real RSS and 500 ms per-PID VRAM samples in GPU mode.

ROOT growth reserve 0.25 GiB, floor 22 + 2 margin => production start 24.25 GiB. DATA transient 8 GiB, durable reserve <0.05 GiB after per-arm SHA capture/deletion of own raw; start 412.5 GiB enforces the requested 404.5 floor. A different floor/loan requires an explicit manager/ALISIOS ruling and resealed plan; no loan is assumed. At proof closure ROOT 23.538 / DATA 396.522 GiB: production resource admission is NOT satisfied. This packet is READY FOR CRITIC, not armed for GPU.

Manager must supply the booked GA window, current lane lease, CPU lease held at least 30 s, and fresh manager NIGHTLY_DONE admission receipt. GPU only through the official lock. Optional graupel diagnostic is separate and not included in these 65 minutes.
