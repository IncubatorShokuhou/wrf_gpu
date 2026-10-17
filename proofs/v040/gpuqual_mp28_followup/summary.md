# MP28 affected GPU 24 h qualification

Measured immutable main 195cd4194 with the original CPU-WRF reference and unchanged input/namelist hashes. Its model/data/scripts trees match the independently captured changed program in 5ab. One approved GPU forecast completed with 25 hourly frames.

All 62 frozen bounded D6 fields pass across all 25 frames, with zero raw NaN/Inf. Coverage remains INCOMPLETE only for QNBCA and TAOD5502D, the disclosed uncomputed fields. Variable dtypes and attributes match CPU-WRF in every frame. Only TITLE and NTASKS_X/Y differ as execution provenance.

QNIFA2D matches CPU-WRF exactly. QNWFA2D is present and finite but REPORT_ONLY: RMSE 8,920.4125 kg^-1 s^-1, maximum 22,511.8252. No new tolerance or fidelity pass is declared. The difference exists at frame 0 and persists to hour 24.

The source/operator witness identifies a startup caller geometry mismatch. Pristine start_em.F:873 passes interface height (PH2+PHB)/g through module_physics_init.F:4530 to thompson_init. The port's startup emission helper averages interfaces into mass heights. At j21/i18, interface height is 2,479.1777 m; mass height 2,501.9082 m crosses the 2,500 m branch. CPU and the interface formula give 32,030.0977; GPU and the mass formula give about 9,518.27. The literal REAL4 interface formula differs from saved CPU by at most 0.015625, while the mass formula reproduces the 22,511.8 discrepancy. These residuals are diagnostic evidence, not a new oracle tolerance. Producer-owner correction and affected validation remain pending. This lane made no model-code changes.

Actual stepping JIT+AOT E41 coverage is complete, with two entries. E41 FAIL: kInput transpose fusions contain 741 and 1,439 optimized HLO instructions. Maximum stack is 16 B and non-leaf >1k count is zero. Exact paid native artifacts are retained with SHA maps.

F2 remains OPEN: 14,457,017 negative phase-field visits, including 11,816,359 interior visits. These are repeated visits, not unique events or every-event source justification. Raw counter nonfinite count is zero. Full field/phase minima are attached.

Cold wall was 461.968 s, sampled peak VRAM 2,870 MiB, host peak RSS 9,428,480 KiB (8.99 GiB). These are qualification receipts, not S1 or isolated performance measurements. Cache/tmp and the immutable source snapshot were removed after their last E41, scoring and diagnostic consumer. References, old science, 25 new frames and two paid native PINs remain. No GPU jobs/waiters or completed snapshots remain. No release clearance is claimed.
