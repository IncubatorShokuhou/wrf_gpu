# w42-bundle042 handoff (2026-10-19, lane/w42-bundle042)
- Objective: seal the v0.4.2 compile bundle on pin b979188128 (src b3e9b92a8c, version 0.4.2) + RW1–RW4 + tuple draft.
- Bundle [M]: <DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2, bundle_sha256 bfa2dc90…16d2, manifest 7bcd3d98…, P/SHA256SUMS f8ea3fb5 (232 files), BUNDLE_OK.
- Identities Swiss b977712f / PROD 321aa651 EXACT; one sealed key per program (d01 a1662307, fused ac024b4c; icloud0 ae4b9d39 / 73e33b0a).
- RW1 PASS (N72 == A3 219/219, B5 == A1, B6 == A2) · RW2/RW4 PASS · RW3 PASS binding (RW_TABLE_FINAL.json): no >10 % regression
  (d03 stepping 0.942, VRAM 0.994, RSS 1.002, compile 0.974), RW3T PASS (0.9356 + S 0.0819 ≤ 1.10), D6 72 h raw FAIL 5 all
  FLOOR-LIMITED → binding PASS, integrity raw FAIL d03 QGRAUP t72 → manager REGISTERED EXCEPTION 16:49:38Z → PASS.
- icloud=0 QUALIFIED (D6 3/3 PASS 25 frames, CLDFRA 0, E41 STACK PASS). E41 default: STACK + coverage PASS, gate INCOMPLETE (slim AOT, as v0.4.1).
- Hero [M]: PROD 24 h 4.809 s/fc-h CLEAN (19.19×); WN3 par3 6.227 s/case-h DISTURBED (19.81×).
- Found: release notes "COSZEN only" claim wrong (bytes differ in 143 vars; 355/1,212 autotune choices differ = separate compile, E99) → manager ordered relnotes fix 16:56:48Z.
- Commits: 9c377bc9bb … 38181fbcc8, 960c6ff634, this handoff. Experiments: BS42-REHEARSAL, PIN42-REPIN2, BS42-SESSION, HERO42, RW42-FOLLOWUP, AT42-DIFF.
- GPU used: rehearsal 30 + session 119 + hero 15 GPU-min, all rc 0 (rehearsal stopped by HOLD).
- Open risks: N2/O2 first tries aborted by foreign load (repeated); 72 h arms miss watch_cover (disclosed secondary); opt-out flag not primed.
- Next: one critic (cr-w42-bundle042); after its verdict delete by exact path snap_b979188128e0, snap_v041, scratch gpu/ rw3t/ hero/ cpu72 tmp (P and $L/followup kept).
