# Manager decision RW3 (v0.4.4 RC 286e2442a9), 2026-10-27T18:43:27Z — separate file; RW3_VERDICT.json stays unchanged (E212)

Owner approval in chat (2026-10-27, AskUserQuestion "RW3-Urteil": "Ja, nach Bestätigung").
1. d03 RAINNC lead 67: C67 = 0.8905983272943488. It was recovered from the archived ORIGINAL CPU P0227-vs-R pair series
   docs/release/evidence/v033/w2/W2bR/metrics.json (sha256 d3577c22…); the same series reproduces all 6 frozen C values (4491e479) byte-exact.
   Independently confirmed by cr-w44-rc at 2026-10-27T10:56:38Z (rerun byte-identical). Frozen rule unchanged: C/X = 0.8401 >= 0.75 → FLOOR-LIMITED.
   binding72_A3 therefore PASS (0 REAL FAIL, 6 FLOOR-LIMITED). The raw D6 exceedances at leads 67–72 remain DISCLOSED.
2. d03 QGRAUP last frame (GPU exactly 0, CPU max 6.5775e-12 kg/kg): same class as the v0.4.2 registered exception
   (INTEGRITY_ADJUDICATION 2026-10-19). Registered again as a known exception and disclosed.
3. Disclosure in the release notes: d03 RAINNC at lead 67 rose from X 0.50 (v0.4.2/v0.4.3) and 0.76 (v0.4.1) to 1.06 (v0.3.3 had 0.95).
   It is within the CPU-vs-CPU twin spread (0.89). The WN3 programs change through RC items 1 TOPO + 5 QSQ + 6 E84; there is no per-item causal proof.
   O72 was not run (disk); speed and memory come from the 3 same-session 24 h pairs (ratio 0.998, PASS).
RESULT: RW3 PASS by manager decision with disclosed exceptions → release continues.
