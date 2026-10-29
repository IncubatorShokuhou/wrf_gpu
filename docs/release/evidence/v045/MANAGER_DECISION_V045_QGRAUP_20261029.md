# Manager-Entscheidung 2026-10-29T07:20:11Z: v0.4.5 — QGRAUP-Trace-Disposition, Release freigegeben zur Endprüfung

**Stand:**
- Release-Gates: RW1, RW2, RW4 und Swiss PASS.
- RW3-Fidelity: reguläres A3 PASS (5 FLOOR-LIMITED, 0 REAL FAIL).
- RW3-Timing/Memory: PASS (R6: N/O 0,9965, Oberkante 1,0146 < 1,10; VRAM 0,9735, RSS 0,9995).
- Compile: +3,48 %.
- E41: HARD_PASS (HLO UNKNOWN, offengelegt).
- FINAL_RW_VERDICT.json: overall MANAGER_DECISION, nur wegen der QGRAUP-Trace-Klasse.

**QGRAUP d03 t72:** GPU 0 vs CPU 6,58e-12 kg/kg, identisch mit v0.4.4. Belege von v45-graupel (RC_TRAJECTORY_REPORT, Commit 340d8d8b56):
- Der Fix 720d8d2937 ist im tatsächlich geladenen GPU-Programm vorhanden und wird verwendet (PTX/SASS).
- Pristine WRF auf den rekonstruierten RC-t72-Säulen liefert ebenfalls 0.
- Die Trace-Spur bei h71 ist ein anderes Ereignis (46 Zellen, max. 6,4e-11).
- Die Ursache des Prozesses zwischen h71 und h72 ist OFFEN.

**Entscheidung** (Owner informiert: AN-20261029-022834 und -034606; keine Gegenweisung; Owner-Regel „autonom entscheiden, offenlegen“):
- v0.4.5 geht mit **offengelegter** QGRAUP-Trace-Disposition in die Endprüfung, nach v0.4.2-Präzedenz und mit neuen Belegen.
- Es gibt keine stille Ausnahme und keine Änderung an Schwellen oder Regeln.
- Die Release-Notes nennen Belege und offene Ursache.
- Die Ursachenanalyse (h70-Checkpoint plus CPU-Replay, R5G-18s-Daten) wird ein v0.4.6-Punkt.

**Weiter:** Release-Docs final (Speed-Zeilen), final.sh, Wheel-Smoke, Release-Critic cr-w44-rc, dann Tag und Push.
