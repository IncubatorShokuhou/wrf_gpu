"""w44-rc v0.4.4: write the immutable pre-registration of the release real-world gate (RW1-RW4, RW3T timing, RW3 D6,
Swiss identity, E41) BEFORE any v0.4.4 GPU number exists. Called once by prepare.sh. Stdlib only.

usage: python3 -I prereg_v044.py PIN_MANIFEST.json OUT_PREREG.json
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

man_path, out = Path(sys.argv[1]), Path(sys.argv[2])
assert not out.exists(), "PREREG is immutable"
man = json.loads(man_path.read_text())
tools = Path(man["tools_dir"])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rec = {
    "schema": "w44-rc.prereg.v1",
    "registered_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "manifest": str(man_path), "manifest_sha256": sha(man_path), "pin": man["pin"], "src_tree": man["src_tree"],
    "standin": man["standin"],
    "tools_sha256": {p.name: sha(p) for p in sorted(tools.iterdir()) if p.is_file() and p.suffix in (".sh", ".py", ".txt")
                     and not p.name.startswith(("PIN_MANIFEST", "PREREG"))},
    "authority": "AGENTS.md real-world gate (MANAGER.md 'Real-world gate'); _WAVE_V044_COMMON.md; RW3T A/B rule and verdict "
                 "formula unchanged from v0.4.3 (proofs/w43_release/tools/prereg.py, rw3t_verdict.py)",
    "numerics_expectation": {
        "swiss": "IDENTICAL to v0.4.3 ONLY if the candidate release identity is still b977712f (see rules.swiss_identity: "
                 "a WRF-fidelity lane such as w44-constfold may change it before the RC freeze)",
        "prod_wn3": "INTENTIONALLY CHANGED (w44-topo WRF-faithful topo_shading/slope_rad nest fix; PROD/WN3 run topo_shading=1) "
                    "-> no carry-forward; RW3 = full fresh 72 h D6 vs the original CPU-WRF"},
    "rules": {
        "RW1": "BINDING: warm replays through the sealed bundle consumer: B1_wn3_0227_3h == A3_wn3_0227_3h, B5_swiss_3h == A1_swiss, "
               "B6_prod_3h == A2_prod (whole-file wrfout SHA, every file) AND each replay rc 0, BUNDLE_CONSUMER compiled_new=0, "
               "check before/after BUNDLE_OK missing=0 changed=0 new=0, every '[gpuwrf:nested-aot]' program source=aot_blob; "
               "NEW, no v0.4.3 precedent (run-length independence): N72 frames of leads 0-3 == A3_wn3_0227_3h and "
               "B9_swiss_24h frames 0-3 == A1_swiss (same names, same SHA); a difference = MANAGER_DECISION with the first "
               "differing frame/variable, not an automatic STOP. "
               "k1/startup (d01 lead-1 publication s) of B1 vs v0.4.3 B4_wn3_0227in48h_3h receipt: DISCLOSED.",
        "RW2": "B2_wn3_0502_3h (second date) and B4_wn3_0227in48h_3h (48 h input) and B8_wn3_0502_icloud0_3h: replay rules as "
               "RW1 (0 new compiles); one cheap key per program over the WN3 default arms, every key a sealed blob.",
        "RW4": ">= 3 dates through the sealed bundle (B2 0502, B4 0227-48h-input, B3 0614) replay rules as RW1 + CPU "
               "`gpuwrf aot-key` pairs on the pinned source (k0227_v_0502_h72, k0227_v_0614_h48, k0227_v_0227in48_h3, "
               "ki0227_v_ki0502_h3) all SHARED.",
        "RW3_fidelity": "N72 vs the ORIGINAL CPU-WRF WN3 0227 72 h (v0.3.3 W3P0227 full72 frames, staged read-only through "
                        "alisios_guard.facade_stage = the v0.4.2 stage_cpu72 recipe): <snapshot>/scripts/wn3_score.py --hours 72, "
                        "frozen D6 manifest, 73 paired frames x 3 domains; floor72_new.py (frozen v0.3.3 A3 rule, C verbatim "
                        "from FLOOR72_FROM_EXISTING_C.json) binding72_A3 == PASS AND output_integrity_pass == True. "
                        "A d03 QGRAUP 'degenerate' finding where CPU-WRF max < 1e-9 kg/kg (v0.4.2 registered-exception class) "
                        "is reported as MANAGER_DECISION, never auto-PASS. Any other D6/integrity failure = STOP.",
        "RW3_speed": "rw3t_verdict.py unchanged: counted 24 h arms (accept.py), PRIMARY d03_step_median; pooled N/O + S <= 1.10 "
                     "PASS, pooled N/O - S > 1.10 REAL_REGRESSION = STOP, else INCONCLUSIVE (manager). N72/O72 = one disclosed "
                     "pair. Core watch 3,13,19,29 abort P>100 %/A>200 % >10 s (max 3 tries), shadow 20/50 disclosed, pre-gate "
                     "60 s quiet stretch max 180 s, arm-window <DATA_ROOT> await > 50 ms = whole-run DISTURBED (disclosed).",
        "RW3_memory": "VRAM peak per PID (nvidia-smi compute-apps 0.5 s) and host max RSS (/usr/bin/time) of N vs O (pooled "
                      "counted 24 h arms; N72 vs O72 disclosed): ratio > 1.10 = STOP and explain.",
        "RW3_compile": "cold-minus-warm proof wall A3_wn3_0227_3h - B1_wn3_0227_3h (v0.4.4, same session) vs v0.4.3 811.3 s "
                       "(A3 72 h - N72, v0.4.3 RW_VERDICT, CROSS-SESSION and host-loaded, disclosed): ratio > 1.10 = STOP and "
                       "explain (manager may rule cross-session noise).",
        "swiss_identity": "BRANCH on the measured candidate identity (proofs/s1_port/release_identity.py on the pinned source, "
                          "CPU, before any v0.4.4 GPU number; followup/identity_pre.sh). (a) Swiss sha == b977712fe325c953f3eb"
                          "bf8f83ea1ca0b9260833ef70981e48bd8a49410e142a -> C1/C2 (Swiss d01 program optimized-HLO SHA + native "
                          "record-2 SHA equal to a v0.4.3 d01 program) and C3 (A1_swiss / B5_swiss_3h wrfout whole-file SHA == "
                          "v0.4.3 receipts A1_swiss / B5_swiss_3h) must be IDENTICAL; a difference = STOP and diagnose. "
                          "(b) otherwise (e.g. w44-constfold landed) Swiss is a CHANGED program: B9_swiss_24h vs the original "
                          "CPU-WRF Swiss 24 h reference <USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu (FINDINGS D30; inputs "
                          "byte-identical to the arm input [M]) with scripts/wn3_score.py --domains d01 --hours 24 (frozen D6 "
                          "manifest, 25 frames) must PASS incl. output integrity; the identity table records the new sha. "
                          "PROD: identity expected CHANGED (topo; old 321aa651067f...) -> recorded; RW1 bytes only.",
        "E41": "scripts/wave_cubin_gate.py on P/jax (every rich JIT + AOT program incl. fused, E201) with the snapshot "
               "PYTHONPATH: hard_stack_gate PASS required; e41_gate INCOMPLETE accepted only as in v0.4.3 (compact AOT HLO "
               "kInput UNKNOWN, conservative_flags False); FAIL = STOP.",
        "icloud0": "A4/B8 prime + replay (programs in the bundle, replay rules, one key per program). NO D6: the w42 CPU-WRF "
                   "icloud=0 twin was deleted after v0.4.2 ([M] search 2026-10-24) and topo changes WN3 numerics, so the "
                   "icloud=0 qualification is NOT carried forward (disclosed; owner decision).",
        "disk_ram": "every arm refuses when <DATA_ROOT> free - forecast < 365 GiB; GPU payload in one RAM slot (MemoryMax 20G); "
                    "wrfouts deleted after their SHA receipts except A1-A4 (follow-up) and N72 (D6 input)."},
}
out.write_text(json.dumps(rec, indent=1) + "\n")
print("PREREG", out, sha(out))
