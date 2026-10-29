"""v45-rc v0.4.5: write the immutable pre-registration of the release real-world gate (RW1-RW4, RW3T timing, RW3 D6,
Swiss identity, E41) BEFORE any v0.4.5 GPU number exists. Called once by prepare.sh. Stdlib only.

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
    "schema": "v45-rc.prereg.v1",
    "registered_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "manifest": str(man_path), "manifest_sha256": sha(man_path), "pin": man["pin"], "src_tree": man["src_tree"],
    "standin": man["standin"],
    "tools_sha256": {p.name: sha(p) for p in sorted(tools.iterdir()) if p.is_file() and p.suffix in (".sh", ".py", ".txt")
                     and not p.name.startswith(("PIN_MANIFEST", "PREREG"))},
    "authority": "AGENTS.md real-world gate (MANAGER.md 'Real-world gate'); _WAVE_V045_COMMON.md; RW3T A/B rule and verdict "
                 "formula unchanged from v0.4.4 (proofs/w43_release/tools/prereg.py, rw3t_verdict.py)",
    "numerics_expectation": {
        "all": "Fresh schema-2/full-constant identities against v0.4.4. Changed programs require affected original CPU-WRF fidelity gates; GPU-vs-GPU is regression only."},
    "rules": {
        "RW1": "BINDING: warm replays through the sealed bundle consumer: B1_wn3_0227_3h == A3_wn3_0227_3h, B5_swiss_3h == A1_swiss, "
               "B6_prod_3h == A2_prod (whole-file wrfout SHA, every file) AND each replay rc 0, BUNDLE_CONSUMER compiled_new=0, "
               "check before/after BUNDLE_OK missing=0 changed=0 new=0, every '[gpuwrf:nested-aot]' program source=aot_blob; "
               "NEW, no v0.4.4 precedent (run-length independence): N72 frames of leads 0-3 == A3_wn3_0227_3h and "
               "B9_swiss_24h frames 0-3 == A1_swiss (same names, same SHA); a difference = MANAGER_DECISION with the first "
               "differing frame/variable, not an automatic STOP. "
               "k1/startup (d01 lead-1 publication s) of B1 vs v0.4.4 B4_wn3_0227in48h_3h receipt: DISCLOSED.",
        "RW2": "B2_wn3_0502_3h (second date) and B4_wn3_0227in48h_3h (48 h input) and B8_wn3_0502_icloud0_3h: replay rules as "
               "RW1 (0 new compiles); one cheap key per program over the WN3 default arms, every key a sealed blob.",
        "RW4": ">= 3 dates through the sealed bundle (B2 0502, B4 0227-48h-input, B3 0614) replay rules as RW1 + CPU "
               "`gpuwrf aot-key` pairs on the pinned source (k0227_v_0502_h72, k0227_v_0614_h48, k0227_v_0227in48_h3, "
               "ki0227_v_ki0502_h3) all SHARED.",
        "RW3_fidelity": "N72 vs ORIGINAL CPU-WRF WN3 0227, 73 frames x3 domains, approved guarded read-only in-place219-file adapter (no copy/write/chmod). Pinned scripts/wn3_score.py --hours 72 --sequential-domains --rw3-floor; full frozen reference and full original CPU-pair series SHA bound in manifest. Parse rw3_fidelity_verdict and output integrity; raw D6 disclosed separately. Scientific limits unchanged; no borrowed/interpolated C. QGRAUP trace class requires existing registered authority, never an invented waiver.",
        "RW3_speed": "rw3t_verdict.py unchanged: counted 24 h arms (accept.py), PRIMARY d03_step_median; pooled N/O + S <= 1.10 "
                     "PASS, pooled N/O - S > 1.10 REAL_REGRESSION = STOP, else INCONCLUSIVE (manager). N72/O72 is optional (own precheck free>=411GiB), absence HOLD disclosed; no invented 72h O. "
                     "pair. Core watch 3,13,19,29 abort P>100 %/A>200 % >10 s (max 3 tries), shadow 20/50 disclosed, pre-gate "
                     "60 s quiet stretch max 180 s, arm-window <DATA_ROOT> await > 50 ms = whole-run DISTURBED (disclosed).",
        "RW3_memory": "VRAM peak per PID (nvidia-smi compute-apps 0.5 s) and host max RSS (/usr/bin/time) of N vs O (pooled "
                      "counted 24 h arms; N72 vs O72 disclosed): ratio > 1.10 = STOP and explain.",
        "RW3_compile": "cold-minus-warm proof wall A3_wn3_0227_3h - B1_wn3_0227_3h (v0.4.5, same session) vs v0.4.4 473.4201860839967 s "
                       "(same-session A3 3h cold - B1 3h warm; SHA-pinned V044_COMPILE_BASELINE.json; fresh-empty versus old seeded autotune disclosed): ratio > 1.10 = STOP and "
                       "explain (manager may rule cross-session noise).",
        "swiss_identity": "Measured schema2 identity determines required follow-up. If changed, B9 Swiss24h vs ORIGINAL CPU-WRF RD11/cpu (25frames), frozen D6/output integrity required. Same-source warm A1/B5 whole-file SHA binding remains mandatory. No v0.4.3 identity assumptions carried.",
        "E41": "scripts/wave_cubin_gate.py on P/jax (every rich JIT + AOT program incl. fused, E201) with the snapshot "
               "PYTHONPATH: hard_stack_gate PASS required; e41_gate INCOMPLETE accepted only as in v0.4.4 (compact AOT HLO "
               "kInput UNKNOWN, conservative_flags False); FAIL = STOP.",
        "icloud0": "A4/B8 prime + replay (programs in the bundle, replay rules, one key per program). NO D6: the w42 CPU-WRF "
                   "icloud=0 twin was deleted after v0.4.2 ([M] search 2026-10-24) and topo changes WN3 numerics, so the "
                   "icloud=0 qualification is NOT carried forward (disclosed; owner decision).",
        "disk_ram": "every arm refuses when <DATA_ROOT> free - forecast < 365 GiB; GPU payload in one RAM slot (MemoryMax 20G); "
                    "ROOT aggregate<=8GiB/free>=20GiB. DATA writes only after manager Nightly DONE; mandatory peak31GiB/floor365GiB. O72 only if ownprecheckfree>=411GiB, else HOLD/unstarted/no registry start. wrfouts deleted after SHA receipts except A1-A4 and N72."},
}
rec["CPU_admission"] = dict(kind="CARRIED v0.4.4 authority", authority=man["resource_authority"], Swiss="actual CPU first root after full argv alignment", PROD_WN3="exact CLI --dry-run PLAN_ONLY plus all5 fresh fullshape CUDA CompileOnly prescreens", GPU="mandatory fullsize cold primes failfast; no GPU/science PASS until actual runs")
rec["rw3_floor"] = man["rw3_floor"]
out.write_text(json.dumps(rec, indent=1) + "\n")
print("PREREG", out, sha(out))
