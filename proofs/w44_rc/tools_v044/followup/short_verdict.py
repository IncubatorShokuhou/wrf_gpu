"""w44-rc v0.4.4: RW1/RW2/RW4 + Swiss-branch + E41 verdict of the SHORT entry (CPU, reads files only).

usage: python3 -I short_verdict.py RUN_DIR BUNDLE_DIR OUT_JSON --bytes BYTES.json --identity IDENTITY_BRANCH.json
         [--swiss-native SWISS_NATIVE.json] [--swiss-d6 D6_DIR] [--aotkey DIR] [--aotkey-x DIR]
Rules = PREREG.json rules (RW1/RW2/RW4/swiss_identity/E41/icloud0). Missing evidence -> MISSING/INCOMPLETE, never PASS.
"""
import json
import re
import sys
from pathlib import Path

run, bundle, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
opt = {sys.argv[i]: sys.argv[i + 1] for i in range(4, len(sys.argv) - 1, 2)}
R = run / "runs"
AOT = re.compile(r"\[gpuwrf:nested-aot\] domain=(\S+) loaded=(\S+) source=(\S+)(.*)")


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError, TypeError):
        return None


def txt(p):
    try:
        return Path(p).read_text(errors="replace")
    except OSError:
        return ""


def arm(name):
    d = R / name
    if not d.is_dir():
        return dict(present=False)
    rec = dict(present=True, rc=txt(d / "rc.txt").strip() or None)
    cons = [l for l in txt(R / f"{name}.scope.log").splitlines() if l.startswith("BUNDLE_CONSUMER rc=")]
    rec["consumer"] = cons[-1] if cons else None
    rec["consumer_counts"] = {k: int(v) for k, v in re.findall(r"(\w+_new|layer_compiled)=(\d+)", rec["consumer"] or "")}
    for c in ("check_before", "check_after"):
        rec[c] = (txt(d / f"{c}.txt").strip().splitlines() or [None])[-1]
    lines = []
    for line in txt(d / "run.log").splitlines():
        g = AOT.search(line)
        if g:
            key = re.search(r"/k_([0-9a-f]{64})\.xlaexec", g.group(4)) or re.search(r"cheap_key=([0-9a-f]{64})", g.group(4))
            lines.append(dict(domain=g.group(1), loaded=g.group(2) == "true", source=g.group(3), key=key.group(1) if key else None))
    rec["aot"] = lines
    rec["all_loaded"] = bool(lines) and all(x["loaded"] and x["source"] == "aot_blob" for x in lines)
    r = load(d / "RECEIPT.json") or {}
    rec.update(max_rss_gib=r.get("max_rss_gib"), n_wrfout=r.get("n_wrfout"),
               proof_wall_total_s=(r.get("proof") or {}).get("wall_clock_total_s"),
               all_domains_finite=(r.get("proof") or {}).get("all_domains_finite"))
    peak = 0
    for row in txt(d / "vram_apps.csv").splitlines():
        f = [x.strip() for x in row.split(",")]
        if len(f) >= 3 and f[2].isdigit():
            peak = max(peak, int(f[2]))
    rec["vram_peak_pid_mib"] = peak or None
    return rec


def replay_ok(a):
    c = a.get("consumer_counts") or {}
    return bool(a.get("present") and a.get("rc") == "0" and c.get("compiled_new") == 0 and a.get("all_loaded")
                and (a.get("check_after") or "").startswith("BUNDLE_OK missing=0 changed=0 new=0 ")
                and (a.get("check_before") or "").startswith("BUNDLE_OK missing=0 changed=0 new=0 "))


PRIME = ["A1_swiss", "A2_prod", "A3_wn3_0227_3h", "A4_wn3_0227_icloud0_3h"]
REPLAY = ["B1_wn3_0227_3h", "B2_wn3_0502_3h", "B4_wn3_0227in48h_3h", "B5_swiss_3h", "B6_prod_3h", "B9_swiss_24h",
          "B8_wn3_0502_icloud0_3h", "B3_wn3_0614_3h"]
arms = {n: arm(n) for n in PRIME + REPLAY}
byt = load(opt.get("--bytes")) or {}
pairs = byt.get("pairs") or {}
res = dict(run=str(run), bundle=str(bundle), arms={k: {kk: vv for kk, vv in v.items() if kk != "aot"} for k, v in arms.items()})
res["prime_ok"] = {n: arms[n].get("rc") == "0" and arms[n].get("all_domains_finite") is not False for n in PRIME}
rw1_bin = {w: dict(replay_ok=replay_ok(arms[w]), bytes=(pairs.get(f"RW1 {w}") or {}).get("verdict", "MISSING"))
           for w in ("B1_wn3_0227_3h", "B5_swiss_3h", "B6_prod_3h")}
res["RW1"] = dict(binding=rw1_bin, new_prefix_checks={k: v.get("verdict") for k, v in pairs.items() if k.startswith("PREFIX")},
                  verdict="PASS" if all(v["replay_ok"] and v["bytes"] == "PASS" for v in rw1_bin.values()) else "FAIL_OR_INCOMPLETE")
manifest = load(bundle / "sealed/BUNDLE_MANIFEST.json") or {}
sealed = {re.sub(r".*/(\w+)/k_([0-9a-f]{64})\.xlaexec$", r"\1:\2", f) for f in (manifest.get("files") or {})
          if re.search(r"/k_[0-9a-f]{64}\.xlaexec$", f)}


def key_group(names):
    keys = {}
    for n in names:
        for x in arms[n].get("aot") or []:
            if x["key"]:
                keys.setdefault(x["domain"], {}).setdefault(n, set()).add(x["key"])
    one = {dom: len(set().union(*v.values())) == 1 and set(v) == set(names) for dom, v in keys.items()}
    in_seal = {dom: all(f"{dom.replace('/', '_')}:{k}" in sealed for k in set().union(*v.values())) for dom, v in keys.items()}
    return {d: {n: sorted(s) for n, s in v.items()} for d, v in keys.items()}, one, in_seal


wn3 = ["A3_wn3_0227_3h", "B1_wn3_0227_3h", "B2_wn3_0502_3h", "B4_wn3_0227in48h_3h", "B3_wn3_0614_3h"]
keys, one, in_seal = key_group(wn3)
ikeys, ione, iseal = key_group(["A4_wn3_0227_icloud0_3h", "B8_wn3_0502_icloud0_3h"])


def aotkey(d):
    out_ = {}
    for f in sorted(Path(d).glob("*.json")) if d else []:
        rep = load(f) or {}
        out_[f.stem] = {name: row.get("shared") for name, row in (rep.get("comparison") or {}).items()}
    return out_


ak, akx = aotkey(opt.get("--aotkey")), aotkey(opt.get("--aotkey-x"))
ak_ok = bool(ak) and len(ak) >= 4 and all(v and all(v.values()) for v in ak.values())
res["RW2"] = dict(arms={n: replay_ok(arms[n]) for n in ("B2_wn3_0502_3h", "B4_wn3_0227in48h_3h", "B8_wn3_0502_icloud0_3h")},
                  gpu_keys=keys, one_key_per_program=one, keys_sealed=in_seal)
res["RW2"]["verdict"] = "PASS" if (all(res["RW2"]["arms"].values()) and keys and all(one.values()) and all(in_seal.values())) else "FAIL_OR_INCOMPLETE"
res["RW4"] = dict(dates={n: replay_ok(arms[n]) for n in ("B2_wn3_0502_3h", "B4_wn3_0227in48h_3h", "B3_wn3_0614_3h")},
                  cpu_aot_key_pairs=ak, cpu_aot_key_icloud_flip=akx)
res["RW4"]["verdict"] = "PASS" if all(res["RW4"]["dates"].values()) and ak_ok else "FAIL_OR_INCOMPLETE"
res["icloud0"] = dict(keys=ikeys, one_key_per_program=ione, keys_sealed=iseal, replay_ok=replay_ok(arms["B8_wn3_0502_icloud0_3h"]),
                      d6="NOT RUN: CPU-WRF icloud=0 twin deleted after v0.4.2 (owner decision)")
ident = load(opt.get("--identity")) or {}
branch = ident.get("swiss_branch", "MISSING")
sw = dict(branch=branch, swiss_sha=ident.get("swiss_sha"), prod_sha=ident.get("prod_sha"), prod_changed=ident.get("prod_changed"),
          schema2_verdicts={c: (v or {}).get("verdict") for c, v in (ident.get("schema2") or {}).items()},
          expectation_mismatches=ident.get("expectation_mismatches"), convention_disagreement=ident.get("convention_disagreement"))
if branch.startswith("a_"):
    nat = load(opt.get("--swiss-native")) or {}
    c3 = {k: (pairs.get(k) or {}).get("verdict", "MISSING") for k in ("C3 A1_swiss", "C3 B5_swiss_3h")}
    sw.update(native=nat.get("verdict", "MISSING"), native_checks=nat.get("checks"), bytes=c3,
              verdict="IDENTICAL" if nat.get("verdict") == "PASS" and all(v == "PASS" for v in c3.values()) else "STOP_DIAGNOSE")
elif branch.startswith("b_"):
    d6 = load(Path(opt.get("--swiss-d6", "/nonexistent")) / "summary.json") or {}
    dom = (d6.get("domains") or {}).get("d01") or {}
    sw.update(d6_pass=d6.get("pass"), integrity=d6.get("output_integrity_pass"), paired=(dom.get("pairing") or {}).get("paired_file_count"),
              policy_failures=dom.get("policy_failures"), verdict=("PASS" if d6.get("pass") and d6.get("output_integrity_pass")
                                                                   and (dom.get("pairing") or {}).get("paired_file_count") == 25
                                                                   else ("MISSING" if not d6 else "STOP")))
elif branch == "INVALID_STOP":
    sw["verdict"] = "STOP"   # schema-2 identity INVALID_INPUT/BLIND/ERROR: the Swiss branch is unproven
    sw["schema2"] = ident.get("schema2")
else:
    sw["verdict"] = "MISSING (run identity_pre.sh)"
res["swiss_identity"] = sw
e41 = load(run / "e41/E41_POST.json") or {}
g = (e41.get("gates") or {}).get("all") or {}
res["E41"] = dict(post=e41, verdict=("PASS" if g.get("hard_stack_gate") == "PASS" and g.get("e41_gate") == "PASS" else
                                     "INCOMPLETE_AS_V043" if g.get("hard_stack_gate") == "PASS" and g.get("e41_gate") == "INCOMPLETE"
                                     and g.get("conservative_flags") is False else ("MISSING" if not g else "FAIL")))
policy = load(bundle / "meta/E41_RC_POLICY.json") or {}
res["E41"]["raw_legacy_verdict"] = res["E41"]["verdict"]
res["E41"]["RC_policy"] = policy
res["E41"]["verdict"] = policy.get("status", "MISSING_RC_POLICY") if policy.get("hard_gate")=="PASS" else "FAIL_OR_INCOMPLETE"

a3, b1 = arms["A3_wn3_0227_3h"], arms["B1_wn3_0227_3h"]
res["compile_s_cold_minus_warm"] = dict(v044=(round(a3["proof_wall_total_s"] - b1["proof_wall_total_s"], 1)
                                              if a3.get("proof_wall_total_s") and b1.get("proof_wall_total_s") else None),
                                        v043=811.3, basis="v0.4.4: A3 3 h cold - B1 3 h warm (same session); v0.4.3: A3 72 h - N72 (cross-session)")
if res["compile_s_cold_minus_warm"]["v044"]:
    res["compile_s_cold_minus_warm"]["ratio"] = round(res["compile_s_cold_minus_warm"]["v044"] / 811.3, 4)
    res["compile_s_cold_minus_warm"]["over_10pct"] = res["compile_s_cold_minus_warm"]["ratio"] > 1.10
out.write_text(json.dumps(res, indent=1, default=str) + "\n")
print(json.dumps({k: (v.get("verdict") if isinstance(v, dict) else v) for k, v in res.items() if k not in ("arms", "run", "bundle")}, default=str))
