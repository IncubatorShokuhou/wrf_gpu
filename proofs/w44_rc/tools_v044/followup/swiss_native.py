"""w44-rc v0.4.4 Swiss C1/C2 (branch a of PREREG swiss_identity): compiled-artefact identity of the Swiss d01 AOT
program of the new bundle vs the v0.4.3 bundle. CPU only, read-only; content hashes only (no st_dev/st_ino).

usage: JAX_PLATFORMS=cpu python3 swiss_native.py SNAPSHOT NEW_REPLAY_RUN_LOG OLD_REPLAY_RUN_LOG OUT_JSON NEW_SRC_PREFIX OLD_SRC_PREFIX
The blob of each side is the one the warm Swiss replay actually loaded ('[gpuwrf:nested-aot] domain=d01 loaded=true
source=aot_blob path=...'): v0.4.4 B5_swiss_3h run.log and v0.4.3 P43/receipts/B5_swiss_3h/run.log. Compared: optimized HLO
(decoded from the blob) raw and path-normalized SHA, Riegeli record SHAs (path-normalized), native record 2 SHA, CUDA ELF
SHAs. PASS = optimized HLO (normalized) + record 2 + ELFs + every normalized record equal (= the v0.4.3 C1/C2 rule).
"""
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

snap, new_log, old_log, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
prefixes = [p.encode() for p in sys.argv[5:7]]
sys.path.insert(0, str(snap / "src"))
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm  # noqa: E402
spec = importlib.util.spec_from_file_location("gate", snap / "scripts/wave_cubin_gate.py")
gate = importlib.util.module_from_spec(spec); sys.modules["gate"] = gate; spec.loader.exec_module(gate)
sha = lambda b: hashlib.sha256(b).hexdigest()


def norm(b):
    for p in prefixes:
        b = b.replace(p, b"<SRC>")
    return b


def blob_of(log):
    m = re.findall(r"\[gpuwrf:nested-aot\] domain=d01 loaded=true source=aot_blob path=(\S+\.xlaexec)", Path(log).read_text(errors="replace"))
    if not m:
        raise SystemExit(f"no loaded d01 aot_blob line in {log}")
    return Path(m[-1])


def ident(path):
    blob = path.read_bytes()
    rec = dict(path=str(path), blob_sha256=sha(blob), format=ax._gpu_blob_format(blob))
    recs = []
    for i in range(100000):
        try:
            r = gate._record(blob, i, ax, asm)
        except ValueError as e:
            if "missing Riegeli record" not in str(e):
                rec["records_error"] = repr(e)[:200]
            break
        recs.append(dict(index=i, sha256=sha(r), sha256_pathnorm=sha(norm(r))))
    rec["records"] = recs
    native = gate._record(blob, 2, ax, asm)
    rec["native_record2_sha256"] = sha(native)
    rec["native_elfs"] = [sha(e) for e in gate._elfs(native)]
    hlo = gate._optimized_hlo(blob, ax, asm)
    hb = hlo.encode() if isinstance(hlo, str) else hlo
    rec.update(optimized_hlo_sha256=sha(hb), optimized_hlo_sha256_pathnorm=sha(norm(hb)))
    return rec


a, b = ident(blob_of(new_log)), ident(blob_of(old_log))
checks = dict(optimized_hlo_pathnorm=a["optimized_hlo_sha256_pathnorm"] == b["optimized_hlo_sha256_pathnorm"],
              native_record2=a["native_record2_sha256"] == b["native_record2_sha256"],
              elfs=a["native_elfs"] == b["native_elfs"],
              records_pathnorm=[r["sha256_pathnorm"] for r in a["records"]] == [r["sha256_pathnorm"] for r in b["records"]])
res = dict(new=a, old=b, checks=checks, verdict="PASS" if all(checks.values()) else "DIFF")
out.write_text(json.dumps(res, indent=1) + "\n")
print("SWISS_NATIVE", res["verdict"], checks)
