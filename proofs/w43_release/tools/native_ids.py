"""Per-program compiled-artefact identities of a sealed bundle AOT dir (CPU only, read-only).

usage: JAX_PLATFORMS=cpu python3 native_ids.py SRC_TREE AOT_VERSION_DIR OUT_JSON [PATH_PREFIX ...]
For every <program>/<hlo_sha256>.xlaexec (k_* hardlink aliases grouped by inode): whole-blob SHA, Riegeli record SHAs
(record 2 = native binary), the CUDA ELF SHAs inside record 2, the optimized-HLO text SHA with every PATH_PREFIX
replaced by '<SRC>' (Pallas/Triton configs embed absolute source paths, E58/E109), raw-path occurrence offsets per
record, and the .meta fields (hlo_sha256, cheap_key, fingerprint, executable_format, retained_module).
"""
import hashlib, importlib.util, json, os, pickle, sys
from pathlib import Path

src, aot, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
prefixes = [p.encode() for p in sys.argv[4:]]
sys.path.insert(0, str(src / "src"))
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm  # noqa: E402
spec = importlib.util.spec_from_file_location("gate", src / "scripts/wave_cubin_gate.py")
gate = importlib.util.module_from_spec(spec); sys.modules["gate"] = gate; spec.loader.exec_module(gate)

def sha(b): return hashlib.sha256(b).hexdigest()
def norm(b):
    for p in prefixes:
        b = b.replace(p, b"<SRC>")
    return b

res = {"aot_dir": str(aot), "path_prefixes": sys.argv[4:], "programs": {}}
for prog in sorted(d for d in aot.iterdir() if d.is_dir()):
    by_inode = {}
    for f in sorted(prog.glob("*.xlaexec")):
        by_inode.setdefault(f.stat().st_ino, []).append(f.name)
    for names in by_inode.values():
        main = next(n for n in names if not n.startswith("k_")) if any(not n.startswith("k_") for n in names) else names[0]
        blob = (prog / main).read_bytes()
        rec = {"aliases": sorted(names), "blob_sha256": sha(blob), "blob_bytes": len(blob),
               "format": ax._gpu_blob_format(blob), "path_hits_blob": sum(blob.count(p) for p in prefixes)}
        if rec["format"] == ax.GPU_THUNK_FORMAT:
            recs = []
            for i in range(0, 100000):
                try:
                    r = gate._record(blob, i, ax, asm)
                except ValueError as e:  # "missing Riegeli record i" = past the last record
                    if "missing Riegeli record" not in str(e):
                        rec["records_error"] = repr(e)[:200]
                    break
                recs.append({"index": i, "bytes": len(r), "sha256": sha(r), "sha256_pathnorm": sha(norm(r)),
                             "path_hits": sum(r.count(p) for p in prefixes)})
            rec["records"] = recs
            native = gate._record(blob, 2, ax, asm)
            rec["native_record2_sha256"] = sha(native)
            rec["native_elfs"] = [sha(e) for e in gate._elfs(native)]
        try:
            hlo = gate._optimized_hlo(blob, ax, asm)
            hb = hlo.encode() if isinstance(hlo, str) else hlo
            rec["optimized_hlo_sha256"] = sha(hb); rec["optimized_hlo_sha256_pathnorm"] = sha(norm(hb))
            rec["optimized_hlo_path_hits"] = sum(hb.count(p) for p in prefixes)
        except Exception as e:
            rec["optimized_hlo_error"] = repr(e)[:200]
        mp = prog / (main[:-len(".xlaexec")] + ".meta")
        if mp.exists():
            try:
                m = pickle.loads(mp.read_bytes())
                rec["meta"] = {k: (getattr(m, k, None) if isinstance(getattr(m, k, None), (str, int, float, type(None)))
                                   else repr(getattr(m, k, None))[:300])
                               for k in ("hlo_sha256", "cheap_key", "key_schema", "blob_sha256", "executable_format",
                                         "retained_module", "fingerprint")}
            except Exception as e:
                rec["meta_error"] = repr(e)[:200]
        res["programs"].setdefault(prog.name, []).append(rec)
        print(prog.name, main[:12], rec.get("native_record2_sha256", "-")[:16], rec.get("path_hits_blob"), flush=True)
out.write_text(json.dumps(res, indent=1, default=str) + "\n")
