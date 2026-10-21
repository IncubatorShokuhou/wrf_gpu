"""G-INT item tables for the qualification note + per-case deletion receipts with the correct G-INT wording.

usage: python3 gint_tables.py OUT_MD SCORE_ROOT NAME ...   (reads SCORE_ROOT/NAME/integ/{INTEGRITY.json,annex_postpass.txt})
       python3 gint_tables.py --deletion-receipt SCORE_DIR   (writes SCORE_DIR/DELETION_RECEIPT.json; manifest bytes untouched)
FAIL items stay FAIL per TEACHER_PREREG v3 §3 (annex (i)-(iv) only); nothing here reinterprets them.
"""
import hashlib, json, re, sys
from pathlib import Path

ROW = re.compile(r"^\s+(d0\d) (\S+?)\s*@(\S+?):\s*(.*?)\s*->\s*(EXPLAINED|REAL FAIL)\s*$")


def items(sdir):
    out = []
    for line in (sdir / "integ/annex_postpass.txt").read_text().splitlines():
        m = ROW.match(line)
        if m:
            out.append(dict(zip(("domain", "field", "frame", "detail", "annex"), m.groups())))
    return out


def notes(sdir):
    j = json.loads((sdir / "integ/INTEGRITY.json").read_text())
    return j.get("new", [])


def gint(sdir):
    j = json.loads((sdir / "integ/INTEGRITY.json").read_text())
    return {"pass": j.get("pass"), "registry_verdict": j.get("registry_verdict"), "annex": j.get("annex")}


def deletion_receipt(sdir):
    man = sdir / "GPU_WRFOUT_MANIFEST.json"
    m = json.loads(man.read_text())
    g = gint(sdir)
    v = json.loads((sdir / "VERDICT.json").read_text())
    rec = {"schema": "w43-teacher.gpu_deletion_receipt.v1", "case": sdir.name,
           "basis": "D6 scored; G-INT all-frame chain COMPLETED (verdict below, PASS or FAIL); GPU teacher extract readback PASS; "
                    "per-file sha256 re-verified vs the run's wrfout_sha.txt and Times recorded before deletion",
           "d6_bounded_worst": {d: x["bounded_worst"] for d, x in v["domains"].items()}, "case_teacher": v["case_teacher"],
           "gint": g, "manifest": man.name, "manifest_sha256": hashlib.sha256(man.read_bytes()).hexdigest(),
           "n_files": m["n_files"], "total_bytes": m["total_bytes"],
           "wording_correction": "the MAILBOX DISK line said 'after D6+G-INT+extract readback PASS'; correct reading: D6 + G-INT "
                                 "completed (G-INT verdict as recorded here) + extract readback PASS"}
    (sdir / "DELETION_RECEIPT.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def main():
    if sys.argv[1] == "--deletion-receipt":
        r = deletion_receipt(Path(sys.argv[2])); print(r["case"], r["gint"]["pass"], r["manifest_sha256"][:12]); return
    out, root, names = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    L = ["| case | G-INT | domain | field | frame (valid UTC) | annex (i)-(iv) check | registry note (frozen classify) | annex result |",
         "|---|---|---|---|---|---|---|---|"]
    S = ["| case | G-INT pass | REAL FAIL items | annex-EXPLAINED items |", "|---|---|---|---|"]
    for n in names:
        sd = root / n
        g, it = gint(sd), items(sd)
        real = [i for i in it if i["annex"] == "REAL FAIL"]
        S.append(f"| {n} | {g['pass']} | {len(real)} | {len(it) - len(real)} |")
        nt = notes(sd)
        for i in it:
            key = f"{i['domain']} {i['field']}@{i['frame']}"
            note = "; ".join(x[len(key):].strip(" ()") for x in nt if x.startswith(key + " ")) or "-"
            note = note.replace("|max|", "max abs")
            det = re.sub(r"\s+", " ", i["detail"]).replace("|", ";")
            L.append(f"| {n} | {'PASS' if g['pass'] else 'FAIL'} | {i['domain']} | {i['field']} | {i['frame']} | {det} | {note} | {i['annex']} |")
    out.write_text("\n".join(S) + "\n\n" + "\n".join(L) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
