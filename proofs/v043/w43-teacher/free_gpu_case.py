"""Free one scored GPU case's raw wrfouts (manager DISK URGENT 2026-10-21 14:25Z).

usage: python3 free_gpu_case.py RUN_DIR SCORE_DIR EXTRACT_DIR [--delete]
Preconditions (all required, else refuse rc 2): SCORE_DIR/VERDICT.json present, SCORE_DIR/integ/INTEGRITY.json pass is not
null (G-INT ran), EXTRACT_DIR/READBACK.json status PASS. Writes SCORE_DIR/GPU_WRFOUT_MANIFEST.json: every wrfout re-hashed
(sha256 must equal RUN_DIR/wrfout_sha.txt), size, netCDF Times; copies wrfout/proofs/* next to it. With --delete, removes
exactly the listed files (os.remove per file, then rmdir) - never a recursive delete.
"""
import hashlib, json, os, shutil, sys
from pathlib import Path

import netCDF4


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    run, sdir, xdir = map(Path, sys.argv[1:4])
    delete = "--delete" in sys.argv[4:]
    integ = json.loads((sdir / "integ/INTEGRITY.json").read_text())
    rb = json.loads((xdir / "READBACK.json").read_text())
    if not (sdir / "VERDICT.json").exists() or integ.get("pass") is None or rb.get("status") != "PASS":
        print("REFUSE: preconditions", (sdir / "VERDICT.json").exists(), integ.get("pass"), rb.get("status")); sys.exit(2)
    want = {}
    for line in (run / "wrfout_sha.txt").read_text().splitlines():
        h, name = line.split()
        want[Path(name).name] = h
    wdir = run / "wrfout"
    files = sorted(p for p in wdir.iterdir() if p.is_file())
    if {p.name for p in files} != set(want) or len(want) != 219:
        print("REFUSE: file set != wrfout_sha.txt", len(files), len(want)); sys.exit(2)
    rows, bad = [], []
    for p in files:
        h = sha256(p)
        with netCDF4.Dataset(p) as ds:
            times = [b"".join(t).decode() for t in ds.variables["Times"][:].data]
        rows.append({"file": p.name, "bytes": p.stat().st_size, "sha256": h, "Times": times})
        if h != want[p.name]:
            bad.append(p.name)
    if bad:
        print("REFUSE: sha mismatch", bad[:5]); sys.exit(2)
    pdir = wdir / "proofs"
    kept = []
    if pdir.is_dir():
        (sdir / "gpu_proofs").mkdir(exist_ok=True)
        for q in sorted(pdir.iterdir()):
            shutil.copy2(q, sdir / "gpu_proofs" / q.name); kept.append(q.name)
    total = sum(r["bytes"] for r in rows)
    man = {"run_dir": str(run), "receipt_sha256": sha256(run / "receipt.json"), "n_files": len(rows), "total_bytes": total,
           "sha_list_verified": True, "proofs_copied": kept, "files": rows}
    (sdir / "GPU_WRFOUT_MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
    shutil.copy2(run / "receipt.json", sdir / "gpu_receipt.json")
    freed = 0
    if delete:
        for r in rows:
            os.remove(wdir / r["file"]); freed += r["bytes"]
        if pdir.is_dir():
            for q in kept:
                os.remove(pdir / q)
            pdir.rmdir()
        wdir.rmdir()
    print(f"OK {run.name} files {len(rows)} total {total/2**30:.2f} GiB sha verified; deleted {delete} freed {freed/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
