"""Frozen G-INT chain on CACHED reads, so the ALISIOS reference side can be read inside a guarded session (where every
netCDF open must be an admitted target) while the candidate side (our own files) is read outside it. The frozen tools run
UNCHANGED (runpy / import of the sha-pinned copies); only `netCDF4.Dataset` is replaced by CacheDataset, which serves exactly
what those tools read:
  - every variable: dtype, shape and the 2-value summary [nanmin, nanmax] of the masked->NaN float64 array ([nan] if no
    finite value) — output_integrity only uses isfinite().any(), nanmin, nanmax of `ds[name][:]`, so it is decision-identical;
  - FULL arrays for the R4 fields (r4_eval) and for the annex item fields/frames.
Any read the cache cannot serve raises (fail closed). Equivalence vs the plain frozen chain is tested on local pairs
(results/GI09_equivalence.json) before use.

usage (JAX_PLATFORMS=cpu):
  gint_cached.py cache DIR DOMAINS OUT.pkl [--full VAR,VAR,..] [--frames-json ITEMS.json]   # inside guarded_run.py for ALISIOS
  gint_cached.py run REF_DIR CAND_DIR REF.pkl[,..] CAND.pkl[,..] OUTDIR                     # integrity + r4 + classify
  gint_cached.py annex REF_DIR CAND_DIR REF.pkl[,..] CAND.pkl[,..] OUTDIR                   # needs item caches
"""
import glob, json, os, pickle, re, runpy, sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import netCDF4

HERE = Path(__file__).resolve().parent
FZ = (HERE / "../../v043/w43-teacher/frozen").resolve()
R4_NAMES = ("ECAN", "EVC", "CANLIQ", "CANWAT", "FWET", "RAINNC", "RAINC")
_REAL = netCDF4.Dataset


def _summary(x):
    a = np.ma.asarray(x, dtype=np.float64).filled(np.nan)
    f = np.isfinite(a)
    return np.array([np.nanmin(a), np.nanmax(a)]) if f.any() else np.array([np.nan])


def cache(dirpath, domains, out, full=(), items=None):
    rec = {}
    want_items = {}
    if items:
        for it in json.load(open(items)):
            want_items.setdefault((it["dom"], it["frame"]), set()).add(it["field"])
    for dom in domains.split(","):
        for p in sorted(glob.glob(f"{dirpath}/wrfout_{dom}_*")):
            if p.endswith(".tmp"):
                continue
            fr = os.path.basename(p)[11:24]
            item_vars = want_items.get((dom, fr), set())
            if items and not item_vars:
                continue
            with _REAL(p) as ds:
                vs = {}
                for n, v in ds.variables.items():
                    numeric = np.issubdtype(v.dtype, np.number)
                    e = {"dtype": v.dtype, "shape": v.shape}
                    if numeric and (not items or n in full or n in item_vars):
                        x = v[:]                              # ONE read, exactly as the frozen tools read it
                        if not items:
                            e["summary"] = _summary(x)
                        if n in full or n in item_vars:
                            e["full"] = x
                    vs[n] = e
                rec[os.path.abspath(p)] = {"vars": vs, "attrs": list(ds.ncattrs())}
    pickle.dump(rec, open(out, "wb"), protocol=4)
    print("cached", len(rec), "files ->", out)


class _Var:
    def __init__(self, e, key):
        self.e, self.key = e, key
        self.dtype, self.shape = e["dtype"], e["shape"]

    def __getitem__(self, idx):
        if "full" in self.e:
            return self.e["full"][idx]
        if (idx == slice(None) or idx == Ellipsis) and "summary" in self.e:
            return self.e["summary"]
        raise KeyError(f"cache cannot serve {self.key}[{idx}]")


class CacheDataset:
    CACHE = {}

    def __init__(self, path, *a, **k):
        key = os.path.abspath(os.fspath(path))
        if key not in self.CACHE:
            raise FileNotFoundError(f"not cached: {key}")
        self._r, self._k = self.CACHE[key], key
        self.variables = {n: _Var(e, (key, n)) for n, e in self._r["vars"].items()}

    def ncattrs(self):
        return list(self._r["attrs"])

    def __getitem__(self, n):
        return self.variables[n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self):
        pass


def _load(pkls):
    for p in pkls.split(","):
        for k, v in pickle.load(open(p, "rb")).items():
            if k in CacheDataset.CACHE:   # merge item caches into summary caches
                CacheDataset.CACHE[k]["vars"].update({n: {**CacheDataset.CACHE[k]["vars"].get(n, {}), **e} for n, e in v["vars"].items()})
            else:
                CacheDataset.CACHE[k] = v
    netCDF4.Dataset = CacheDataset


def _runpy(tool, argv):
    sys.argv = [str(tool), *argv]
    try:
        runpy.run_path(str(tool), run_name="__main__")
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    return 0


def run(ref, cand, rp, cp, outdir, doms="d01,d02,d03"):
    """doms != all three (root-only candidates): integrity/merge/classify over those domains only; r4_eval gets an auxiliary
    integrity file whose absent domains have empty frames_checked (it then reads nothing for them)."""
    _load(rp); _load(cp)
    doms = tuple(doms.split(","))
    I = Path(outdir); I.mkdir(parents=True, exist_ok=False)
    for d in doms:
        rc = _runpy(FZ / "integrity_all/scripts/wn3_integrity.py", ["--cpu-dir", ref, "--gpu-dir", cand, "--out", str(I / f"{d}.json"),
                                                                   "--domains", d, "--all-frames"])
        print("integrity", d, "rc", rc)
    dd = {d: json.load(open(I / f"{d}.json"))["domains"][d] for d in doms}   # classify_all.sh merge (verbatim for all 3 domains)
    first = json.load(open(I / f"{doms[0]}.json")); first["domains"] = dd; first["pass"] = all(v["pass"] for v in dd.values())
    (I / "integrity_all_72h.json").write_text(json.dumps(first, indent=1) + "\n")
    r4_integ = I / "integrity_all_72h.json"
    if len(doms) < 3:
        aux = json.loads(r4_integ.read_text())
        for d in ("d01", "d02", "d03"):
            aux["domains"].setdefault(d, {"frames_checked": [], "note": "absent in root-only candidate"})
        r4_integ = I / "r4_input_integrity.json"; r4_integ.write_text(json.dumps(aux, indent=1) + "\n")
    rc = _runpy(FZ / "gate/r4_eval.py", ["--cpu-dir", ref, "--gpu-dir", cand, "--out", str(I / "r4_watch_all_72h.json"),
                                         "--integrity", str(r4_integ)])
    print("r4 rc", rc)
    sys.path.insert(0, str(FZ / "gate"))
    from integ_classes import classify, compact
    v = classify(json.load(open(I / "integrity_all_72h.json")), json.load(open(I / "r4_watch_all_72h.json")))
    json.dump(v, open(I / "classes_all_72h.json", "w"), indent=1, default=str)
    (I / "classify.log").write_text(compact(v) + "\n"); print(compact(v)[:400])
    items = []
    for it in v.get("new", []):
        m = re.match(r"(d0\d) ([A-Z0-9_]+)(?:\(companion\))?@(\d{4}-\d\d-\d\d_\d\d)", it)
        if m:
            dom, fld, t = m.groups()
            items.append({"dom": dom, "field": fld, "frame": t})
            if fld == "ACSNOM":
                items.append({"dom": dom, "field": fld, "frame": (datetime.strptime(t, "%Y-%m-%d_%H") - timedelta(hours=1)).strftime("%Y-%m-%d_%H")})
    json.dump(items, open(I / "annex_items.json", "w"), indent=1)
    print("annex items", len(items))


def annex(ref, cand, rp, cp, outdir):
    _load(rp); _load(cp)
    I = Path(outdir)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _runpy(FZ / "integrity_all/annex_postpass.py", [str(I / "classes_all_72h.json"), ref, cand])
    (I / "annex_postpass.txt").write_text(buf.getvalue()); print(buf.getvalue()); return rc


if __name__ == "__main__":
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    cmd = sys.argv[1]
    if cmd == "cache":
        a = sys.argv[2:]
        full = a[a.index("--full") + 1].split(",") if "--full" in a else ()
        items = a[a.index("--frames-json") + 1] if "--frames-json" in a else None
        cache(a[0], a[1], a[2], full=full, items=items)
    elif cmd == "run":
        run(*sys.argv[2:8])
    elif cmd == "annex":
        sys.exit(annex(*sys.argv[2:7]))
