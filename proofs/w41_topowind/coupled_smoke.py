"""CPU coupled smoke of topo_wind through the REAL pipeline path (lane w41-topowind).

Loads real Swiss cases through nested_pipeline._load_domains (namelist binding,
admission, topo_wind static from wrfinput + resolved terrain) and runs N operational
steps of the release-default program on CPU (Pallas interpreted, CPU Thomas harness of
the o1 probe), census ON. Matched controls: the SAME case dir with topo_wind=0.
Arms: c02 (YSU/sfclay1) tw0/tw1/tw2, c07 (Shin-Hong) tw0/tw1. Records finiteness, guard
counts, the bound static, and how U10/V10, UST-driven low-level u/v respond where
CTOPO/CTOPO2 differ from 1 (YSU/Shin-Hong really consume them; history U10 blend).
usage: JAX_PLATFORMS=cpu python coupled_smoke.py <repo> <case_root> <out_dir> [steps]
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO, CASES, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
os.environ.setdefault("GPUWRF_CENSUS", "1")
sys.path[:0] = [str(REPO / "src"), str(REPO / "proofs/o1_nlbind")]
import jax  # noqa: E402
import numpy as np  # noqa: E402

import cpu_safe  # noqa: E402

cpu_safe.install()
from gpuwrf.backends import pallas as facade  # noqa: E402
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402

assert jax.devices()[0].platform == "cpu" and om.__file__.startswith(str(REPO))
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
facade.cuda_native = lambda: False
_real = pl.pallas_call


def _interpret(*a, **k):
    k["interpret"] = True
    k.pop("compiler_params", None)
    return _real(*a, **k)


pl.pallas_call = _interpret
ARMS = {"c02_tw0": ("c02tw1", 0), "c02_tw1": ("c02tw1", None), "c02_tw2": ("c02tw2", None),
        "c07_tw0": ("c07tw1", 0), "c07_tw1": ("c07tw1", None)}
OUT.mkdir(parents=True, exist_ok=True)
report = {"repo": str(REPO), "steps": STEPS, "census": os.environ["GPUWRF_CENSUS"], "arms": {}}
fields = {}
for arm, (case_name, override) in ARMS.items():
    if len(sys.argv) > 5 and arm not in sys.argv[5:]:
        continue
    t0 = time.time()
    rec = {}
    src = CASES / case_name / "input"
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        case = Path(tmp) / "case"
        case.mkdir()
        for f in src.iterdir():
            if f.name != "namelist.input":
                (case / f.name).symlink_to(f.resolve())
        text = (src / "namelist.input").read_text()
        if override is not None:
            text = "\n".join(f" topo_wind = {override}," if line.strip().lower().startswith("topo_wind") else line
                             for line in text.splitlines()) + "\n"
        (case / "namelist.input").write_text(text)
        cfg = pipeline.NestedPipelineConfig(case, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=1)
        _h, bundles, _m, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
    ns, carry = bundles["d01"].namelist, carries["d01"]
    static = ns.topo_wind_static
    rec["static"] = None if static is None else {
        k: (None if getattr(static, k) is None else
            dict(min=float(np.min(getattr(static, k))), max=float(np.max(getattr(static, k))),
                 ne1=int(np.sum(np.asarray(getattr(static, k)) != 1))))
        for k in static._fields}
    rec["bl_pbl_physics"] = int(ns.bl_pbl_physics)
    om._resolve_operational_suite(ns)
    cb = om.build_clock_base(ns)
    # The loaded carry shares some buffers between leaves; the production step donates its carry.
    c = jax.tree.map(lambda x: x.copy() if isinstance(x, jax.Array) else x, carry)
    t1 = time.time()
    for i in range(STEPS):
        c = om._advance_chunk_fori(c, ns, i + 1, cb, n_steps=1, cadence=int(ns.radiation_cadence_steps))
    jax.block_until_ready(c)
    rec["exec_s"] = round(time.time() - t1, 1)
    leaves = [np.asarray(x) for x in jax.tree.leaves(c)]
    rec["nonfinite_values"] = int(sum(int((~np.isfinite(x)).sum()) for x in leaves if np.issubdtype(x.dtype, np.floating)))
    census = getattr(c, "census", None)
    if census is not None:
        rec["census"] = {k: int(np.asarray(v).sum()) for k, v in census._asdict().items()
                         if hasattr(v, "shape") and np.issubdtype(np.asarray(v).dtype, np.number)}
    h = c.history_diagnostics
    fields[arm] = dict(u10=np.asarray(h.u10), v10=np.asarray(h.v10), ust=np.asarray(c.state.ustar),
                       u0=np.asarray(c.state.u[0]), v0=np.asarray(c.state.v[0]), theta0=np.asarray(c.state.theta[0]),
                       ctopo=None if static is None else np.asarray(static.ctopo),
                       ctopo2=None if static is None or static.ctopo2 is None else np.asarray(static.ctopo2))
    rec["wall_s"] = round(time.time() - t0, 1)
    report["arms"][arm] = rec
    print(arm, json.dumps(rec), flush=True)
    np.savez(OUT / f"{arm}.npz", **{k: v for k, v in fields[arm].items() if v is not None})
    (OUT / f"{arm}.json").write_text(json.dumps(rec, indent=1))
    jax.clear_caches()

# Arms may run in separate processes (memory): compare from the saved per-arm files.
for arm in ARMS:
    if arm not in fields and (OUT / f"{arm}.npz").exists():
        with np.load(OUT / f"{arm}.npz") as z:
            fields[arm] = {k: z[k] for k in z.files}
        fields[arm].setdefault("ctopo", None); fields[arm].setdefault("ctopo2", None)
        report["arms"][arm] = json.loads((OUT / f"{arm}.json").read_text())

cmp = {}
for base, other in (("c02_tw0", "c02_tw1"), ("c02_tw0", "c02_tw2"), ("c07_tw0", "c07_tw1")):
    if base not in fields or other not in fields:
        continue
    a, b = fields[base], fields[other]
    ct, ct2 = b["ctopo"], b["ctopo2"]
    one = np.ones_like(a["u10"]) if ct2 is None else ct2
    row = {}
    for k in ("u10", "v10", "ust", "theta0"):
        d = np.abs(b[k].astype(np.float64) - a[k].astype(np.float64))
        row[k] = dict(max_abs=float(d.max()), cells_differ=int(np.count_nonzero(d)))
    if ct2 is not None:
        blend = one != 1
        row["u10_blend_cells"] = int(blend.sum())
        row["u10_blend_cells_changed"] = int(np.count_nonzero(b["u10"][blend] != a["u10"][blend]))
    cmp[f"{other}_vs_{base}"] = row
report["compare"] = cmp
(OUT / "COUPLED_SMOKE.json").write_text(json.dumps(report, indent=1))
print(json.dumps(cmp, indent=1))
