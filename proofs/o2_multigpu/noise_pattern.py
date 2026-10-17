"""Where do decomposed-vs-reference round-off differences sit? (lane o2-multigpu M3 analysis, NumPy only)

  python3 noise_pattern.py DIR arm_2x1_h25_h25_s2_n1_k1 [...]
For each saved arm state (arm_*.npz + .json) vs DIR/reference_<tag>.npz: per differing leaf, the x columns with
differences, their residues mod 8 / mod 4, the tile they belong to, distance from that tile's local edges.
"""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
for name in sys.argv[2:]:
    rep = json.loads((root / f"{name}.json").read_text())
    tag = name.split("_s", 1)[1]
    ref = np.load(root / f"reference_s{tag}.npz")
    got = np.load(root / f"{name}.npz")
    tiles = rep["spec"]["tiles"]
    print(f"== {name}: tiles {[(t['own_x'], t['ext_x']) for t in tiles]}")
    allx = []
    for key in sorted(got.files, key=lambda k: int(k[1:])):
        a, b = got[key], ref[key]
        if a.shape != b.shape or a.ndim < 2 or np.array_equal(a, b, equal_nan=True):
            continue
        d = np.abs(a.astype(np.float64) - b.astype(np.float64)).reshape(-1, *a.shape[-2:]).max(axis=0)
        xs = np.unique(np.argwhere(d > 0)[:, 1])
        allx.extend(xs.tolist())
        rel = float(d.max() / (np.abs(b).max() or 1.0))
        print(f"  {key} {a.shape} rel {rel:.2g} x {xs[:12].tolist()}{'...' if len(xs) > 12 else ''}")
    xs = np.array(sorted(set(allx)))
    if xs.size:
        owner = [next(i for i, t in enumerate(tiles) if t["own_x"][0] <= x < t["own_x"][1] + (1 if x == rep["case"]["nx"] else 0))
                 for x in xs]
        local = [int(x - tiles[o]["ext_x"][0]) for x, o in zip(xs, owner)]
        print(f"  columns: {xs.tolist()}")
        print(f"  local x in owning tile: {local}")
        print(f"  local x mod 8: {sorted(set(l % 8 for l in local))}  global x mod 8: {sorted(set(int(x) % 8 for x in xs))}")
