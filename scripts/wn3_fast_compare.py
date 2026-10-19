"""Run the frozen scripts/compare_wrfout_grid.py (entry point kept for wn3_score and frozen scorer copies).

History: this wrapper used to hold one read-only handle per wrfout for the whole run (E69: a netCDF-4 open of a
375-variable WRF history file costs ~30-60 ms of header parsing). Held handles cost ~18 MB native memory per open file
plus the per-variable chunk cache (0227 d02 24 h, 50 files: 3.2 GB) and OOMed 72 h scores (146 files per domain,
w41-swddirscreen 2026-10-18). The comparator now opens ONE frame pair at a time and streams the fields through in a
few passes (see build_report), so this wrapper adds nothing but the chunk-cache setting. Arguments and outputs are
the same as calling the comparator directly.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import netCDF4

ROOT = Path(__file__).resolve().parents[1]
# Full-variable reads bypass the cache; only one frame pair is open at a time.
netCDF4.set_chunk_cache(1 << 20, 521, 0.75)

spec = importlib.util.spec_from_file_location("compare_wrfout_grid", ROOT / "scripts/compare_wrfout_grid.py")
compare = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compare
spec.loader.exec_module(compare)

if __name__ == "__main__":
    raise SystemExit(compare.main(sys.argv[1:]))
