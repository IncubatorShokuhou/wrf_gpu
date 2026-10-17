"""CPU offline cost receipt on tracked real fields; NOT forecast/GPU timing."""
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

from gpuwrf.post import read_wrfout, run_products, write_products
from gpuwrf.post.registry import required_inputs

output = Path(sys.argv[1])
source = Path(__file__).resolve().parents[2]
fixtures = sorted((source/"tests/fixtures/post").glob("cpu_[0-9].nc"))
groups = [
    ["wind10", "wind-height", "cloud", "precip", "precip-type", "snow"],
    ["thermal", "instability", "srh", "freezing"],
]
results = []
options = {"heights": [50., 100.], "storm_u": 0., "storm_v": 0.}
for names in groups:
    t0, c0 = time.perf_counter(), time.process_time()
    ds = read_wrfout(fixtures, required_inputs(names))
    products = run_products(ds, names, options)
    tmp = output.parent/"post_benchmark.nc"
    write_products(products, tmp)
    size = tmp.stat().st_size
    tmp.unlink()
    results.append({"products": names, "source_dimensions": dict(ds.sizes), "samples": len(ds.Time),
        "selected_input_array_bytes": int(ds.nbytes), "output_array_bytes": int(products.nbytes), "output_file_bytes": size,
        "wall_s": time.perf_counter()-t0, "process_cpu_s": time.process_time()-c0})
import jax._src.xla_bridge as bridge
assert not bridge.backends_are_initialized()
receipt = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
    "kind": "CPU offline post cost, small real fixtures, no forecast or GPU claim",
    "jax_backend_initialized": False, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "options": options, "groups": results}
output.write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps(receipt))
