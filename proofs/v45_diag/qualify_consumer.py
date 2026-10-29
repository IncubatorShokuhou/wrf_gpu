"""Transport retained pristine-WRF MYNN columns through the offline consumer.

No original solver or forecast is rerun. This checks the consumer against
already pinned original outputs, not the coupled GPU producer's qualification.
"""
import hashlib
import json
from pathlib import Path
import resource
import time

from netCDF4 import Dataset
import numpy as np

from gpuwrf.optimize import diagnostics as D


def qualify(reference_dir, output_dir):
    reference_dir, output_dir = Path(reference_dir), Path(output_dir)
    output_dir.mkdir(exist_ok=False)
    reference = json.loads((reference_dir / "result.json").read_text())
    results = {}
    for case, info in reference["cases"].items():
        npz = reference_dir / (case + ".npz")
        before = hashlib.sha256(npz.read_bytes()).hexdigest()
        assert before == info["output_sha256"], (case, "original reference SHA")
        with np.load(npz) as saved:
            values = {name: saved["dqke" if name == "Dtke" else name.lower()] for name in D.METADATA}
        nc, output = output_dir / (case + ".nc"), output_dir / (case + ".json")
        n, nz = values["Dtke"].shape
        with Dataset(nc, "w") as ds:
            for name, size in (("Time", 1), ("DateStrLen", 19), ("bottom_top", nz),
                               ("bottom_top_stag", nz+1), ("south_north", 1), ("west_east", n)):
                ds.createDimension(name, size)
            ds.GRID_ID = 3
            ds.SIMULATION_START_DATE = "2026-02-27_00:00:00"
            ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array([list("2026-02-27_01:00:00")], "S1")
            for name, data in values.items():
                staggered = D.METADATA[name][2] == "Z"
                zdim = "bottom_top_stag" if staggered else "bottom_top"
                var = ds.createVariable(name, "f4", ("Time", zdim, "south_north", "west_east"))
                var.units = D.METADATA[name][1]
                data = np.pad(data, ((0, 0), (0, 1))) if staggered else data
                var[:] = data.T[None, :, None, :]
        # Geometry/time here are only an IO carrier for the original columns;
        # no claim about real north/sea sites, forecast time or cloud profiles.
        points = [dict(id=f"original_column_{i}", region="north_windward", domain="d03", j=0, i=i) for i in range(n)]
        started = time.perf_counter()
        report = D.write_profile_report([nc], points, output)
        wall = time.perf_counter() - started
        by_id = {row["point"]: row["metrics"] for row in report["rows"]}
        ref = {name: value.astype(np.float64) for name, value in values.items()}
        residual = ref["Dtke"] - (ref["qSHEAR"] + ref["qBUOY"] + ref["qWT"] - ref["qDISS"])
        assert not np.count_nonzero(residual[:, -1]), "original unused top"
        for i in range(n):
            m = by_id[f"original_column_{i}"]
            for name in values:
                np.testing.assert_array_equal(m[name]["value"], ref[name][i], err_msg=f"{case}/{i}/{name}")
            np.testing.assert_array_equal(m["tke_budget_residual"]["value"], residual[i, :-1])
        maximum = max(m["tke_budget_residual"]["max_abs"] for m in by_id.values())
        assert maximum == info["closure_reference_only"]["max_abs"], (case, "frozen original closure maximum")
        assert hashlib.sha256(npz.read_bytes()).hexdigest() == before
        results[case] = dict(reference_sha256=before, columns=n, levels=nz, exact_term_profiles=21*n,
                             exact_residual_profiles=n, max_abs_residual_m2_s3=maximum,
                             extract_wall_s=wall, sample_bytes=report["sample_bytes"], report_bytes=output.stat().st_size)
        nc.unlink()
        output.unlink()
    summary = dict(scope="Read-only consumer against retained pristine-WRF outputs; no new producer/forecast qualification",
                   cases=results, maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   caches="NONE", gpu=False)
    (output_dir / "receipt.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


if __name__ == "__main__":
    import sys
    print(json.dumps(qualify(*sys.argv[1:])))
