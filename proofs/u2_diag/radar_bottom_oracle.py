"""Additional pristine cases with active melting snow at the bottom level."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

from radar_oracle import build, execute


def generate(source, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = build(source, output / "build")
    original = Path(__file__).with_name("radar_oracle.npz")
    with np.load(original) as data:
        base = data["inputs"][31].copy()  # mixed rain/snow/graupel warm column.
    a = np.repeat(base[None], 8, axis=0)
    for i, bottom_snow in enumerate((1e-4, 3e-4, .001, 2e-6, 1e-4, 1e-4, 1e-4, 1e-4)):
        a[i, 4] = np.linspace(bottom_snow, .001, a.shape[-1])
        a[i, 2, :10] = .001
    a[4, 4, 9:13] = 0  # ice at melt-start may be graupel only.
    a[5, 2, :8] = 0  # melt-start remains caller/whole-column dependent.
    a[6, 2] = 0  # no rain -> no bright-band trigger.
    a[7, 8] = 255  # no warm layer -> no bright band.
    bottom = execute(output / "build/oracle", a, ke_diag=1)
    full = execute(output / "build/oracle", a)
    np.testing.assert_array_equal(bottom[:, 0], full[:, 0])
    fixture = output / "radar_bottom_oracle.npz"
    np.savez_compressed(fixture, inputs=a, bottom=bottom)
    manifest.update(fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                    original_fixture_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                    tolerance={"max_absolute_dbz": 5e-5},
                    scope="original calc_refl10cm bottom-only with active wet snow; whole-column melting detection preserved",
                    census={"bottom_active_snow_columns": int((a[:, 4, 0] > 1e-6).sum()),
                            "warm_raining_bottom_columns": int(((a[:, 8, 0] > 273.15) & (a[:, 2, 0] > 1e-12)).sum())})
    (output / "radar_bottom_oracle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("<USER_HOME>/src/wrf_pristine/WRF"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.output)
