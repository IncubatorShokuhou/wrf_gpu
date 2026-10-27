"""Causal control: hold the ON apply path fixed, change only operand constness.

Compare one compiled frozen-default ON result with the already measured OFF
leaf hashes. No numerical reference is recomputed; no physics code is edited.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--reference", required=True)
p.add_argument("--out", required=True)
p.add_argument("--native-reference", type=Path, help="retained OFF raw arrays; do not rerun reference")
p.add_argument("--cold", action="store_true")
a = p.parse_args()
root = Path(a.repo)
sys.path[:0] = [str(root / "src"), str(root / "proofs/w44_lsmparams")]
import jax
import jax.numpy as jnp
import numpy as np
from columns import build
from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
from gpuwrf.runtime.operational_mode import OperationalNamelist, build_clock_base
from gpuwrf.surface_params import controls as sc

assert jax.devices()[0].platform == "cpu"
reference = json.loads(Path(a.reference).read_text())
if a.native_reference:
    with np.load(a.native_reference, allow_pickle=False) as data:
        reference_hashes = [hashlib.sha256(data[f"OFF_{i}"].tobytes()).hexdigest() for i in range(48)]
    names = sorted(set(sc.LSM_CONTROLS) - {"rsmin"})
else:
    assert reference["cold"] == a.cold
    reference_hashes = reference["output_sha256"][0]
    names = reference["names"]
land, forcing, static, ep, rp, fixture_nl, rows = build(cold=a.cold)
dt = rows[0]["dt"]
frozen = sc.prior_controls(fixture_nl, names, device=False)
nl = OperationalNamelist(grid=None, tendencies=None, metrics=None, dt_s=dt,
                         use_noahmp=True, noahmp_static=static, noahmp_land=land,
                         noahmp_energy_params=ep, noahmp_rad_params=rp,
                         surface_controls=sc.prior_controls(fixture_nl, names))
cb = build_clock_base(nl)


def program(land, forcing, nl, cb):
    s, ep, rp = nl.noahmp_static, nl.noahmp_energy_params, nl.noahmp_rad_params
    values = sc.traced_controls(nl, cb)
    if values is not None:
        values = {name: jnp.asarray(value) for name, value in frozen.items()}
        s, ep, rp = sc.apply_surface_controls(s, ep, rp, values)
    return noah_mp_step(land, forcing, s, dt, energy_params=ep, rad_params=rp,
                        options=nl.noahmp_options, return_diag=True)


fn = jax.jit(program)
result = fn(land, forcing, nl, cb)
arrays = [np.asarray(v) for v in jax.tree.leaves(result)]
assert all(np.isfinite(v).all() for v in arrays)
hashes = [hashlib.sha256(v.tobytes()).hexdigest() for v in arrays]
same = [x == y for x, y in zip(hashes, reference_hashes, strict=True)]
receipt = {"source": str(root), "names": names, "reference": a.reference,
           "reference_sha256": hashlib.sha256(Path(a.reference).read_bytes()).hexdigest(),
           "platform": "cpu", "gpu_executed": False, "output_leaves": len(same),
           "cold": a.cold, "native_reference": str(a.native_reference) if a.native_reference else None,
           "frozen_default_equal_leaves": sum(same), "frozen_default_byte_exact": all(same),
           "dynamic_default_equal_leaves": reference["default_equal_leaves"],
           "output_sha256": hashes,
           "intervention": "only ON control operands captured as constants; same apply path and unchanged model equations"}
Path(a.out).write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k != "output_sha256"}), flush=True)
assert all(same), "constant substitution did not explain all differences; association class NOT proven"
