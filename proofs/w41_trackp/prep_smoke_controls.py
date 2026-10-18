"""CPU prep for the Track P GPU smoke: control files from the case bases (NumPy only after the basis load).

usage: JAX_PLATFORMS=cpu python prep_smoke_controls.py <work_dir> <spec_T.json>
expects <work_dir>/basis_swiss.npz and <work_dir>/basis_prod.npz (gpuwrf surface-params basis), writes
controls/{swiss_prior,swiss_T,prod_prior,prod_A,prod_B}.npz + controls/MANIFEST.json (digests per file).
"""
import json
import sys
from pathlib import Path

from gpuwrf.surface_params.cli import expand_from_basis, load_basis
from gpuwrf.surface_params.controls import load_controls, save_controls

W, SPEC = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
out = W / "controls"
out.mkdir(exist_ok=True)


def active(manifest, dom):
    return sorted(n for n, v in manifest["domains"][dom]["controls"].items() if v["active"])


def write(name, basis_name, per_domain):
    basis, manifest = load_basis(W / basis_name)
    fields = expand_from_basis(basis, manifest, per_domain)
    path = out / f"{name}.npz"
    save_controls(path, fields, provenance={"basis": basis_name, "multipliers": json.loads(json.dumps(per_domain))})
    return {d: e["digest"] for d, e in load_controls(path)[1]["domains"].items()}


_, ms = load_basis(W / "basis_swiss.npz")
_, mp = load_basis(W / "basis_prod.npz")
spec_t = {k: ({int(c): f for c, f in v.items()} if isinstance(v, dict) else v) for k, v in SPEC.items()}
forest = {c: 1.25 for c in (1, 2, 3, 4, 5)}
shrub = {c: 0.8 for c in (6, 7, 8, 9, 10)}
digests = {
    "swiss_prior": write("swiss_prior", "basis_swiss.npz", {"d01": {n: 1.0 for n in active(ms, "d01")}}),
    # same control NAMES as swiss_prior (one program signature): unedited controls at multiplier 1
    "swiss_T": write("swiss_T", "basis_swiss.npz", {"d01": {**{n: 1.0 for n in active(ms, "d01")}, **spec_t}}),
    "prod_prior": write("prod_prior", "basis_prod.npz", {d: {n: 1.0 for n in active(mp, d)} for d in mp["domains"]}),
    "prod_A": write("prod_A", "basis_prod.npz", {d: {**{n: 1.0 for n in active(mp, d)}, "z0mvt": forest,
                                                     "albsat": 1.1} for d in mp["domains"]}),
    "prod_B": write("prod_B", "basis_prod.npz", {d: {**{n: 1.0 for n in active(mp, d)}, "z0mvt": shrub,
                                                     "laim": 0.9} for d in mp["domains"]}),
}
(out / "MANIFEST.json").write_text(json.dumps(digests, indent=1))
print(json.dumps(digests, indent=1))
