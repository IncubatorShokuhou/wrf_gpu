"""Track P restart gate on the real product identity (Swiss d01, CPU): resume refuses changed/missing controls.

usage: JAX_PLATFORMS=cpu python cpu_restart_identity.py <src_root> <out.json>
Loads the case once through nested_pipeline._load_domains and evaluates runtime.restart_store.nested_identity (the
manifest identity a resumed nested/native run must reproduce exactly, restart_store:699) for:
  OFF (no controls), A, A' (A rebuilt from its file), B (other values, same names), C (other names).
Gates: OFF has no 'surface_controls' key (released identity layout unchanged); A == A'; A != B; A != OFF; A != C.
Also exercises the single-domain checkpoint digest validator on the same namelists.
"""
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

SRC, OUT = str(Path(sys.argv[1]).resolve()), sys.argv[2]
sys.path.insert(0, SRC + "/src")
sys.path.insert(0, SRC + "/proofs/o1_nlbind")
import jax  # noqa: E402

import cpu_safe  # noqa: E402

cpu_safe.install()
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import restart_store  # noqa: E402
from gpuwrf.surface_params import controls as sc  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402

assert jax.devices()[0].platform == "cpu"
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
CASE = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/w41-trackp")
_real = pl.pallas_call
pl.pallas_call = lambda *a, **k: _real(*a, **{**{x: y for x, y in k.items() if x != "compiler_params"},
                                              "interpret": True})
with tempfile.TemporaryDirectory(dir=LANE) as tmp:
    tmp = Path(tmp)
    case = tmp / "case"
    case.mkdir()
    for f in CASE.iterdir():
        (case / f.name).symlink_to(f)
    cfg = pipeline.NestedPipelineConfig(case, tmp / "out", tmp / "proof", hours=1, max_dom=1)
    hier, bundles, meta, run_start, dt, carries = pipeline._load_domains(cfg, ("d01",))
    ns = bundles["d01"].namelist
    fields = {"A": sc.expand_controls(ns, {"z0mvt": 1.3, "laim": 0.9}, device=False),
              "B": sc.expand_controls(ns, {"z0mvt": 1.3, "laim": 0.91}, device=False),
              "C": sc.expand_controls(ns, {"z0mvt": 1.3}, device=False)}
    sc.save_controls(tmp / "A.npz", {"d01": fields["A"]})
    namelists = {"OFF": ns,
                 "A": sc.bind_domain_controls(ns, {"d01": fields["A"]}, "d01"),
                 "A_file": sc.bind_domain_controls(ns, tmp / "A.npz", "d01"),
                 "B": sc.bind_domain_controls(ns, {"d01": fields["B"]}, "d01"),
                 "C": sc.bind_domain_controls(ns, {"d01": fields["C"]}, "d01")}
    ident = {k: restart_store.nested_identity(cfg, {"d01": dataclasses.replace(bundles["d01"], namelist=v)},
                                              run_start) for k, v in namelists.items()}
    refusals = {}
    for want, have in (("A", "B"), ("A", "OFF"), ("OFF", "A")):
        try:
            sc.validate_controls_restart(sc.restart_controls_digest(namelists[want]), namelists[have])
            refusals[f"{want}->{have}"] = "accepted"
        except ValueError as exc:
            refusals[f"{want}->{have}"] = f"refused: {exc}"
    sc.validate_controls_restart(sc.restart_controls_digest(namelists["A"]), namelists["A_file"])
out = {
    "identity_surface_controls": {k: v.get("surface_controls") for k, v in ident.items()},
    "checkpoint_validator": refusals,
    "gates": {
        "off_layout_unchanged": "surface_controls" not in ident["OFF"],
        "same_controls_same_identity": ident["A"] == ident["A_file"],
        "value_change_refused": ident["A"] != ident["B"],
        "added_controls_refused": ident["A"] != ident["OFF"],
        "name_change_refused": ident["A"] != ident["C"],
        "checkpoint_validator_refuses": all(v.startswith("refused") for v in refusals.values()),
    },
}
out["gates"]["PASS"] = all(out["gates"].values())
Path(OUT).write_text(json.dumps(out, indent=1))
print(json.dumps(out["gates"]))
