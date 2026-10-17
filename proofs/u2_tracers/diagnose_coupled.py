"""Compare externally saved forecast operands with original WRF PBL sources.

CPU diagnostic only. The forecast source tree is immutable; no observer is
inserted in its compiled loop. Run after coupled_probe's process has exited.
"""
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys

root, forecast, original, output = map(lambda p: Path(p).resolve(), sys.argv[1:])
sys.path.insert(0, str(root / "src"))
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.coupling.physics_couplers import _mynn_column_from_state, _from_columns
from gpuwrf.tracers.pbl import tracer_pbl_tendency

assert jax.devices()[0].platform == "cpu"
receipt = json.loads(forecast.read_text())
with np.load(original) as archive:
    truth = {k: archive[k] for k in archive.files}
frames = []
for frame in receipt["frames"]:
    with gzip.open(frame["saved_carry"], "rb") as stream:
        frames.append(pickle.load(stream))


def difference(a, b, index=None):
    delta = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    where = tuple(int(i) for i in np.unravel_index(np.argmax(np.abs(delta)), delta.shape))
    result = dict(max_abs=float(np.max(np.abs(delta))), index=where,
                  rmse=float(np.sqrt(np.mean(delta * delta))),
                  nonfinite=int((~np.isfinite(delta)).sum()))
    if index is not None:
        result["delta_at_tracer_max"] = float(delta[index])
    return result


program = jax.jit(tracer_pbl_tendency)
rows = []
for step in range(1, 4):
    previous, nl = frames[step - 1]
    current, _ = frames[step]
    i = step - 1
    location = tuple(receipt["frames"][step]["tracers"]["tr17_1"]["max_index_zyx"])
    column = _mynn_column_from_state(previous.state, nl.grid)
    rho = np.asarray(_from_columns(column.rho), np.float32)
    dz = np.asarray(_from_columns(column.dz), np.float32)
    expected = truth["tendency"][i]
    literal = np.asarray(program(*(jnp.asarray(truth[k][i]) for k in
        ("tracers", "rho", "dz", "exch_h", "exch_m", "dt"))))
    geometry = np.asarray(program(jnp.asarray(previous.state.tracers), jnp.asarray(rho),
        jnp.asarray(dz), jnp.asarray(truth["exch_h"][i]), jnp.asarray(truth["exch_m"][i]),
        jnp.asarray(truth["dt"][i])))
    actual = np.asarray(current.tracer_tendencies)
    row = dict(step=step, tracer_max_zyx=location,
        tracer_max_error=receipt["frames"][step]["tracers"]["tr17_1"]["max_abs"],
        matched_original_pbl=difference(literal, expected),
        rho=difference(rho, truth["rho"][i], location),
        dz=difference(dz, truth["dz"][i], location),
        previous_tracers=difference(previous.state.tracers, truth["tracers"][i]),
        all_pbl_rate_delta=difference(actual[0], expected[0], location),
        port_geometry_original_exchange_delta=difference(geometry[0], expected[0], location),
        held_exchange_residual=difference(actual[0], geometry[0], location))
    rows.append(row)
    print(json.dumps(row), flush=True)
output.write_text(json.dumps(dict(backend="cpu", scope="raw source/input localization; not a new trajectory or GPU claim",
    source=str(root), forecast=str(forecast), original=str(original),
    diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows), indent=2) + "\n")
