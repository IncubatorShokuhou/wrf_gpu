"""The actual fp64 L3b caller consumes the original driver's held inputs."""
import functools
import json
from types import SimpleNamespace

from run_oracles import REFERENCE, RETAINED, load, sha

RUNNER = load('p0l3b7c_runner')


def test_original_driver_emission_is_held_in_fp64_column(tmp_path, monkeypatch):
    from gpuwrf.kernels import phys_thompson_sedimentation as sedimentation
    for name in ('fill_down', 'sediment_one_species'):
        monkeypatch.setattr(sedimentation, name, functools.partial(getattr(sedimentation, name), interpret=True))
    manifest = REFERENCE/'p0c_aero_column_manifest_r7.json'
    column = next(iter(json.loads(manifest.read_text())['columns']))
    compare = REFERENCE/'p0_gate2_compare.py'
    tol = REFERENCE/'p0c_mp28_tolerances.json'
    gate2 = REFERENCE/'gate2_tolerances.json'
    result = tmp_path/'fp64.json'
    args = SimpleNamespace(compare_module=str(compare), compare_sha256=sha(compare),
        manifest=str(manifest), manifest_sha256=sha(manifest), column=column,
        npz=str(RETAINED/'p0l3b7c_r2/columns'/f'port_column_{column}.npz'),
        wrf_output=str(RETAINED/'p0l3b7c_r2/run'/f'OUTPUT_{column}.txt'),
        tolerances=str(tol), tolerances_sha256=sha(tol),
        gate2_tolerances=str(gate2), gate2_tolerances_sha256=sha(gate2),
        state_dtype='fp64', out=str(result), repo=str(REFERENCE.parents[3]))
    assert RUNNER.run(args) == 0
    receipt = json.loads(result.read_text())
    assert receipt['verdict'] == 'PASS'
    assert receipt['emission_source'] == 'frozen original-driver held inputs; consumption/preservation checked'
