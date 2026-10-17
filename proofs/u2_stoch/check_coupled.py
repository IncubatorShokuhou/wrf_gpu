"""Three actual Swiss operational steps, guarded CPU harness, original pattern truth.

Uses the established O1 native loader/step harness and its CPU Thomas safety
shim. --allow-staging bypasses ONLY the catalog in this measured process; the
branch stays refused until the complete feature gate closes.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import sys
import pickle


def qualify_guard_control(candidate,control):
    """Manager standing policy: same field/count/location class on matched work."""
    a,b=(r["arms"]["release"] for r in (candidate,control))
    assert candidate["case"]==control["case"]
    assert candidate["exec_steps"]==control["exec_steps"]==3
    assert a["bound"]==b["bound"] and a["resolved_work"]==b["resolved_work"]
    assert a["nonfinite_values"]==b["nonfinite_values"]==0
    assert b["stochastic_disabled_control"]
    assert a["guard_events"]==b["guard_events"]
    assert a["negative_location_classes"]==b["negative_location_classes"]
    return "PASS matched release-path field/count/location-class control; pre-existing events"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tree",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--allow-staging",action="store_true")
    p.add_argument("--control",action="store_true")
    p.add_argument("--guard-control",type=Path)
    args=p.parse_args()
    tree=args.tree.resolve()
    probe=tree/"proofs/o1_nlbind/probe.py"
    source=probe.read_text()
    old='LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")'
    assert source.count(old)==1
    source=source.replace(old,f"LANE = Path({str(args.output.resolve().parent)!r})")
    hook='''
_stochastic_capture = {}
_original_advance = om._advance_chunk_fori
def _capture_stochastic(*a,**k):
    # The predecessor probe calls the private donating jit directly. Production
    # runners prepare aliases through _advance_chunk/unalias_donated_carry.
    prepared=(a[0] if any(isinstance(x,jax.core.Tracer) for x in jax.tree.leaves(a[0]))
              else om._commit_to_operational_device(a[0]))
    a = (om.unalias_donated_carry(prepared,by_buffer=True),*a[1:])
    value = _original_advance(*a,**k)
    _stochastic_capture['carry'] = value
    _stochastic_capture['namelist'] = a[1]
    return value
om._advance_chunk_fori = _capture_stochastic
'''
    if args.allow_staging:
        hook+='''
from gpuwrf.io import scheme_catalog as _catalog
from gpuwrf.io.stochastic_config import SWITCHES as _switches
_catalog._IMPLEMENTED_FEATURE_VALUES = {**_catalog._IMPLEMENTED_FEATURE_VALUES,
    **{key:frozenset({1}) for key in _switches}}
'''
    needle='_real_pallas_call = pl.pallas_call'
    assert source.count(needle)==1
    source=source.replace(needle,needle+"\n"+hook)
    # Retain the initial diagnostic state before the donating call consumes it;
    # this host snapshot is outside stepping, solely for the probe's delta.
    source=source.replace('            c = carry\n',
        '            _initial_state_for_delta = jax.tree.map(lambda x: np.array(x,copy=True),carry.state)\n'
        '            c = carry\n')
    source=source.replace('s0, s1 = carry.state, c.state',
                          's0, s1 = _initial_state_for_delta, c.state')
    if args.control:
        switches=("skebs","sppt","spp","rand_perturb","spp_conv","spp_pbl","spp_lsm","stoch_force_opt")
        source=source.replace('edits = {} if arm == "release"',
            f'edits = {dict.fromkeys(switches,"0")!r} if arm == "release"')
    os.environ.update(O1_CASE=str(args.reference.resolve()),O1_EXEC_STEPS="3",O1_INTERPRET_EXEC="1")
    sys.argv=[str(probe),str(tree),str(args.output.resolve()),"release"]
    namespace={"__file__":str(probe),"__name__":"__main__"}
    exec(compile(source,str(probe),"exec"),namespace)
    result=json.loads(args.output.read_text())
    record=result["arms"]["release"]
    assert "error" not in record,record
    assert record["nonfinite_values"]==0,record
    import jax
    import numpy as np
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.stochastic_schema import history_fields,SEED_NAMES,PATTERN_FIELDS
    box=namespace["_stochastic_capture"]
    carry,nml=box["carry"],box["namelist"]
    assert jax.devices()[0].platform=="cpu"
    from gpuwrf.diagnostics.census import GUARDS,EVENTS
    census=carry.census
    record.update(stochastic_disabled_control=args.control,
        resolved_work=dict(steps=3,dt_s=nml.dt_s,shape=[nml.grid.nz,nml.grid.ny,nml.grid.nx]),
        guard_events={f"{guard}.{event}":int(np.asarray(census.guards)[i,j])
                      for i,guard in enumerate(GUARDS) for j,event in enumerate(EVENTS)},
        negative_location_classes=np.asarray(census.f2_negative_location).tolist(),
        f2_counts=np.asarray(census.f2_counts).tolist(),
        f2_first_index=np.asarray(census.f2_first_index).tolist(),
        f2_min_negative=np.asarray(census.f2_min_negative).tolist(),
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Keep the scientific output before a qualification assertion so a control
    # or scoring fix never repeats this expensive CPU compile unnecessarily.
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    if args.control:
        assert nml.stochastic_config is None and carry.stochastic is None
        print(json.dumps(record,indent=2))
        return
    with args.output.with_suffix(".stochastic.pkl").open("wb") as f:
        pickle.dump(jax.device_get(carry.stochastic),f)
    assert all(nml.stochastic_config.produce) and nml.stochastic_config.apply_pbl
    fields=history_fields(carry.stochastic,nml.grid.nz,nml.grid.ny,nml.grid.nx)
    with Dataset(sorted(args.reference.glob("wrfout_d01_*"))[-1]) as truth:
        for name in PATTERN_FIELDS:
            expected=np.asarray(truth[name][0])
            np.testing.assert_allclose(fields[name],expected,rtol=8.e-6,
                atol=64*np.finfo(np.float32).eps*np.max(np.abs(expected)),err_msg=name)
    with Dataset(sorted(args.reference.glob("wrfrst_d01_*"))[-1]) as truth:
        for name in SEED_NAMES:
            np.testing.assert_array_equal(fields[name],truth[name][0],err_msg=name)
    record.update(staging_catalog_bypass=args.allow_staging,
        donation_preparation="production device commitment + unalias_donated_carry(by_buffer=True)",
        stochastic_config=asdict(nml.stochastic_config),
        original_WRF_final_pattern_fields=8,original_WRF_final_seed_words_exact=48,
        stochastic_clip_counts=np.asarray(carry.stochastic.clip_counts).tolist(),
        stochastic_resident_bytes=sum(np.asarray(x).nbytes for x in jax.tree.leaves(carry.stochastic)),
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        qualification="actual CPU coupled mechanics + original WRF patterns/seeds; full forecast fidelity/GPU pending")
    record["guard_qualification"]=("PASS zero events" if record["guard_total"]==0
        else qualify_guard_control(result,json.loads(args.guard_control.read_text()))
        if args.guard_control and args.guard_control.is_file() else "PENDING matched release-path control")
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(record,indent=2))


if __name__=="__main__":
    main()
