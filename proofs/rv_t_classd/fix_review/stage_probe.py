from pathlib import Path
import importlib.util,sys,json,hashlib,numpy as np
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/fix_review/snapshot')
OUT=ROOT.parent
stage=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/r1')
spec=importlib.util.spec_from_file_location('stage',ROOT/'proofs/t_classd/stage_c08_radar_cpu.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
reference=module.REFERENCE
names=('namelist.input','wrfinput_d01','wrfbdy_d01')
before={n:module.sha(reference/'input'/n) for n in names}
module.main(stage)
after={n:module.sha(reference/'input'/n) for n in names}
assert before==after
sealed=json.loads((reference/'cpu_run.json').read_text())['input_sha256']
assert before=={n:sealed[n] for n in names}
for n in names:
    assert (stage/n).is_file() and not (stage/n).is_symlink()
    assert (stage/n).stat().st_ino!=(reference/'input'/n).stat().st_ino
links={p.name:str(p.resolve()) for p in stage.iterdir() if p.is_symlink()}
assert all(Path(p).is_relative_to(module.PRISTINE.resolve()) for p in links.values())
# Genuine alias negative control: the function must refuse before following a symlink.
protected=OUT/'protected';protected.write_bytes(b'UNCHANGED')
alias=OUT/'write_alias';alias.symlink_to(protected)
try:
    module.write_private(alias,b'MUTANT')
    raise AssertionError('aliased write accepted')
except SystemExit as e:refusal=str(e)
assert protected.read_bytes()==b'UNCHANGED'
# Verify the retained rerun, its actual measured input files and original executable lineage.
receipt=json.loads((ROOT/'proofs/t_classd/C08R_CPU_RUN.json').read_text())
run=Path('<USER_HOME>/wrf_gpu2_lanes/t-classd/c08r2/run')
for name in names:
    entry=receipt['inputs'][name]
    assert not (run/name).is_symlink()
    assert module.sha(run/name)==entry['used_sha256']==entry['measured_sha256']
    assert before[name]==entry['original_sealed_sha256']
frames={name:module.sha(run/name) for name in receipt['wrfout_sha256']}
assert frames==receipt['wrfout_sha256']
old=json.loads(Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/RV10_c08_reference_mutation.json').read_text())['producer_claim']
assert frames==old['wrfout_sha256']
a=(module.PRISTINE/'main/wrf.exe').read_bytes();b=(run/'wrf.exe').read_bytes()
reconstructed=bytearray(a)
for patch in receipt['executable']['patches']:
    at=patch['offset'];oldpath=patch['old'].encode();newpath=patch['new'].encode()
    assert a[at:at+len(oldpath)]==oldpath
    reconstructed[at:at+len(oldpath)]=newpath+b'\0'*(len(oldpath)-len(newpath))
assert bytes(reconstructed)==b
assert hashlib.sha256(a).hexdigest()==receipt['executable']['original_sha256']
assert hashlib.sha256(b).hexdigest()==receipt['executable']['private_sha256']
record=dict(reference_before=before,reference_after=after,foreign_bytes_unchanged=True,
private_inputs_nonaliased=True,table_links=len(links),table_links_resolve_into_pristine=True,
alias_write_refused=refusal,protected_bytes_unchanged=True,retained_rerun_inputs_measured_sha_exact=True,
retained_rerun_frames_sha_exact=len(frames),frames_byte_identical_to_superseded_rerun=len(frames),
executable_only_declared_path_patches=True,attribution_hypothesis_disclosed='HYPOTHESIS' in (ROOT/'proofs/t_classd/MORR_REFL_OUTLIERS_ORACLE.json').read_text())
assert record['attribution_hypothesis_disclosed']
(OUT/'RV17_staging.json').write_text(json.dumps(record,indent=2)+'\n')
print('STAGING/PROVENANCE PASS',record)
