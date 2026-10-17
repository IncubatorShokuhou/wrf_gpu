"""Seal the completed same-critic CPU qualification delta; preserve old packet."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    packet=root/'proofs/t_icediag';receipts=packet/'delta_receipts';receipts.mkdir(exist_ok=True)
    original=packet/'QUALIFICATION.json'
    assert hashlib.sha256(original.read_bytes()).hexdigest()=='e69f806bdf621879202d8315a1744300c970ddfd906ec6c9354c51513957093c'
    def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
    assert git('rev-parse','HEAD:src/gpuwrf')=='53334b3c965fe03238d4df55ec8795dee82ee567'
    result=dict(lane='t-icediag',status='READY_FOR_SAME_CRITIC_DISCLOSED_JENSEN_EXCESS',
        created_utc=datetime.now(timezone.utc).isoformat(),source=git('rev-parse','HEAD'),
        model_tree=git('rev-parse','HEAD:src/gpuwrf'),same_critic='rv-t-icediag',
        original_packet_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        scope='CPU numerical expanded native projections, real Swiss two-step/full restart and five standard consumers. No GPU/E41/forecast fidelity/performance claim.',
        native={},frontend={},standard={},swiss={},receipts={},limitations=[])
    def archive(path,name=None):
        raw=path.read_bytes();name=name or path.name
        if path.suffix in ('.xml','.log'):name+='.gz';data=gzip.compress(raw,mtime=0)
        else:data=raw
        destination=receipts/name;destination.write_bytes(data)
        entry=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),original=str(path))
        if path.suffix in ('.xml','.log'):
            entry.update(raw_sha256=hashlib.sha256(raw).hexdigest(),raw_bytes=len(raw),
                         compression='gzip; exact original bytes preserved')
        result['receipts'][name]=entry
    def read(stem):return json.loads((a.data/(stem+'.json')).read_text())
    native_stems={50:'IC12b-mp50',51:'IC12-mp51',52:'IC12c-mp52',53:'IC12-mp53',55:'IC12b-mp55'}
    for mp,stem in native_stems.items():
        row=read(stem);assert row['mp']==mp and not row['GPU']
        if mp!=55:assert row['status']=='PASS' and not any(x['outside_frozen_tolerance'] for x in row['fields'].values())
        else:assert row['existing_native_outputs_BYTE'] and row['status']=='DISCLOSED_EXCESS'
        result['native'][str(mp)]=row
        for suffix in ('.json','.npz','.log','-resource.json'):archive(a.data/(stem+suffix))
        stem=f'IC13-mp{mp}';front=read(stem)
        assert all(x['pallas_calls']==1 for x in front['arms'].values())
        assert front['arms']['previous']['fp64_equations']==front['arms']['retained']['fp64_equations']
        result['frontend'][str(mp)]=front
        for suffix in ('.json','.log'):archive(a.data/(stem+suffix))
        stem=f'IC15-mp{mp}';nodes=list(ET.parse(a.data/(stem+'.xml')).iter('testcase'))
        assert len(nodes)==1 and not list(nodes[0])
        result['standard'][str(mp)]=dict(node=nodes[0].attrib,reused_with='IC16-comparison.json')
        for suffix in ('.xml','.log','-resource.json'):archive(a.data/(stem+suffix))
        stem=f'IC18-mp{mp}';coupled=json.loads((a.data/stem/'report.json').read_text())
        assert coupled['status']=='PASS' and coupled['restart_all_leaves_BYTE']
        assert coupled['microphysics_itimestep']==[1,2] and len(coupled['steps'])==2
        for step in coupled['steps']:
            assert step['nonfinite']==0 and not any(v for row in step['guards'] for v in row)
            assert all(step['producer_to_carry_interior_BYTE'].values())
            assert all(x['dtype']=='float32' for x in step['ice_diagnostics'].values())
        result['swiss'][str(mp)]=coupled
        archive(a.data/stem/'report.json',stem+'-report.json')
        for suffix in ('.log','-resource.json'):archive(a.data/(stem+suffix))
    comparison=read('IC16-comparison');assert comparison['status']=='PASS_ALL5_EXACT'
    result['standard_program_reuse']=comparison
    for stem in ('IC16-old','IC16-new'):
        for suffix in ('.json','.log','-resource.json'):archive(a.data/(stem+suffix))
    archive(a.data/'IC16-comparison.json')
    nodes=list(ET.parse(a.data/'IC17-composed.xml').iter('testcase'))
    assert len(nodes)==145 and all(not list(n) for n in nodes)
    result['composed_light_tests']=dict(PASS=145,scope='affected append-only State/E78/writer composition')
    for suffix in ('.xml','.log','-resource.json'):archive(a.data/('IC17-composed'+suffix))
    excess=read('IC14-jensen-excess');assert excess['outside_values']==165
    result['Jensen_strict_enclosure']=dict(status='FAIL_165_DISCLOSED_NOT_WAIVED',values=8160,
        outside=165,old_operational_outputs_BYTE=True,max_boundary_float32_steps=138,
        max_relative_excess=1.4297135713903125e-5,same_critic_assessment_required=True)
    archive(a.data/'IC14-jensen-excess.json')
    for label,expected in (('Swiss','f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d'),
                           ('PROD','6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b')):
        stem='IC19b-'+label;identity=read(stem)
        assert identity['arms']['release']['sha']==expected and 'error' not in identity['arms']['release']
        result.setdefault('FULL_default_program_identities',{})[label]=expected
        for suffix in ('.json','.log','-resource.json'):archive(a.data/(stem+suffix))
    for path in ('IC12-mp52-resource.json','IC12-mp52.log','IC12-mp55.log','IC12-mp50.log'):
        archive(a.data/path)
    for suffix in ('.json','.log','-resource.json'):
        archive(a.data/('IC19-Swiss'+suffix))
    result['superseded_arms']=dict(IC06='All five TTIR selectors used a different output order; IC13 is the actual adapter order.',
        IC12_mp52='6G arm ended at own 4.75GiB early-stop, no numerical outputs, no OOM; exact12G retry passed.',
        IC12_mp55='Cancelled wrong proof ABI order before accepted numerical receipt; corrected IC12b qualifies actual order.',
        IC19_Swiss='Instrumented CENSUS1 hash dd2e036f, not the registered default CENSUS0 program; IC19b uses exact frozen flags.')
    result['limitations']=[
        'Jensen 165 strict original enclosure excesses remain; no widened tolerance or adjusted values. Standing small-compiled-deviation policy requires SAME critic assessment.',
        'Swiss first-two-step diagnostic maxima may be zero; active original column and independent IO crops are separate evidence, not a coherent forecast trajectory.',
        'Original P3 GPU44 component has 8 disclosed FMA residuals; MP52 compiler blocker unresolved. Owner ZERO reference disposition bound separately.',
        'CPU Pallas interpretation and CUDA TTIR frontend are distinct from actual GPU execution, cubin/E41, launch/register pressure, peak VRAM, readiness and full forecast D6.']
    originals=json.loads((packet/'SOURCE_REUSE.json').read_text())['files']
    tree={line.split('\t',1)[1]:line.split()[2] for line in git('ls-tree','-r','HEAD').splitlines()}
    assert all(tree[name]==sha for name,sha in originals.items())
    result['original_producer_asset_oracle_blobs_exact']=len(originals)
    result['old_sealed_source_io_tests']=dict(distinct_PASS=433,mutant='genuine DI/rho AssertionError / zero setup errors',
        reference_IO='525 original finite field-frames, 21 active independent crops, 75 full original file SHA exact',
        scope='prior sealed packet retained; affected composition rechecked in IC17')
    (packet/'DELTA_QUALIFICATION.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status'],result['model_tree'],len(result['receipts']),'sealed receipts',flush=True)


if __name__=='__main__':main()
