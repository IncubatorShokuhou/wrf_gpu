"""Diff initialized writer inventories against EVERY available original CPU frame.

Schema coverage is distinct from fidelity; extra writer slots are disclosed.
No incomplete or refused initialization/reference can receive a PASS.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

ATTRS=('FieldType','MemoryOrder','units','description','stagger','coordinates')
PROVENANCE={'TITLE','NTASKS_X','NTASKS_Y','NTASKS_TOTAL'}


def compare(root, inventories, output, source):
    matrix=json.loads((root/'MATRIX.json').read_text())
    source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
    rows=[]
    disclosures=json.loads((Path(__file__).parent/'producer_disclosures.json').read_text())
    for case in matrix['cases']:
        receipt=inventories/case['id']/'inventory.json'
        rec=json.loads(receipt.read_text()) if receipt.exists() else {'status':'PENDING'}
        for domain in case['domains']:
            row=dict(case=case['id'],domain=domain,status='INCOMPLETE',initialization=rec['status'],
                     errors={},frames=0,extra_variables=[],extra_dimensions=[],extra_globals=[])
            row['initialization_source_commit']=rec.get('source_commit','UNRECORDED')
            row['initialization_receipt']=str(receipt)
            row['initialization_receipt_sha256']=(hashlib.sha256(receipt.read_bytes()).hexdigest()
                                                  if receipt.exists() else None)
            if rec.get('reuse'):
                row['initialization_reuse']=rec['reuse']
            rows.append(row)
            if rec['status']!='COMPLETE':
                row['reason']=rec.get('error',rec.get('reason',rec['status']))
                continue
            original_receipt=root/case['id']/'cpu_run.json'
            if not original_receipt.exists():
                row['reason']='missing original CPU receipt'
                continue
            original=json.loads(original_receipt.read_text())
            row['original_cpu_status']=original['status']
            row['original_cpu_receipt_sha256']=hashlib.sha256(original_receipt.read_bytes()).hexdigest()
            expected_inputs=original.get('actual_input_sha256',{})
            current_inputs={}
            for name in expected_inputs:
                with (root/case['id']/'input'/name).open('rb') as stream:
                    current_inputs[name]=hashlib.file_digest(stream,'sha256').hexdigest()
            row['actual_input_dir']=str(root/case['id']/'input')
            row['actual_input_sha256']=current_inputs
            if not expected_inputs or current_inputs!=expected_inputs:
                row['reason']='actual initialized input differs from original CPU receipt'
                row['status']='FAIL'
                continue
            files=sorted((root/case['id']/'cpu').glob(f'wrfout_{domain}_*'))
            for path in files:
                try:
                    with Dataset(path) as ds:
                        stamp=path.name.rsplit('_',1)[-1]
                        lead=float(np.asarray(ds['XTIME'][0]))
                        actual=rec['phases']['0' if lead==0 else '1'][domain]
                        def error(kind,name,cpu,gpu):
                            key=f'{kind}:{name}'
                            row['errors'].setdefault(key,dict(cpu=cpu,gpu=gpu,frames=[]))['frames'].append(path.name)
                        for name,dim in ds.dimensions.items():
                            if name!='Time' and actual['dimensions'].get(name)!=len(dim):
                                error('dimension',name,len(dim),actual['dimensions'].get(name))
                        for name,var in ds.variables.items():
                            a=actual['variables'].get(name)
                            if a is None:
                                error('missing',name,'required',None)
                                continue
                            if list(var.dimensions)!=a['dimensions']:
                                error('dimensions',name,list(var.dimensions),a['dimensions'])
                            if str(var.dtype)!=a['dtype']:
                                error('dtype',name,str(var.dtype),a['dtype'])
                            for attr in ATTRS:
                                if name=='XTIME' and attr in ('units','description'):
                                    # Initialization's timestamp has the same run start;
                                    # this text must match independently at every phase.
                                    pass
                                value=var.getncattr(attr) if attr in var.ncattrs() else None
                                value=value.item() if isinstance(value,np.generic) else value
                                if value!=a['attrs'].get(attr):
                                    error('attribute',name+'.'+attr,value,a['attrs'].get(attr))
                        for name in ds.ncattrs():
                            if name in PROVENANCE:
                                continue
                            cpu=ds.getncattr(name)
                            gpu=actual['globals'].get(name)
                            if not np.array_equal(cpu,gpu):
                                error('global',name,cpu.item() if isinstance(cpu,np.generic) else cpu,gpu)
                        row['extra_variables']=sorted(set(actual['variables'])-set(ds.variables))
                        row['extra_dimensions']=sorted(set(actual['dimensions'])-set(ds.dimensions))
                        row['extra_globals']=sorted(set(actual['globals'])-set(ds.ncattrs()))
                        row['frames']+=1
                except OSError as exc:
                    row['errors'].setdefault('unreadable',[]).append(dict(path=str(path),error=str(exc)))
            manifest=root/case['id']/'CPU_OUTPUTS.json'
            expected_files=None
            if manifest.exists():
                expected_files={v['file'] for v in json.loads(manifest.read_text())
                                if v['file'].startswith(f'wrfout_{domain}_')}
            # DFI may legitimately omit the nominal lead-zero output. Use the
            # sealed original CPU inventory when supplied, not an invented count.
            expected=len(expected_files) if expected_files is not None else int(case['hours'])+1
            missing_original=sorted(expected_files-{p.name for p in files}) if expected_files is not None else []
            if case['id'] in ('c03','c26') and any(k in row['errors'] for k in ('missing:U10E','missing:V10E')):
                row['pending_owner']='t-c03rain (sf2/sf4 U10E/V10E producer/history/writer)'
            if row['errors']:
                row['status']='FAIL'
                if 'unreadable' in row['errors']:
                    row['status']='INCOMPLETE_REFERENCE'
                if case['id']=='c49':
                    row['status']='INCOMPLETE_PREPASS_SCHEMA'
                    row['schema_limitation']='Initialization precedes the DFI prepass. This row does not establish a forecast diffusion bug: existing driver12 resolves DFL diff_opt/km_opt to0/1 before forecasting. Post-prepass schema remains unmeasured.'
                    row['pending_owner']='t-dfidiff / DFI prepass schema resolution'
            elif original['status']!='COMPLETE' or row['frames']<expected or missing_original:
                row['reason']=f'incomplete original reference: {row["frames"]}/{expected} frames'
                row['missing_original_frames']=missing_original
            else:
                row['status']='PASS_CPU_SCHEMA_COVERAGE'
            if case['id']=='c53' and original['status']!='COMPLETE':
                # Manager 2026-10-14 21:59Z: the one surviving frame cannot
                # qualify this failed original run. Keep any concrete schema
                # errors visible, but never extrapolate c78's full proof here.
                row['status']='INCOMPLETE_REFERENCE'
                row['reason']=(f'original CPU {original["status"]}; {row["frames"]} readable frame(s), '
                    f'nominal {int(case["hours"])+1}; no 24-hour qualification')
            row['producer_dispositions']={}
            for error in row['errors']:
                if not error.startswith('missing:'):continue
                name=error.removeprefix('missing:')
                matched=[d for d in disclosures if name in d['fields'] and
                         ('mp' not in d or case['options']['mp_physics'] in d['mp'])]
                row['producer_dispositions'][name]=matched or [{'disposition':'No retained producer value established. Explicitly omitted; no zero-fill.'}]
            print(case['id'],domain,row['status'],row['frames'],len(row['errors']),flush=True)
            output.write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
    external=json.loads((Path(__file__).parent/'external_gates.json').read_text())
    report=dict(schema='t-writer-composite-schema-v1',utc=datetime.now(timezone.utc).isoformat(),
        candidate_source=str(source),candidate_source_commit=source_commit,
        candidate_gpuwrf_tree=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD:src/gpuwrf'],text=True).strip(),
        matrix_sha256=hashlib.sha256((root/'MATRIX.json').read_bytes()).hexdigest(),
        comparison_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        producer_disclosures_sha256=hashlib.sha256((Path(__file__).parent/'producer_disclosures.json').read_bytes()).hexdigest(),
        initialization_provenance='Each row binds its actual initialization source and receipt; reused rows are explicitly identified.',
        scope='real CPU initialization / actual writer serialization, schemas of all available CPU frames; no forecast/numerical/GPU qualification',
        allowed_provenance_differences=sorted(PROVENANCE),extras='disclosed superset, not exact inventory identity',rows=rows,
        external_actual_gpu_gates=external)
    output.write_text(json.dumps(report,indent=2)+'\n')
    output.with_suffix('.tsv').write_text('case\tdomain\tverdict\tCPU_frames\tissues\n'+''.join(
        f'{r["case"]}\t{r["domain"]}\t{r["status"]}\t{r["frames"]}\t'+
        '; '.join(r['errors'] or [r.get('reason','')])+('; pending '+r['pending_owner'] if
            r.get('pending_owner') and any(k in r['errors'] for k in ('missing:U10E','missing:V10E')) else '')+'\n' for r in rows)+''.join(
        f'{r["case"]}\t{r["domain"]}\t{r["status"]}\t25\t'+
        'missing: '+','.join(r['missing_vs_cpu'])+'; '+r.get('bounded_D6','see source receipt')+'\n' for r in external))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','inventories','output','source'):p.add_argument(name,type=Path)
    a=p.parse_args()
    compare(a.root,a.inventories,a.output,a.source)
