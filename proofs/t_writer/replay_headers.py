"""Reuse initialized inventories; replay only the final pure host header delta.

The numerical model and emission source must be byte-identical to the final
affected-case initialization snapshot. Every receipt records its actual older
initialization source; this does not turn an older arm into a new measurement.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from gpuwrf.io.gen2_accessor import parse_namelist
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrf_history_metadata import wrf_history_global_attributes

root,baseline,updates,output,source=map(Path,sys.argv[1:])
source=source.resolve()
candidate=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
upstream='8c9b004f4606246434dfd728237e659deca38b1f'
# Compose the qualified diagnostic source with the approved upstream train.
# This operation writes Git objects only; it never edits the immutable tree.
baseline_tree=subprocess.check_output(['git','-C',str(source),'merge-tree','--write-tree',
                                       '55c7c0ddd',upstream],text=True).strip()
assert len(baseline_tree)==40,baseline_tree
changed=subprocess.check_output(['git','-C',str(source),'diff','--name-only',baseline_tree,candidate,
                                '--','src/gpuwrf'],text=True).splitlines()
host_functions={
    'src/gpuwrf/io/wrf_history_metadata.py':{'wrf_history_global_attributes'},
    'src/gpuwrf/io/wrfout_writer.py':{'prepare_wrfout_payload','_dimension_sizes'},
    'src/gpuwrf/integration/nested_pipeline.py':{'_materialize_and_submit'},
}
metadata_file='src/gpuwrf/io/wrf_variable_metadata.py'
assert set(changed)<=set(host_functions)|{metadata_file},changed
for name in changed:
    before=subprocess.check_output(['git','-C',str(source),'show',baseline_tree+':'+name],text=True)
    after=(source/name).read_text()
    if name==metadata_file:
        def table(text):
            return ast.literal_eval(next(n.value for n in ast.parse(text).body
                                         if isinstance(n,ast.Assign)))
        old,new=table(before),table(after)
        assert all(new[k]==v for k,v in old.items())
        continue
    class HostOnly(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            return ast.Pass() if node.name in host_functions[name] else self.generic_visit(node)
    assert ast.dump(HostOnly().visit(ast.parse(before)))==ast.dump(HostOnly().visit(ast.parse(after))),name
old_manifest=json.loads((baseline.parent/'TW08_source_manifest.json').read_text())
affected=set(json.loads((baseline.parent/'TW14_cases.json').read_text()))
matrix=json.loads((root/'MATRIX.json').read_text())
output.mkdir(parents=True,exist_ok=True)
for case in matrix['cases']:
    selected=case['id'] in affected
    raw=(updates if selected else baseline)/case['id']/'inventory.json'
    digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    if not selected:
        assert digest==old_manifest['receipts'][case['id']],case['id']
    rec=json.loads(raw.read_text())
    rec['source_commit']=rec.get('source_commit',old_manifest['commit']
                                if rec['status']=='COMPLETE' else 'NOT_INITIALIZED')
    rec['reuse']=dict(original_receipt=str(raw),original_receipt_sha256=digest,
        candidate_source_commit=candidate,affected_configuration_reinitialized=selected,
        upstream_main_commit=upstream,composed_baseline_tree=baseline_tree,
        scope=f'Schema-only carry-forward. {len(affected)} affected/new configurations are reinitialized or explicitly ineligible; other initialized inventories retain their recorded source. No numerical/GPU claim.',
        host_replay='Protected model/initialization AST equals qualified55c7c0ddd composed with approved upstream main. Remaining deltas are restricted to history header/metadata and host writer functions; affected emission configurations are reinitialized. Final NetCDF globals replayed below.')
    if rec['status']=='COMPLETE':
        namelist=parse_namelist(root/case['id']/'input/namelist.input')
        for phase,domains in rec['phases'].items():
            for domain,inventory in domains.items():
                attrs=wrf_history_global_attributes(root/case['id']/f'input/wrfinput_{domain}',
                                                   namelist,domain,inventory['globals']['DT'])
                header=output/'header.nc'
                with Dataset(header,'w') as ds:ds.setncatts(attrs)
                with Dataset(header) as ds:
                    inventory['globals'].update({k:(v.tolist() if isinstance(v,np.ndarray) else
                                                    v.item() if isinstance(v,np.generic) else v)
                                                 for k in ds.ncattrs() for v in (ds.getncattr(k),)})
                header.unlink()
                inventory['serialized_bytes_scope']='Original initialization arm, before final host-header-only replay.'
    folder=output/case['id'];folder.mkdir(exist_ok=True)
    (folder/'inventory.json').write_text(json.dumps(rec,indent=2)+'\n')
print('SEALED',len(matrix['cases']),'cases;',len(affected),'affected/new configuration receipts')
