"""Relabel old CPP -P observations using CPP's original file/line markers.

The nonempty preprocessed streams must match exactly. AST identity before/after
proves this operation changes comments only; it does not retranslate science.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
WRF=Path('<USER_HOME>/src/wrf_pristine/WRF/phys')
MODULES=dict(wv_saturation='module_cam_wv_saturation',
    module_mp_cammgmp_driver='module_mp_cammgmp_driver',
    cldwat2m_micro='module_cam_mp_cldwat2m_micro',
    ndrop='module_cam_mp_ndrop',microp_aero='module_cam_mp_microp_aero',
    error_function='module_cam_error_function')


def mapping(module):
    source=WRF/(module+'.F')
    command=['cpp','-traditional-cpp','-DWRF_CHEM=0','-DEM_CORE=1',str(source)]
    old=subprocess.check_output(command[:1]+['-P']+command[1:],text=True).splitlines()
    marked=subprocess.check_output(command,text=True).splitlines()
    contents=[];line=1;file=''
    for value in marked:
        match=re.match(r'#\s+(\d+)\s+"([^"]+)"',value)
        if match:line=int(match[1]);file=match[2];continue
        if value.strip():contents.append((value,Path(file).name,line))
        line+=1
    original=[(i,value) for i,value in enumerate(old,1) if value.strip()]
    assert [v for _,v in original]==[v for v,_,_ in contents],module
    return {i:f'{file}:{line}' for (i,_),(_,file,line) in zip(original,contents,strict=True)}


def main():
    maps={name:mapping(module) for name,module in MODULES.items()};receipt={};planned={}
    # These two helper units were translated directly from the original .F.
    for namespace in ("wv_saturation","error_function"):
        module=MODULES[namespace]
        maps[namespace]={i:f"{module}.F:{i}" for i in range(1,len((WRF/(module+".F")).read_text().splitlines())+1)}
    for name in ('cloud','macro','micro','aerosol','ice_activation','error_function'):
        path=ROOT/'src/gpuwrf/physics/cammg'/(name+'.py')
        before=subprocess.check_output(['git','show','770728c02:'+str(path.relative_to(ROOT))],cwd=ROOT,text=True)
        assert ast.dump(ast.parse(before))==ast.dump(ast.parse(path.read_text())),name
        namespace=None
        output=[]
        for line in before.splitlines():
            match=re.search(r"v\['__modules'\]\.get\('([^']+)'",line)
            if match:namespace=match[1]
            match=re.search(r'(?:# WRF:|# Original WRF fatal event:)(\d+)',line)
            if match:
                source=maps[namespace][int(match[1])]
                line=line[:match.start(1)]+source+line[match.end(1):]
            output.append(line)
        after='\n'.join(output)+'\n'
        assert ast.dump(ast.parse(before))==ast.dump(ast.parse(after)),name
        planned[path]=after
        receipt[name]={'AST_identical':True,'before':hashlib.sha256(before.encode()).hexdigest(),
                       'after':hashlib.sha256(after.encode()).hexdigest()}
    for path,after in planned.items():path.write_text(after)
    (ROOT/'proofs/s2_cammg/SOURCE_LINE_IDENTITY.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
