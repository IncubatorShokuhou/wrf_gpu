"""DLF2 (mp11) history proof: original CPU-WRF c89 frames + pristine source audit (CPU, read-only).

Writes proofs/t_cammgws/DLF2_CPU_ALL_FRAMES.json and DLF2_SOURCE_AUDIT.json.
usage: build_dlf2_proof.py TREE CPU_DIR PRISTINE_WRF
"""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


def main():
    tree,cpu,wrf=map(Path,sys.argv[1:4])
    sys.path.insert(0,str(tree/'src'))
    from gpuwrf.io.netcdf_lock import Dataset
    frames=[];fields=None;glob_attrs=None
    for path in sorted(cpu.glob('wrfout_d01_*')):
        with Dataset(path) as ds:
            var=ds.variables['DLF2'];bits=np.asarray(var[:],np.float32).view(np.uint32)
            meta=dict(dimensions=list(var.dimensions),dtype=str(var.dtype),
                      **{k:str(var.getncattr(k)) for k in var.ncattrs()})
            attrs={k:str(ds.getncattr(k)) for k in ('MP_PHYSICS','SHCU_PHYSICS','BL_PBL_PHYSICS','CU_PHYSICS',
                                                   'SF_SFCLAY_PHYSICS','SF_SURFACE_PHYSICS') if k in ds.ncattrs()}
        fields=fields or meta;glob_attrs=glob_attrs or attrs
        assert meta==fields and attrs==glob_attrs,(path,'metadata drift')
        frames.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           shape=list(bits.shape),bits=int(bits.flat[0]),all_equal_bits=bool(np.all(bits==bits.flat[0])),
                           nonzero_bits=int(np.count_nonzero(bits))))
    proof=dict(case='c89',source=str(cpu),global_attributes=glob_attrs,field=fields,frame_count=len(frames),
               all_frames_positive_zero=all(f['bits']==0 and f['all_equal_bits'] for f in frames),frames=frames)
    (tree/'proofs/t_cammgws/DLF2_CPU_ALL_FRAMES.json').write_text(json.dumps(proof,indent=1)+'\n')
    # Every pristine source line naming dlf2 (original .F/.Registry only, not generated .f90/.inc).
    sites=[]
    for path in sorted([*wrf.glob('phys/*.F'),*wrf.glob('dyn_em/*.F'),*wrf.glob('share/*.F'),
                        *wrf.glob('frame/*.F'),*wrf.glob('Registry/*')]):
        if not path.is_file():continue
        for number,line in enumerate(path.read_text(errors='replace').splitlines(),1):
            if re.search(r'\bdlf2(_3d)?\b',line,re.I):
                sites.append(dict(file=str(path.relative_to(wrf)),line=number,text=line.strip()[:200]))
    files=sorted({s['file'] for s in sites})
    audit=dict(pristine=str(wrf),site_count=len(sites),files={f:hashlib.sha256((wrf/f).read_bytes()).hexdigest() for f in files},
               sites=sites,
               classification={
                 'registry':'Registry/registry.cam:73 state real dlf2 ikj misc io rh "DLF2" kg/kg/s; allocated by cammgmpscheme '
                            '(Registry.EM_COMMON:3031, mp_physics==11) and camuwshcuscheme (:3229, shcu_physics==2)',
                 'initial_value':'allocation sets grid%dlf2=initial_data_value (inc/allocs_25.F); get_initial_data_value '
                                 'returns 0. (frame/wrf_num_bytes_between.c, NAN_VALUE disabled); io rh: not read from wrfinput',
                 'only_writer':'phys/module_shcu_camuwshcu_driver.F:400 dlf2(i,k,j)=qc2(1,kflip); called only from '
                                'phys/module_shallowcu_driver.F (IF (shcu_physics .eq. 0) return; SELECT CASE(shcu_physics) '
                                'CASE (CAMUWSHCUSCHEME) -> camuwshcu_driver(...,DLF2=dlf2,...) :545)',
                 'readers':'phys/module_mp_cammgmp_driver.F:184 dlf2_3d intent(in); :344 local real(r8) dlf2 (:758/:766/:844 '
                           'assign the LOCAL copy, never the grid array); '
                           'phys/module_microphysics_driver.F passes DLF2_3D=dlf2; dyn_em/solve_em.F and '
                           'module_first_rk_step_part1.F pass grid%dlf2 through',
                 'port':'gpuwrf refuses shcu_physics!=0 (io/namelist_binding.py), so no admitted configuration runs the writer'})
    (tree/'proofs/t_cammgws/DLF2_SOURCE_AUDIT.json').write_text(json.dumps(audit,indent=1)+'\n')
    print(json.dumps(dict(frames=len(frames),all_zero=proof['all_frames_positive_zero'],attrs=glob_attrs,field=fields,
                          sites=len(sites),files=files)))


if __name__=='__main__':main()
