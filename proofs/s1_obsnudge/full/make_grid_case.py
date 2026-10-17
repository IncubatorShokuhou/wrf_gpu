"""Private original-WRF case with BOTH analysis and station nudging active."""
import hashlib,importlib.util,json,shutil,sys
from pathlib import Path
import netCDF4
import numpy as np
s=importlib.util.spec_from_file_location('make_case',Path(__file__).with_name('make_cpu_case.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def main(out):
    m.main(out,True)
    base=out.parent
    shutil.copyfile(base/'cpu_probe/wrf.exe',out/'wrf.exe')
    for name in ('qr_acr_qg_V4.dat','qr_acr_qsV2.dat','freezeH2O.dat'):
        p=base/'cpu_on'/name
        if p.exists() and not (out/name).exists():(out/name).symlink_to(p)
    analysis=Path('<USER_HOME>/wrf_gpu2_lanes/o2-fdda/case/wrffdda_d01')
    # ext_ncd_read_field returns before reading if FieldType is absent (SN09).
    # Repair metadata in a private copy, retaining every analysis value byte.
    shutil.copyfile(analysis,out/'wrffdda_d01')
    with netCDF4.Dataset(analysis) as original,netCDF4.Dataset(out/'wrffdda_d01','r+') as private:
        with netCDF4.Dataset(out/'wrfinput_d01') as initial:
            private.setncatts({k:initial.getncattr(k) for k in initial.ncattrs() if k!='TITLE'})
        private.TITLE='WRF V4.7.1-compatible auxinput10 analyses assembled from original CPU-WRF history'
        for key,var in private.variables.items():
            if key!='Times':
                var.FieldType=104
                var.MemoryOrder='XYZ' if var.ndim==4 else 'XY'
                var.stagger={'U':'X','V':'Y','PH':'Z'}.get(key.split('_NDG_')[0],'')
            assert np.asarray(var[:]).tobytes()==np.asarray(original[key][:]).tobytes()
    text=(out/'namelist.input').read_text()
    options=dict(grid_fdda=1,gfdda_interval_m=60,gfdda_end_h=6,guv=.0003,gt=.0003,gq=.0003)
    options['gfdda_inname']="'wrffdda_d<domain>'"
    for key,value in options.items():text=m.edit(text,key,str(value),'fdda')
    # This clean DM build retains the generic auxinput10 aliases at startup;
    # make them explicitly resolve to the same GFdda stream and cadence.
    for key,value in dict(auxinput10_inname="'wrffdda_d<domain>'",io_form_auxinput10=2,
                          auxinput10_interval_m=60,auxinput10_end_h=6).items():
        text=m.edit(text,key,str(value),'time_control')
    text=m.edit(text,'debug_level','10','time_control')
    (out/'namelist.input').write_text(text)
    meta=json.loads((out/'PROVENANCE.json').read_text())
    meta['options'].update(options)
    meta['executable_source']=str(base/'cpu_probe/wrf.exe')
    meta['executable_sha256']=hashlib.sha256((out/'wrf.exe').read_bytes()).hexdigest()
    for key in ('namelist.input','wrffdda_d01'):meta['inputs'][key]=hashlib.sha256((out/key).read_bytes()).hexdigest()
    (out/'PROVENANCE.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
