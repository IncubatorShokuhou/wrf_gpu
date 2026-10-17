"""Pack remaining CLI features using authentic or explicitly replay analyses."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from prepare import ROOT, RETAINED, SLOTS, private_executable, write_json
from extra_cases import stage
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.netcdf_lock import Dataset

CORRECTED_ANALYSIS=Path('<USER_HOME>/wrf_gpu2_lanes/t-fdda/TF02/wrffdda_d01')


def surface_analyses():
    out = ROOT / 'analyses'
    out.mkdir(exist_ok=True)
    path = out / 'wrfsfdda_d01'
    if path.exists():
        return path
    source = ROOT / 'c00/cpu'
    with Dataset(source / 'wrfout_d01_2023-01-15_00:00:00') as ds:
        nx,ny=len(ds.dimensions['west_east']),len(ds.dimensions['south_north'])
        attrs={k:ds.getncattr(k) for k in ds.ncattrs() if k!='TITLE'}
    with Dataset(path,'w') as ds:
        for n,size in (('Time',24),('DateStrLen',19),('west_east',nx),('south_north',ny),
                       ('west_east_stag',nx+1),('south_north_stag',ny+1)):
            ds.createDimension(n,size)
        times=ds.createVariable('Times','S1',('Time','DateStrLen'))
        for name in ('U10','V10','TH2','Q2','ODIS','T2','RH','PSL','PS','TOB'):
            dims=('south_north','west_east_stag') if name=='U10' else (
                 ('south_north_stag','west_east') if name=='V10' else ('south_north','west_east'))
            for suffix in ('OLD','NEW'):
                v=ds.createVariable(name+'_NDG_'+suffix,'f4',('Time',*dims))
                v.FieldType=104;v.MemoryOrder='XY';v.stagger={'U10':'X','V10':'Y'}.get(name,'')
        for hour in range(24):
            stamp=f'2023-01-15_{hour:02d}:00:00'
            times[hour]=np.frombuffer(stamp.encode(),dtype='S1')
            for suffix,h in (('OLD',hour),('NEW',hour+1)):
                date='2023-01-15' if h<24 else '2023-01-16'
                with Dataset(source/f'wrfout_d01_{date}_{h%24:02d}:00:00') as frame:
                    for name in ('U10','V10','TH2','Q2','ODIS','T2','RH','PSL','PS','TOB'):
                        a=(np.asarray(frame['PSFC' if name=='PS' else name][0],np.float32)
                           if name in ('U10','V10','TH2','Q2','T2','PS') else np.zeros((ny,nx),np.float32))
                        if name=='U10':a=np.pad(a,((0,0),(0,1)),mode='edge')
                        if name=='V10':a=np.pad(a,((0,1),(0,0)),mode='edge')
                        ds[name+'_NDG_'+suffix][hour]=a
        ds.setncatts(attrs)
        ds.TITLE='WRF V4.7.1-compatible surface replay analyses from original CPU-WRF hourly history'
        ds.GPUWRF_PROVENANCE='CPU-WRF replay fields; ODIS/RH/PSL/TOB explicit zero controls, no external observation/skill claim'
    return path


def add_case(matrix, ident, groups, base, features, extras=None, zero=False, files=None):
    if any(r['id']==ident for r in matrix['cases']):
        return
    row=dict(id=ident,geography='MMF-lowland' if ident=='c47' else 'Swiss',hours=24,domains=['d01'],
        options={k:groups['physics'][k] for k in SLOTS},extra=extras or {},features=features)
    row=stage(row,base,groups,files)
    if zero:
        private_executable(ROOT/ident,ROOT/'reference_zero')
        row['reference_semantics']='compiler-defined zero, inactive canonical Morrison-aero object'
    matrix['cases'].append(row)
    write_json(ROOT/ident/'inputs.json',row)
    write_json(ROOT/'MATRIX.json',matrix)
    print(json.dumps({'id':ident,'errors':row['admission_errors']}),flush=True)


if __name__=='__main__':
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    source=Path(sys.argv[1])
    matrix=json.loads((ROOT/'MATRIX.json').read_text())
    swiss=RETAINED/'source_inputs/Swiss'
    template=source/'examples/switzerland_d01/namelist.input'
    g=parse_namelist(ROOT/'mmf24_real/namelist.input')
    g['physics']['wtddt']=.75
    g['noah_mp']=dict(opt_run=5,opt_inf=1)
    add_case(matrix,'c47',g,ROOT/'mmf24_real',['MMF5-authentic-active-groundwater','opt_inf1'],zero=True)
    g=parse_namelist(template);g['dynamics'].update(diff_opt=2,km_opt=3)
    add_case(matrix,'c48',g,swiss,['KM3-pristine-zero-reference-probe'],zero=True)
    for ident,code,original in (('c49',1,Path('<USER_HOME>/wrf_gpu2_lanes/s1-dfi12/oracle/opt1')),
                                ('c50',2,Path('<USER_HOME>/wrf_gpu2_lanes/s1-dfi12/oracle/opt2')),
                                ('c51',3,RETAINED/'dfi3/input')):
        g=parse_namelist(template)
        g['dfi_control']=parse_namelist(original/'namelist.input')['dfi_control']
        g['dfi_control']['dfi_opt']=code
        add_case(matrix,ident,g,swiss,[f'DFI{code}-original-phase-windows;clock-inventory-required'],zero=True)
    sf=surface_analyses()
    for ident,mode in (('c52',1),('c53',2)):
        if any(r['id']==ident for r in matrix['cases']):continue
        g=parse_namelist(template)
        g['physics']['swint_opt']=1
        g['fdda']=dict(grid_fdda=1,gfdda_inname='wrffdda_d<domain>',gfdda_interval_m=60,gfdda_end_h=24,
            io_form_gfdda=2,guv=.0003,gt=.0003,gq=.0003,fgdt=.9,
            grid_sfdda=mode,sgfdda_inname='wrfsfdda_d<domain>',sgfdda_interval_m=60,sgfdda_end_h=24,
            guv_sfc=.0003,gt_sfc=.0003,gq_sfc=.0003,rinblw=100.)
        add_case(matrix,ident,g,swiss,[f'gridFDDA1+surfaceFDDA{mode}','solar-swint1','original-CPU-replay-analyses'],zero=True)
        case=ROOT/ident
        for name,p in (('wrffdda_d01',CORRECTED_ANALYSIS),('wrfsfdda_d01',sf)):
            (case/'input'/name).symlink_to(p)
            (case/'cpu'/name).symlink_to(p)
        from prepare import sha
        row=next(r for r in matrix['cases'] if r['id']==ident)
        row['input_sha256']={p.name:sha(p) for p in (case/'input').iterdir() if p.is_file()}
        write_json(case/'inputs.json',row);write_json(ROOT/'MATRIX.json',matrix)
    g=parse_namelist(template)
    g['fdda']=dict(grid_fdda=2,gfdda_inname='wrffdda_d<domain>',gfdda_interval_m=60,gfdda_end_h=24,
        io_form_gfdda=2,guv=.0003,gt=.0003,gph=.0003,gq=0.,xwavenum=2,ywavenum=2,fgdt=0)
    g['dynamics'].update(diff_opt=0,km_opt=0,damp_opt=0,diff_6th_opt=0,w_damping=0,moist_adv_opt=0,scalar_adv_opt=0)
    add_case(matrix,'c54',g,swiss,['spectralFDDA2','diffusion-off','damping-off','standard-scalar-advection0'],zero=True)
    case=ROOT/'c54'
    if not (case/'input/wrffdda_d01').exists():
        (case/'input/wrffdda_d01').symlink_to(CORRECTED_ANALYSIS)
        (case/'cpu/wrffdda_d01').symlink_to(CORRECTED_ANALYSIS)
        from prepare import sha
        row=next(r for r in matrix['cases'] if r['id']=='c54')
        row['input_sha256']={p.name:sha(p) for p in (case/'input').iterdir() if p.is_file()}
        write_json(case/'inputs.json',row);write_json(ROOT/'MATRIX.json',matrix)
    original=Path('<USER_HOME>/wrf_gpu2_lanes/s1-obsnudge/full/cpu_on')
    g=parse_namelist(template);g['fdda']=parse_namelist(original/'namelist.input')['fdda']
    g['dynamics'].update(diff_opt=0,km_opt=1,moist_adv_opt=2,scalar_adv_opt=2)
    add_case(matrix,'c55',g,swiss,['observation-nudging1','monotonic-advection2','diff0/km1-not-constantK'],zero=True,
        files={'OBS_DOMAIN101':(original/'OBS_DOMAIN101').read_text()})
