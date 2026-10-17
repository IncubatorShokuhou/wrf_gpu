"""Real Swiss loader: WRF cold LAI/XSAI precede input-LAI PHENOLOGY."""
import importlib.util
import json
from pathlib import Path
import re
import shutil
import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SWISS=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')


@pytest.mark.parametrize('dveg',(7,8,9))
def test_real_loader_cold_lai_threshold_and_nonvegetated_cells(tmp_path,monkeypatch,cpu_pallas_interpret,dveg):
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.physics.noahmp.phenology import noahmp_phenology_table
    # Retained CPU oracle harness solver; scoped and restored after this test.
    spec=importlib.util.spec_from_file_location('input_lai_cpu_solver',ROOT/'proofs/o1_nlbind/cpu_safe.py')
    solver=importlib.util.module_from_spec(spec);spec.loader.exec_module(solver)
    monkeypatch.setattr(jax.lax.linalg,'tridiagonal_solve',solver._thomas)
    monkeypatch.setenv('GPUWRF_NOAHMP_NATIVE_REAL','1')
    monkeypatch.setenv('GPUWRF_NOAHMP_COLUMN_KERNELS','0')
    jax.clear_caches()
    case=tmp_path/'case';case.mkdir()
    for path in SWISS.iterdir():
        if path.name not in ('wrfinput_d01','namelist.input'):
            (case/path.name).symlink_to(path)
    shutil.copyfile(SWISS/'wrfinput_d01',case/'wrfinput_d01')
    text=re.sub(r'^\s*dveg\s*=.*$','',(SWISS/'namelist.input').read_text(),flags=re.I|re.M)
    if re.search(r'&noah_mp\b',text,re.I):
        text=re.sub(r'(&noah_mp\s*\n)',lambda m:m[1]+f'dveg={dveg},\n',text,count=1,flags=re.I)
    else:
        text+=f'\n&noah_mp\ndveg={dveg},\n/\n'
    (case/'namelist.input').write_text(text)
    threshold=np.float32(.05)
    raw=np.asarray([0.,.01,.04,np.nextafter(threshold,np.float32(0)),threshold,
                    np.nextafter(threshold,np.float32(1)),.5,2.],np.float32)
    with Dataset(case/'wrfinput_d01','a') as ds:
        veg=np.asarray(ds['IVGTYP'][0]);mask=np.asarray(ds['LANDMASK'][0])>.5
        selected=np.argwhere(mask&~np.isin(veg,(13,15,16,17))&(np.asarray(ds['SNOWH'][0])==0.))[:12]
        assert selected.shape==(12,2)
        natural,nonveg=selected[:8],selected[8:]
        for (y,x),value in zip(natural,raw,strict=True):
            ds['LAI'][0,y,x]=value
            if 'XSAI' in ds.variables:ds['XSAI'][0,y,x]=8.
        for (y,x),category in zip(nonveg,(13,15,16,17),strict=True):
            ds['IVGTYP'][0,y,x]=category;ds['LAI'][0,y,x]=2.
    cfg=pipeline.NestedPipelineConfig(case,tmp_path/'out',tmp_path/'proof',hours=1,max_dom=1)
    _,bundles,_,_,_,carries=pipeline._load_domains(cfg,('d01',))
    nl=bundles['d01'].namelist;land=carries['d01'].noahmp_land
    assert nl.noahmp_options.dveg==dveg
    # Independent WRF NOAHMP_INIT:2181–2182 expectations at and either side
    # of the REAL4 .05 threshold; the previous XSAI is not an input here.
    expected_lai=np.asarray([threshold]*5+[raw[5],.5,2.],np.float32)
    expected_sai=np.asarray([threshold]*7+[.2],np.float32)
    pick=lambda field,cells:np.asarray(field)[tuple(cells.T)]
    np.testing.assert_array_equal(pick(land.lai,natural),expected_lai)
    np.testing.assert_array_equal(pick(land.sai,natural),expected_sai)
    np.testing.assert_array_equal(pick(land.lai,nonveg),np.zeros(4,np.float32))
    np.testing.assert_array_equal(pick(land.sai,nonveg),np.zeros(4,np.float32))
    phen=noahmp_phenology_table(land,None,nl.noahmp_static,dveg=dveg)
    np.testing.assert_array_equal(pick(phen.lai,natural),expected_lai)
    assert np.all(pick(phen.fveg,natural)>0)
    np.testing.assert_array_equal(pick(phen.fveg,nonveg),np.zeros(4,np.float32))
    # Evolved, legitimately subthreshold LAI still reaches the original zero
    # canopy branch. Initialization is not a new per-step masking clamp.
    evolved=np.asarray(land.lai).copy();evolved[tuple(natural[:3].T)]=np.float32(.01)
    phen=noahmp_phenology_table(land.replace(lai=jnp.asarray(evolved)),None,nl.noahmp_static,dveg=dveg)
    np.testing.assert_array_equal(pick(phen.lai,natural[:3]),np.zeros(3,np.float32))
    np.testing.assert_array_equal(pick(phen.fveg,natural[:3]),np.zeros(3,np.float32))
    print(json.dumps(dict(dveg=dveg,raw_lai=raw.tolist(),loaded_lai=pick(land.lai,natural).tolist(),
        loaded_sai=pick(land.sai,natural).tolist(),nonveg_categories=[13,15,16,17],
        source='pristine module_sf_noahmpdrv.F:2163-2182',scope='real cold-loader/first PHENOLOGY, not forecast fidelity')))
    jax.clear_caches()
