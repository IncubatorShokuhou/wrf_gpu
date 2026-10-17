"""Independent radar oracles and failure-sensitive product contracts (CPU)."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from gpuwrf.post.radar import radar, column_products
from gpuwrf.post.radar_thompson import thompson_reflectivity
from gpuwrf.post.radar_rip import rip_reflectivity
from gpuwrf.post.radar_scattering import mie_backscatter, dielectric, dry_table

FIXTURE=Path(__file__).resolve().parents[1]/'proofs/u2_radar'


def test_thompson_original_fortran():
    with np.load(FIXTURE/'thompson_oracle.npz') as f:
        a,ref=f['inputs'],f['dbz']
    actual=thompson_reflectivity(a[:,0],a[:,2],a[:,3],a[:,4],a[:,5],a[:,6],a[:,8],a[:,9])
    assert np.isfinite(actual).all() and np.isfinite(ref).all()
    # Frozen before final composition; <= about 2 ulp on this active fixture.
    np.testing.assert_allclose(actual,ref,atol=5e-6,rtol=0)
    assert actual.shape==(40,44)
    assert actual.max()>35 and actual.min()==-35  # pristine maximum 37.6246


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize('field',range(8))
def test_thompson_nonfinite_rejected(field,bad):
    a=[np.zeros((1,3),dtype='f4') for _ in range(8)]
    a[-2][:]=280;a[-1][:]=90000;a[field][0,1]=bad
    with pytest.raises(ValueError,match='NaN/Inf'):
        thompson_reflectivity(*a)


def test_thompson_positive_number_required():
    a=[np.zeros((1,3)) for _ in range(8)];a[-2][:]=280;a[-1][:]=90000
    a[1][:]=.001
    with pytest.raises(ValueError,match='number'):
        thompson_reflectivity(*a)


def test_rip_original_ncar_fortran():
    f=json.loads((FIXTURE/'rip_oracle.json').read_text())
    actual=[]
    for row in f['rows']:
        actual.append(float(rip_reflectivity(*row[:6],variable_intercept=bool(row[6]),liquid_skin=bool(row[7]),snow_present=bool(row[8]))))
    np.testing.assert_allclose(actual,f['dbz'],atol=1e-12,rtol=0)


# Published MIEV0/Wiscombe Qback values transcribed from the validation tables at
# https://miepython.readthedocs.io/en/3.0.1/09_backscattering.html
@pytest.mark.parametrize('m,x,qback',[
    (1.55+0j,2*np.pi*.525/.6328,2.92534),
    (.75+0j,.099,1.108554e-5),
    (.75+0j,10.,.04658462),
    (1.33-.00001j,1.,.08462494),
    (1.5-1j,.055,1.695493e-5),
    (1.5-1j,1.,.5730036),
    (10-10j,1.,3.308998),
])
def test_mie_published_backscatter(m,x,qback):
    pytest.importorskip('miepython')
    wavelength=.032
    d=x*wavelength/np.pi
    actual=mie_backscatter(d,m,wavelength)/(np.pi*d*d/4)
    np.testing.assert_allclose(actual,qback,rtol=6e-6,atol=0)


def test_mie_small_particle_limit_and_large_drop_difference():
    pytest.importorskip('miepython')
    m,_=dielectric(.032)
    d=1e-6
    rayleigh=abs((m*m-1)/(m*m+2))**2*np.pi**5*d**6/.032**4
    np.testing.assert_allclose(mie_backscatter(d,m,.032),rayleigh,rtol=1e-5)
    d=.008
    rayleigh=abs((m*m-1)/(m*m+2))**2*np.pi**5*d**6/.032**4
    assert abs(mie_backscatter(d,m,.032)/rayleigh-1)>.1


def test_column_products_independent_greene_clark():
    # NWS OHX published formula; fixed reflectivity gives analytic column integral.
    z=np.array([[40.,40.,40.],[-35.,-35.,-35.],[30.,20.,0.]])
    h=np.tile([100.,1100.,2100.],(3,1))
    composite,top,status,vil=column_products(z,h)
    np.testing.assert_allclose(composite,[40.,-35.,30.])
    np.testing.assert_allclose(top,[2100.,0.,1200.])
    np.testing.assert_array_equal(status,[2,0,1])
    assert vil[0]==pytest.approx(3.44e-6*10000**(4/7)*2000)
    assert vil[1]==0
    expected=3.44e-6*((550.)**(4/7)+50.5**(4/7))*1000
    assert vil[2]==pytest.approx(expected)


def test_vil_hail_cap_no_extrapolation():
    _,_,_,vil=column_products([[70.,80.]],[[500.,1500.]])
    assert float(vil[0])==pytest.approx(3.44e-6*(10**5.6)**(4/7)*1000)


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_column_nonfinite_rejected(bad):
    with pytest.raises(ValueError,match='finite'):
        column_products([[30.,bad]],[[100.,200.]])
    with pytest.raises(ValueError,match='finite'):
        column_products([[30.,20.]],[[100.,bad]])


def sample():
    dims=('Time','bottom_top','south_north','west_east')
    shape=(1,3,1,1)
    ds=xr.Dataset({'REFL_10CM':(dims,np.array([10.,30.,0.],dtype='f4').reshape(shape)),
                   'PH':(('Time','bottom_top_stag','south_north','west_east'),np.array([0,200,2000,4000.]).reshape(1,4,1,1)*9.81),
                   'PHB':(('Time','bottom_top_stag','south_north','west_east'),np.zeros((1,4,1,1))),
                   'HGT':(('Time','south_north','west_east'),np.zeros((1,1,1)))},
                  coords={'Time':np.array(['2023-01-15T00'],dtype='datetime64[ns]')},attrs={'MP_PHYSICS':8})
    for key,value in {'P':0.,'PB':90000.,'T':0.,'QVAPOR':.005,'QRAIN':1e-4,'QNRAIN':1e4,'QSNOW':0.,'QGRAUP':0.}.items():
        ds[key]=(dims,np.full(shape,value))
    return ds


def test_saved_reflectivity_is_exact_and_method_explicit():
    ds=sample();out=radar(ds)
    np.testing.assert_array_equal(out.radar_dbz_s,ds.REFL_10CM)
    assert out.radar_dbz_s.dtype==ds.REFL_10CM.dtype
    assert out.radar_dbz_s.attrs['radar_method']=='saved'
    assert float(out.radar_composite_s.values.item())==30
    assert out.radar_echo_top_s.attrs['units']=='m'


def test_thompson_missing_ng_refuses_only_active_graupel():
    ds=sample().drop_vars('REFL_10CM')
    assert np.isfinite(radar(ds).radar_dbz_s).all()
    ds.QGRAUP[:]=1e-3
    with pytest.raises(ValueError,match='QNGRAUPEL'):
        radar(ds)


def test_no_implicit_fallback_or_band_rescaling():
    ds=sample().drop_vars('REFL_10CM');ds.attrs['MP_PHYSICS']=6
    with pytest.raises(ValueError,match='explicitly'):
        radar(ds)
    with pytest.raises(ValueError,match='S-band only'):
        radar(ds,{'radar_band':'X','radar_method':'rip'})
    with pytest.raises(ValueError,match='S-band only'):
        radar(sample(),{'radar_band':'C','radar_method':'saved'})
    assert radar(ds,{'radar_method':'rip'}).radar_dbz_s.attrs['radar_method']=='rip'


def test_band_products_are_distinct_and_finite():
    pytest.importorskip('miepython')
    ds=sample().drop_vars('REFL_10CM');ds.QRAIN[:]=.003;ds.QNRAIN[:]=1e3
    c=radar(ds,{'radar_band':'C'});x=radar(ds,{'radar_band':'X'})
    assert np.isfinite(c.radar_dbz_c).all() and np.isfinite(x.radar_dbz_x).all()
    assert float(abs(c.radar_dbz_c-x.radar_dbz_x).max())>.1
    assert dry_table.cache_info().currsize <= 8


def test_mie_all_species_and_wet_snow_are_exercised():
    pytest.importorskip('miepython')
    with np.load(FIXTURE/'thompson_oracle.npz') as f:
        a=f['inputs'][-9:-8]  # rain+snow+graupel, warm layer and ice above
    values=[]
    for band in ('S','C','X'):
        z=thompson_reflectivity(a[:,0],a[:,2],a[:,3],a[:,4],a[:,5],a[:,6],a[:,8],a[:,9],band=band)
        assert np.isfinite(z).all()
        values.append(z)
    assert np.max(abs(values[1]-values[2]))>.1
    # Remove each species separately: its contribution must have been active.
    for field in (2,4,5):
        b=a.copy();b[:,field]=0
        z=thompson_reflectivity(b[:,0],b[:,2],b[:,3],b[:,4],b[:,5],b[:,6],b[:,8],b[:,9],band='X')
        assert np.max(abs(z-values[2]))>.01


@pytest.mark.parametrize('field',['REFL_10CM','PH','PHB','HGT','P','PB','T','QVAPOR','QRAIN','QNRAIN','QSNOW','QGRAUP','QNGRAUPEL'])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_runner_rejects_nonfinite_used_inputs(field,bad):
    ds=sample()
    if field not in ('REFL_10CM','PH','PHB','HGT'):
        ds=ds.drop_vars('REFL_10CM')
        ds['QNGRAUPEL']=xr.zeros_like(ds.QRAIN)
    ds[field].values.flat[0]=bad
    with pytest.raises(ValueError,match='NaN/Inf'):
        radar(ds)


def test_echo_highest_disconnected_and_exact_threshold():
    z=np.array([[19.,0.,25.,0.],[18.,0.,0.,0.]])
    h=np.tile([100.,1100.,2100.,3100.],(2,1))
    _,top,flags,_=column_products(z,h)
    np.testing.assert_allclose(top,[2380.,100.])
    np.testing.assert_array_equal(flags,[1,1])


def test_invalid_height_and_threshold():
    for h in ([[100.,100.]],[[100.,50.]],[[-1.,100.]]):
        with pytest.raises(ValueError,match='heights'):
            column_products([[30.,0.]],h)
    with pytest.raises(ValueError,match='threshold'):
        column_products([[30.,0.]],[[100.,200.]],echo_threshold=-35.)


def test_dependency_error_is_actionable(monkeypatch):
    import builtins
    original=builtins.__import__
    def missing(name,*args,**kwargs):
        if name=='miepython':
            raise ImportError('missing optional library')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',missing)
    with pytest.raises(ImportError,match=r'gpuwrf\[radar\]'):
        mie_backscatter(.001,1.33-.1j,.032)


def test_saved_original_swiss_frames():
    with np.load(FIXTURE/'saved_swiss.npz') as f:
        ds=xr.Dataset({name:((('Time','south_north','west_east') if name=='HGT' else
                              ('Time','bottom_top_stag','south_north','west_east') if name in ('PH','PHB') else
                              ('Time','bottom_top','south_north','west_east')), f[name]) for name in f.files})
    out=radar(ds)
    assert out.radar_dbz_s.values.tobytes()==ds.REFL_10CM.values.tobytes()
    np.testing.assert_array_equal(out.radar_composite_s,ds.REFL_10CM.max(dim='bottom_top'))
    assert out.radar_dbz_s.shape==(4,44,3,3)
    assert float(out.radar_composite_s.max())>20
    assert (out.radar_echo_top_status_s.values==1).any()


def test_rip_nonfinite_rejected():
    for bad in [np.nan,np.inf,-np.inf]:
        for i in range(6):
            a=[90000.,280.,.005,.001,0.,0.];a[i]=bad
            with pytest.raises(ValueError,match='NaN/Inf'):
                rip_reflectivity(*a)


def test_psd_table_quadrature_convergence():
    pytest.importorskip('miepython')
    # Independent analytic mass moments normalize rain/graupel PSDs; table
    # refinement tests numerical integration only, not observational fidelity.
    from math import gamma
    for species,am in [('rain',np.pi*1000/6),('graupel',np.pi*400/6)]:
        slope=1500.;n0=8e6
        moments=[]
        for bins in [256,512]:
            d,width,sigma=dry_table(species,.032,bins)
            mass=np.sum(n0*np.exp(-slope*d)*am*d**3*width)
            assert mass==pytest.approx(n0*am*gamma(4)/slope**4,rel=3e-4)
            moments.append(float(np.sum(n0*np.exp(-slope*d)*sigma*width)))
        assert 10*abs(np.log10(moments[0]/moments[1]))<.002


def test_registry_and_product_composition():
    from gpuwrf.post.registry import required_inputs,run_products
    ds=sample()
    ds['U10']=(('Time','south_north','west_east'),np.full((1,1,1),3.))
    ds['V10']=(('Time','south_north','west_east'),np.full((1,1,1),4.))
    assert 'REFL_10CM' in required_inputs(['radar'])
    for names in (['radar','wind-speed10'],['wind-speed10','radar']):
        out=run_products(ds,names)
        np.testing.assert_array_equal(out.radar_dbz_s,ds.REFL_10CM)
        assert len(out.data_vars)==6
        assert 'instantaneous' in out.radar_dbz_s.attrs['sampling']


def cli(args,cwd):
    env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')+os.pathsep+os.environ.get('PYTHONPATH',''),
         'JAX_PLATFORMS':'cpu','GPUWRF_JAX_CACHE':'0'}
    return subprocess.run([sys.executable,'-m','gpuwrf.cli',*args],cwd=cwd,env=env,text=True,capture_output=True,timeout=40)


def test_cli_list_and_saved_netcdf_roundtrip(tmp_path):
    listed=cli(['post','--list-products'],tmp_path)
    assert listed.returncode==0,listed.stderr
    assert 'radar' in listed.stdout.splitlines()
    ds=sample();ds.attrs['Times']='2023-01-15_00:00:00'
    path=tmp_path/'wrfout';ds.to_netcdf(path)
    outpath=tmp_path/'radar.nc'
    result=cli(['post','--input',str(path),'--output',str(outpath),'--product','radar'],tmp_path)
    assert result.returncode==0,result.stderr
    with xr.open_dataset(outpath) as out:
        np.testing.assert_array_equal(out.radar_dbz_s,ds.REFL_10CM)
        np.testing.assert_array_equal(out.Time,ds.Time)
        assert out.attrs['Conventions']=='CF-1.10'
        assert out.radar_echo_top_status_s.attrs['flag_meanings']=='no_echo resolved model_top_censored'
    refused=cli(['post','--input',str(path),'--output',str(tmp_path/'no.nc'),'--product','radar',
                 '--option','radar_band="X"','--option','radar_method="saved"'],tmp_path)
    assert refused.returncode==2 and 'S-band only' in refused.stderr
    assert not (tmp_path/'no.nc').exists()
