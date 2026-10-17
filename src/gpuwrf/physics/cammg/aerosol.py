"""Generated offline from pristine WRF CAM-MG; see proofs/s2_cammg/port.py.
Qualified against original WRF public/staged calls; native GPU qualification pending.
"""
import jax.numpy as jnp
from ._support import _r,_d,_int,_get,_put,_slice,_loop,_cond,_cast,_dtype,_pow,_powi,_retry,_zeros,_literal_array,_static,_neg

from .error_function import erf
from .saturation_table import estblf
from .ice_activation import nucleati
from .cloud import polysvp
from .cloud import vqsatd_water
def explmix(q,src,ekkp,ekkm,overlapp,overlapm,qold,surfrate,flxconv,pver,dt,is_unact,qactold,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('ndrop',{}))
    v['q']=q
    v['src']=src
    v['ekkp']=ekkp
    v['ekkm']=ekkm
    v['overlapp']=overlapp
    v['overlapm']=overlapm
    v['qold']=qold
    v['surfrate']=surfrate
    v['flxconv']=flxconv
    v['pver']=pver
    v['dt']=dt
    v['is_unact']=is_unact
    v['qactold']=qactold
    v['_jump']=_int(0)
    v['k']=_zeros((),'i32',xp)
    v['kp1']=_zeros((),'i32',xp)
    v['km1']=_zeros((),'i32',xp)
    _outer1=dict(v)
    def _yes1(_carry):
        v=dict(_outer1);v.update(zip(('k', 'km1', 'kp1', 'q'),_carry))
        _outer2=dict(v)
        def _body2(_k,_carry):
            v=dict(_outer2);v.update(zip(('_jump', 'k', 'km1', 'kp1', 'q'),_carry));v['k']=_int(_k)
            _put(v,'kp1',None,xp.minimum(v['k'] + 1, v['pver']))  # WRF:module_cam_mp_ndrop.F:980
            _put(v,'km1',None,xp.maximum(v['k'] - 1, 1))  # WRF:module_cam_mp_ndrop.F:981
            _put(v,'q',(v['k'],),_get(v['qold'], (v['k'],)) + v['dt'] * (_neg(_get(v['src'], (v['k'],))) + _get(v['ekkp'], (v['k'],)) * (_get(v['qold'], (v['kp1'],)) - _get(v['qold'], (v['k'],)) + _get(v['qactold'], (v['kp1'],)) * (_r(1.0) - _get(v['overlapp'], (v['k'],)))) + _get(v['ekkm'], (v['k'],)) * (_get(v['qold'], (v['km1'],)) - _get(v['qold'], (v['k'],)) + _get(v['qactold'], (v['km1'],)) * (_r(1.0) - _get(v['overlapm'], (v['k'],))))))  # WRF:module_cam_mp_ndrop.F:982
            _put(v,'q',(v['k'],),xp.maximum(_get(v['q'], (v['k'],)), _d(0.0)))  # WRF:module_cam_mp_ndrop.F:989
            return tuple(v[_n] for _n in ('_jump', 'k', 'km1', 'kp1', 'q'))
        _result=_loop(1,v['pver'],1,_body2,tuple(v[_n] for _n in ('_jump', 'k', 'km1', 'kp1', 'q')),xp,jump_index=0)
        v.update(zip(('_jump', 'k', 'km1', 'kp1', 'q'),_result))
        _put(v,'q',(v['pver'],),_get(v['q'], (v['pver'],)) - v['surfrate'] * _get(v['qold'], (v['pver'],)) * v['dt'] + v['flxconv'] * v['dt'])  # WRF:module_cam_mp_ndrop.F:993
        _put(v,'q',(v['pver'],),xp.maximum(_get(v['q'], (v['pver'],)), _d(0.0)))  # WRF:module_cam_mp_ndrop.F:997
        return tuple(v[_n] for _n in ('k', 'km1', 'kp1', 'q'))
    def _no1(_carry):
        v=dict(_outer1);v.update(zip(('k', 'km1', 'kp1', 'q'),_carry))
        _outer3=dict(v)
        def _body3(_k,_carry):
            v=dict(_outer3);v.update(zip(('_jump', 'k', 'km1', 'kp1', 'q'),_carry));v['k']=_int(_k)
            _put(v,'kp1',None,xp.minimum(v['k'] + 1, v['pver']))  # WRF:module_cam_mp_ndrop.F:1001
            _put(v,'km1',None,xp.maximum(v['k'] - 1, 1))  # WRF:module_cam_mp_ndrop.F:1002
            _put(v,'q',(v['k'],),_get(v['qold'], (v['k'],)) + v['dt'] * (_get(v['src'], (v['k'],)) + _get(v['ekkp'], (v['k'],)) * (_get(v['overlapp'], (v['k'],)) * _get(v['qold'], (v['kp1'],)) - _get(v['qold'], (v['k'],))) + _get(v['ekkm'], (v['k'],)) * (_get(v['overlapm'], (v['k'],)) * _get(v['qold'], (v['km1'],)) - _get(v['qold'], (v['k'],)))))  # WRF:module_cam_mp_ndrop.F:1003
            _put(v,'q',(v['k'],),xp.maximum(_get(v['q'], (v['k'],)), _d(0.0)))  # WRF:module_cam_mp_ndrop.F:1008
            return tuple(v[_n] for _n in ('_jump', 'k', 'km1', 'kp1', 'q'))
        _result=_loop(1,v['pver'],1,_body3,tuple(v[_n] for _n in ('_jump', 'k', 'km1', 'kp1', 'q')),xp,jump_index=0)
        v.update(zip(('_jump', 'k', 'km1', 'kp1', 'q'),_result))
        _put(v,'q',(v['pver'],),_get(v['q'], (v['pver'],)) - v['surfrate'] * _get(v['qold'], (v['pver'],)) * v['dt'] + v['flxconv'] * v['dt'])  # WRF:module_cam_mp_ndrop.F:1012
        _put(v,'q',(v['pver'],),xp.maximum(_get(v['q'], (v['pver'],)), _d(0.0)))  # WRF:module_cam_mp_ndrop.F:1016
        return tuple(v[_n] for _n in ('k', 'km1', 'kp1', 'q'))
    _result=_cond(v['is_unact'],_yes1,_no1,tuple(v[_n] for _n in ('k', 'km1', 'kp1', 'q')),xp)  # WRF:module_cam_mp_ndrop.F:977
    v.update(zip(('k', 'km1', 'kp1', 'q'),_result))
    return (v['q'],)

def maxsat(zeta,eta,nmode,smc,smax,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('ndrop',{}))
    v['zeta']=zeta
    v['eta']=eta
    v['nmode']=nmode
    v['smc']=smc
    v['smax']=smax
    v['_jump']=_int(0)
    v['m']=_zeros((),'i32',xp)
    v['sum']=_zeros((),'f64',xp)
    v['g1']=_zeros((),'f64',xp)
    v['g2']=_zeros((),'f64',xp)
    v['g1sqrt']=_zeros((),'f64',xp)
    v['g2sqrt']=_zeros((),'f64',xp)
    _outer1=dict(v)
    def _yes1(_carry):
        v=dict(_outer1);v.update(zip(('_jump', 'm', 'smax'),_carry))
        _outer2=dict(v)
        def _body2(_k,_carry):
            v=dict(_outer2);v.update(zip(('_jump', 'm', 'smax'),_carry));v['m']=_int(_k)
            _outer3=dict(v)
            def _yes3(_carry):
                v=dict(_outer3);v.update(zip(('_jump', 'smax'),_carry))
                _outer4=dict(v)
                def _yes4(_carry):
                    v=dict(_outer4);v.update(zip(('_jump', 'smax'),_carry))
                    _put(v,'smax',None,_d(1e-20))  # WRF:module_cam_mp_ndrop.F:1567
                    return tuple(v[_n] for _n in ('_jump', 'smax'))
                def _no4(_carry):
                    v=dict(_outer4);v.update(zip(('_jump', 'smax'),_carry))
                    v['_jump']=_int(1)  # WRF:module_cam_mp_ndrop.F:1570
                    return tuple(v[_n] for _n in ('_jump', 'smax'))
                _result=_cond(xp.logical_or(_get(v['zeta'], (v['m'],)) > _d(100000.0) * _get(v['eta'], (v['m'],)), _get(v['smc'], (v['m'],)) * _get(v['smc'], (v['m'],)) > _d(100000.0) * _get(v['eta'], (v['m'],))),_yes4,_no4,tuple(v[_n] for _n in ('_jump', 'smax')),xp)  # WRF:module_cam_mp_ndrop.F:1565
                v.update(zip(('_jump', 'smax'),_result))
                return tuple(v[_n] for _n in ('_jump', 'smax'))
            def _no3(_carry):
                v=dict(_outer3);v.update(zip(('_jump', 'smax'),_carry))
                return tuple(v[_n] for _n in ('_jump', 'smax'))
            _result=_cond(v['_jump'] == 0,_yes3,_no3,tuple(v[_n] for _n in ('_jump', 'smax')),xp)  # WRF:module_cam_mp_ndrop.F:1565
            v.update(zip(('_jump', 'smax'),_result))
            return tuple(v[_n] for _n in ('_jump', 'm', 'smax'))
        _result=_loop(1,v['nmode'],1,_body2,tuple(v[_n] for _n in ('_jump', 'm', 'smax')),xp,jump_index=0)
        v.update(zip(('_jump', 'm', 'smax'),_result))
        return tuple(v[_n] for _n in ('_jump', 'm', 'smax'))
    def _no1(_carry):
        v=dict(_outer1);v.update(zip(('_jump', 'm', 'smax'),_carry))
        return tuple(v[_n] for _n in ('_jump', 'm', 'smax'))
    _result=_cond(v['_jump'] == 0,_yes1,_no1,tuple(v[_n] for _n in ('_jump', 'm', 'smax')),xp)  # WRF:module_cam_mp_ndrop.F:1564
    v.update(zip(('_jump', 'm', 'smax'),_result))
    _outer5=dict(v)
    def _yes5(_carry):
        v=dict(_outer5);v.update(zip(('_jump',),_carry))
        v['_jump']=_int(-1)
        return tuple(v[_n] for _n in ('_jump',))
    def _no5(_carry):
        v=dict(_outer5);v.update(zip(('_jump',),_carry))
        return tuple(v[_n] for _n in ('_jump',))
    _result=_cond(v['_jump'] == 0,_yes5,_no5,tuple(v[_n] for _n in ('_jump',)),xp)  # WRF:module_cam_mp_ndrop.F:1574
    v.update(zip(('_jump',),_result))
    v['_jump']=xp.where(v['_jump']==1,_int(0),v['_jump'])
    _outer6=dict(v)
    def _yes6(_carry):
        v=dict(_outer6);v.update(zip(('sum',),_carry))
        _put(v,'sum',None,0)  # WRF:module_cam_mp_ndrop.F:1578
        return tuple(v[_n] for _n in ('sum',))
    def _no6(_carry):
        v=dict(_outer6);v.update(zip(('sum',),_carry))
        return tuple(v[_n] for _n in ('sum',))
    _result=_cond(v['_jump'] == 0,_yes6,_no6,tuple(v[_n] for _n in ('sum',)),xp)  # WRF:module_cam_mp_ndrop.F:1578
    v.update(zip(('sum',),_result))
    _outer7=dict(v)
    def _yes7(_carry):
        v=dict(_outer7);v.update(zip(('g1', 'g1sqrt', 'g2', 'g2sqrt', 'm', 'sum'),_carry))
        _outer8=dict(v)
        def _body8(_k,_carry):
            v=dict(_outer8);v.update(zip(('_jump', 'g1', 'g1sqrt', 'g2', 'g2sqrt', 'm', 'sum'),_carry));v['m']=_int(_k)
            _outer9=dict(v)
            def _yes9(_carry):
                v=dict(_outer9);v.update(zip(('g1', 'g1sqrt', 'g2', 'g2sqrt', 'sum'),_carry))
                _put(v,'g1',None,_get(v['zeta'], (v['m'],)) / _get(v['eta'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:1581
                _put(v,'g1sqrt',None,xp.sqrt(v['g1']))  # WRF:module_cam_mp_ndrop.F:1582
                _put(v,'g1',None,v['g1sqrt'] * v['g1'])  # WRF:module_cam_mp_ndrop.F:1583
                _put(v,'g2',None,_get(v['smc'], (v['m'],)) / xp.sqrt(_get(v['eta'], (v['m'],)) + _d(3.0) * _get(v['zeta'], (v['m'],))))  # WRF:module_cam_mp_ndrop.F:1584
                _put(v,'g2sqrt',None,xp.sqrt(v['g2']))  # WRF:module_cam_mp_ndrop.F:1585
                _put(v,'g2',None,v['g2sqrt'] * v['g2'])  # WRF:module_cam_mp_ndrop.F:1586
                _put(v,'sum',None,v['sum'] + (_get(v['f1'], (v['m'],)) * v['g1'] + _get(v['f2'], (v['m'],)) * v['g2']) / (_get(v['smc'], (v['m'],)) * _get(v['smc'], (v['m'],))))  # WRF:module_cam_mp_ndrop.F:1587
                return tuple(v[_n] for _n in ('g1', 'g1sqrt', 'g2', 'g2sqrt', 'sum'))
            def _no9(_carry):
                v=dict(_outer9);v.update(zip(('g1', 'g1sqrt', 'g2', 'g2sqrt', 'sum'),_carry))
                _put(v,'sum',None,_d(1e+20))  # WRF:module_cam_mp_ndrop.F:1589
                return tuple(v[_n] for _n in ('g1', 'g1sqrt', 'g2', 'g2sqrt', 'sum'))
            _result=_cond(_get(v['eta'], (v['m'],)) > _d(1e-20),_yes9,_no9,tuple(v[_n] for _n in ('g1', 'g1sqrt', 'g2', 'g2sqrt', 'sum')),xp)  # WRF:module_cam_mp_ndrop.F:1580
            v.update(zip(('g1', 'g1sqrt', 'g2', 'g2sqrt', 'sum'),_result))
            return tuple(v[_n] for _n in ('_jump', 'g1', 'g1sqrt', 'g2', 'g2sqrt', 'm', 'sum'))
        _result=_loop(1,v['nmode'],1,_body8,tuple(v[_n] for _n in ('_jump', 'g1', 'g1sqrt', 'g2', 'g2sqrt', 'm', 'sum')),xp,jump_index=0)
        v.update(zip(('_jump', 'g1', 'g1sqrt', 'g2', 'g2sqrt', 'm', 'sum'),_result))
        return tuple(v[_n] for _n in ('g1', 'g1sqrt', 'g2', 'g2sqrt', 'm', 'sum'))
    def _no7(_carry):
        v=dict(_outer7);v.update(zip(('g1', 'g1sqrt', 'g2', 'g2sqrt', 'm', 'sum'),_carry))
        return tuple(v[_n] for _n in ('g1', 'g1sqrt', 'g2', 'g2sqrt', 'm', 'sum'))
    _result=_cond(v['_jump'] == 0,_yes7,_no7,tuple(v[_n] for _n in ('g1', 'g1sqrt', 'g2', 'g2sqrt', 'm', 'sum')),xp)  # WRF:module_cam_mp_ndrop.F:1579
    v.update(zip(('g1', 'g1sqrt', 'g2', 'g2sqrt', 'm', 'sum'),_result))
    _outer10=dict(v)
    def _yes10(_carry):
        v=dict(_outer10);v.update(zip(('smax',),_carry))
        _put(v,'smax',None,_d(1.0) / xp.sqrt(v['sum']))  # WRF:module_cam_mp_ndrop.F:1593
        return tuple(v[_n] for _n in ('smax',))
    def _no10(_carry):
        v=dict(_outer10);v.update(zip(('smax',),_carry))
        return tuple(v[_n] for _n in ('smax',))
    _result=_cond(v['_jump'] == 0,_yes10,_no10,tuple(v[_n] for _n in ('smax',)),xp)  # WRF:module_cam_mp_ndrop.F:1593
    v.update(zip(('smax',),_result))
    return (v['smax'],)

def activate_modal(wbar,sigw,wdiab,wminf,wmaxf,tair,rhoair,na,pmode,nmode,volume,sigman,hygro,fn,fm,fluxn,fluxm,flux_fullact,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('ndrop',{}))
    v['wbar']=wbar
    v['sigw']=sigw
    v['wdiab']=wdiab
    v['wminf']=wminf
    v['wmaxf']=wmaxf
    v['tair']=tair
    v['rhoair']=rhoair
    v['na']=na
    v['pmode']=pmode
    v['nmode']=nmode
    v['volume']=volume
    v['sigman']=sigman
    v['hygro']=hygro
    v['fn']=fn
    v['fm']=fm
    v['fluxn']=fluxn
    v['fluxm']=fluxm
    v['flux_fullact']=flux_fullact
    v['_jump']=_int(0)
    v['_fatal']=_int(0)
    v['nx']=_static(200,'i32')
    v['integ']=_zeros((),'f64',xp)
    v['integf']=_zeros((),'f64',xp)
    v['p0']=_cast(_d(101325.0),'f64',xp)
    v['pres']=_zeros((),'f64',xp)
    v['diff0']=_zeros((),'f64',xp)
    v['conduct0']=_zeros((),'f64',xp)
    v['es']=_zeros((),'f64',xp)
    v['qs']=_zeros((),'f64',xp)
    v['dqsdt']=_zeros((),'f64',xp)
    v['g']=_zeros((),'f64',xp)
    v['zeta']=_zeros((v['ntot_amode'],),'f64',xp)
    v['eta']=_zeros((v['ntot_amode'],),'f64',xp)
    v['lnsmax']=_zeros((),'f64',xp)
    v['alpha']=_zeros((),'f64',xp)
    v['gamma']=_zeros((),'f64',xp)
    v['beta']=_zeros((),'f64',xp)
    v['sqrtg']=_zeros((v['ntot_amode'],),'f64',xp)
    v['amcube']=_zeros((v['ntot_amode'],),'f64',xp)
    v['lnsm']=_zeros((v['ntot_amode'],),'f64',xp)
    v['smc']=_zeros((v['ntot_amode'],),'f64',xp)
    v['sumflx_fullact']=_zeros((),'f64',xp)
    v['sumflxn']=_zeros((v['ntot_amode'],),'f64',xp)
    v['sumflxm']=_zeros((v['ntot_amode'],),'f64',xp)
    v['sumfn']=_zeros((v['ntot_amode'],),'f64',xp)
    v['sumfm']=_zeros((v['ntot_amode'],),'f64',xp)
    v['fnold']=_zeros((v['ntot_amode'],),'f64',xp)
    v['fmold']=_zeros((v['ntot_amode'],),'f64',xp)
    v['wold']=_zeros((),'f64',xp)
    v['gold']=_zeros((),'f64',xp)
    v['wmin']=_zeros((),'f64',xp)
    v['wmax']=_zeros((),'f64',xp)
    v['w']=_zeros((),'f64',xp)
    v['dw']=_zeros((),'f64',xp)
    v['dwmax']=_zeros((),'f64',xp)
    v['dwmin']=_zeros((),'f64',xp)
    v['wnuc']=_zeros((),'f64',xp)
    v['dwnew']=_zeros((),'f64',xp)
    v['wb']=_zeros((),'f64',xp)
    v['dfmin']=_zeros((),'f64',xp)
    v['dfmax']=_zeros((),'f64',xp)
    v['fnew']=_zeros((),'f64',xp)
    v['fold']=_zeros((),'f64',xp)
    v['fnmin']=_zeros((),'f64',xp)
    v['fnbar']=_zeros((),'f64',xp)
    v['fmbar']=_zeros((),'f64',xp)
    v['alw']=_zeros((),'f64',xp)
    v['sqrtalw']=_zeros((),'f64',xp)
    v['smax']=_zeros((),'f64',xp)
    v['x']=_zeros((),'f64',xp)
    v['arg']=_zeros((),'f64',xp)
    v['xmincoeff']=_zeros((),'f64',xp)
    v['z']=_zeros((),'f64',xp)
    v['z1']=_zeros((),'f64',xp)
    v['z2']=_zeros((),'f64',xp)
    v['wf1']=_zeros((),'f64',xp)
    v['wf2']=_zeros((),'f64',xp)
    v['zf1']=_zeros((),'f64',xp)
    v['zf2']=_zeros((),'f64',xp)
    v['gf1']=_zeros((),'f64',xp)
    v['gf2']=_zeros((),'f64',xp)
    v['gf']=_zeros((),'f64',xp)
    v['etafactor1']=_zeros((),'f64',xp)
    v['etafactor2']=_zeros((v['ntot_amode'],),'f64',xp)
    v['etafactor2max']=_zeros((),'f64',xp)
    v['m']=_zeros((),'i32',xp)
    v['n']=_zeros((),'i32',xp)
    v['eps']=_cast(_d(0.3),'f64',xp)
    v['fmax']=_cast(_d(0.99),'f64',xp)
    v['sds']=_cast(_d(3.0),'f64',xp)
    v['ndist']=_zeros((v['nx'],),'i32',xp)
    _outer2=dict(v)
    def _yes2(_carry):
        v=dict(_outer2);v.update(zip(('_fatal',),_carry))
        _outer3=dict(v)
        def _yes3(_carry):
            v=dict(_outer3);v.update(zip(('_fatal',),_carry))
            v['_fatal']=v['_fatal']+_int(1)  # Original WRF fatal event:module_cam_mp_ndrop.F:1213; caller must refuse
            return tuple(v[_n] for _n in ('_fatal',))
        def _no3(_carry):
            v=dict(_outer3);v.update(zip(('_fatal',),_carry))
            return tuple(v[_n] for _n in ('_fatal',))
        _result=_cond(v['ntot_amode'] < v['pmode'],_yes3,_no3,tuple(v[_n] for _n in ('_fatal',)),xp)  # WRF:module_cam_mp_ndrop.F:1208
        v.update(zip(('_fatal',),_result))
        return tuple(v[_n] for _n in ('_fatal',))
    def _no2(_carry):
        v=dict(_outer2);v.update(zip(('_fatal',),_carry))
        return tuple(v[_n] for _n in ('_fatal',))
    _result=_cond(v['_jump'] == 0,_yes2,_no2,tuple(v[_n] for _n in ('_fatal',)),xp)  # WRF:module_cam_mp_ndrop.F:1208
    v.update(zip(('_fatal',),_result))
    _outer4=dict(v)
    def _yes4(_carry):
        v=dict(_outer4);v.update(zip(('fn',),_carry))
        _put(v,'fn',(_slice(None,None,None),),_d(0.0))  # WRF:module_cam_mp_ndrop.F:1216
        return tuple(v[_n] for _n in ('fn',))
    def _no4(_carry):
        v=dict(_outer4);v.update(zip(('fn',),_carry))
        return tuple(v[_n] for _n in ('fn',))
    _result=_cond(v['_jump'] == 0,_yes4,_no4,tuple(v[_n] for _n in ('fn',)),xp)  # WRF:module_cam_mp_ndrop.F:1216
    v.update(zip(('fn',),_result))
    _outer5=dict(v)
    def _yes5(_carry):
        v=dict(_outer5);v.update(zip(('fm',),_carry))
        _put(v,'fm',(_slice(None,None,None),),_d(0.0))  # WRF:module_cam_mp_ndrop.F:1217
        return tuple(v[_n] for _n in ('fm',))
    def _no5(_carry):
        v=dict(_outer5);v.update(zip(('fm',),_carry))
        return tuple(v[_n] for _n in ('fm',))
    _result=_cond(v['_jump'] == 0,_yes5,_no5,tuple(v[_n] for _n in ('fm',)),xp)  # WRF:module_cam_mp_ndrop.F:1217
    v.update(zip(('fm',),_result))
    _outer6=dict(v)
    def _yes6(_carry):
        v=dict(_outer6);v.update(zip(('fluxn',),_carry))
        _put(v,'fluxn',(_slice(None,None,None),),_d(0.0))  # WRF:module_cam_mp_ndrop.F:1218
        return tuple(v[_n] for _n in ('fluxn',))
    def _no6(_carry):
        v=dict(_outer6);v.update(zip(('fluxn',),_carry))
        return tuple(v[_n] for _n in ('fluxn',))
    _result=_cond(v['_jump'] == 0,_yes6,_no6,tuple(v[_n] for _n in ('fluxn',)),xp)  # WRF:module_cam_mp_ndrop.F:1218
    v.update(zip(('fluxn',),_result))
    _outer7=dict(v)
    def _yes7(_carry):
        v=dict(_outer7);v.update(zip(('fluxm',),_carry))
        _put(v,'fluxm',(_slice(None,None,None),),_d(0.0))  # WRF:module_cam_mp_ndrop.F:1219
        return tuple(v[_n] for _n in ('fluxm',))
    def _no7(_carry):
        v=dict(_outer7);v.update(zip(('fluxm',),_carry))
        return tuple(v[_n] for _n in ('fluxm',))
    _result=_cond(v['_jump'] == 0,_yes7,_no7,tuple(v[_n] for _n in ('fluxm',)),xp)  # WRF:module_cam_mp_ndrop.F:1219
    v.update(zip(('fluxm',),_result))
    _outer8=dict(v)
    def _yes8(_carry):
        v=dict(_outer8);v.update(zip(('flux_fullact',),_carry))
        _put(v,'flux_fullact',None,_d(0.0))  # WRF:module_cam_mp_ndrop.F:1220
        return tuple(v[_n] for _n in ('flux_fullact',))
    def _no8(_carry):
        v=dict(_outer8);v.update(zip(('flux_fullact',),_carry))
        return tuple(v[_n] for _n in ('flux_fullact',))
    _result=_cond(v['_jump'] == 0,_yes8,_no8,tuple(v[_n] for _n in ('flux_fullact',)),xp)  # WRF:module_cam_mp_ndrop.F:1220
    v.update(zip(('flux_fullact',),_result))
    _outer9=dict(v)
    def _yes9(_carry):
        v=dict(_outer9);v.update(zip(('_jump',),_carry))
        _outer10=dict(v)
        def _yes10(_carry):
            v=dict(_outer10);v.update(zip(('_jump',),_carry))
            v['_jump']=_int(-1)
            return tuple(v[_n] for _n in ('_jump',))
        def _no10(_carry):
            v=dict(_outer10);v.update(zip(('_jump',),_carry))
            return tuple(v[_n] for _n in ('_jump',))
        _result=_cond(xp.logical_and(v['nmode'] == 1, _get(v['na'], (1,)) < _d(1e-20)),_yes10,_no10,tuple(v[_n] for _n in ('_jump',)),xp)  # WRF:module_cam_mp_ndrop.F:1222
        v.update(zip(('_jump',),_result))
        return tuple(v[_n] for _n in ('_jump',))
    def _no9(_carry):
        v=dict(_outer9);v.update(zip(('_jump',),_carry))
        return tuple(v[_n] for _n in ('_jump',))
    _result=_cond(v['_jump'] == 0,_yes9,_no9,tuple(v[_n] for _n in ('_jump',)),xp)  # WRF:module_cam_mp_ndrop.F:1222
    v.update(zip(('_jump',),_result))
    _outer11=dict(v)
    def _yes11(_carry):
        v=dict(_outer11);v.update(zip(('_jump',),_carry))
        _outer12=dict(v)
        def _yes12(_carry):
            v=dict(_outer12);v.update(zip(('_jump',),_carry))
            v['_jump']=_int(-1)
            return tuple(v[_n] for _n in ('_jump',))
        def _no12(_carry):
            v=dict(_outer12);v.update(zip(('_jump',),_carry))
            return tuple(v[_n] for _n in ('_jump',))
        _result=_cond(xp.logical_and(v['sigw'] <= _d(1e-05), v['wbar'] <= _r(0.0)),_yes12,_no12,tuple(v[_n] for _n in ('_jump',)),xp)  # WRF:module_cam_mp_ndrop.F:1224
        v.update(zip(('_jump',),_result))
        return tuple(v[_n] for _n in ('_jump',))
    def _no11(_carry):
        v=dict(_outer11);v.update(zip(('_jump',),_carry))
        return tuple(v[_n] for _n in ('_jump',))
    _result=_cond(v['_jump'] == 0,_yes11,_no11,tuple(v[_n] for _n in ('_jump',)),xp)  # WRF:module_cam_mp_ndrop.F:1224
    v.update(zip(('_jump',),_result))
    _outer13=dict(v)
    def _yes13(_carry):
        v=dict(_outer13);v.update(zip(('pres',),_carry))
        _put(v,'pres',None,v['rair'] * v['rhoair'] * v['tair'])  # WRF:module_cam_mp_ndrop.F:1226
        return tuple(v[_n] for _n in ('pres',))
    def _no13(_carry):
        v=dict(_outer13);v.update(zip(('pres',),_carry))
        return tuple(v[_n] for _n in ('pres',))
    _result=_cond(v['_jump'] == 0,_yes13,_no13,tuple(v[_n] for _n in ('pres',)),xp)  # WRF:module_cam_mp_ndrop.F:1226
    v.update(zip(('pres',),_result))
    _outer14=dict(v)
    def _yes14(_carry):
        v=dict(_outer14);v.update(zip(('diff0',),_carry))
        _put(v,'diff0',None,_d(2.11e-05) * (v['p0'] / v['pres']) * _pow(v['tair'] / v['t0'], _r(1.94), xp))  # WRF:module_cam_mp_ndrop.F:1227
        return tuple(v[_n] for _n in ('diff0',))
    def _no14(_carry):
        v=dict(_outer14);v.update(zip(('diff0',),_carry))
        return tuple(v[_n] for _n in ('diff0',))
    _result=_cond(v['_jump'] == 0,_yes14,_no14,tuple(v[_n] for _n in ('diff0',)),xp)  # WRF:module_cam_mp_ndrop.F:1227
    v.update(zip(('diff0',),_result))
    _outer15=dict(v)
    def _yes15(_carry):
        v=dict(_outer15);v.update(zip(('conduct0',),_carry))
        _put(v,'conduct0',None,(_d(5.69) + _d(0.017) * (v['tair'] - v['t0'])) * _d(418.6) * _d(1e-05))  # WRF:module_cam_mp_ndrop.F:1228
        return tuple(v[_n] for _n in ('conduct0',))
    def _no15(_carry):
        v=dict(_outer15);v.update(zip(('conduct0',),_carry))
        return tuple(v[_n] for _n in ('conduct0',))
    _result=_cond(v['_jump'] == 0,_yes15,_no15,tuple(v[_n] for _n in ('conduct0',)),xp)  # WRF:module_cam_mp_ndrop.F:1228
    v.update(zip(('conduct0',),_result))
    _outer16=dict(v)
    def _yes16(_carry):
        v=dict(_outer16);v.update(zip(('es',),_carry))
        _put(v,'es',None,estblf(v['tair'], xp=xp, _globals=_globals))  # WRF:module_cam_mp_ndrop.F:1229
        return tuple(v[_n] for _n in ('es',))
    def _no16(_carry):
        v=dict(_outer16);v.update(zip(('es',),_carry))
        return tuple(v[_n] for _n in ('es',))
    _result=_cond(v['_jump'] == 0,_yes16,_no16,tuple(v[_n] for _n in ('es',)),xp)  # WRF:module_cam_mp_ndrop.F:1229
    v.update(zip(('es',),_result))
    _outer17=dict(v)
    def _yes17(_carry):
        v=dict(_outer17);v.update(zip(('qs',),_carry))
        _put(v,'qs',None,v['epsilo'] * v['es'] / (v['pres'] - (_d(1.0) - v['epsqs']) * v['es']))  # WRF:module_cam_mp_ndrop.F:1230
        return tuple(v[_n] for _n in ('qs',))
    def _no17(_carry):
        v=dict(_outer17);v.update(zip(('qs',),_carry))
        return tuple(v[_n] for _n in ('qs',))
    _result=_cond(v['_jump'] == 0,_yes17,_no17,tuple(v[_n] for _n in ('qs',)),xp)  # WRF:module_cam_mp_ndrop.F:1230
    v.update(zip(('qs',),_result))
    _outer18=dict(v)
    def _yes18(_carry):
        v=dict(_outer18);v.update(zip(('dqsdt',),_carry))
        _put(v,'dqsdt',None,v['latvap'] / (v['rh2o'] * v['tair'] * v['tair']) * v['qs'])  # WRF:module_cam_mp_ndrop.F:1231
        return tuple(v[_n] for _n in ('dqsdt',))
    def _no18(_carry):
        v=dict(_outer18);v.update(zip(('dqsdt',),_carry))
        return tuple(v[_n] for _n in ('dqsdt',))
    _result=_cond(v['_jump'] == 0,_yes18,_no18,tuple(v[_n] for _n in ('dqsdt',)),xp)  # WRF:module_cam_mp_ndrop.F:1231
    v.update(zip(('dqsdt',),_result))
    _outer19=dict(v)
    def _yes19(_carry):
        v=dict(_outer19);v.update(zip(('alpha',),_carry))
        _put(v,'alpha',None,v['gravit'] * (v['latvap'] / (v['cpair'] * v['rh2o'] * v['tair'] * v['tair']) - _r(1.0) / (v['rair'] * v['tair'])))  # WRF:module_cam_mp_ndrop.F:1232
        return tuple(v[_n] for _n in ('alpha',))
    def _no19(_carry):
        v=dict(_outer19);v.update(zip(('alpha',),_carry))
        return tuple(v[_n] for _n in ('alpha',))
    _result=_cond(v['_jump'] == 0,_yes19,_no19,tuple(v[_n] for _n in ('alpha',)),xp)  # WRF:module_cam_mp_ndrop.F:1232
    v.update(zip(('alpha',),_result))
    _outer20=dict(v)
    def _yes20(_carry):
        v=dict(_outer20);v.update(zip(('gamma',),_carry))
        _put(v,'gamma',None,(1 + v['latvap'] / v['cpair'] * v['dqsdt']) / (v['rhoair'] * v['qs']))  # WRF:module_cam_mp_ndrop.F:1233
        return tuple(v[_n] for _n in ('gamma',))
    def _no20(_carry):
        v=dict(_outer20);v.update(zip(('gamma',),_carry))
        return tuple(v[_n] for _n in ('gamma',))
    _result=_cond(v['_jump'] == 0,_yes20,_no20,tuple(v[_n] for _n in ('gamma',)),xp)  # WRF:module_cam_mp_ndrop.F:1233
    v.update(zip(('gamma',),_result))
    _outer21=dict(v)
    def _yes21(_carry):
        v=dict(_outer21);v.update(zip(('etafactor2max',),_carry))
        _put(v,'etafactor2max',None,_r(10000000000.0) / _pow(v['alpha'] * v['wmaxf'], _r(1.5), xp))  # WRF:module_cam_mp_ndrop.F:1234
        return tuple(v[_n] for _n in ('etafactor2max',))
    def _no21(_carry):
        v=dict(_outer21);v.update(zip(('etafactor2max',),_carry))
        return tuple(v[_n] for _n in ('etafactor2max',))
    _result=_cond(v['_jump'] == 0,_yes21,_no21,tuple(v[_n] for _n in ('etafactor2max',)),xp)  # WRF:module_cam_mp_ndrop.F:1234
    v.update(zip(('etafactor2max',),_result))
    _outer22=dict(v)
    def _yes22(_carry):
        v=dict(_outer22);v.update(zip(('amcube', 'beta', 'etafactor2', 'g', 'lnsm', 'm', 'smc', 'sqrtg'),_carry))
        _outer23=dict(v)
        def _body23(_k,_carry):
            v=dict(_outer23);v.update(zip(('_jump', 'amcube', 'beta', 'etafactor2', 'g', 'lnsm', 'm', 'smc', 'sqrtg'),_carry));v['m']=_int(_k)
            _outer24=dict(v)
            def _yes24(_carry):
                v=dict(_outer24);v.update(zip(('amcube', 'beta', 'etafactor2', 'g', 'smc', 'sqrtg'),_carry))
                _put(v,'amcube',(v['m'],),_r(3.0) * _get(v['volume'], (v['m'],)) / (_r(4.0) * v['pi'] * _get(v['exp45logsig'], (v['m'],)) * _get(v['na'], (v['m'],))))  # WRF:module_cam_mp_ndrop.F:1240
                _put(v,'g',None,_d(1.0) / (v['rhoh2o'] / (v['diff0'] * v['rhoair'] * v['qs']) + v['latvap'] * v['rhoh2o'] / (v['conduct0'] * v['tair']) * (v['latvap'] / (v['rh2o'] * v['tair']) - _d(1.0))))  # WRF:module_cam_mp_ndrop.F:1245
                _put(v,'sqrtg',(v['m'],),xp.sqrt(v['g']))  # WRF:module_cam_mp_ndrop.F:1247
                _put(v,'beta',None,_d(2.0) * v['pi'] * v['rhoh2o'] * v['g'] * v['gamma'])  # WRF:module_cam_mp_ndrop.F:1248
                _put(v,'etafactor2',(v['m'],),_d(1.0) / (_get(v['na'], (v['m'],)) * v['beta'] * _get(v['sqrtg'], (v['m'],))))  # WRF:module_cam_mp_ndrop.F:1249
                _outer25=dict(v)
                def _yes25(_carry):
                    v=dict(_outer25);v.update(zip(('smc',),_carry))
                    _put(v,'smc',(v['m'],),_r(2.0) * v['aten'] * xp.sqrt(v['aten'] / (_r(27.0) * _get(v['hygro'], (v['m'],)) * _get(v['amcube'], (v['m'],)))))  # WRF:module_cam_mp_ndrop.F:1251
                    return tuple(v[_n] for _n in ('smc',))
                def _no25(_carry):
                    v=dict(_outer25);v.update(zip(('smc',),_carry))
                    _put(v,'smc',(v['m'],),_r(100.0))  # WRF:module_cam_mp_ndrop.F:1253
                    return tuple(v[_n] for _n in ('smc',))
                _result=_cond(_get(v['hygro'], (v['m'],)) > _r(1e-10),_yes25,_no25,tuple(v[_n] for _n in ('smc',)),xp)  # WRF:module_cam_mp_ndrop.F:1250
                v.update(zip(('smc',),_result))
                return tuple(v[_n] for _n in ('amcube', 'beta', 'etafactor2', 'g', 'smc', 'sqrtg'))
            def _no24(_carry):
                v=dict(_outer24);v.update(zip(('amcube', 'beta', 'etafactor2', 'g', 'smc', 'sqrtg'),_carry))
                _put(v,'g',None,_d(1.0) / (v['rhoh2o'] / (v['diff0'] * v['rhoair'] * v['qs']) + v['latvap'] * v['rhoh2o'] / (v['conduct0'] * v['tair']) * (v['latvap'] / (v['rh2o'] * v['tair']) - _d(1.0))))  # WRF:module_cam_mp_ndrop.F:1257
                _put(v,'sqrtg',(v['m'],),xp.sqrt(v['g']))  # WRF:module_cam_mp_ndrop.F:1259
                _put(v,'smc',(v['m'],),_d(1.0))  # WRF:module_cam_mp_ndrop.F:1260
                _put(v,'etafactor2',(v['m'],),v['etafactor2max'])  # WRF:module_cam_mp_ndrop.F:1261
                return tuple(v[_n] for _n in ('amcube', 'beta', 'etafactor2', 'g', 'smc', 'sqrtg'))
            _result=_cond(xp.logical_and(_get(v['volume'], (v['m'],)) > _d(1e-39), _get(v['na'], (v['m'],)) > _d(1e-39)),_yes24,_no24,tuple(v[_n] for _n in ('amcube', 'beta', 'etafactor2', 'g', 'smc', 'sqrtg')),xp)  # WRF:module_cam_mp_ndrop.F:1237
            v.update(zip(('amcube', 'beta', 'etafactor2', 'g', 'smc', 'sqrtg'),_result))
            _put(v,'lnsm',(v['m'],),xp.log(_get(v['smc'], (v['m'],))))  # WRF:module_cam_mp_ndrop.F:1263
            return tuple(v[_n] for _n in ('_jump', 'amcube', 'beta', 'etafactor2', 'g', 'lnsm', 'm', 'smc', 'sqrtg'))
        _result=_loop(1,v['nmode'],1,_body23,tuple(v[_n] for _n in ('_jump', 'amcube', 'beta', 'etafactor2', 'g', 'lnsm', 'm', 'smc', 'sqrtg')),xp,jump_index=0)
        v.update(zip(('_jump', 'amcube', 'beta', 'etafactor2', 'g', 'lnsm', 'm', 'smc', 'sqrtg'),_result))
        return tuple(v[_n] for _n in ('amcube', 'beta', 'etafactor2', 'g', 'lnsm', 'm', 'smc', 'sqrtg'))
    def _no22(_carry):
        v=dict(_outer22);v.update(zip(('amcube', 'beta', 'etafactor2', 'g', 'lnsm', 'm', 'smc', 'sqrtg'),_carry))
        return tuple(v[_n] for _n in ('amcube', 'beta', 'etafactor2', 'g', 'lnsm', 'm', 'smc', 'sqrtg'))
    _result=_cond(v['_jump'] == 0,_yes22,_no22,tuple(v[_n] for _n in ('amcube', 'beta', 'etafactor2', 'g', 'lnsm', 'm', 'smc', 'sqrtg')),xp)  # WRF:module_cam_mp_ndrop.F:1236
    v.update(zip(('amcube', 'beta', 'etafactor2', 'g', 'lnsm', 'm', 'smc', 'sqrtg'),_result))
    _outer26=dict(v)
    def _yes26(_carry):
        v=dict(_outer26);v.update(zip(('_fatal', '_jump', 'alw', 'arg', 'dfmax', 'dfmin', 'dw', 'dwmax', 'dwmin', 'dwnew', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnew', 'fnmin', 'fnold', 'fold', 'g', 'gf', 'gf1', 'gf2', 'gold', 'integ', 'integf', 'lnsmax', 'm', 'n', 'ndist', 'smax', 'sqrtalw', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'w', 'wb', 'wf1', 'wf2', 'wmax', 'wmin', 'wnuc', 'wold', 'x', 'xmincoeff', 'z', 'z1', 'z2', 'zeta', 'zf1', 'zf2'),_carry))
        _outer27=dict(v)
        def _yes27(_carry):
            v=dict(_outer27);v.update(zip(('_fatal', '_jump', 'alw', 'arg', 'dfmax', 'dfmin', 'dw', 'dwmax', 'dwmin', 'dwnew', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnew', 'fnmin', 'fnold', 'fold', 'g', 'gf', 'gf1', 'gf2', 'gold', 'integ', 'integf', 'lnsmax', 'm', 'n', 'ndist', 'smax', 'sqrtalw', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'w', 'wb', 'wf1', 'wf2', 'wmax', 'wmin', 'wnuc', 'wold', 'x', 'xmincoeff', 'z', 'z1', 'z2', 'zeta', 'zf1', 'zf2'),_carry))
            _put(v,'wmax',None,xp.minimum(v['wmaxf'], v['wbar'] + v['sds'] * v['sigw']))  # WRF:module_cam_mp_ndrop.F:1270
            _put(v,'wmin',None,xp.maximum(v['wminf'], _neg(v['wdiab'])))  # WRF:module_cam_mp_ndrop.F:1271
            _put(v,'wmin',None,xp.maximum(v['wmin'], v['wbar'] - v['sds'] * v['sigw']))  # WRF:module_cam_mp_ndrop.F:1272
            _put(v,'w',None,v['wmin'])  # WRF:module_cam_mp_ndrop.F:1273
            _put(v,'dwmax',None,v['eps'] * v['sigw'])  # WRF:module_cam_mp_ndrop.F:1274
            _put(v,'dw',None,v['dwmax'])  # WRF:module_cam_mp_ndrop.F:1275
            _put(v,'dfmax',None,_d(0.2))  # WRF:module_cam_mp_ndrop.F:1276
            _put(v,'dfmin',None,_d(0.1))  # WRF:module_cam_mp_ndrop.F:1277
            _outer28=dict(v)
            def _yes28(_carry):
                v=dict(_outer28);v.update(zip(('_jump', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'm'),_carry))
                _outer29=dict(v)
                def _body29(_k,_carry):
                    v=dict(_outer29);v.update(zip(('_jump', 'fluxm', 'fluxn', 'fm', 'fn', 'm'),_carry));v['m']=_int(_k)
                    _put(v,'fluxn',(v['m'],),_r(0.0))  # WRF:module_cam_mp_ndrop.F:1280
                    _put(v,'fn',(v['m'],),_r(0.0))  # WRF:module_cam_mp_ndrop.F:1281
                    _put(v,'fluxm',(v['m'],),_r(0.0))  # WRF:module_cam_mp_ndrop.F:1282
                    _put(v,'fm',(v['m'],),_r(0.0))  # WRF:module_cam_mp_ndrop.F:1283
                    return tuple(v[_n] for _n in ('_jump', 'fluxm', 'fluxn', 'fm', 'fn', 'm'))
                _result=_loop(1,v['nmode'],1,_body29,tuple(v[_n] for _n in ('_jump', 'fluxm', 'fluxn', 'fm', 'fn', 'm')),xp,jump_index=0)
                v.update(zip(('_jump', 'fluxm', 'fluxn', 'fm', 'fn', 'm'),_result))
                _put(v,'flux_fullact',None,_d(0.0))  # WRF:module_cam_mp_ndrop.F:1285
                v['_jump']=_int(-1)
                return tuple(v[_n] for _n in ('_jump', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'm'))
            def _no28(_carry):
                v=dict(_outer28);v.update(zip(('_jump', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'm'),_carry))
                return tuple(v[_n] for _n in ('_jump', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'm'))
            _result=_cond(v['wmax'] <= v['w'],_yes28,_no28,tuple(v[_n] for _n in ('_jump', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'm')),xp)  # WRF:module_cam_mp_ndrop.F:1278
            v.update(zip(('_jump', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'm'),_result))
            _outer30=dict(v)
            def _body30(_k,_carry):
                v=dict(_outer30);v.update(zip(('_jump', 'fmold', 'fnold', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn'),_carry));v['m']=_int(_k)
                _put(v,'sumflxn',(v['m'],),_d(0.0))  # WRF:module_cam_mp_ndrop.F:1289
                _put(v,'sumfn',(v['m'],),_d(0.0))  # WRF:module_cam_mp_ndrop.F:1290
                _put(v,'fnold',(v['m'],),_d(0.0))  # WRF:module_cam_mp_ndrop.F:1291
                _put(v,'sumflxm',(v['m'],),_d(0.0))  # WRF:module_cam_mp_ndrop.F:1292
                _put(v,'sumfm',(v['m'],),_d(0.0))  # WRF:module_cam_mp_ndrop.F:1293
                _put(v,'fmold',(v['m'],),_d(0.0))  # WRF:module_cam_mp_ndrop.F:1294
                return tuple(v[_n] for _n in ('_jump', 'fmold', 'fnold', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn'))
            _result=_loop(1,v['nmode'],1,_body30,tuple(v[_n] for _n in ('_jump', 'fmold', 'fnold', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn')),xp,jump_index=0)
            v.update(zip(('_jump', 'fmold', 'fnold', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn'),_result))
            _put(v,'sumflx_fullact',None,_d(0.0))  # WRF:module_cam_mp_ndrop.F:1296
            _put(v,'fold',None,_d(0.0))  # WRF:module_cam_mp_ndrop.F:1298
            _put(v,'wold',None,_d(0.0))  # WRF:module_cam_mp_ndrop.F:1299
            _put(v,'gold',None,_d(0.0))  # WRF:module_cam_mp_ndrop.F:1300
            _put(v,'dwmin',None,xp.minimum(v['dwmax'], _d(0.01)))  # WRF:module_cam_mp_ndrop.F:1302
            _outer31=dict(v)
            def _body31(_k,_carry):
                v=dict(_outer31);v.update(zip(('_jump', 'alw', 'arg', 'dw', 'dwnew', 'eta', 'etafactor1', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnew', 'fnmin', 'fnold', 'fold', 'g', 'gold', 'lnsmax', 'm', 'n', 'smax', 'sqrtalw', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'w', 'wb', 'wnuc', 'wold', 'x', 'xmincoeff', 'z', 'zeta'),_carry));v['n']=_int(_k)
                _outer32=dict(v)
                def _yes32(_carry):
                    v=dict(_outer32);v.update(zip(('_jump', 'alw', 'dw', 'dwnew', 'eta', 'etafactor1', 'fnew', 'lnsmax', 'm', 'smax', 'sqrtalw', 'w', 'wnuc', 'x', 'zeta'),_carry))
                    _outer33=dict(v)
                    def _retrybody33(_carry):
                        v=dict(_outer33);v.update(zip(('_jump', 'alw', 'dw', 'dwnew', 'eta', 'etafactor1', 'fnew', 'lnsmax', 'm', 'smax', 'sqrtalw', 'w', 'wnuc', 'x', 'zeta'),_carry))
                        _outer34=dict(v)
                        def _yes34(_carry):
                            v=dict(_outer34);v.update(zip(('wnuc',),_carry))
                            _put(v,'wnuc',None,v['w'] + v['wdiab'])  # WRF:module_cam_mp_ndrop.F:1305
                            return tuple(v[_n] for _n in ('wnuc',))
                        def _no34(_carry):
                            v=dict(_outer34);v.update(zip(('wnuc',),_carry))
                            return tuple(v[_n] for _n in ('wnuc',))
                        _result=_cond(v['_jump'] == 0,_yes34,_no34,tuple(v[_n] for _n in ('wnuc',)),xp)  # WRF:module_cam_mp_ndrop.F:1305
                        v.update(zip(('wnuc',),_result))
                        _outer35=dict(v)
                        def _yes35(_carry):
                            v=dict(_outer35);v.update(zip(('alw',),_carry))
                            _put(v,'alw',None,v['alpha'] * v['wnuc'])  # WRF:module_cam_mp_ndrop.F:1307
                            return tuple(v[_n] for _n in ('alw',))
                        def _no35(_carry):
                            v=dict(_outer35);v.update(zip(('alw',),_carry))
                            return tuple(v[_n] for _n in ('alw',))
                        _result=_cond(v['_jump'] == 0,_yes35,_no35,tuple(v[_n] for _n in ('alw',)),xp)  # WRF:module_cam_mp_ndrop.F:1307
                        v.update(zip(('alw',),_result))
                        _outer36=dict(v)
                        def _yes36(_carry):
                            v=dict(_outer36);v.update(zip(('sqrtalw',),_carry))
                            _put(v,'sqrtalw',None,xp.sqrt(v['alw']))  # WRF:module_cam_mp_ndrop.F:1308
                            return tuple(v[_n] for _n in ('sqrtalw',))
                        def _no36(_carry):
                            v=dict(_outer36);v.update(zip(('sqrtalw',),_carry))
                            return tuple(v[_n] for _n in ('sqrtalw',))
                        _result=_cond(v['_jump'] == 0,_yes36,_no36,tuple(v[_n] for _n in ('sqrtalw',)),xp)  # WRF:module_cam_mp_ndrop.F:1308
                        v.update(zip(('sqrtalw',),_result))
                        _outer37=dict(v)
                        def _yes37(_carry):
                            v=dict(_outer37);v.update(zip(('etafactor1',),_carry))
                            _put(v,'etafactor1',None,v['alw'] * v['sqrtalw'])  # WRF:module_cam_mp_ndrop.F:1309
                            return tuple(v[_n] for _n in ('etafactor1',))
                        def _no37(_carry):
                            v=dict(_outer37);v.update(zip(('etafactor1',),_carry))
                            return tuple(v[_n] for _n in ('etafactor1',))
                        _result=_cond(v['_jump'] == 0,_yes37,_no37,tuple(v[_n] for _n in ('etafactor1',)),xp)  # WRF:module_cam_mp_ndrop.F:1309
                        v.update(zip(('etafactor1',),_result))
                        _outer38=dict(v)
                        def _yes38(_carry):
                            v=dict(_outer38);v.update(zip(('eta', 'm', 'zeta'),_carry))
                            _outer39=dict(v)
                            def _body39(_k,_carry):
                                v=dict(_outer39);v.update(zip(('_jump', 'eta', 'm', 'zeta'),_carry));v['m']=_int(_k)
                                _put(v,'eta',(v['m'],),v['etafactor1'] * _get(v['etafactor2'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:1312
                                _put(v,'zeta',(v['m'],),v['twothird'] * v['sqrtalw'] * v['aten'] / _get(v['sqrtg'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:1313
                                return tuple(v[_n] for _n in ('_jump', 'eta', 'm', 'zeta'))
                            _result=_loop(1,v['nmode'],1,_body39,tuple(v[_n] for _n in ('_jump', 'eta', 'm', 'zeta')),xp,jump_index=0)
                            v.update(zip(('_jump', 'eta', 'm', 'zeta'),_result))
                            return tuple(v[_n] for _n in ('eta', 'm', 'zeta'))
                        def _no38(_carry):
                            v=dict(_outer38);v.update(zip(('eta', 'm', 'zeta'),_carry))
                            return tuple(v[_n] for _n in ('eta', 'm', 'zeta'))
                        _result=_cond(v['_jump'] == 0,_yes38,_no38,tuple(v[_n] for _n in ('eta', 'm', 'zeta')),xp)  # WRF:module_cam_mp_ndrop.F:1311
                        v.update(zip(('eta', 'm', 'zeta'),_result))
                        _outer40=dict(v)
                        def _yes40(_carry):
                            v=dict(_outer40);v.update(zip(('smax',),_carry))
                            _call=maxsat(v['zeta'],v['eta'],v['nmode'],v['smc'],v['smax'],_globals=_globals,xp=xp)
                            _put(v,'smax',None,_call[0])  # WRF:module_cam_mp_ndrop.F:1316
                            return tuple(v[_n] for _n in ('smax',))
                        def _no40(_carry):
                            v=dict(_outer40);v.update(zip(('smax',),_carry))
                            return tuple(v[_n] for _n in ('smax',))
                        _result=_cond(v['_jump'] == 0,_yes40,_no40,tuple(v[_n] for _n in ('smax',)),xp)  # WRF:module_cam_mp_ndrop.F:1316
                        v.update(zip(('smax',),_result))
                        _outer41=dict(v)
                        def _yes41(_carry):
                            v=dict(_outer41);v.update(zip(('lnsmax',),_carry))
                            _put(v,'lnsmax',None,xp.log(v['smax']))  # WRF:module_cam_mp_ndrop.F:1319
                            return tuple(v[_n] for _n in ('lnsmax',))
                        def _no41(_carry):
                            v=dict(_outer41);v.update(zip(('lnsmax',),_carry))
                            return tuple(v[_n] for _n in ('lnsmax',))
                        _result=_cond(v['_jump'] == 0,_yes41,_no41,tuple(v[_n] for _n in ('lnsmax',)),xp)  # WRF:module_cam_mp_ndrop.F:1319
                        v.update(zip(('lnsmax',),_result))
                        _outer42=dict(v)
                        def _yes42(_carry):
                            v=dict(_outer42);v.update(zip(('x',),_carry))
                            _put(v,'x',None,v['twothird'] * (_get(v['lnsm'], (v['nmode'],)) - v['lnsmax']) / (v['sq2'] * _get(v['alogsig'], (v['nmode'],))))  # WRF:module_cam_mp_ndrop.F:1321
                            return tuple(v[_n] for _n in ('x',))
                        def _no42(_carry):
                            v=dict(_outer42);v.update(zip(('x',),_carry))
                            return tuple(v[_n] for _n in ('x',))
                        _result=_cond(v['_jump'] == 0,_yes42,_no42,tuple(v[_n] for _n in ('x',)),xp)  # WRF:module_cam_mp_ndrop.F:1321
                        v.update(zip(('x',),_result))
                        _outer43=dict(v)
                        def _yes43(_carry):
                            v=dict(_outer43);v.update(zip(('fnew',),_carry))
                            _put(v,'fnew',None,_d(0.5) * (_d(1.0) - erf(v['x'], xp=xp, _globals=_globals)))  # WRF:module_cam_mp_ndrop.F:1322
                            return tuple(v[_n] for _n in ('fnew',))
                        def _no43(_carry):
                            v=dict(_outer43);v.update(zip(('fnew',),_carry))
                            return tuple(v[_n] for _n in ('fnew',))
                        _result=_cond(v['_jump'] == 0,_yes43,_no43,tuple(v[_n] for _n in ('fnew',)),xp)  # WRF:module_cam_mp_ndrop.F:1322
                        v.update(zip(('fnew',),_result))
                        _outer44=dict(v)
                        def _yes44(_carry):
                            v=dict(_outer44);v.update(zip(('dwnew',),_carry))
                            _put(v,'dwnew',None,v['dw'])  # WRF:module_cam_mp_ndrop.F:1325
                            return tuple(v[_n] for _n in ('dwnew',))
                        def _no44(_carry):
                            v=dict(_outer44);v.update(zip(('dwnew',),_carry))
                            return tuple(v[_n] for _n in ('dwnew',))
                        _result=_cond(v['_jump'] == 0,_yes44,_no44,tuple(v[_n] for _n in ('dwnew',)),xp)  # WRF:module_cam_mp_ndrop.F:1325
                        v.update(zip(('dwnew',),_result))
                        _outer45=dict(v)
                        def _yes45(_carry):
                            v=dict(_outer45);v.update(zip(('_jump', 'dw', 'dwnew', 'w'),_carry))
                            _outer46=dict(v)
                            def _yes46(_carry):
                                v=dict(_outer46);v.update(zip(('_jump', 'dw', 'dwnew', 'w'),_carry))
                                _outer47=dict(v)
                                def _yes47(_carry):
                                    v=dict(_outer47);v.update(zip(('_jump', 'dw', 'dwnew', 'w'),_carry))
                                    _put(v,'dw',None,_d(0.7) * v['dw'])  # WRF:module_cam_mp_ndrop.F:1329
                                    _put(v,'dw',None,xp.maximum(v['dw'], v['dwmin']))  # WRF:module_cam_mp_ndrop.F:1330
                                    _put(v,'w',None,v['wold'] + v['dw'])  # WRF:module_cam_mp_ndrop.F:1331
                                    v['_jump']=_int(100)  # WRF:module_cam_mp_ndrop.F:1332
                                    return tuple(v[_n] for _n in ('_jump', 'dw', 'dwnew', 'w'))
                                def _no47(_carry):
                                    v=dict(_outer47);v.update(zip(('_jump', 'dw', 'dwnew', 'w'),_carry))
                                    _put(v,'dwnew',None,v['dwmin'])  # WRF:module_cam_mp_ndrop.F:1334
                                    return tuple(v[_n] for _n in ('_jump', 'dw', 'dwnew', 'w'))
                                _result=_cond(v['dw'] > _r(1.01) * v['dwmin'],_yes47,_no47,tuple(v[_n] for _n in ('_jump', 'dw', 'dwnew', 'w')),xp)  # WRF:module_cam_mp_ndrop.F:1328
                                v.update(zip(('_jump', 'dw', 'dwnew', 'w'),_result))
                                return tuple(v[_n] for _n in ('_jump', 'dw', 'dwnew', 'w'))
                            def _no46(_carry):
                                v=dict(_outer46);v.update(zip(('_jump', 'dw', 'dwnew', 'w'),_carry))
                                return tuple(v[_n] for _n in ('_jump', 'dw', 'dwnew', 'w'))
                            _result=_cond(xp.logical_and(v['fnew'] - v['fold'] > v['dfmax'], v['n'] > 1),_yes46,_no46,tuple(v[_n] for _n in ('_jump', 'dw', 'dwnew', 'w')),xp)  # WRF:module_cam_mp_ndrop.F:1326
                            v.update(zip(('_jump', 'dw', 'dwnew', 'w'),_result))
                            return tuple(v[_n] for _n in ('_jump', 'dw', 'dwnew', 'w'))
                        def _no45(_carry):
                            v=dict(_outer45);v.update(zip(('_jump', 'dw', 'dwnew', 'w'),_carry))
                            return tuple(v[_n] for _n in ('_jump', 'dw', 'dwnew', 'w'))
                        _result=_cond(v['_jump'] == 0,_yes45,_no45,tuple(v[_n] for _n in ('_jump', 'dw', 'dwnew', 'w')),xp)  # WRF:module_cam_mp_ndrop.F:1326
                        v.update(zip(('_jump', 'dw', 'dwnew', 'w'),_result))
                        return tuple(v[_n] for _n in ('_jump', 'alw', 'dw', 'dwnew', 'eta', 'etafactor1', 'fnew', 'lnsmax', 'm', 'smax', 'sqrtalw', 'w', 'wnuc', 'x', 'zeta'))
                    _result=_retry(100,_retrybody33,tuple(v[_n] for _n in ('_jump', 'alw', 'dw', 'dwnew', 'eta', 'etafactor1', 'fnew', 'lnsmax', 'm', 'smax', 'sqrtalw', 'w', 'wnuc', 'x', 'zeta')),xp,0)
                    v.update(zip(('_jump', 'alw', 'dw', 'dwnew', 'eta', 'etafactor1', 'fnew', 'lnsmax', 'm', 'smax', 'sqrtalw', 'w', 'wnuc', 'x', 'zeta'),_result))
                    return tuple(v[_n] for _n in ('_jump', 'alw', 'dw', 'dwnew', 'eta', 'etafactor1', 'fnew', 'lnsmax', 'm', 'smax', 'sqrtalw', 'w', 'wnuc', 'x', 'zeta'))
                def _no32(_carry):
                    v=dict(_outer32);v.update(zip(('_jump', 'alw', 'dw', 'dwnew', 'eta', 'etafactor1', 'fnew', 'lnsmax', 'm', 'smax', 'sqrtalw', 'w', 'wnuc', 'x', 'zeta'),_carry))
                    return tuple(v[_n] for _n in ('_jump', 'alw', 'dw', 'dwnew', 'eta', 'etafactor1', 'fnew', 'lnsmax', 'm', 'smax', 'sqrtalw', 'w', 'wnuc', 'x', 'zeta'))
                _result=_cond(v['_jump'] == 0,_yes32,_no32,tuple(v[_n] for _n in ('_jump', 'alw', 'dw', 'dwnew', 'eta', 'etafactor1', 'fnew', 'lnsmax', 'm', 'smax', 'sqrtalw', 'w', 'wnuc', 'x', 'zeta')),xp)  # WRF:module_cam_mp_ndrop.F:1305
                v.update(zip(('_jump', 'alw', 'dw', 'dwnew', 'eta', 'etafactor1', 'fnew', 'lnsmax', 'm', 'smax', 'sqrtalw', 'w', 'wnuc', 'x', 'zeta'),_result))
                _outer48=dict(v)
                def _yes48(_carry):
                    v=dict(_outer48);v.update(zip(('dwnew',),_carry))
                    _outer49=dict(v)
                    def _yes49(_carry):
                        v=dict(_outer49);v.update(zip(('dwnew',),_carry))
                        _put(v,'dwnew',None,xp.minimum(_d(1.5) * v['dw'], v['dwmax']))  # WRF:module_cam_mp_ndrop.F:1340
                        return tuple(v[_n] for _n in ('dwnew',))
                    def _no49(_carry):
                        v=dict(_outer49);v.update(zip(('dwnew',),_carry))
                        return tuple(v[_n] for _n in ('dwnew',))
                    _result=_cond(v['fnew'] - v['fold'] < v['dfmin'],_yes49,_no49,tuple(v[_n] for _n in ('dwnew',)),xp)  # WRF:module_cam_mp_ndrop.F:1338
                    v.update(zip(('dwnew',),_result))
                    return tuple(v[_n] for _n in ('dwnew',))
                def _no48(_carry):
                    v=dict(_outer48);v.update(zip(('dwnew',),_carry))
                    return tuple(v[_n] for _n in ('dwnew',))
                _result=_cond(v['_jump'] == 0,_yes48,_no48,tuple(v[_n] for _n in ('dwnew',)),xp)  # WRF:module_cam_mp_ndrop.F:1338
                v.update(zip(('dwnew',),_result))
                _outer50=dict(v)
                def _yes50(_carry):
                    v=dict(_outer50);v.update(zip(('fold',),_carry))
                    _put(v,'fold',None,v['fnew'])  # WRF:module_cam_mp_ndrop.F:1342
                    return tuple(v[_n] for _n in ('fold',))
                def _no50(_carry):
                    v=dict(_outer50);v.update(zip(('fold',),_carry))
                    return tuple(v[_n] for _n in ('fold',))
                _result=_cond(v['_jump'] == 0,_yes50,_no50,tuple(v[_n] for _n in ('fold',)),xp)  # WRF:module_cam_mp_ndrop.F:1342
                v.update(zip(('fold',),_result))
                _outer51=dict(v)
                def _yes51(_carry):
                    v=dict(_outer51);v.update(zip(('z',),_carry))
                    _put(v,'z',None,(v['w'] - v['wbar']) / (v['sigw'] * v['sq2']))  # WRF:module_cam_mp_ndrop.F:1344
                    return tuple(v[_n] for _n in ('z',))
                def _no51(_carry):
                    v=dict(_outer51);v.update(zip(('z',),_carry))
                    return tuple(v[_n] for _n in ('z',))
                _result=_cond(v['_jump'] == 0,_yes51,_no51,tuple(v[_n] for _n in ('z',)),xp)  # WRF:module_cam_mp_ndrop.F:1344
                v.update(zip(('z',),_result))
                _outer52=dict(v)
                def _yes52(_carry):
                    v=dict(_outer52);v.update(zip(('g',),_carry))
                    _put(v,'g',None,xp.exp(_neg(v['z']) * v['z']))  # WRF:module_cam_mp_ndrop.F:1345
                    return tuple(v[_n] for _n in ('g',))
                def _no52(_carry):
                    v=dict(_outer52);v.update(zip(('g',),_carry))
                    return tuple(v[_n] for _n in ('g',))
                _result=_cond(v['_jump'] == 0,_yes52,_no52,tuple(v[_n] for _n in ('g',)),xp)  # WRF:module_cam_mp_ndrop.F:1345
                v.update(zip(('g',),_result))
                _outer53=dict(v)
                def _yes53(_carry):
                    v=dict(_outer53);v.update(zip(('fnmin',),_carry))
                    _put(v,'fnmin',None,_d(1.0))  # WRF:module_cam_mp_ndrop.F:1346
                    return tuple(v[_n] for _n in ('fnmin',))
                def _no53(_carry):
                    v=dict(_outer53);v.update(zip(('fnmin',),_carry))
                    return tuple(v[_n] for _n in ('fnmin',))
                _result=_cond(v['_jump'] == 0,_yes53,_no53,tuple(v[_n] for _n in ('fnmin',)),xp)  # WRF:module_cam_mp_ndrop.F:1346
                v.update(zip(('fnmin',),_result))
                _outer54=dict(v)
                def _yes54(_carry):
                    v=dict(_outer54);v.update(zip(('xmincoeff',),_carry))
                    _put(v,'xmincoeff',None,v['alogaten'] - v['twothird'] * (v['lnsmax'] - v['alog2']) - v['alog3'])  # WRF:module_cam_mp_ndrop.F:1347
                    return tuple(v[_n] for _n in ('xmincoeff',))
                def _no54(_carry):
                    v=dict(_outer54);v.update(zip(('xmincoeff',),_carry))
                    return tuple(v[_n] for _n in ('xmincoeff',))
                _result=_cond(v['_jump'] == 0,_yes54,_no54,tuple(v[_n] for _n in ('xmincoeff',)),xp)  # WRF:module_cam_mp_ndrop.F:1347
                v.update(zip(('xmincoeff',),_result))
                _outer55=dict(v)
                def _yes55(_carry):
                    v=dict(_outer55);v.update(zip(('arg', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnmin', 'fnold', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'wb', 'x'),_carry))
                    _outer56=dict(v)
                    def _body56(_k,_carry):
                        v=dict(_outer56);v.update(zip(('_jump', 'arg', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnmin', 'fnold', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'wb', 'x'),_carry));v['m']=_int(_k)
                        _put(v,'x',None,v['twothird'] * (_get(v['lnsm'], (v['m'],)) - v['lnsmax']) / (v['sq2'] * _get(v['alogsig'], (v['m'],))))  # WRF:module_cam_mp_ndrop.F:1351
                        _put(v,'fn',(v['m'],),_d(0.5) * (_r(1.0) - erf(v['x'], xp=xp, _globals=_globals)))  # WRF:module_cam_mp_ndrop.F:1352
                        _put(v,'fnmin',None,xp.minimum(_get(v['fn'], (v['m'],)), v['fnmin']))  # WRF:module_cam_mp_ndrop.F:1353
                        _put(v,'fnbar',None,_get(v['fn'], (v['m'],)) * v['g'] + _get(v['fnold'], (v['m'],)) * v['gold'])  # WRF:module_cam_mp_ndrop.F:1356
                        _put(v,'arg',None,v['x'] - _d(1.5) * v['sq2'] * _get(v['alogsig'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:1357
                        _put(v,'fm',(v['m'],),_d(0.5) * (_d(1.0) - erf(v['arg'], xp=xp, _globals=_globals)))  # WRF:module_cam_mp_ndrop.F:1358
                        _put(v,'fmbar',None,_get(v['fm'], (v['m'],)) * v['g'] + _get(v['fmold'], (v['m'],)) * v['gold'])  # WRF:module_cam_mp_ndrop.F:1359
                        _put(v,'wb',None,v['w'] + v['wold'])  # WRF:module_cam_mp_ndrop.F:1360
                        _outer57=dict(v)
                        def _yes57(_carry):
                            v=dict(_outer57);v.update(zip(('sumflxm', 'sumflxn'),_carry))
                            _put(v,'sumflxn',(v['m'],),_get(v['sumflxn'], (v['m'],)) + v['sixth'] * (v['wb'] * v['fnbar'] + (_get(v['fn'], (v['m'],)) * v['g'] * v['w'] + _get(v['fnold'], (v['m'],)) * v['gold'] * v['wold'])) * v['dw'])  # WRF:module_cam_mp_ndrop.F:1362
                            _put(v,'sumflxm',(v['m'],),_get(v['sumflxm'], (v['m'],)) + v['sixth'] * (v['wb'] * v['fmbar'] + (_get(v['fm'], (v['m'],)) * v['g'] * v['w'] + _get(v['fmold'], (v['m'],)) * v['gold'] * v['wold'])) * v['dw'])  # WRF:module_cam_mp_ndrop.F:1364
                            return tuple(v[_n] for _n in ('sumflxm', 'sumflxn'))
                        def _no57(_carry):
                            v=dict(_outer57);v.update(zip(('sumflxm', 'sumflxn'),_carry))
                            return tuple(v[_n] for _n in ('sumflxm', 'sumflxn'))
                        _result=_cond(v['w'] > _r(0.0),_yes57,_no57,tuple(v[_n] for _n in ('sumflxm', 'sumflxn')),xp)  # WRF:module_cam_mp_ndrop.F:1361
                        v.update(zip(('sumflxm', 'sumflxn'),_result))
                        _put(v,'sumfn',(v['m'],),_get(v['sumfn'], (v['m'],)) + _d(0.5) * v['fnbar'] * v['dw'])  # WRF:module_cam_mp_ndrop.F:1367
                        _put(v,'fnold',(v['m'],),_get(v['fn'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:1369
                        _put(v,'sumfm',(v['m'],),_get(v['sumfm'], (v['m'],)) + _d(0.5) * v['fmbar'] * v['dw'])  # WRF:module_cam_mp_ndrop.F:1370
                        _put(v,'fmold',(v['m'],),_get(v['fm'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:1371
                        return tuple(v[_n] for _n in ('_jump', 'arg', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnmin', 'fnold', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'wb', 'x'))
                    _result=_loop(1,v['nmode'],1,_body56,tuple(v[_n] for _n in ('_jump', 'arg', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnmin', 'fnold', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'wb', 'x')),xp,jump_index=0)
                    v.update(zip(('_jump', 'arg', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnmin', 'fnold', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'wb', 'x'),_result))
                    return tuple(v[_n] for _n in ('arg', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnmin', 'fnold', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'wb', 'x'))
                def _no55(_carry):
                    v=dict(_outer55);v.update(zip(('arg', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnmin', 'fnold', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'wb', 'x'),_carry))
                    return tuple(v[_n] for _n in ('arg', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnmin', 'fnold', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'wb', 'x'))
                _result=_cond(v['_jump'] == 0,_yes55,_no55,tuple(v[_n] for _n in ('arg', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnmin', 'fnold', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'wb', 'x')),xp)  # WRF:module_cam_mp_ndrop.F:1349
                v.update(zip(('arg', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnmin', 'fnold', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'wb', 'x'),_result))
                _outer58=dict(v)
                def _yes58(_carry):
                    v=dict(_outer58);v.update(zip(('sumflx_fullact',),_carry))
                    _put(v,'sumflx_fullact',None,v['sumflx_fullact'] + v['sixth'] * (v['wb'] * (v['g'] + v['gold']) + (v['g'] * v['w'] + v['gold'] * v['wold'])) * v['dw'])  # WRF:module_cam_mp_ndrop.F:1374
                    return tuple(v[_n] for _n in ('sumflx_fullact',))
                def _no58(_carry):
                    v=dict(_outer58);v.update(zip(('sumflx_fullact',),_carry))
                    return tuple(v[_n] for _n in ('sumflx_fullact',))
                _result=_cond(v['_jump'] == 0,_yes58,_no58,tuple(v[_n] for _n in ('sumflx_fullact',)),xp)  # WRF:module_cam_mp_ndrop.F:1374
                v.update(zip(('sumflx_fullact',),_result))
                _outer59=dict(v)
                def _yes59(_carry):
                    v=dict(_outer59);v.update(zip(('gold',),_carry))
                    _put(v,'gold',None,v['g'])  # WRF:module_cam_mp_ndrop.F:1377
                    return tuple(v[_n] for _n in ('gold',))
                def _no59(_carry):
                    v=dict(_outer59);v.update(zip(('gold',),_carry))
                    return tuple(v[_n] for _n in ('gold',))
                _result=_cond(v['_jump'] == 0,_yes59,_no59,tuple(v[_n] for _n in ('gold',)),xp)  # WRF:module_cam_mp_ndrop.F:1377
                v.update(zip(('gold',),_result))
                _outer60=dict(v)
                def _yes60(_carry):
                    v=dict(_outer60);v.update(zip(('wold',),_carry))
                    _put(v,'wold',None,v['w'])  # WRF:module_cam_mp_ndrop.F:1378
                    return tuple(v[_n] for _n in ('wold',))
                def _no60(_carry):
                    v=dict(_outer60);v.update(zip(('wold',),_carry))
                    return tuple(v[_n] for _n in ('wold',))
                _result=_cond(v['_jump'] == 0,_yes60,_no60,tuple(v[_n] for _n in ('wold',)),xp)  # WRF:module_cam_mp_ndrop.F:1378
                v.update(zip(('wold',),_result))
                _outer61=dict(v)
                def _yes61(_carry):
                    v=dict(_outer61);v.update(zip(('dw',),_carry))
                    _put(v,'dw',None,v['dwnew'])  # WRF:module_cam_mp_ndrop.F:1379
                    return tuple(v[_n] for _n in ('dw',))
                def _no61(_carry):
                    v=dict(_outer61);v.update(zip(('dw',),_carry))
                    return tuple(v[_n] for _n in ('dw',))
                _result=_cond(v['_jump'] == 0,_yes61,_no61,tuple(v[_n] for _n in ('dw',)),xp)  # WRF:module_cam_mp_ndrop.F:1379
                v.update(zip(('dw',),_result))
                _outer62=dict(v)
                def _yes62(_carry):
                    v=dict(_outer62);v.update(zip(('_jump',),_carry))
                    _outer63=dict(v)
                    def _yes63(_carry):
                        v=dict(_outer63);v.update(zip(('_jump',),_carry))
                        v['_jump']=_int(20)  # WRF:module_cam_mp_ndrop.F:1380
                        return tuple(v[_n] for _n in ('_jump',))
                    def _no63(_carry):
                        v=dict(_outer63);v.update(zip(('_jump',),_carry))
                        return tuple(v[_n] for _n in ('_jump',))
                    _result=_cond(xp.logical_and(v['n'] > 1, xp.logical_or(v['w'] > v['wmax'], v['fnmin'] > v['fmax'])),_yes63,_no63,tuple(v[_n] for _n in ('_jump',)),xp)  # WRF:module_cam_mp_ndrop.F:1380
                    v.update(zip(('_jump',),_result))
                    return tuple(v[_n] for _n in ('_jump',))
                def _no62(_carry):
                    v=dict(_outer62);v.update(zip(('_jump',),_carry))
                    return tuple(v[_n] for _n in ('_jump',))
                _result=_cond(v['_jump'] == 0,_yes62,_no62,tuple(v[_n] for _n in ('_jump',)),xp)  # WRF:module_cam_mp_ndrop.F:1380
                v.update(zip(('_jump',),_result))
                _outer64=dict(v)
                def _yes64(_carry):
                    v=dict(_outer64);v.update(zip(('w',),_carry))
                    _put(v,'w',None,v['w'] + v['dw'])  # WRF:module_cam_mp_ndrop.F:1381
                    return tuple(v[_n] for _n in ('w',))
                def _no64(_carry):
                    v=dict(_outer64);v.update(zip(('w',),_carry))
                    return tuple(v[_n] for _n in ('w',))
                _result=_cond(v['_jump'] == 0,_yes64,_no64,tuple(v[_n] for _n in ('w',)),xp)  # WRF:module_cam_mp_ndrop.F:1381
                v.update(zip(('w',),_result))
                return tuple(v[_n] for _n in ('_jump', 'alw', 'arg', 'dw', 'dwnew', 'eta', 'etafactor1', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnew', 'fnmin', 'fnold', 'fold', 'g', 'gold', 'lnsmax', 'm', 'n', 'smax', 'sqrtalw', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'w', 'wb', 'wnuc', 'wold', 'x', 'xmincoeff', 'z', 'zeta'))
            _result=_loop(1,200,1,_body31,tuple(v[_n] for _n in ('_jump', 'alw', 'arg', 'dw', 'dwnew', 'eta', 'etafactor1', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnew', 'fnmin', 'fnold', 'fold', 'g', 'gold', 'lnsmax', 'm', 'n', 'smax', 'sqrtalw', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'w', 'wb', 'wnuc', 'wold', 'x', 'xmincoeff', 'z', 'zeta')),xp,jump_index=0)
            v.update(zip(('_jump', 'alw', 'arg', 'dw', 'dwnew', 'eta', 'etafactor1', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnew', 'fnmin', 'fnold', 'fold', 'g', 'gold', 'lnsmax', 'm', 'n', 'smax', 'sqrtalw', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'w', 'wb', 'wnuc', 'wold', 'x', 'xmincoeff', 'z', 'zeta'),_result))
            v['_fatal']=v['_fatal']+_int(1)  # Original WRF fatal event:module_cam_mp_ndrop.F:1438; caller must refuse
            v['_jump']=xp.where(v['_jump']==20,_int(0),v['_jump'])
            _put(v,'ndist',(v['n'],),_get(v['ndist'], (v['n'],)) + 1)  # WRF:module_cam_mp_ndrop.F:1440
            _outer65=dict(v)
            def _yes65(_carry):
                v=dict(_outer65);v.update(zip(('g', 'gf', 'gf1', 'gf2', 'integ', 'integf', 'm', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'wf1', 'wf2', 'wnuc', 'z1', 'z2', 'zf1', 'zf2'),_carry))
                _put(v,'wnuc',None,v['w'] + v['wdiab'])  # WRF:module_cam_mp_ndrop.F:1445
                _put(v,'z1',None,(v['w'] - v['wbar']) / (v['sigw'] * v['sq2']))  # WRF:module_cam_mp_ndrop.F:1447
                _put(v,'z2',None,(v['wmaxf'] - v['wbar']) / (v['sigw'] * v['sq2']))  # WRF:module_cam_mp_ndrop.F:1448
                _put(v,'g',None,xp.exp(_neg(v['z1']) * v['z1']))  # WRF:module_cam_mp_ndrop.F:1449
                _put(v,'integ',None,v['sigw'] * _r(0.5) * v['sq2'] * v['sqpi'] * (erf(v['z2'], xp=xp, _globals=_globals) - erf(v['z1'], xp=xp, _globals=_globals)))  # WRF:module_cam_mp_ndrop.F:1450
                _put(v,'wf1',None,xp.maximum(v['w'], v['zero']))  # WRF:module_cam_mp_ndrop.F:1452
                _put(v,'zf1',None,(v['wf1'] - v['wbar']) / (v['sigw'] * v['sq2']))  # WRF:module_cam_mp_ndrop.F:1453
                _put(v,'gf1',None,xp.exp(_neg(v['zf1']) * v['zf1']))  # WRF:module_cam_mp_ndrop.F:1454
                _put(v,'wf2',None,xp.maximum(v['wmaxf'], v['zero']))  # WRF:module_cam_mp_ndrop.F:1455
                _put(v,'zf2',None,(v['wf2'] - v['wbar']) / (v['sigw'] * v['sq2']))  # WRF:module_cam_mp_ndrop.F:1456
                _put(v,'gf2',None,xp.exp(_neg(v['zf2']) * v['zf2']))  # WRF:module_cam_mp_ndrop.F:1457
                _put(v,'gf',None,v['gf1'] - v['gf2'])  # WRF:module_cam_mp_ndrop.F:1458
                _put(v,'integf',None,v['wbar'] * v['sigw'] * _r(0.5) * v['sq2'] * v['sqpi'] * (erf(v['zf2'], xp=xp, _globals=_globals) - erf(v['zf1'], xp=xp, _globals=_globals)) + v['sigw'] * v['sigw'] * v['gf'])  # WRF:module_cam_mp_ndrop.F:1459
                _outer66=dict(v)
                def _body66(_k,_carry):
                    v=dict(_outer66);v.update(zip(('_jump', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn'),_carry));v['m']=_int(_k)
                    _put(v,'sumflxn',(v['m'],),_get(v['sumflxn'], (v['m'],)) + v['integf'] * _get(v['fn'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:1462
                    _put(v,'sumfn',(v['m'],),_get(v['sumfn'], (v['m'],)) + _get(v['fn'], (v['m'],)) * v['integ'])  # WRF:module_cam_mp_ndrop.F:1463
                    _put(v,'sumflxm',(v['m'],),_get(v['sumflxm'], (v['m'],)) + v['integf'] * _get(v['fm'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:1464
                    _put(v,'sumfm',(v['m'],),_get(v['sumfm'], (v['m'],)) + _get(v['fm'], (v['m'],)) * v['integ'])  # WRF:module_cam_mp_ndrop.F:1465
                    return tuple(v[_n] for _n in ('_jump', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn'))
                _result=_loop(1,v['nmode'],1,_body66,tuple(v[_n] for _n in ('_jump', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn')),xp,jump_index=0)
                v.update(zip(('_jump', 'm', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn'),_result))
                _put(v,'sumflx_fullact',None,v['sumflx_fullact'] + v['integf'])  # WRF:module_cam_mp_ndrop.F:1468
                return tuple(v[_n] for _n in ('g', 'gf', 'gf1', 'gf2', 'integ', 'integf', 'm', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'wf1', 'wf2', 'wnuc', 'z1', 'z2', 'zf1', 'zf2'))
            def _no65(_carry):
                v=dict(_outer65);v.update(zip(('g', 'gf', 'gf1', 'gf2', 'integ', 'integf', 'm', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'wf1', 'wf2', 'wnuc', 'z1', 'z2', 'zf1', 'zf2'),_carry))
                return tuple(v[_n] for _n in ('g', 'gf', 'gf1', 'gf2', 'integ', 'integf', 'm', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'wf1', 'wf2', 'wnuc', 'z1', 'z2', 'zf1', 'zf2'))
            _result=_cond(v['w'] < v['wmaxf'],_yes65,_no65,tuple(v[_n] for _n in ('g', 'gf', 'gf1', 'gf2', 'integ', 'integf', 'm', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'wf1', 'wf2', 'wnuc', 'z1', 'z2', 'zf1', 'zf2')),xp)  # WRF:module_cam_mp_ndrop.F:1441
            v.update(zip(('g', 'gf', 'gf1', 'gf2', 'integ', 'integf', 'm', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'wf1', 'wf2', 'wnuc', 'z1', 'z2', 'zf1', 'zf2'),_result))
            _outer67=dict(v)
            def _body67(_k,_carry):
                v=dict(_outer67);v.update(zip(('_fatal', '_jump', 'fluxm', 'fluxn', 'fm', 'fn', 'm'),_carry));v['m']=_int(_k)
                _put(v,'fn',(v['m'],),_get(v['sumfn'], (v['m'],)) / (v['sq2'] * v['sqpi'] * v['sigw']))  # WRF:module_cam_mp_ndrop.F:1474
                _outer68=dict(v)
                def _yes68(_carry):
                    v=dict(_outer68);v.update(zip(('_fatal',),_carry))
                    v['_fatal']=v['_fatal']+_int(1)  # Original WRF fatal event:module_cam_mp_ndrop.F:1489; caller must refuse
                    return tuple(v[_n] for _n in ('_fatal',))
                def _no68(_carry):
                    v=dict(_outer68);v.update(zip(('_fatal',),_carry))
                    return tuple(v[_n] for _n in ('_fatal',))
                _result=_cond(_get(v['fn'], (v['m'],)) > _r(1.01),_yes68,_no68,tuple(v[_n] for _n in ('_fatal',)),xp)  # WRF:module_cam_mp_ndrop.F:1476
                v.update(zip(('_fatal',),_result))
                _put(v,'fluxn',(v['m'],),_get(v['sumflxn'], (v['m'],)) / (v['sq2'] * v['sqpi'] * v['sigw']))  # WRF:module_cam_mp_ndrop.F:1491
                _put(v,'fm',(v['m'],),_get(v['sumfm'], (v['m'],)) / (v['sq2'] * v['sqpi'] * v['sigw']))  # WRF:module_cam_mp_ndrop.F:1492
                _put(v,'fluxm',(v['m'],),_get(v['sumflxm'], (v['m'],)) / (v['sq2'] * v['sqpi'] * v['sigw']))  # WRF:module_cam_mp_ndrop.F:1500
                return tuple(v[_n] for _n in ('_fatal', '_jump', 'fluxm', 'fluxn', 'fm', 'fn', 'm'))
            _result=_loop(1,v['nmode'],1,_body67,tuple(v[_n] for _n in ('_fatal', '_jump', 'fluxm', 'fluxn', 'fm', 'fn', 'm')),xp,jump_index=1)
            v.update(zip(('_fatal', '_jump', 'fluxm', 'fluxn', 'fm', 'fn', 'm'),_result))
            _put(v,'flux_fullact',None,v['sumflx_fullact'] / (v['sq2'] * v['sqpi'] * v['sigw']))  # WRF:module_cam_mp_ndrop.F:1503
            return tuple(v[_n] for _n in ('_fatal', '_jump', 'alw', 'arg', 'dfmax', 'dfmin', 'dw', 'dwmax', 'dwmin', 'dwnew', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnew', 'fnmin', 'fnold', 'fold', 'g', 'gf', 'gf1', 'gf2', 'gold', 'integ', 'integf', 'lnsmax', 'm', 'n', 'ndist', 'smax', 'sqrtalw', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'w', 'wb', 'wf1', 'wf2', 'wmax', 'wmin', 'wnuc', 'wold', 'x', 'xmincoeff', 'z', 'z1', 'z2', 'zeta', 'zf1', 'zf2'))
        def _no27(_carry):
            v=dict(_outer27);v.update(zip(('_fatal', '_jump', 'alw', 'arg', 'dfmax', 'dfmin', 'dw', 'dwmax', 'dwmin', 'dwnew', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnew', 'fnmin', 'fnold', 'fold', 'g', 'gf', 'gf1', 'gf2', 'gold', 'integ', 'integf', 'lnsmax', 'm', 'n', 'ndist', 'smax', 'sqrtalw', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'w', 'wb', 'wf1', 'wf2', 'wmax', 'wmin', 'wnuc', 'wold', 'x', 'xmincoeff', 'z', 'z1', 'z2', 'zeta', 'zf1', 'zf2'),_carry))
            _put(v,'wnuc',None,v['wbar'] + v['wdiab'])  # WRF:module_cam_mp_ndrop.F:1508
            _outer70=dict(v)
            def _yes70(_carry):
                v=dict(_outer70);v.update(zip(('alw', 'arg', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'lnsmax', 'm', 'smax', 'sqrtalw', 'w', 'x', 'xmincoeff', 'zeta'),_carry))
                _put(v,'w',None,v['wbar'])  # WRF:module_cam_mp_ndrop.F:1512
                _put(v,'alw',None,v['alpha'] * v['wnuc'])  # WRF:module_cam_mp_ndrop.F:1513
                _put(v,'sqrtalw',None,xp.sqrt(v['alw']))  # WRF:module_cam_mp_ndrop.F:1514
                _put(v,'etafactor1',None,v['alw'] * v['sqrtalw'])  # WRF:module_cam_mp_ndrop.F:1515
                _outer71=dict(v)
                def _body71(_k,_carry):
                    v=dict(_outer71);v.update(zip(('_jump', 'eta', 'm', 'zeta'),_carry));v['m']=_int(_k)
                    _put(v,'eta',(v['m'],),v['etafactor1'] * _get(v['etafactor2'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:1518
                    _put(v,'zeta',(v['m'],),v['twothird'] * v['sqrtalw'] * v['aten'] / _get(v['sqrtg'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:1519
                    return tuple(v[_n] for _n in ('_jump', 'eta', 'm', 'zeta'))
                _result=_loop(1,v['nmode'],1,_body71,tuple(v[_n] for _n in ('_jump', 'eta', 'm', 'zeta')),xp,jump_index=0)
                v.update(zip(('_jump', 'eta', 'm', 'zeta'),_result))
                _call=maxsat(v['zeta'],v['eta'],v['nmode'],v['smc'],v['smax'],_globals=_globals,xp=xp)
                _put(v,'smax',None,_call[0])  # WRF:module_cam_mp_ndrop.F:1522
                _put(v,'lnsmax',None,xp.log(v['smax']))  # WRF:module_cam_mp_ndrop.F:1524
                _put(v,'xmincoeff',None,v['alogaten'] - v['twothird'] * (v['lnsmax'] - v['alog2']) - v['alog3'])  # WRF:module_cam_mp_ndrop.F:1525
                _outer72=dict(v)
                def _body72(_k,_carry):
                    v=dict(_outer72);v.update(zip(('_jump', 'arg', 'fluxm', 'fluxn', 'fm', 'fn', 'm', 'x'),_carry));v['m']=_int(_k)
                    _put(v,'x',None,v['twothird'] * (_get(v['lnsm'], (v['m'],)) - v['lnsmax']) / (v['sq2'] * _get(v['alogsig'], (v['m'],))))  # WRF:module_cam_mp_ndrop.F:1530
                    _put(v,'fn',(v['m'],),_d(0.5) * (_d(1.0) - erf(v['x'], xp=xp, _globals=_globals)))  # WRF:module_cam_mp_ndrop.F:1531
                    _put(v,'arg',None,v['x'] - _d(1.5) * v['sq2'] * _get(v['alogsig'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:1532
                    _put(v,'fm',(v['m'],),_d(0.5) * (_d(1.0) - erf(v['arg'], xp=xp, _globals=_globals)))  # WRF:module_cam_mp_ndrop.F:1533
                    _outer73=dict(v)
                    def _yes73(_carry):
                        v=dict(_outer73);v.update(zip(('fluxm', 'fluxn'),_carry))
                        _put(v,'fluxn',(v['m'],),_get(v['fn'], (v['m'],)) * v['w'])  # WRF:module_cam_mp_ndrop.F:1535
                        _put(v,'fluxm',(v['m'],),_get(v['fm'], (v['m'],)) * v['w'])  # WRF:module_cam_mp_ndrop.F:1536
                        return tuple(v[_n] for _n in ('fluxm', 'fluxn'))
                    def _no73(_carry):
                        v=dict(_outer73);v.update(zip(('fluxm', 'fluxn'),_carry))
                        return tuple(v[_n] for _n in ('fluxm', 'fluxn'))
                    _result=_cond(v['wbar'] > _d(0.0),_yes73,_no73,tuple(v[_n] for _n in ('fluxm', 'fluxn')),xp)  # WRF:module_cam_mp_ndrop.F:1534
                    v.update(zip(('fluxm', 'fluxn'),_result))
                    return tuple(v[_n] for _n in ('_jump', 'arg', 'fluxm', 'fluxn', 'fm', 'fn', 'm', 'x'))
                _result=_loop(1,v['nmode'],1,_body72,tuple(v[_n] for _n in ('_jump', 'arg', 'fluxm', 'fluxn', 'fm', 'fn', 'm', 'x')),xp,jump_index=0)
                v.update(zip(('_jump', 'arg', 'fluxm', 'fluxn', 'fm', 'fn', 'm', 'x'),_result))
                _put(v,'flux_fullact',None,v['w'])  # WRF:module_cam_mp_ndrop.F:1539
                return tuple(v[_n] for _n in ('alw', 'arg', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'lnsmax', 'm', 'smax', 'sqrtalw', 'w', 'x', 'xmincoeff', 'zeta'))
            def _no70(_carry):
                v=dict(_outer70);v.update(zip(('alw', 'arg', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'lnsmax', 'm', 'smax', 'sqrtalw', 'w', 'x', 'xmincoeff', 'zeta'),_carry))
                return tuple(v[_n] for _n in ('alw', 'arg', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'lnsmax', 'm', 'smax', 'sqrtalw', 'w', 'x', 'xmincoeff', 'zeta'))
            _result=_cond(v['wnuc'] > _d(0.0),_yes70,_no70,tuple(v[_n] for _n in ('alw', 'arg', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'lnsmax', 'm', 'smax', 'sqrtalw', 'w', 'x', 'xmincoeff', 'zeta')),xp)  # WRF:module_cam_mp_ndrop.F:1510
            v.update(zip(('alw', 'arg', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'lnsmax', 'm', 'smax', 'sqrtalw', 'w', 'x', 'xmincoeff', 'zeta'),_result))
            return tuple(v[_n] for _n in ('_fatal', '_jump', 'alw', 'arg', 'dfmax', 'dfmin', 'dw', 'dwmax', 'dwmin', 'dwnew', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnew', 'fnmin', 'fnold', 'fold', 'g', 'gf', 'gf1', 'gf2', 'gold', 'integ', 'integf', 'lnsmax', 'm', 'n', 'ndist', 'smax', 'sqrtalw', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'w', 'wb', 'wf1', 'wf2', 'wmax', 'wmin', 'wnuc', 'wold', 'x', 'xmincoeff', 'z', 'z1', 'z2', 'zeta', 'zf1', 'zf2'))
        _result=_cond(v['sigw'] > _d(1e-05),_yes27,_no27,tuple(v[_n] for _n in ('_fatal', '_jump', 'alw', 'arg', 'dfmax', 'dfmin', 'dw', 'dwmax', 'dwmin', 'dwnew', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnew', 'fnmin', 'fnold', 'fold', 'g', 'gf', 'gf1', 'gf2', 'gold', 'integ', 'integf', 'lnsmax', 'm', 'n', 'ndist', 'smax', 'sqrtalw', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'w', 'wb', 'wf1', 'wf2', 'wmax', 'wmin', 'wnuc', 'wold', 'x', 'xmincoeff', 'z', 'z1', 'z2', 'zeta', 'zf1', 'zf2')),xp)  # WRF:module_cam_mp_ndrop.F:1268
        v.update(zip(('_fatal', '_jump', 'alw', 'arg', 'dfmax', 'dfmin', 'dw', 'dwmax', 'dwmin', 'dwnew', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnew', 'fnmin', 'fnold', 'fold', 'g', 'gf', 'gf1', 'gf2', 'gold', 'integ', 'integf', 'lnsmax', 'm', 'n', 'ndist', 'smax', 'sqrtalw', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'w', 'wb', 'wf1', 'wf2', 'wmax', 'wmin', 'wnuc', 'wold', 'x', 'xmincoeff', 'z', 'z1', 'z2', 'zeta', 'zf1', 'zf2'),_result))
        return tuple(v[_n] for _n in ('_fatal', '_jump', 'alw', 'arg', 'dfmax', 'dfmin', 'dw', 'dwmax', 'dwmin', 'dwnew', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnew', 'fnmin', 'fnold', 'fold', 'g', 'gf', 'gf1', 'gf2', 'gold', 'integ', 'integf', 'lnsmax', 'm', 'n', 'ndist', 'smax', 'sqrtalw', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'w', 'wb', 'wf1', 'wf2', 'wmax', 'wmin', 'wnuc', 'wold', 'x', 'xmincoeff', 'z', 'z1', 'z2', 'zeta', 'zf1', 'zf2'))
    def _no26(_carry):
        v=dict(_outer26);v.update(zip(('_fatal', '_jump', 'alw', 'arg', 'dfmax', 'dfmin', 'dw', 'dwmax', 'dwmin', 'dwnew', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnew', 'fnmin', 'fnold', 'fold', 'g', 'gf', 'gf1', 'gf2', 'gold', 'integ', 'integf', 'lnsmax', 'm', 'n', 'ndist', 'smax', 'sqrtalw', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'w', 'wb', 'wf1', 'wf2', 'wmax', 'wmin', 'wnuc', 'wold', 'x', 'xmincoeff', 'z', 'z1', 'z2', 'zeta', 'zf1', 'zf2'),_carry))
        return tuple(v[_n] for _n in ('_fatal', '_jump', 'alw', 'arg', 'dfmax', 'dfmin', 'dw', 'dwmax', 'dwmin', 'dwnew', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnew', 'fnmin', 'fnold', 'fold', 'g', 'gf', 'gf1', 'gf2', 'gold', 'integ', 'integf', 'lnsmax', 'm', 'n', 'ndist', 'smax', 'sqrtalw', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'w', 'wb', 'wf1', 'wf2', 'wmax', 'wmin', 'wnuc', 'wold', 'x', 'xmincoeff', 'z', 'z1', 'z2', 'zeta', 'zf1', 'zf2'))
    _result=_cond(v['_jump'] == 0,_yes26,_no26,tuple(v[_n] for _n in ('_fatal', '_jump', 'alw', 'arg', 'dfmax', 'dfmin', 'dw', 'dwmax', 'dwmin', 'dwnew', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnew', 'fnmin', 'fnold', 'fold', 'g', 'gf', 'gf1', 'gf2', 'gold', 'integ', 'integf', 'lnsmax', 'm', 'n', 'ndist', 'smax', 'sqrtalw', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'w', 'wb', 'wf1', 'wf2', 'wmax', 'wmin', 'wnuc', 'wold', 'x', 'xmincoeff', 'z', 'z1', 'z2', 'zeta', 'zf1', 'zf2')),xp)  # WRF:module_cam_mp_ndrop.F:1268
    v.update(zip(('_fatal', '_jump', 'alw', 'arg', 'dfmax', 'dfmin', 'dw', 'dwmax', 'dwmin', 'dwnew', 'eta', 'etafactor1', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fmbar', 'fmold', 'fn', 'fnbar', 'fnew', 'fnmin', 'fnold', 'fold', 'g', 'gf', 'gf1', 'gf2', 'gold', 'integ', 'integf', 'lnsmax', 'm', 'n', 'ndist', 'smax', 'sqrtalw', 'sumflx_fullact', 'sumflxm', 'sumflxn', 'sumfm', 'sumfn', 'w', 'wb', 'wf1', 'wf2', 'wmax', 'wmin', 'wnuc', 'wold', 'x', 'xmincoeff', 'z', 'z1', 'z2', 'zeta', 'zf1', 'zf2'),_result))
    return (v['fn'],v['fm'],v['fluxn'],v['fluxm'],v['flux_fullact'],v['_fatal'],)

def loadaer(raer,qqcw,istart,istop,k,m,cs,npv1,phase,naerosol,vaerosol,hygro,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('ndrop',{}))
    v['raer']=raer
    v['qqcw']=qqcw
    v['istart']=istart
    v['istop']=istop
    v['k']=k
    v['m']=m
    v['cs']=cs
    v['npv1']=npv1
    v['phase']=phase
    v['naerosol']=naerosol
    v['vaerosol']=vaerosol
    v['hygro']=hygro
    v['_jump']=_int(0)
    v['_fatal']=_int(0)
    v['vol']=_zeros((v['pcols'],),'f64',xp)
    v['i']=_zeros((),'i32',xp)
    v['lnum']=_zeros((),'i32',xp)
    v['lnumcw']=_zeros((),'i32',xp)
    v['l']=_zeros((),'i32',xp)
    v['ltype']=_zeros((),'i32',xp)
    v['lmass']=_zeros((),'i32',xp)
    v['lmasscw']=_zeros((),'i32',xp)
    _outer1=dict(v)
    def _body1(_k,_carry):
        v=dict(_outer1);v.update(zip(('_jump', 'hygro', 'i', 'vaerosol'),_carry));v['i']=_int(_k)
        _put(v,'vaerosol',(v['i'],),_d(0.0))  # WRF:module_cam_mp_ndrop.F:1747
        _put(v,'hygro',(v['i'],),_d(0.0))  # WRF:module_cam_mp_ndrop.F:1748
        return tuple(v[_n] for _n in ('_jump', 'hygro', 'i', 'vaerosol'))
    _result=_loop(v['istart'],v['istop'],1,_body1,tuple(v[_n] for _n in ('_jump', 'hygro', 'i', 'vaerosol')),xp,jump_index=0)
    v.update(zip(('_jump', 'hygro', 'i', 'vaerosol'),_result))
    _outer2=dict(v)
    def _body2(_k,_carry):
        v=dict(_outer2);v.update(zip(('_fatal', '_jump', 'hygro', 'i', 'l', 'lmass', 'lmasscw', 'ltype', 'vaerosol', 'vol'),_carry));v['l']=_int(_k)
        _put(v,'lmass',None,_get(v['lmassptr_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:1752
        _put(v,'lmasscw',None,_get(v['lmassptrcw_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:1753
        _put(v,'ltype',None,_get(v['lspectype_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:1754
        _outer3=dict(v)
        def _yes3(_carry):
            v=dict(_outer3);v.update(zip(('_fatal', 'i', 'vol'),_carry))
            _outer4=dict(v)
            def _body4(_k,_carry):
                v=dict(_outer4);v.update(zip(('_jump', 'i', 'vol'),_carry));v['i']=_int(_k)
                _put(v,'vol',(v['i'],),xp.maximum(_get(v['raer'], (v['i'], v['k'], v['lmass'])) + _get(v['qqcw'], (v['i'], v['k'], v['lmasscw'])), _d(0.0)) / _get(v['specdens_amode'], (v['ltype'],)))  # WRF:module_cam_mp_ndrop.F:1758
                return tuple(v[_n] for _n in ('_jump', 'i', 'vol'))
            _result=_loop(v['istart'],v['istop'],1,_body4,tuple(v[_n] for _n in ('_jump', 'i', 'vol')),xp,jump_index=0)
            v.update(zip(('_jump', 'i', 'vol'),_result))
            return tuple(v[_n] for _n in ('_fatal', 'i', 'vol'))
        def _no3(_carry):
            v=dict(_outer3);v.update(zip(('_fatal', 'i', 'vol'),_carry))
            _outer5=dict(v)
            def _yes5(_carry):
                v=dict(_outer5);v.update(zip(('_fatal', 'i', 'vol'),_carry))
                _outer6=dict(v)
                def _body6(_k,_carry):
                    v=dict(_outer6);v.update(zip(('_jump', 'i', 'vol'),_carry));v['i']=_int(_k)
                    _put(v,'vol',(v['i'],),xp.maximum(_get(v['qqcw'], (v['i'], v['k'], v['lmasscw'])), _d(0.0)) / _get(v['specdens_amode'], (v['ltype'],)))  # WRF:module_cam_mp_ndrop.F:1762
                    return tuple(v[_n] for _n in ('_jump', 'i', 'vol'))
                _result=_loop(v['istart'],v['istop'],1,_body6,tuple(v[_n] for _n in ('_jump', 'i', 'vol')),xp,jump_index=0)
                v.update(zip(('_jump', 'i', 'vol'),_result))
                return tuple(v[_n] for _n in ('_fatal', 'i', 'vol'))
            def _no5(_carry):
                v=dict(_outer5);v.update(zip(('_fatal', 'i', 'vol'),_carry))
                _outer7=dict(v)
                def _yes7(_carry):
                    v=dict(_outer7);v.update(zip(('_fatal', 'i', 'vol'),_carry))
                    _outer8=dict(v)
                    def _body8(_k,_carry):
                        v=dict(_outer8);v.update(zip(('_jump', 'i', 'vol'),_carry));v['i']=_int(_k)
                        _put(v,'vol',(v['i'],),xp.maximum(_get(v['raer'], (v['i'], v['k'], v['lmass'])), _d(0.0)) / _get(v['specdens_amode'], (v['ltype'],)))  # WRF:module_cam_mp_ndrop.F:1766
                        return tuple(v[_n] for _n in ('_jump', 'i', 'vol'))
                    _result=_loop(v['istart'],v['istop'],1,_body8,tuple(v[_n] for _n in ('_jump', 'i', 'vol')),xp,jump_index=0)
                    v.update(zip(('_jump', 'i', 'vol'),_result))
                    return tuple(v[_n] for _n in ('_fatal', 'i', 'vol'))
                def _no7(_carry):
                    v=dict(_outer7);v.update(zip(('_fatal', 'i', 'vol'),_carry))
                    v['_fatal']=v['_fatal']+_int(1)  # Original WRF fatal event:module_cam_mp_ndrop.F:1773; caller must refuse
                    return tuple(v[_n] for _n in ('_fatal', 'i', 'vol'))
                _result=_cond(v['phase'] == 1,_yes7,_no7,tuple(v[_n] for _n in ('_fatal', 'i', 'vol')),xp)  # WRF:module_cam_mp_ndrop.F:1764
                v.update(zip(('_fatal', 'i', 'vol'),_result))
                return tuple(v[_n] for _n in ('_fatal', 'i', 'vol'))
            _result=_cond(v['phase'] == 2,_yes5,_no5,tuple(v[_n] for _n in ('_fatal', 'i', 'vol')),xp)  # WRF:module_cam_mp_ndrop.F:1760
            v.update(zip(('_fatal', 'i', 'vol'),_result))
            return tuple(v[_n] for _n in ('_fatal', 'i', 'vol'))
        _result=_cond(v['phase'] == 3,_yes3,_no3,tuple(v[_n] for _n in ('_fatal', 'i', 'vol')),xp)  # WRF:module_cam_mp_ndrop.F:1755
        v.update(zip(('_fatal', 'i', 'vol'),_result))
        _outer9=dict(v)
        def _body9(_k,_carry):
            v=dict(_outer9);v.update(zip(('_jump', 'hygro', 'i', 'vaerosol'),_carry));v['i']=_int(_k)
            _put(v,'vaerosol',(v['i'],),_get(v['vaerosol'], (v['i'],)) + _get(v['vol'], (v['i'],)))  # WRF:module_cam_mp_ndrop.F:1776
            _put(v,'hygro',(v['i'],),_get(v['hygro'], (v['i'],)) + _get(v['vol'], (v['i'],)) * _get(v['spechygro'], (v['ltype'],)))  # WRF:module_cam_mp_ndrop.F:1777
            return tuple(v[_n] for _n in ('_jump', 'hygro', 'i', 'vaerosol'))
        _result=_loop(v['istart'],v['istop'],1,_body9,tuple(v[_n] for _n in ('_jump', 'hygro', 'i', 'vaerosol')),xp,jump_index=0)
        v.update(zip(('_jump', 'hygro', 'i', 'vaerosol'),_result))
        return tuple(v[_n] for _n in ('_fatal', '_jump', 'hygro', 'i', 'l', 'lmass', 'lmasscw', 'ltype', 'vaerosol', 'vol'))
    _result=_loop(1,_get(v['nspec_amode'], (v['m'],)),1,_body2,tuple(v[_n] for _n in ('_fatal', '_jump', 'hygro', 'i', 'l', 'lmass', 'lmasscw', 'ltype', 'vaerosol', 'vol')),xp,jump_index=1)
    v.update(zip(('_fatal', '_jump', 'hygro', 'i', 'l', 'lmass', 'lmasscw', 'ltype', 'vaerosol', 'vol'),_result))
    _outer10=dict(v)
    def _body10(_k,_carry):
        v=dict(_outer10);v.update(zip(('_jump', 'hygro', 'i', 'vaerosol'),_carry));v['i']=_int(_k)
        _outer11=dict(v)
        def _yes11(_carry):
            v=dict(_outer11);v.update(zip(('hygro', 'vaerosol'),_carry))
            _put(v,'hygro',(v['i'],),_get(v['hygro'], (v['i'],)) / _get(v['vaerosol'], (v['i'],)))  # WRF:module_cam_mp_ndrop.F:1782
            _put(v,'vaerosol',(v['i'],),_get(v['vaerosol'], (v['i'],)) * _get(v['cs'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:1783
            return tuple(v[_n] for _n in ('hygro', 'vaerosol'))
        def _no11(_carry):
            v=dict(_outer11);v.update(zip(('hygro', 'vaerosol'),_carry))
            _put(v,'hygro',(v['i'],),_d(0.0))  # WRF:module_cam_mp_ndrop.F:1785
            _put(v,'vaerosol',(v['i'],),_d(0.0))  # WRF:module_cam_mp_ndrop.F:1786
            return tuple(v[_n] for _n in ('hygro', 'vaerosol'))
        _result=_cond(_get(v['vaerosol'], (v['i'],)) > _d(1e-30),_yes11,_no11,tuple(v[_n] for _n in ('hygro', 'vaerosol')),xp)  # WRF:module_cam_mp_ndrop.F:1781
        v.update(zip(('hygro', 'vaerosol'),_result))
        return tuple(v[_n] for _n in ('_jump', 'hygro', 'i', 'vaerosol'))
    _result=_loop(v['istart'],v['istop'],1,_body10,tuple(v[_n] for _n in ('_jump', 'hygro', 'i', 'vaerosol')),xp,jump_index=0)
    v.update(zip(('_jump', 'hygro', 'i', 'vaerosol'),_result))
    _put(v,'lnum',None,_get(v['numptr_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:1790
    _put(v,'lnumcw',None,_get(v['numptrcw_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:1791
    _outer12=dict(v)
    def _yes12(_carry):
        v=dict(_outer12);v.update(zip(('i', 'naerosol'),_carry))
        _outer13=dict(v)
        def _yes13(_carry):
            v=dict(_outer13);v.update(zip(('i', 'naerosol'),_carry))
            _outer14=dict(v)
            def _body14(_k,_carry):
                v=dict(_outer14);v.update(zip(('_jump', 'i', 'naerosol'),_carry));v['i']=_int(_k)
                _put(v,'naerosol',(v['i'],),(_get(v['raer'], (v['i'], v['k'], v['lnum'])) + _get(v['qqcw'], (v['i'], v['k'], v['lnumcw']))) * _get(v['cs'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:1797
                return tuple(v[_n] for _n in ('_jump', 'i', 'naerosol'))
            _result=_loop(v['istart'],v['istop'],1,_body14,tuple(v[_n] for _n in ('_jump', 'i', 'naerosol')),xp,jump_index=0)
            v.update(zip(('_jump', 'i', 'naerosol'),_result))
            return tuple(v[_n] for _n in ('i', 'naerosol'))
        def _no13(_carry):
            v=dict(_outer13);v.update(zip(('i', 'naerosol'),_carry))
            _outer15=dict(v)
            def _yes15(_carry):
                v=dict(_outer15);v.update(zip(('i', 'naerosol'),_carry))
                _outer16=dict(v)
                def _body16(_k,_carry):
                    v=dict(_outer16);v.update(zip(('_jump', 'i', 'naerosol'),_carry));v['i']=_int(_k)
                    _put(v,'naerosol',(v['i'],),_get(v['qqcw'], (v['i'], v['k'], v['lnumcw'])) * _get(v['cs'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:1801
                    return tuple(v[_n] for _n in ('_jump', 'i', 'naerosol'))
                _result=_loop(v['istart'],v['istop'],1,_body16,tuple(v[_n] for _n in ('_jump', 'i', 'naerosol')),xp,jump_index=0)
                v.update(zip(('_jump', 'i', 'naerosol'),_result))
                return tuple(v[_n] for _n in ('i', 'naerosol'))
            def _no15(_carry):
                v=dict(_outer15);v.update(zip(('i', 'naerosol'),_carry))
                _outer17=dict(v)
                def _body17(_k,_carry):
                    v=dict(_outer17);v.update(zip(('_jump', 'i', 'naerosol'),_carry));v['i']=_int(_k)
                    _put(v,'naerosol',(v['i'],),_get(v['raer'], (v['i'], v['k'], v['lnum'])) * _get(v['cs'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:1805
                    return tuple(v[_n] for _n in ('_jump', 'i', 'naerosol'))
                _result=_loop(v['istart'],v['istop'],1,_body17,tuple(v[_n] for _n in ('_jump', 'i', 'naerosol')),xp,jump_index=0)
                v.update(zip(('_jump', 'i', 'naerosol'),_result))
                return tuple(v[_n] for _n in ('i', 'naerosol'))
            _result=_cond(v['phase'] == 2,_yes15,_no15,tuple(v[_n] for _n in ('i', 'naerosol')),xp)  # WRF:module_cam_mp_ndrop.F:1799
            v.update(zip(('i', 'naerosol'),_result))
            return tuple(v[_n] for _n in ('i', 'naerosol'))
        _result=_cond(v['phase'] == 3,_yes13,_no13,tuple(v[_n] for _n in ('i', 'naerosol')),xp)  # WRF:module_cam_mp_ndrop.F:1794
        v.update(zip(('i', 'naerosol'),_result))
        _outer18=dict(v)
        def _body18(_k,_carry):
            v=dict(_outer18);v.update(zip(('_jump', 'i', 'naerosol'),_carry));v['i']=_int(_k)
            _put(v,'naerosol',(v['i'],),xp.maximum(_get(v['naerosol'], (v['i'],)), _get(v['vaerosol'], (v['i'],)) * _get(v['voltonumbhi_amode'], (v['m'],))))  # WRF:module_cam_mp_ndrop.F:1810
            _put(v,'naerosol',(v['i'],),xp.minimum(_get(v['naerosol'], (v['i'],)), _get(v['vaerosol'], (v['i'],)) * _get(v['voltonumblo_amode'], (v['m'],))))  # WRF:module_cam_mp_ndrop.F:1811
            return tuple(v[_n] for _n in ('_jump', 'i', 'naerosol'))
        _result=_loop(v['istart'],v['istop'],1,_body18,tuple(v[_n] for _n in ('_jump', 'i', 'naerosol')),xp,jump_index=0)
        v.update(zip(('_jump', 'i', 'naerosol'),_result))
        return tuple(v[_n] for _n in ('i', 'naerosol'))
    def _no12(_carry):
        v=dict(_outer12);v.update(zip(('i', 'naerosol'),_carry))
        _outer19=dict(v)
        def _body19(_k,_carry):
            v=dict(_outer19);v.update(zip(('_jump', 'i', 'naerosol'),_carry));v['i']=_int(_k)
            _put(v,'naerosol',(v['i'],),_get(v['vaerosol'], (v['i'],)) * v['npv1'])  # WRF:module_cam_mp_ndrop.F:1816
            _put(v,'naerosol',(v['i'],),xp.maximum(_get(v['naerosol'], (v['i'],)), _d(0.0)))  # WRF:module_cam_mp_ndrop.F:1817
            return tuple(v[_n] for _n in ('_jump', 'i', 'naerosol'))
        _result=_loop(v['istart'],v['istop'],1,_body19,tuple(v[_n] for _n in ('_jump', 'i', 'naerosol')),xp,jump_index=0)
        v.update(zip(('_jump', 'i', 'naerosol'),_result))
        return tuple(v[_n] for _n in ('i', 'naerosol'))
    _result=_cond(v['lnum'] > 0,_yes12,_no12,tuple(v[_n] for _n in ('i', 'naerosol')),xp)  # WRF:module_cam_mp_ndrop.F:1792
    v.update(zip(('i', 'naerosol'),_result))
    return (v['naerosol'],v['vaerosol'],v['hygro'],v['_fatal'],)

def dropmixnuc(lchnk,ncol,ncldwtr,tendnd,temp,omega,pmid,pint,pdel,rpdel,zm,kvh,wsub,cldn,cldo,raer,cflx,raertend,dtmicro,qqcw_inout,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('ndrop',{}))
    v['lchnk']=lchnk
    v['ncol']=ncol
    v['ncldwtr']=ncldwtr
    v['tendnd']=tendnd
    v['temp']=temp
    v['omega']=omega
    v['pmid']=pmid
    v['pint']=pint
    v['pdel']=pdel
    v['rpdel']=rpdel
    v['zm']=zm
    v['kvh']=kvh
    v['wsub']=wsub
    v['cldn']=cldn
    v['cldo']=cldo
    v['raer']=raer
    v['cflx']=cflx
    v['raertend']=raertend
    v['dtmicro']=dtmicro
    v['qqcw_inout']=qqcw_inout
    v['_jump']=_int(0)
    v['_fatal']=_int(0)
    v['depvel']=_zeros((v['pcols'],v['pcnst'],),'f64',xp)
    v['qcld']=_zeros((v['pver'],),'f64',xp)
    v['wtke']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['wtke_cen']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['zn']=_zeros((v['pver'],),'f64',xp)
    v['zs']=_zeros((v['pver'],),'f64',xp)
    v['zkmin']=_cast(_d(0.01),'f64',xp)
    v['zkmax']=_cast(_d(100.0),'f64',xp)
    v['w']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['cs']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['dz']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['zero']=_zeros((),'f64',xp)
    v['flxconv']=_zeros((),'f64',xp)
    v['wdiab']=_zeros((),'f64',xp)
    v['wmixmin']=_cast(_r(0.1),'f64',xp)
    v['qncld']=_zeros((v['pver'],),'f64',xp)
    v['ekd']=_zeros((v['pver'],),'f64',xp)
    v['ekk']=_zeros(((v['pver'])-(0)+1,),'f64',xp)
    v['srcn']=_zeros((v['pver'],),'f64',xp)
    v['sq2pi']=_cast(_d(2.5066283),'f64',xp)
    v['dtinv']=_zeros((),'f64',xp)
    v['km1']=_zeros((),'i32',xp)
    v['kp1']=_zeros((),'i32',xp)
    v['wbar']=_zeros((),'f64',xp)
    v['wmix']=_zeros((),'f64',xp)
    v['wmin']=_zeros((),'f64',xp)
    v['wmax']=_zeros((),'f64',xp)
    v['dum']=_zeros((),'f64',xp)
    v['dumc']=_zeros((),'f64',xp)
    v['dact']=_zeros((),'f64',xp)
    v['fluxntot']=_zeros((),'f64',xp)
    v['fac_srflx']=_zeros((),'f64',xp)
    v['surfrate']=_zeros((v['pcnst'],),'f64',xp)
    v['surfratemax']=_zeros((),'f64',xp)
    v['dtmin']=_zeros((),'f64',xp)
    v['tinv']=_zeros((),'f64',xp)
    v['dtt']=_zeros((),'f64',xp)
    v['nsubmix']=_zeros((),'i32',xp)
    v['nsubmix_bnd']=_zeros((),'i32',xp)
    v['i']=_zeros((),'i32',xp)
    v['k']=_zeros((),'i32',xp)
    v['m']=_zeros((),'i32',xp)
    v['n']=_zeros((),'i32',xp)
    v['dtmix']=_zeros((),'f64',xp)
    v['alogarg']=_zeros((),'f64',xp)
    v['pi']=_zeros((),'f64',xp)
    v['nnew']=_zeros((),'i32',xp)
    v['nsav']=_zeros((),'i32',xp)
    v['ntemp']=_zeros((),'i32',xp)
    v['overlapp']=_zeros((v['pver'],),'f64',xp)
    v['overlapm']=_zeros((v['pver'],),'f64',xp)
    v['ekkp']=_zeros((v['pver'],),'f64',xp)
    v['ekkm']=_zeros((v['pver'],),'f64',xp)
    v['count_submix']=_zeros((100,),'i32',xp)
    v['kvhmax']=_zeros((),'f64',xp)
    v['nsource']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['ndropmix']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['ndropcol']=_zeros((v['pcols'],),'f64',xp)
    v['lnum']=_zeros((),'i32',xp)
    v['lnumcw']=_zeros((),'i32',xp)
    v['lmass']=_zeros((),'i32',xp)
    v['lmasscw']=_zeros((),'i32',xp)
    v['na']=_zeros((v['pcols'],),'f64',xp)
    v['va']=_zeros((v['pcols'],),'f64',xp)
    v['hy']=_zeros((v['pcols'],),'f64',xp)
    v['naermod']=_zeros((v['ntot_amode'],),'f64',xp)
    v['sigmag']=_zeros((v['ntot_amode'],),'f64',xp)
    v['hygro']=_zeros((v['ntot_amode'],),'f64',xp)
    v['vaerosol']=_zeros((v['ntot_amode'],),'f64',xp)
    v['raercol']=_zeros((v['pver'],v['pcnst'],2,),'f64',xp)
    v['raercol_cw']=_zeros((v['pver'],v['pcnst'],2,),'f64',xp)
    v['surfrate_cw']=_zeros((v['pcnst'],),'f64',xp)
    v['source']=_zeros((v['pver'],),'f64',xp)
    v['fn']=_zeros((v['ntot_amode'],),'f64',xp)
    v['fm']=_zeros((v['ntot_amode'],),'f64',xp)
    v['fluxn']=_zeros((v['ntot_amode'],),'f64',xp)
    v['fluxm']=_zeros((v['ntot_amode'],),'f64',xp)
    v['nact']=_zeros((v['pver'],v['ntot_amode'],),'f64',xp)
    v['mact']=_zeros((v['pver'],v['ntot_amode'],),'f64',xp)
    v['qqcwtend']=_zeros((v['pcols'],v['pver'],v['pcnst'],),'f64',xp)
    v['tmpa']=_zeros((),'f64',xp)
    v['csbot']=_zeros((v['pver'],),'f64',xp)
    v['csbot_cscen']=_zeros((v['pver'],),'f64',xp)
    v['flux_fullact']=_zeros((v['pver'],),'f64',xp)
    v['taumix_internal_pver_inv']=_zeros((),'f64',xp)
    v['cldo_tmp']=_zeros((),'f64',xp)
    v['cldn_tmp']=_zeros((),'f64',xp)
    v['tau_cld_regenerate']=_zeros((),'f64',xp)
    v['ixndrop']=_zeros((),'i32',xp)
    v['l']=_zeros((),'i32',xp)
    v['arg']=_zeros((),'f64',xp)
    v['phase']=_zeros((),'i32',xp)
    _put(v,'arg',None,_d(1.0))  # WRF:module_cam_mp_ndrop.F:228
    _outer1=dict(v)
    def _yes1(_carry):
        v=dict(_outer1);v.update(zip(('_fatal',),_carry))
        v['_fatal']=v['_fatal']+_int(1)  # Original WRF fatal event:module_cam_mp_ndrop.F:234; caller must refuse
        return tuple(v[_n] for _n in ('_fatal',))
    def _no1(_carry):
        v=dict(_outer1);v.update(zip(('_fatal',),_carry))
        return tuple(v[_n] for _n in ('_fatal',))
    _result=_cond(xp.abs(_d(0.8427) - erf(v['arg'], xp=xp, _globals=_globals)) / _d(0.8427) > _d(0.001),_yes1,_no1,tuple(v[_n] for _n in ('_fatal',)),xp)  # WRF:module_cam_mp_ndrop.F:229
    v.update(zip(('_fatal',),_result))
    _put(v,'arg',None,_r(0.0))  # WRF:module_cam_mp_ndrop.F:236
    _outer2=dict(v)
    def _yes2(_carry):
        v=dict(_outer2);v.update(zip(('_fatal',),_carry))
        v['_fatal']=v['_fatal']+_int(1)  # Original WRF fatal event:module_cam_mp_ndrop.F:246; caller must refuse
        return tuple(v[_n] for _n in ('_fatal',))
    def _no2(_carry):
        v=dict(_outer2);v.update(zip(('_fatal',),_carry))
        return tuple(v[_n] for _n in ('_fatal',))
    _result=_cond(erf(v['arg'], xp=xp, _globals=_globals) != _r(0.0),_yes2,_no2,tuple(v[_n] for _n in ('_fatal',)),xp)  # WRF:module_cam_mp_ndrop.F:237
    v.update(zip(('_fatal',),_result))
    _put(v,'zero',None,_d(0.0))  # WRF:module_cam_mp_ndrop.F:248
    _put(v,'pi',None,_d(4.0) * xp.arctan(_d(1.0)))  # WRF:module_cam_mp_ndrop.F:251
    _put(v,'dtinv',None,_r(1.0) / v['dtmicro'])  # WRF:module_cam_mp_ndrop.F:252
    _put(v,'ixndrop',None,_int(4))  # WRF:module_cam_mp_ndrop.F:260
    _put(v,'depvel',(_slice(None,None,None),_slice(None,None,None),),_d(0.0))  # WRF:module_cam_mp_ndrop.F:261
    _outer3=dict(v)
    def _body3(_k,_carry):
        v=dict(_outer3);v.update(zip(('_jump', 'l', 'm'),_carry));v['m']=_int(_k)
        _outer4=dict(v)
        def _body4(_k,_carry):
            v=dict(_outer4);v.update(zip(('_jump', 'l'),_carry));v['l']=_int(_k)
            return tuple(v[_n] for _n in ('_jump', 'l'))
        _result=_loop(1,_get(v['nspec_amode'], (v['m'],)),1,_body4,tuple(v[_n] for _n in ('_jump', 'l')),xp,jump_index=0)
        v.update(zip(('_jump', 'l'),_result))
        return tuple(v[_n] for _n in ('_jump', 'l', 'm'))
    _result=_loop(1,v['ntot_amode'],1,_body3,tuple(v[_n] for _n in ('_jump', 'l', 'm')),xp,jump_index=0)
    v.update(zip(('_jump', 'l', 'm'),_result))
    _outer5=dict(v)
    def _body5(_k,_carry):
        v=dict(_outer5);v.update(zip(('_fatal', '_jump', 'alogarg', 'cldn_tmp', 'cldo_tmp', 'count_submix', 'cs', 'csbot', 'csbot_cscen', 'dact', 'dtmin', 'dtmix', 'dtt', 'dum', 'dumc', 'dz', 'ekd', 'ekk', 'ekkm', 'ekkp', 'fac_srflx', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'flxconv', 'fm', 'fn', 'hy', 'hygro', 'i', 'k', 'km1', 'kp1', 'kvhmax', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'mact', 'n', 'na', 'nact', 'naermod', 'ndropcol', 'ndropmix', 'nnew', 'nsav', 'nsource', 'nsubmix', 'nsubmix_bnd', 'ntemp', 'overlapm', 'overlapp', 'phase', 'qcld', 'qncld', 'qqcw_inout', 'qqcwtend', 'raercol', 'raercol_cw', 'raertend', 'source', 'srcn', 'surfrate', 'surfrate_cw', 'surfratemax', 'tau_cld_regenerate', 'taumix_internal_pver_inv', 'tendnd', 'tinv', 'tmpa', 'va', 'vaerosol', 'w', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix', 'wtke', 'wtke_cen', 'zn', 'zs'),_carry));v['i']=_int(_k)
        _outer6=dict(v)
        def _body6(_k,_carry):
            v=dict(_outer6);v.update(zip(('_jump', 'k', 'zs'),_carry));v['k']=_int(_k)
            _put(v,'zs',(v['k'],),_d(1.0) / (_get(v['zm'], (v['i'], v['k'])) - _get(v['zm'], (v['i'], v['k'] + 1))))  # WRF:module_cam_mp_ndrop.F:292
            return tuple(v[_n] for _n in ('_jump', 'k', 'zs'))
        _result=_loop(1,v['pver'] - 1,1,_body6,tuple(v[_n] for _n in ('_jump', 'k', 'zs')),xp,jump_index=0)
        v.update(zip(('_jump', 'k', 'zs'),_result))
        _put(v,'zs',(v['pver'],),_get(v['zs'], (v['pver'] - 1,)))  # WRF:module_cam_mp_ndrop.F:294
        _outer7=dict(v)
        def _body7(_k,_carry):
            v=dict(_outer7);v.update(zip(('_jump', 'cs', 'csbot', 'csbot_cscen', 'dz', 'ekd', 'k', 'kvhmax', 'm', 'mact', 'nact', 'nsource', 'qcld', 'qncld', 'srcn', 'w', 'wtke', 'wtke_cen', 'zn'),_carry));v['k']=_int(_k)
            _put(v,'qcld',(v['k'],),_get(v['ncldwtr'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:297
            _put(v,'qncld',(v['k'],),_d(0.0))  # WRF:module_cam_mp_ndrop.F:298
            _put(v,'srcn',(v['k'],),_d(0.0))  # WRF:module_cam_mp_ndrop.F:299
            _put(v,'cs',(v['i'],v['k'],),_get(v['pmid'], (v['i'], v['k'])) / (v['rair'] * _get(v['temp'], (v['i'], v['k']))))  # WRF:module_cam_mp_ndrop.F:300
            _put(v,'dz',(v['i'],v['k'],),_d(1.0) / (_get(v['cs'], (v['i'], v['k'])) * v['gravit'] * _get(v['rpdel'], (v['i'], v['k']))))  # WRF:module_cam_mp_ndrop.F:301
            _put(v,'w',(v['i'],v['k'],),_neg(_d(1.0)) * _get(v['omega'], (v['i'], v['k'])) / (_get(v['cs'], (v['i'], v['k'])) * v['gravit']))  # WRF:module_cam_mp_ndrop.F:302
            _outer8=dict(v)
            def _body8(_k,_carry):
                v=dict(_outer8);v.update(zip(('_jump', 'm', 'mact', 'nact'),_carry));v['m']=_int(_k)
                _put(v,'nact',(v['k'],v['m'],),_d(0.0))  # WRF:module_cam_mp_ndrop.F:304
                _put(v,'mact',(v['k'],v['m'],),_d(0.0))  # WRF:module_cam_mp_ndrop.F:305
                return tuple(v[_n] for _n in ('_jump', 'm', 'mact', 'nact'))
            _result=_loop(1,v['ntot_amode'],1,_body8,tuple(v[_n] for _n in ('_jump', 'm', 'mact', 'nact')),xp,jump_index=0)
            v.update(zip(('_jump', 'm', 'mact', 'nact'),_result))
            _put(v,'zn',(v['k'],),v['gravit'] * _get(v['rpdel'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:307
            _put(v,'kvhmax',None,xp.maximum(_get(v['kvh'], (v['i'], v['k'])), v['kvhmax']))  # WRF:module_cam_mp_ndrop.F:308
            _outer9=dict(v)
            def _yes9(_carry):
                v=dict(_outer9);v.update(zip(('csbot', 'csbot_cscen', 'ekd'),_carry))
                _put(v,'ekd',(v['k'],),_get(v['kvh'], (v['i'], v['k'] + 1)))  # WRF:module_cam_mp_ndrop.F:310
                _put(v,'ekd',(v['k'],),xp.maximum(_get(v['ekd'], (v['k'],)), v['zkmin']))  # WRF:module_cam_mp_ndrop.F:311
                _put(v,'ekd',(v['k'],),xp.minimum(_get(v['ekd'], (v['k'],)), v['zkmax']))  # WRF:module_cam_mp_ndrop.F:312
                _put(v,'csbot',(v['k'],),_d(2.0) * _get(v['pint'], (v['i'], v['k'] + 1)) / (v['rair'] * (_get(v['temp'], (v['i'], v['k'])) + _get(v['temp'], (v['i'], v['k'] + 1)))))  # WRF:module_cam_mp_ndrop.F:313
                _put(v,'csbot_cscen',(v['k'],),_get(v['csbot'], (v['k'],)) / _get(v['cs'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:314
                return tuple(v[_n] for _n in ('csbot', 'csbot_cscen', 'ekd'))
            def _no9(_carry):
                v=dict(_outer9);v.update(zip(('csbot', 'csbot_cscen', 'ekd'),_carry))
                _put(v,'ekd',(v['k'],),_d(0.0))  # WRF:module_cam_mp_ndrop.F:316
                _put(v,'csbot',(v['k'],),_get(v['cs'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:317
                _put(v,'csbot_cscen',(v['k'],),_d(1.0))  # WRF:module_cam_mp_ndrop.F:318
                return tuple(v[_n] for _n in ('csbot', 'csbot_cscen', 'ekd'))
            _result=_cond(v['k'] < v['pver'],_yes9,_no9,tuple(v[_n] for _n in ('csbot', 'csbot_cscen', 'ekd')),xp)  # WRF:module_cam_mp_ndrop.F:309
            v.update(zip(('csbot', 'csbot_cscen', 'ekd'),_result))
            _outer10=dict(v)
            def _yes10(_carry):
                v=dict(_outer10);v.update(zip(('wtke',),_carry))
                _put(v,'wtke',(v['i'],v['k'],),v['sq2pi'] * _get(v['depvel'], (v['i'], v['ixndrop'])))  # WRF:module_cam_mp_ndrop.F:322
                _put(v,'wtke',(v['i'],v['k'],),xp.maximum(_get(v['wtke'], (v['i'], v['k'])), v['wmixmin']))  # WRF:module_cam_mp_ndrop.F:324
                return tuple(v[_n] for _n in ('wtke',))
            def _no10(_carry):
                v=dict(_outer10);v.update(zip(('wtke',),_carry))
                _put(v,'wtke',(v['i'],v['k'],),v['sq2pi'] * _get(v['ekd'], (v['k'],)) / _get(v['dz'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:326
                return tuple(v[_n] for _n in ('wtke',))
            _result=_cond(v['k'] == v['pver'],_yes10,_no10,tuple(v[_n] for _n in ('wtke',)),xp)  # WRF:module_cam_mp_ndrop.F:321
            v.update(zip(('wtke',),_result))
            _put(v,'wtke_cen',(v['i'],v['k'],),_get(v['wsub'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:337
            _put(v,'wtke',(v['i'],v['k'],),_get(v['wsub'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:338
            _put(v,'wtke_cen',(v['i'],v['k'],),xp.maximum(_get(v['wtke_cen'], (v['i'], v['k'])), v['wmixmin']))  # WRF:module_cam_mp_ndrop.F:340
            _put(v,'wtke',(v['i'],v['k'],),xp.maximum(_get(v['wtke'], (v['i'], v['k'])), v['wmixmin']))  # WRF:module_cam_mp_ndrop.F:341
            _put(v,'nsource',(v['i'],v['k'],),_d(0.0))  # WRF:module_cam_mp_ndrop.F:342
            return tuple(v[_n] for _n in ('_jump', 'cs', 'csbot', 'csbot_cscen', 'dz', 'ekd', 'k', 'kvhmax', 'm', 'mact', 'nact', 'nsource', 'qcld', 'qncld', 'srcn', 'w', 'wtke', 'wtke_cen', 'zn'))
        _result=_loop(1,v['pver'],1,_body7,tuple(v[_n] for _n in ('_jump', 'cs', 'csbot', 'csbot_cscen', 'dz', 'ekd', 'k', 'kvhmax', 'm', 'mact', 'nact', 'nsource', 'qcld', 'qncld', 'srcn', 'w', 'wtke', 'wtke_cen', 'zn')),xp,jump_index=0)
        v.update(zip(('_jump', 'cs', 'csbot', 'csbot_cscen', 'dz', 'ekd', 'k', 'kvhmax', 'm', 'mact', 'nact', 'nsource', 'qcld', 'qncld', 'srcn', 'w', 'wtke', 'wtke_cen', 'zn'),_result))
        _put(v,'surfratemax',None,_d(0.0))  # WRF:module_cam_mp_ndrop.F:345
        _put(v,'nsav',None,1)  # WRF:module_cam_mp_ndrop.F:346
        _put(v,'nnew',None,2)  # WRF:module_cam_mp_ndrop.F:347
        _put(v,'surfrate',(v['ixndrop'],),_get(v['depvel'], (v['i'], v['ixndrop'])) / _get(v['dz'], (v['i'], v['pver'])))  # WRF:module_cam_mp_ndrop.F:348
        _put(v,'surfratemax',None,xp.maximum(v['surfratemax'], _get(v['surfrate'], (v['ixndrop'],))))  # WRF:module_cam_mp_ndrop.F:349
        _outer11=dict(v)
        def _body11(_k,_carry):
            v=dict(_outer11);v.update(zip(('_jump', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw', 'surfrate', 'surfrate_cw', 'surfratemax'),_carry));v['m']=_int(_k)
            _put(v,'lnum',None,_get(v['numptr_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:351
            _put(v,'lnumcw',None,_get(v['numptrcw_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:352
            _outer12=dict(v)
            def _yes12(_carry):
                v=dict(_outer12);v.update(zip(('raercol', 'raercol_cw', 'surfrate', 'surfrate_cw', 'surfratemax'),_carry))
                _put(v,'surfrate',(v['lnum'],),_get(v['depvel'], (v['i'], v['lnum'])) / _get(v['dz'], (v['i'], v['pver'])))  # WRF:module_cam_mp_ndrop.F:354
                _put(v,'surfrate_cw',(v['lnumcw'],),_get(v['surfrate'], (v['ixndrop'],)))  # WRF:module_cam_mp_ndrop.F:356
                _put(v,'surfratemax',None,xp.maximum(v['surfratemax'], _get(v['surfrate'], (v['lnum'],))))  # WRF:module_cam_mp_ndrop.F:357
                _put(v,'raercol_cw',(_slice(None,None,None),v['lnumcw'],v['nsav'],),_get(v['qqcw_inout'], (v['i'], _slice(1, None, None), v['lnumcw'])))  # WRF:module_cam_mp_ndrop.F:359
                _put(v,'raercol',(_slice(None,None,None),v['lnum'],v['nsav'],),_get(v['raer'], (v['i'], _slice(1, None, None), v['lnum'])))  # WRF:module_cam_mp_ndrop.F:360
                return tuple(v[_n] for _n in ('raercol', 'raercol_cw', 'surfrate', 'surfrate_cw', 'surfratemax'))
            def _no12(_carry):
                v=dict(_outer12);v.update(zip(('raercol', 'raercol_cw', 'surfrate', 'surfrate_cw', 'surfratemax'),_carry))
                return tuple(v[_n] for _n in ('raercol', 'raercol_cw', 'surfrate', 'surfrate_cw', 'surfratemax'))
            _result=_cond(v['lnum'] > 0,_yes12,_no12,tuple(v[_n] for _n in ('raercol', 'raercol_cw', 'surfrate', 'surfrate_cw', 'surfratemax')),xp)  # WRF:module_cam_mp_ndrop.F:353
            v.update(zip(('raercol', 'raercol_cw', 'surfrate', 'surfrate_cw', 'surfratemax'),_result))
            _outer13=dict(v)
            def _body13(_k,_carry):
                v=dict(_outer13);v.update(zip(('_jump', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw', 'surfrate', 'surfrate_cw', 'surfratemax'),_carry));v['l']=_int(_k)
                _put(v,'lmass',None,_get(v['lmassptr_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:363
                _put(v,'lmasscw',None,_get(v['lmassptrcw_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:364
                _put(v,'surfrate',(v['lmass'],),_get(v['depvel'], (v['i'], v['lmass'])) / _get(v['dz'], (v['i'], v['pver'])))  # WRF:module_cam_mp_ndrop.F:365
                _put(v,'surfrate_cw',(v['lmasscw'],),_get(v['surfrate'], (v['ixndrop'],)))  # WRF:module_cam_mp_ndrop.F:367
                _put(v,'surfratemax',None,xp.maximum(v['surfratemax'], _get(v['surfrate'], (v['lmass'],))))  # WRF:module_cam_mp_ndrop.F:368
                _put(v,'raercol_cw',(_slice(None,None,None),v['lmasscw'],v['nsav'],),_get(v['qqcw_inout'], (v['i'], _slice(1, None, None), v['lmasscw'])))  # WRF:module_cam_mp_ndrop.F:370
                _put(v,'raercol',(_slice(None,None,None),v['lmass'],v['nsav'],),_get(v['raer'], (v['i'], _slice(1, None, None), v['lmass'])))  # WRF:module_cam_mp_ndrop.F:371
                return tuple(v[_n] for _n in ('_jump', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw', 'surfrate', 'surfrate_cw', 'surfratemax'))
            _result=_loop(1,_get(v['nspec_amode'], (v['m'],)),1,_body13,tuple(v[_n] for _n in ('_jump', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw', 'surfrate', 'surfrate_cw', 'surfratemax')),xp,jump_index=0)
            v.update(zip(('_jump', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw', 'surfrate', 'surfrate_cw', 'surfratemax'),_result))
            return tuple(v[_n] for _n in ('_jump', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw', 'surfrate', 'surfrate_cw', 'surfratemax'))
        _result=_loop(1,v['ntot_amode'],1,_body11,tuple(v[_n] for _n in ('_jump', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw', 'surfrate', 'surfrate_cw', 'surfratemax')),xp,jump_index=0)
        v.update(zip(('_jump', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw', 'surfrate', 'surfrate_cw', 'surfratemax'),_result))
        _put(v,'tau_cld_regenerate',None,_d(3600.0) * _d(3.0))  # WRF:module_cam_mp_ndrop.F:384
        _outer14=dict(v)
        def _body14(_k,_carry):
            v=dict(_outer14);v.update(zip(('_fatal', '_jump', 'cldn_tmp', 'cldo_tmp', 'dact', 'dum', 'dumc', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'hy', 'hygro', 'k', 'km1', 'kp1', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'na', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix'),_carry));v['k']=_int(_k)
            _put(v,'km1',None,xp.maximum(v['k'] - 1, 1))  # WRF:module_cam_mp_ndrop.F:389
            _put(v,'kp1',None,xp.minimum(v['k'] + 1, v['pver']))  # WRF:module_cam_mp_ndrop.F:390
            _put(v,'cldo_tmp',None,_get(v['cldo'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:395
            _put(v,'cldn_tmp',None,_get(v['cldn'], (v['i'], v['k'])) * xp.exp(_neg(v['dtmicro']) / v['tau_cld_regenerate']))  # WRF:module_cam_mp_ndrop.F:396
            _outer15=dict(v)
            def _yes15(_carry):
                v=dict(_outer15);v.update(zip(('dact', 'dumc', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'nsource', 'qcld', 'raercol', 'raercol_cw'),_carry))
                _put(v,'nsource',(v['i'],v['k'],),_get(v['nsource'], (v['i'], v['k'])) + _get(v['qcld'], (v['k'],)) * (v['cldn_tmp'] - v['cldo_tmp']) / v['cldo_tmp'] * v['dtinv'])  # WRF:module_cam_mp_ndrop.F:403
                _put(v,'qcld',(v['k'],),_get(v['qcld'], (v['k'],)) * (_r(1.0) + (v['cldn_tmp'] - v['cldo_tmp']) / v['cldo_tmp']))  # WRF:module_cam_mp_ndrop.F:404
                _put(v,'dumc',None,(v['cldn_tmp'] - v['cldo_tmp']) / v['cldo_tmp'])  # WRF:module_cam_mp_ndrop.F:409
                _outer16=dict(v)
                def _body16(_k,_carry):
                    v=dict(_outer16);v.update(zip(('_jump', 'dact', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw'),_carry));v['m']=_int(_k)
                    _put(v,'lnum',None,_get(v['numptr_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:411
                    _put(v,'lnumcw',None,_get(v['numptrcw_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:412
                    _outer17=dict(v)
                    def _yes17(_carry):
                        v=dict(_outer17);v.update(zip(('dact', 'raercol', 'raercol_cw'),_carry))
                        _put(v,'dact',None,_get(v['raercol_cw'], (v['k'], v['lnumcw'], v['nsav'])) * v['dumc'])  # WRF:module_cam_mp_ndrop.F:414
                        _put(v,'raercol_cw',(v['k'],v['lnumcw'],v['nsav'],),_get(v['raercol_cw'], (v['k'], v['lnumcw'], v['nsav'])) + v['dact'])  # WRF:module_cam_mp_ndrop.F:416
                        _put(v,'raercol',(v['k'],v['lnum'],v['nsav'],),_get(v['raercol'], (v['k'], v['lnum'], v['nsav'])) - v['dact'])  # WRF:module_cam_mp_ndrop.F:417
                        return tuple(v[_n] for _n in ('dact', 'raercol', 'raercol_cw'))
                    def _no17(_carry):
                        v=dict(_outer17);v.update(zip(('dact', 'raercol', 'raercol_cw'),_carry))
                        return tuple(v[_n] for _n in ('dact', 'raercol', 'raercol_cw'))
                    _result=_cond(v['lnum'] > 0,_yes17,_no17,tuple(v[_n] for _n in ('dact', 'raercol', 'raercol_cw')),xp)  # WRF:module_cam_mp_ndrop.F:413
                    v.update(zip(('dact', 'raercol', 'raercol_cw'),_result))
                    _outer18=dict(v)
                    def _body18(_k,_carry):
                        v=dict(_outer18);v.update(zip(('_jump', 'dact', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw'),_carry));v['l']=_int(_k)
                        _put(v,'lmass',None,_get(v['lmassptr_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:420
                        _put(v,'lmasscw',None,_get(v['lmassptrcw_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:421
                        _put(v,'dact',None,_get(v['raercol_cw'], (v['k'], v['lmasscw'], v['nsav'])) * v['dumc'])  # WRF:module_cam_mp_ndrop.F:422
                        _put(v,'raercol_cw',(v['k'],v['lmasscw'],v['nsav'],),_get(v['raercol_cw'], (v['k'], v['lmasscw'], v['nsav'])) + v['dact'])  # WRF:module_cam_mp_ndrop.F:424
                        _put(v,'raercol',(v['k'],v['lmass'],v['nsav'],),_get(v['raercol'], (v['k'], v['lmass'], v['nsav'])) - v['dact'])  # WRF:module_cam_mp_ndrop.F:425
                        return tuple(v[_n] for _n in ('_jump', 'dact', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw'))
                    _result=_loop(1,_get(v['nspec_amode'], (v['m'],)),1,_body18,tuple(v[_n] for _n in ('_jump', 'dact', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw')),xp,jump_index=0)
                    v.update(zip(('_jump', 'dact', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw'),_result))
                    return tuple(v[_n] for _n in ('_jump', 'dact', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw'))
                _result=_loop(1,v['ntot_amode'],1,_body16,tuple(v[_n] for _n in ('_jump', 'dact', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw')),xp,jump_index=0)
                v.update(zip(('_jump', 'dact', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw'),_result))
                return tuple(v[_n] for _n in ('dact', 'dumc', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'nsource', 'qcld', 'raercol', 'raercol_cw'))
            def _no15(_carry):
                v=dict(_outer15);v.update(zip(('dact', 'dumc', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'nsource', 'qcld', 'raercol', 'raercol_cw'),_carry))
                return tuple(v[_n] for _n in ('dact', 'dumc', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'nsource', 'qcld', 'raercol', 'raercol_cw'))
            _result=_cond(v['cldn_tmp'] < v['cldo_tmp'],_yes15,_no15,tuple(v[_n] for _n in ('dact', 'dumc', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'nsource', 'qcld', 'raercol', 'raercol_cw')),xp)  # WRF:module_cam_mp_ndrop.F:400
            v.update(zip(('dact', 'dumc', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'nsource', 'qcld', 'raercol', 'raercol_cw'),_result))
            _put(v,'cldo_tmp',None,v['cldn_tmp'])  # WRF:module_cam_mp_ndrop.F:433
            _put(v,'cldn_tmp',None,_get(v['cldn'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:434
            _outer19=dict(v)
            def _yes19(_carry):
                v=dict(_outer19);v.update(zip(('_fatal', 'dact', 'dum', 'dumc', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'hy', 'hygro', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'na', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix'),_carry))
                _put(v,'wbar',None,_get(v['wtke_cen'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:440
                _put(v,'wmix',None,_d(0.0))  # WRF:module_cam_mp_ndrop.F:441
                _put(v,'wmin',None,_d(0.0))  # WRF:module_cam_mp_ndrop.F:442
                _put(v,'wmax',None,_d(10.0))  # WRF:module_cam_mp_ndrop.F:443
                _put(v,'wdiab',None,0)  # WRF:module_cam_mp_ndrop.F:444
                _put(v,'phase',None,1)  # WRF:module_cam_mp_ndrop.F:447
                _outer20=dict(v)
                def _body20(_k,_carry):
                    v=dict(_outer20);v.update(zip(('_fatal', '_jump', 'hy', 'hygro', 'm', 'na', 'naermod', 'va', 'vaerosol'),_carry));v['m']=_int(_k)
                    _call=loadaer(v['raer'],v['qqcw_inout'],v['i'],v['i'],v['k'],v['m'],v['cs'],_get(v['npv'], (v['m'],)),v['phase'],v['na'],v['va'],v['hy'],_globals=_globals,xp=xp)
                    _put(v,'na',None,_call[0])  # WRF:module_cam_mp_ndrop.F:449
                    _put(v,'va',None,_call[1])  # WRF:module_cam_mp_ndrop.F:449
                    _put(v,'hy',None,_call[2])  # WRF:module_cam_mp_ndrop.F:449
                    v['_fatal']=v['_fatal']+_call[3]
                    _put(v,'naermod',(v['m'],),_get(v['na'], (v['i'],)))  # WRF:module_cam_mp_ndrop.F:451
                    _put(v,'vaerosol',(v['m'],),_get(v['va'], (v['i'],)))  # WRF:module_cam_mp_ndrop.F:452
                    _put(v,'hygro',(v['m'],),_get(v['hy'], (v['i'],)))  # WRF:module_cam_mp_ndrop.F:453
                    return tuple(v[_n] for _n in ('_fatal', '_jump', 'hy', 'hygro', 'm', 'na', 'naermod', 'va', 'vaerosol'))
                _result=_loop(1,v['ntot_amode'],1,_body20,tuple(v[_n] for _n in ('_fatal', '_jump', 'hy', 'hygro', 'm', 'na', 'naermod', 'va', 'vaerosol')),xp,jump_index=1)
                v.update(zip(('_fatal', '_jump', 'hy', 'hygro', 'm', 'na', 'naermod', 'va', 'vaerosol'),_result))
                _call=activate_modal(v['wbar'],v['wmix'],v['wdiab'],v['wmin'],v['wmax'],_get(v['temp'], (v['i'], v['k'])),_get(v['cs'], (v['i'], v['k'])),v['naermod'],v['ntot_amode'],v['ntot_amode'],v['vaerosol'],v['sigmag'],v['hygro'],v['fn'],v['fm'],v['fluxn'],v['fluxm'],_get(v['flux_fullact'], (v['k'],)),_globals=_globals,xp=xp)
                _put(v,'fn',None,_call[0])  # WRF:module_cam_mp_ndrop.F:461
                _put(v,'fm',None,_call[1])  # WRF:module_cam_mp_ndrop.F:461
                _put(v,'fluxn',None,_call[2])  # WRF:module_cam_mp_ndrop.F:461
                _put(v,'fluxm',None,_call[3])  # WRF:module_cam_mp_ndrop.F:461
                _put(v,'flux_fullact',(v['k'],),_call[4])  # WRF:module_cam_mp_ndrop.F:461
                v['_fatal']=v['_fatal']+_call[5]
                _put(v,'dumc',None,v['cldn_tmp'] - v['cldo_tmp'])  # WRF:module_cam_mp_ndrop.F:469
                _outer21=dict(v)
                def _body21(_k,_carry):
                    v=dict(_outer21);v.update(zip(('_jump', 'dact', 'dum', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'nsource', 'qcld', 'raercol', 'raercol_cw'),_carry));v['m']=_int(_k)
                    _put(v,'lnum',None,_get(v['numptr_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:471
                    _put(v,'lnumcw',None,_get(v['numptrcw_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:472
                    _put(v,'dact',None,v['dumc'] * _get(v['fn'], (v['m'],)) * _get(v['raer'], (v['i'], v['k'], v['lnum'])))  # WRF:module_cam_mp_ndrop.F:473
                    _put(v,'qcld',(v['k'],),_get(v['qcld'], (v['k'],)) + v['dact'])  # WRF:module_cam_mp_ndrop.F:474
                    _put(v,'nsource',(v['i'],v['k'],),_get(v['nsource'], (v['i'], v['k'])) + v['dact'] * v['dtinv'])  # WRF:module_cam_mp_ndrop.F:475
                    _outer22=dict(v)
                    def _yes22(_carry):
                        v=dict(_outer22);v.update(zip(('raercol', 'raercol_cw'),_carry))
                        _put(v,'raercol_cw',(v['k'],v['lnumcw'],v['nsav'],),_get(v['raercol_cw'], (v['k'], v['lnumcw'], v['nsav'])) + v['dact'])  # WRF:module_cam_mp_ndrop.F:477
                        _put(v,'raercol',(v['k'],v['lnum'],v['nsav'],),_get(v['raercol'], (v['k'], v['lnum'], v['nsav'])) - v['dact'])  # WRF:module_cam_mp_ndrop.F:478
                        return tuple(v[_n] for _n in ('raercol', 'raercol_cw'))
                    def _no22(_carry):
                        v=dict(_outer22);v.update(zip(('raercol', 'raercol_cw'),_carry))
                        return tuple(v[_n] for _n in ('raercol', 'raercol_cw'))
                    _result=_cond(v['lnum'] > 0,_yes22,_no22,tuple(v[_n] for _n in ('raercol', 'raercol_cw')),xp)  # WRF:module_cam_mp_ndrop.F:476
                    v.update(zip(('raercol', 'raercol_cw'),_result))
                    _put(v,'dum',None,v['dumc'] * _get(v['fm'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:480
                    _outer23=dict(v)
                    def _body23(_k,_carry):
                        v=dict(_outer23);v.update(zip(('_jump', 'dact', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw'),_carry));v['l']=_int(_k)
                        _put(v,'lmass',None,_get(v['lmassptr_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:482
                        _put(v,'lmasscw',None,_get(v['lmassptrcw_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:483
                        _put(v,'dact',None,v['dum'] * _get(v['raer'], (v['i'], v['k'], v['lmass'])))  # WRF:module_cam_mp_ndrop.F:484
                        _put(v,'raercol_cw',(v['k'],v['lmasscw'],v['nsav'],),_get(v['raercol_cw'], (v['k'], v['lmasscw'], v['nsav'])) + v['dact'])  # WRF:module_cam_mp_ndrop.F:485
                        _put(v,'raercol',(v['k'],v['lmass'],v['nsav'],),_get(v['raercol'], (v['k'], v['lmass'], v['nsav'])) - v['dact'])  # WRF:module_cam_mp_ndrop.F:486
                        return tuple(v[_n] for _n in ('_jump', 'dact', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw'))
                    _result=_loop(1,_get(v['nspec_amode'], (v['m'],)),1,_body23,tuple(v[_n] for _n in ('_jump', 'dact', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw')),xp,jump_index=0)
                    v.update(zip(('_jump', 'dact', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw'),_result))
                    return tuple(v[_n] for _n in ('_jump', 'dact', 'dum', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'nsource', 'qcld', 'raercol', 'raercol_cw'))
                _result=_loop(1,v['ntot_amode'],1,_body21,tuple(v[_n] for _n in ('_jump', 'dact', 'dum', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'nsource', 'qcld', 'raercol', 'raercol_cw')),xp,jump_index=0)
                v.update(zip(('_jump', 'dact', 'dum', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'nsource', 'qcld', 'raercol', 'raercol_cw'),_result))
                return tuple(v[_n] for _n in ('_fatal', 'dact', 'dum', 'dumc', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'hy', 'hygro', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'na', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix'))
            def _no19(_carry):
                v=dict(_outer19);v.update(zip(('_fatal', 'dact', 'dum', 'dumc', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'hy', 'hygro', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'na', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix'),_carry))
                return tuple(v[_n] for _n in ('_fatal', 'dact', 'dum', 'dumc', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'hy', 'hygro', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'na', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix'))
            _result=_cond(v['cldn_tmp'] - v['cldo_tmp'] > _r(0.01),_yes19,_no19,tuple(v[_n] for _n in ('_fatal', 'dact', 'dum', 'dumc', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'hy', 'hygro', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'na', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix')),xp)  # WRF:module_cam_mp_ndrop.F:436
            v.update(zip(('_fatal', 'dact', 'dum', 'dumc', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'hy', 'hygro', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'na', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix'),_result))
            return tuple(v[_n] for _n in ('_fatal', '_jump', 'cldn_tmp', 'cldo_tmp', 'dact', 'dum', 'dumc', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'hy', 'hygro', 'k', 'km1', 'kp1', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'na', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix'))
        _result=_loop(1,v['pver'],1,_body14,tuple(v[_n] for _n in ('_fatal', '_jump', 'cldn_tmp', 'cldo_tmp', 'dact', 'dum', 'dumc', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'hy', 'hygro', 'k', 'km1', 'kp1', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'na', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix')),xp,jump_index=1)
        v.update(zip(('_fatal', '_jump', 'cldn_tmp', 'cldo_tmp', 'dact', 'dum', 'dumc', 'flux_fullact', 'fluxm', 'fluxn', 'fm', 'fn', 'hy', 'hygro', 'k', 'km1', 'kp1', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'na', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix'),_result))
        _outer24=dict(v)
        def _body24(_k,_carry):
            v=dict(_outer24);v.update(zip(('_fatal', '_jump', 'alogarg', 'dum', 'dumc', 'ekd', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'fm', 'fn', 'hy', 'hygro', 'k', 'km1', 'kp1', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'mact', 'na', 'nact', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'srcn', 'taumix_internal_pver_inv', 'tmpa', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix'),_carry));v['k']=_int(_k)
            _put(v,'km1',None,xp.maximum(v['k'] - 1, 1))  # WRF:module_cam_mp_ndrop.F:507
            _put(v,'kp1',None,xp.minimum(v['k'] + 1, v['pver']))  # WRF:module_cam_mp_ndrop.F:508
            _put(v,'taumix_internal_pver_inv',None,_r(0.0))  # WRF:module_cam_mp_ndrop.F:509
            _outer25=dict(v)
            def _yes25(_carry):
                v=dict(_outer25);v.update(zip(('_fatal', 'alogarg', 'dum', 'dumc', 'ekd', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'fm', 'fn', 'hy', 'hygro', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'mact', 'na', 'nact', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'srcn', 'taumix_internal_pver_inv', 'tmpa', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix'),_carry))
                _put(v,'wdiab',None,0)  # WRF:module_cam_mp_ndrop.F:514
                _put(v,'wmix',None,_d(0.0))  # WRF:module_cam_mp_ndrop.F:517
                _put(v,'wbar',None,_get(v['wtke'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:518
                _outer26=dict(v)
                def _yes26(_carry):
                    v=dict(_outer26);v.update(zip(('wbar',),_carry))
                    _put(v,'wbar',None,_get(v['wtke_cen'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:520
                    return tuple(v[_n] for _n in ('wbar',))
                def _no26(_carry):
                    v=dict(_outer26);v.update(zip(('wbar',),_carry))
                    return tuple(v[_n] for _n in ('wbar',))
                _result=_cond(v['k'] == v['pver'],_yes26,_no26,tuple(v[_n] for _n in ('wbar',)),xp)  # WRF:module_cam_mp_ndrop.F:520
                v.update(zip(('wbar',),_result))
                _put(v,'wmax',None,_d(10.0))  # WRF:module_cam_mp_ndrop.F:521
                _put(v,'wmin',None,_d(0.0))  # WRF:module_cam_mp_ndrop.F:522
                _outer27=dict(v)
                def _yes27(_carry):
                    v=dict(_outer27);v.update(zip(('_fatal', 'alogarg', 'dum', 'dumc', 'ekd', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'fm', 'fn', 'hy', 'hygro', 'lnum', 'lnumcw', 'm', 'mact', 'na', 'nact', 'naermod', 'nsource', 'phase', 'srcn', 'taumix_internal_pver_inv', 'tmpa', 'va', 'vaerosol', 'wmin'),_carry))
                    _put(v,'ekd',(v['k'],),_get(v['wtke'], (v['i'], v['k'])) * _get(v['dz'], (v['i'], v['k'])) / v['sq2pi'])  # WRF:module_cam_mp_ndrop.F:528
                    _put(v,'ekd',(v['k'],),v['wbar'] / _get(v['zs'], (v['k'],)))  # WRF:module_cam_mp_ndrop.F:539
                    _put(v,'alogarg',None,xp.maximum(_d(1e-20), 1 / _get(v['cldn'], (v['i'], v['k'])) - _d(1.0)))  # WRF:module_cam_mp_ndrop.F:541
                    _put(v,'wmin',None,v['wbar'] + v['wmix'] * _r(0.25) * v['sq2pi'] * xp.log(v['alogarg']))  # WRF:module_cam_mp_ndrop.F:542
                    _put(v,'phase',None,1)  # WRF:module_cam_mp_ndrop.F:543
                    _outer28=dict(v)
                    def _body28(_k,_carry):
                        v=dict(_outer28);v.update(zip(('_fatal', '_jump', 'hy', 'hygro', 'm', 'na', 'naermod', 'va', 'vaerosol'),_carry));v['m']=_int(_k)
                        _call=loadaer(v['raer'],v['qqcw_inout'],v['i'],v['i'],v['kp1'],v['m'],v['cs'],_get(v['npv'], (v['m'],)),v['phase'],v['na'],v['va'],v['hy'],_globals=_globals,xp=xp)
                        _put(v,'na',None,_call[0])  # WRF:module_cam_mp_ndrop.F:548
                        _put(v,'va',None,_call[1])  # WRF:module_cam_mp_ndrop.F:548
                        _put(v,'hy',None,_call[2])  # WRF:module_cam_mp_ndrop.F:548
                        v['_fatal']=v['_fatal']+_call[3]
                        _put(v,'naermod',(v['m'],),_get(v['na'], (v['i'],)))  # WRF:module_cam_mp_ndrop.F:550
                        _put(v,'vaerosol',(v['m'],),_get(v['va'], (v['i'],)))  # WRF:module_cam_mp_ndrop.F:551
                        _put(v,'hygro',(v['m'],),_get(v['hy'], (v['i'],)))  # WRF:module_cam_mp_ndrop.F:552
                        return tuple(v[_n] for _n in ('_fatal', '_jump', 'hy', 'hygro', 'm', 'na', 'naermod', 'va', 'vaerosol'))
                    _result=_loop(1,v['ntot_amode'],1,_body28,tuple(v[_n] for _n in ('_fatal', '_jump', 'hy', 'hygro', 'm', 'na', 'naermod', 'va', 'vaerosol')),xp,jump_index=1)
                    v.update(zip(('_fatal', '_jump', 'hy', 'hygro', 'm', 'na', 'naermod', 'va', 'vaerosol'),_result))
                    _call=activate_modal(v['wbar'],v['wmix'],v['wdiab'],v['wmin'],v['wmax'],_get(v['temp'], (v['i'], v['k'])),_get(v['cs'], (v['i'], v['k'])),v['naermod'],v['ntot_amode'],v['ntot_amode'],v['vaerosol'],v['sigmag'],v['hygro'],v['fn'],v['fm'],v['fluxn'],v['fluxm'],_get(v['flux_fullact'], (v['k'],)),_globals=_globals,xp=xp)
                    _put(v,'fn',None,_call[0])  # WRF:module_cam_mp_ndrop.F:559
                    _put(v,'fm',None,_call[1])  # WRF:module_cam_mp_ndrop.F:559
                    _put(v,'fluxn',None,_call[2])  # WRF:module_cam_mp_ndrop.F:559
                    _put(v,'fluxm',None,_call[3])  # WRF:module_cam_mp_ndrop.F:559
                    _put(v,'flux_fullact',(v['k'],),_call[4])  # WRF:module_cam_mp_ndrop.F:559
                    v['_fatal']=v['_fatal']+_call[5]
                    _outer29=dict(v)
                    def _yes29(_carry):
                        v=dict(_outer29);v.update(zip(('dumc',),_carry))
                        _put(v,'dumc',None,_get(v['cldn'], (v['i'], v['k'])) - _get(v['cldn'], (v['i'], v['kp1'])))  # WRF:module_cam_mp_ndrop.F:567
                        return tuple(v[_n] for _n in ('dumc',))
                    def _no29(_carry):
                        v=dict(_outer29);v.update(zip(('dumc',),_carry))
                        _put(v,'dumc',None,_get(v['cldn'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:569
                        return tuple(v[_n] for _n in ('dumc',))
                    _result=_cond(v['k'] < v['pver'],_yes29,_no29,tuple(v[_n] for _n in ('dumc',)),xp)  # WRF:module_cam_mp_ndrop.F:566
                    v.update(zip(('dumc',),_result))
                    _put(v,'fluxntot',None,0)  # WRF:module_cam_mp_ndrop.F:571
                    _put(v,'dum',None,_get(v['csbot_cscen'], (v['k'],)) / _get(v['dz'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:579
                    _outer30=dict(v)
                    def _yes30(_carry):
                        v=dict(_outer30);v.update(zip(('taumix_internal_pver_inv',),_carry))
                        _put(v,'taumix_internal_pver_inv',None,_get(v['flux_fullact'], (v['k'],)) / _get(v['dz'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:609
                        return tuple(v[_n] for _n in ('taumix_internal_pver_inv',))
                    def _no30(_carry):
                        v=dict(_outer30);v.update(zip(('taumix_internal_pver_inv',),_carry))
                        return tuple(v[_n] for _n in ('taumix_internal_pver_inv',))
                    _result=_cond(v['k'] == v['pver'],_yes30,_no30,tuple(v[_n] for _n in ('taumix_internal_pver_inv',)),xp)  # WRF:module_cam_mp_ndrop.F:608
                    v.update(zip(('taumix_internal_pver_inv',),_result))
                    _outer31=dict(v)
                    def _body31(_k,_carry):
                        v=dict(_outer31);v.update(zip(('_jump', 'fluxm', 'fluxn', 'fluxntot', 'lnum', 'lnumcw', 'm', 'mact', 'nact', 'tmpa'),_carry));v['m']=_int(_k)
                        _put(v,'fluxn',(v['m'],),_get(v['fluxn'], (v['m'],)) * v['dumc'])  # WRF:module_cam_mp_ndrop.F:612
                        _put(v,'fluxm',(v['m'],),_get(v['fluxm'], (v['m'],)) * v['dumc'])  # WRF:module_cam_mp_ndrop.F:613
                        _put(v,'lnum',None,_get(v['numptr_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:614
                        _put(v,'nact',(v['k'],v['m'],),_get(v['nact'], (v['k'], v['m'])) + _get(v['fluxn'], (v['m'],)) * v['dum'])  # WRF:module_cam_mp_ndrop.F:615
                        _put(v,'mact',(v['k'],v['m'],),_get(v['mact'], (v['k'], v['m'])) + _get(v['fluxm'], (v['m'],)) * v['dum'])  # WRF:module_cam_mp_ndrop.F:616
                        _outer32=dict(v)
                        def _yes32(_carry):
                            v=dict(_outer32);v.update(zip(('fluxntot', 'lnumcw', 'tmpa'),_carry))
                            _put(v,'fluxntot',None,v['fluxntot'] + _get(v['fluxn'], (v['m'],)) * _get(v['raercol'], (v['kp1'], v['lnum'], v['nsav'])) * _get(v['cs'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:619
                            return tuple(v[_n] for _n in ('fluxntot', 'lnumcw', 'tmpa'))
                        def _no32(_carry):
                            v=dict(_outer32);v.update(zip(('fluxntot', 'lnumcw', 'tmpa'),_carry))
                            _put(v,'lnumcw',None,_get(v['numptrcw_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:622
                            _put(v,'tmpa',None,_get(v['raercol'], (v['kp1'], v['lnum'], v['nsav'])) * _get(v['fluxn'], (v['m'],)) + _get(v['raercol'], (v['kp1'], v['lnumcw'], v['nsav'])) * (_get(v['fluxn'], (v['m'],)) - v['taumix_internal_pver_inv']))  # WRF:module_cam_mp_ndrop.F:623
                            _put(v,'fluxntot',None,v['fluxntot'] + xp.maximum(_d(0.0), v['tmpa']) * _get(v['cs'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:626
                            return tuple(v[_n] for _n in ('fluxntot', 'lnumcw', 'tmpa'))
                        _result=_cond(v['k'] > v['pver'],_yes32,_no32,tuple(v[_n] for _n in ('fluxntot', 'lnumcw', 'tmpa')),xp)  # WRF:module_cam_mp_ndrop.F:617
                        v.update(zip(('fluxntot', 'lnumcw', 'tmpa'),_result))
                        return tuple(v[_n] for _n in ('_jump', 'fluxm', 'fluxn', 'fluxntot', 'lnum', 'lnumcw', 'm', 'mact', 'nact', 'tmpa'))
                    _result=_loop(1,v['ntot_amode'],1,_body31,tuple(v[_n] for _n in ('_jump', 'fluxm', 'fluxn', 'fluxntot', 'lnum', 'lnumcw', 'm', 'mact', 'nact', 'tmpa')),xp,jump_index=0)
                    v.update(zip(('_jump', 'fluxm', 'fluxn', 'fluxntot', 'lnum', 'lnumcw', 'm', 'mact', 'nact', 'tmpa'),_result))
                    _put(v,'srcn',(v['k'],),_get(v['srcn'], (v['k'],)) + v['fluxntot'] / (_get(v['cs'], (v['i'], v['k'])) * _get(v['dz'], (v['i'], v['k']))))  # WRF:module_cam_mp_ndrop.F:629
                    _put(v,'nsource',(v['i'],v['k'],),_get(v['nsource'], (v['i'], v['k'])) + v['fluxntot'] / (_get(v['cs'], (v['i'], v['k'])) * _get(v['dz'], (v['i'], v['k']))))  # WRF:module_cam_mp_ndrop.F:630
                    return tuple(v[_n] for _n in ('_fatal', 'alogarg', 'dum', 'dumc', 'ekd', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'fm', 'fn', 'hy', 'hygro', 'lnum', 'lnumcw', 'm', 'mact', 'na', 'nact', 'naermod', 'nsource', 'phase', 'srcn', 'taumix_internal_pver_inv', 'tmpa', 'va', 'vaerosol', 'wmin'))
                def _no27(_carry):
                    v=dict(_outer27);v.update(zip(('_fatal', 'alogarg', 'dum', 'dumc', 'ekd', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'fm', 'fn', 'hy', 'hygro', 'lnum', 'lnumcw', 'm', 'mact', 'na', 'nact', 'naermod', 'nsource', 'phase', 'srcn', 'taumix_internal_pver_inv', 'tmpa', 'va', 'vaerosol', 'wmin'),_carry))
                    return tuple(v[_n] for _n in ('_fatal', 'alogarg', 'dum', 'dumc', 'ekd', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'fm', 'fn', 'hy', 'hygro', 'lnum', 'lnumcw', 'm', 'mact', 'na', 'nact', 'naermod', 'nsource', 'phase', 'srcn', 'taumix_internal_pver_inv', 'tmpa', 'va', 'vaerosol', 'wmin'))
                _result=_cond(xp.logical_or(_get(v['cldn'], (v['i'], v['k'])) - _get(v['cldn'], (v['i'], v['kp1'])) > _r(0.01), v['k'] == v['pver']),_yes27,_no27,tuple(v[_n] for _n in ('_fatal', 'alogarg', 'dum', 'dumc', 'ekd', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'fm', 'fn', 'hy', 'hygro', 'lnum', 'lnumcw', 'm', 'mact', 'na', 'nact', 'naermod', 'nsource', 'phase', 'srcn', 'taumix_internal_pver_inv', 'tmpa', 'va', 'vaerosol', 'wmin')),xp)  # WRF:module_cam_mp_ndrop.F:524
                v.update(zip(('_fatal', 'alogarg', 'dum', 'dumc', 'ekd', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'fm', 'fn', 'hy', 'hygro', 'lnum', 'lnumcw', 'm', 'mact', 'na', 'nact', 'naermod', 'nsource', 'phase', 'srcn', 'taumix_internal_pver_inv', 'tmpa', 'va', 'vaerosol', 'wmin'),_result))
                return tuple(v[_n] for _n in ('_fatal', 'alogarg', 'dum', 'dumc', 'ekd', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'fm', 'fn', 'hy', 'hygro', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'mact', 'na', 'nact', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'srcn', 'taumix_internal_pver_inv', 'tmpa', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix'))
            def _no25(_carry):
                v=dict(_outer25);v.update(zip(('_fatal', 'alogarg', 'dum', 'dumc', 'ekd', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'fm', 'fn', 'hy', 'hygro', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'mact', 'na', 'nact', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'srcn', 'taumix_internal_pver_inv', 'tmpa', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix'),_carry))
                _put(v,'nsource',(v['i'],v['k'],),_get(v['nsource'], (v['i'], v['k'])) - _get(v['qcld'], (v['k'],)) * v['dtinv'])  # WRF:module_cam_mp_ndrop.F:639
                _put(v,'qcld',(v['k'],),0)  # WRF:module_cam_mp_ndrop.F:640
                _outer33=dict(v)
                def _body33(_k,_carry):
                    v=dict(_outer33);v.update(zip(('_jump', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw'),_carry));v['m']=_int(_k)
                    _put(v,'lnum',None,_get(v['numptr_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:643
                    _put(v,'lnumcw',None,_get(v['numptrcw_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:644
                    _outer34=dict(v)
                    def _yes34(_carry):
                        v=dict(_outer34);v.update(zip(('raercol', 'raercol_cw'),_carry))
                        _put(v,'raercol',(v['k'],v['lnum'],v['nsav'],),_get(v['raercol'], (v['k'], v['lnum'], v['nsav'])) + _get(v['raercol_cw'], (v['k'], v['lnumcw'], v['nsav'])))  # WRF:module_cam_mp_ndrop.F:646
                        _put(v,'raercol_cw',(v['k'],v['lnumcw'],v['nsav'],),_r(0.0))  # WRF:module_cam_mp_ndrop.F:647
                        return tuple(v[_n] for _n in ('raercol', 'raercol_cw'))
                    def _no34(_carry):
                        v=dict(_outer34);v.update(zip(('raercol', 'raercol_cw'),_carry))
                        return tuple(v[_n] for _n in ('raercol', 'raercol_cw'))
                    _result=_cond(v['lnum'] > 0,_yes34,_no34,tuple(v[_n] for _n in ('raercol', 'raercol_cw')),xp)  # WRF:module_cam_mp_ndrop.F:645
                    v.update(zip(('raercol', 'raercol_cw'),_result))
                    _outer35=dict(v)
                    def _body35(_k,_carry):
                        v=dict(_outer35);v.update(zip(('_jump', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw'),_carry));v['l']=_int(_k)
                        _put(v,'lmass',None,_get(v['lmassptr_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:650
                        _put(v,'lmasscw',None,_get(v['lmassptrcw_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:651
                        _put(v,'raercol',(v['k'],v['lmass'],v['nsav'],),_get(v['raercol'], (v['k'], v['lmass'], v['nsav'])) + _get(v['raercol_cw'], (v['k'], v['lmasscw'], v['nsav'])))  # WRF:module_cam_mp_ndrop.F:652
                        _put(v,'raercol_cw',(v['k'],v['lmasscw'],v['nsav'],),_r(0.0))  # WRF:module_cam_mp_ndrop.F:653
                        return tuple(v[_n] for _n in ('_jump', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw'))
                    _result=_loop(1,_get(v['nspec_amode'], (v['m'],)),1,_body35,tuple(v[_n] for _n in ('_jump', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw')),xp,jump_index=0)
                    v.update(zip(('_jump', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw'),_result))
                    return tuple(v[_n] for _n in ('_jump', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw'))
                _result=_loop(1,v['ntot_amode'],1,_body33,tuple(v[_n] for _n in ('_jump', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw')),xp,jump_index=0)
                v.update(zip(('_jump', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw'),_result))
                return tuple(v[_n] for _n in ('_fatal', 'alogarg', 'dum', 'dumc', 'ekd', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'fm', 'fn', 'hy', 'hygro', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'mact', 'na', 'nact', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'srcn', 'taumix_internal_pver_inv', 'tmpa', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix'))
            _result=_cond(_get(v['cldn'], (v['i'], v['k'])) > _r(0.01),_yes25,_no25,tuple(v[_n] for _n in ('_fatal', 'alogarg', 'dum', 'dumc', 'ekd', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'fm', 'fn', 'hy', 'hygro', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'mact', 'na', 'nact', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'srcn', 'taumix_internal_pver_inv', 'tmpa', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix')),xp)  # WRF:module_cam_mp_ndrop.F:511
            v.update(zip(('_fatal', 'alogarg', 'dum', 'dumc', 'ekd', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'fm', 'fn', 'hy', 'hygro', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'mact', 'na', 'nact', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'srcn', 'taumix_internal_pver_inv', 'tmpa', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix'),_result))
            return tuple(v[_n] for _n in ('_fatal', '_jump', 'alogarg', 'dum', 'dumc', 'ekd', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'fm', 'fn', 'hy', 'hygro', 'k', 'km1', 'kp1', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'mact', 'na', 'nact', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'srcn', 'taumix_internal_pver_inv', 'tmpa', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix'))
        _result=_loop(1,v['pver'],1,_body24,tuple(v[_n] for _n in ('_fatal', '_jump', 'alogarg', 'dum', 'dumc', 'ekd', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'fm', 'fn', 'hy', 'hygro', 'k', 'km1', 'kp1', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'mact', 'na', 'nact', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'srcn', 'taumix_internal_pver_inv', 'tmpa', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix')),xp,jump_index=1)
        v.update(zip(('_fatal', '_jump', 'alogarg', 'dum', 'dumc', 'ekd', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'fm', 'fn', 'hy', 'hygro', 'k', 'km1', 'kp1', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'mact', 'na', 'nact', 'naermod', 'nsource', 'phase', 'qcld', 'raercol', 'raercol_cw', 'srcn', 'taumix_internal_pver_inv', 'tmpa', 'va', 'vaerosol', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix'),_result))
        _put(v,'ntemp',None,v['nsav'])  # WRF:module_cam_mp_ndrop.F:659
        _put(v,'nsav',None,v['nnew'])  # WRF:module_cam_mp_ndrop.F:660
        _put(v,'nnew',None,v['ntemp'])  # WRF:module_cam_mp_ndrop.F:661
        _put(v,'dtmin',None,v['dtmicro'])  # WRF:module_cam_mp_ndrop.F:666
        _put(v,'ekk',((0)+1-(0),),_r(0.0))  # WRF:module_cam_mp_ndrop.F:667
        _put(v,'ekk',((v['pver'])+1-(0),),_r(0.0))  # WRF:module_cam_mp_ndrop.F:668
        _outer36=dict(v)
        def _body36(_k,_carry):
            v=dict(_outer36);v.update(zip(('_jump', 'ekk', 'k'),_carry));v['k']=_int(_k)
            _put(v,'ekk',((v['k'])+1-(0),),_get(v['ekd'], (v['k'],)) * _get(v['csbot'], (v['k'],)))  # WRF:module_cam_mp_ndrop.F:675
            return tuple(v[_n] for _n in ('_jump', 'ekk', 'k'))
        _result=_loop(1,v['pver'] - 1,1,_body36,tuple(v[_n] for _n in ('_jump', 'ekk', 'k')),xp,jump_index=0)
        v.update(zip(('_jump', 'ekk', 'k'),_result))
        _outer37=dict(v)
        def _body37(_k,_carry):
            v=dict(_outer37);v.update(zip(('_jump', 'dtmin', 'dtt', 'ekkm', 'ekkp', 'k', 'km1', 'tinv'),_carry));v['k']=_int(_k)
            _put(v,'km1',None,xp.maximum(v['k'] - 1, 1))  # WRF:module_cam_mp_ndrop.F:678
            _put(v,'ekkp',(v['k'],),_get(v['zn'], (v['k'],)) * _get(v['ekk'], (v['k'] + 1 - 0,)) * _get(v['zs'], (v['k'],)))  # WRF:module_cam_mp_ndrop.F:679
            _put(v,'ekkm',(v['k'],),_get(v['zn'], (v['k'],)) * _get(v['ekk'], (v['k'] - 1 + 1 - 0,)) * _get(v['zs'], (v['km1'],)))  # WRF:module_cam_mp_ndrop.F:680
            _put(v,'tinv',None,_get(v['ekkp'], (v['k'],)) + _get(v['ekkm'], (v['k'],)))  # WRF:module_cam_mp_ndrop.F:681
            _outer38=dict(v)
            def _yes38(_carry):
                v=dict(_outer38);v.update(zip(('tinv',),_carry))
                _put(v,'tinv',None,v['tinv'] + v['surfratemax'])  # WRF:module_cam_mp_ndrop.F:683
                return tuple(v[_n] for _n in ('tinv',))
            def _no38(_carry):
                v=dict(_outer38);v.update(zip(('tinv',),_carry))
                return tuple(v[_n] for _n in ('tinv',))
            _result=_cond(v['k'] == v['pver'],_yes38,_no38,tuple(v[_n] for _n in ('tinv',)),xp)  # WRF:module_cam_mp_ndrop.F:683
            v.update(zip(('tinv',),_result))
            _outer39=dict(v)
            def _yes39(_carry):
                v=dict(_outer39);v.update(zip(('tinv',),_carry))
                _put(v,'tinv',None,v['tinv'] + v['taumix_internal_pver_inv'])  # WRF:module_cam_mp_ndrop.F:693
                return tuple(v[_n] for _n in ('tinv',))
            def _no39(_carry):
                v=dict(_outer39);v.update(zip(('tinv',),_carry))
                return tuple(v[_n] for _n in ('tinv',))
            _result=_cond(v['k'] == v['pver'],_yes39,_no39,tuple(v[_n] for _n in ('tinv',)),xp)  # WRF:module_cam_mp_ndrop.F:693
            v.update(zip(('tinv',),_result))
            _outer40=dict(v)
            def _yes40(_carry):
                v=dict(_outer40);v.update(zip(('dtmin', 'dtt'),_carry))
                _put(v,'dtt',None,_r(1.0) / v['tinv'])  # WRF:module_cam_mp_ndrop.F:696
                _put(v,'dtmin',None,xp.minimum(v['dtmin'], v['dtt']))  # WRF:module_cam_mp_ndrop.F:697
                return tuple(v[_n] for _n in ('dtmin', 'dtt'))
            def _no40(_carry):
                v=dict(_outer40);v.update(zip(('dtmin', 'dtt'),_carry))
                return tuple(v[_n] for _n in ('dtmin', 'dtt'))
            _result=_cond(v['tinv'] > _r(1e-06),_yes40,_no40,tuple(v[_n] for _n in ('dtmin', 'dtt')),xp)  # WRF:module_cam_mp_ndrop.F:695
            v.update(zip(('dtmin', 'dtt'),_result))
            return tuple(v[_n] for _n in ('_jump', 'dtmin', 'dtt', 'ekkm', 'ekkp', 'k', 'km1', 'tinv'))
        _result=_loop(1,v['pver'],1,_body37,tuple(v[_n] for _n in ('_jump', 'dtmin', 'dtt', 'ekkm', 'ekkp', 'k', 'km1', 'tinv')),xp,jump_index=0)
        v.update(zip(('_jump', 'dtmin', 'dtt', 'ekkm', 'ekkp', 'k', 'km1', 'tinv'),_result))
        _put(v,'dtmix',None,_r(0.9) * v['dtmin'])  # WRF:module_cam_mp_ndrop.F:700
        _put(v,'nsubmix',None,v['dtmicro'] / v['dtmix'] + 1)  # WRF:module_cam_mp_ndrop.F:701
        _outer41=dict(v)
        def _yes41(_carry):
            v=dict(_outer41);v.update(zip(('nsubmix_bnd',),_carry))
            _put(v,'nsubmix_bnd',None,100)  # WRF:module_cam_mp_ndrop.F:703
            return tuple(v[_n] for _n in ('nsubmix_bnd',))
        def _no41(_carry):
            v=dict(_outer41);v.update(zip(('nsubmix_bnd',),_carry))
            _put(v,'nsubmix_bnd',None,v['nsubmix'])  # WRF:module_cam_mp_ndrop.F:705
            return tuple(v[_n] for _n in ('nsubmix_bnd',))
        _result=_cond(v['nsubmix'] > 100,_yes41,_no41,tuple(v[_n] for _n in ('nsubmix_bnd',)),xp)  # WRF:module_cam_mp_ndrop.F:702
        v.update(zip(('nsubmix_bnd',),_result))
        _put(v,'count_submix',(v['nsubmix_bnd'],),_get(v['count_submix'], (v['nsubmix_bnd'],)) + 1)  # WRF:module_cam_mp_ndrop.F:707
        _put(v,'dtmix',None,v['dtmicro'] / v['nsubmix'])  # WRF:module_cam_mp_ndrop.F:708
        _put(v,'fac_srflx',None,_r(-1.0) / (_get(v['zn'], (v['pver'],)) * v['nsubmix']))  # WRF:module_cam_mp_ndrop.F:709
        _outer42=dict(v)
        def _body42(_k,_carry):
            v=dict(_outer42);v.update(zip(('_jump', 'k', 'km1', 'kp1', 'overlapm', 'overlapp'),_carry));v['k']=_int(_k)
            _put(v,'kp1',None,xp.minimum(v['k'] + 1, v['pver']))  # WRF:module_cam_mp_ndrop.F:712
            _put(v,'km1',None,xp.maximum(v['k'] - 1, 1))  # WRF:module_cam_mp_ndrop.F:713
            _outer43=dict(v)
            def _yes43(_carry):
                v=dict(_outer43);v.update(zip(('overlapp',),_carry))
                _put(v,'overlapp',(v['k'],),xp.minimum(_get(v['cldn'], (v['i'], v['k'])) / _get(v['cldn'], (v['i'], v['kp1'])), _d(1.0)))  # WRF:module_cam_mp_ndrop.F:716
                return tuple(v[_n] for _n in ('overlapp',))
            def _no43(_carry):
                v=dict(_outer43);v.update(zip(('overlapp',),_carry))
                _put(v,'overlapp',(v['k'],),_r(1.0))  # WRF:module_cam_mp_ndrop.F:718
                return tuple(v[_n] for _n in ('overlapp',))
            _result=_cond(_get(v['cldn'], (v['i'], v['kp1'])) > _d(1e-10),_yes43,_no43,tuple(v[_n] for _n in ('overlapp',)),xp)  # WRF:module_cam_mp_ndrop.F:715
            v.update(zip(('overlapp',),_result))
            _outer44=dict(v)
            def _yes44(_carry):
                v=dict(_outer44);v.update(zip(('overlapm',),_carry))
                _put(v,'overlapm',(v['k'],),xp.minimum(_get(v['cldn'], (v['i'], v['k'])) / _get(v['cldn'], (v['i'], v['km1'])), _d(1.0)))  # WRF:module_cam_mp_ndrop.F:721
                return tuple(v[_n] for _n in ('overlapm',))
            def _no44(_carry):
                v=dict(_outer44);v.update(zip(('overlapm',),_carry))
                _put(v,'overlapm',(v['k'],),_r(1.0))  # WRF:module_cam_mp_ndrop.F:723
                return tuple(v[_n] for _n in ('overlapm',))
            _result=_cond(_get(v['cldn'], (v['i'], v['km1'])) > _d(1e-10),_yes44,_no44,tuple(v[_n] for _n in ('overlapm',)),xp)  # WRF:module_cam_mp_ndrop.F:720
            v.update(zip(('overlapm',),_result))
            return tuple(v[_n] for _n in ('_jump', 'k', 'km1', 'kp1', 'overlapm', 'overlapp'))
        _result=_loop(1,v['pver'],1,_body42,tuple(v[_n] for _n in ('_jump', 'k', 'km1', 'kp1', 'overlapm', 'overlapp')),xp,jump_index=0)
        v.update(zip(('_jump', 'k', 'km1', 'kp1', 'overlapm', 'overlapp'),_result))
        _outer45=dict(v)
        def _body45(_k,_carry):
            v=dict(_outer45);v.update(zip(('_jump', 'k', 'm', 'mact', 'nact'),_carry));v['k']=_int(_k)
            _outer46=dict(v)
            def _body46(_k,_carry):
                v=dict(_outer46);v.update(zip(('_jump', 'm', 'mact', 'nact'),_carry));v['m']=_int(_k)
                _put(v,'nact',(v['k'],v['m'],),xp.minimum(_get(v['nact'], (v['k'], v['m'])), _get(v['ekkp'], (v['k'],))))  # WRF:module_cam_mp_ndrop.F:736
                _put(v,'mact',(v['k'],v['m'],),xp.minimum(_get(v['mact'], (v['k'], v['m'])), _get(v['ekkp'], (v['k'],))))  # WRF:module_cam_mp_ndrop.F:737
                return tuple(v[_n] for _n in ('_jump', 'm', 'mact', 'nact'))
            _result=_loop(1,v['ntot_amode'],1,_body46,tuple(v[_n] for _n in ('_jump', 'm', 'mact', 'nact')),xp,jump_index=0)
            v.update(zip(('_jump', 'm', 'mact', 'nact'),_result))
            return tuple(v[_n] for _n in ('_jump', 'k', 'm', 'mact', 'nact'))
        _result=_loop(1,v['pver'] - 1,1,_body45,tuple(v[_n] for _n in ('_jump', 'k', 'm', 'mact', 'nact')),xp,jump_index=0)
        v.update(zip(('_jump', 'k', 'm', 'mact', 'nact'),_result))
        _outer47=dict(v)
        def _body47(_k,_carry):
            v=dict(_outer47);v.update(zip(('_jump', 'flxconv', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'n', 'nnew', 'nsav', 'ntemp', 'qcld', 'qncld', 'raercol', 'raercol_cw', 'source', 'srcn', 'tmpa'),_carry));v['n']=_int(_k)
            _put(v,'qncld',(_slice(None,None,None),),_get(v['qcld'], (_slice(1, None, None),)))  # WRF:module_cam_mp_ndrop.F:744
            _put(v,'ntemp',None,v['nsav'])  # WRF:module_cam_mp_ndrop.F:746
            _put(v,'nsav',None,v['nnew'])  # WRF:module_cam_mp_ndrop.F:747
            _put(v,'nnew',None,v['ntemp'])  # WRF:module_cam_mp_ndrop.F:748
            _put(v,'srcn',(_slice(None,None,None),),_r(0.0))  # WRF:module_cam_mp_ndrop.F:749
            _outer48=dict(v)
            def _body48(_k,_carry):
                v=dict(_outer48);v.update(zip(('_jump', 'lnum', 'lnumcw', 'm', 'srcn', 'tmpa'),_carry));v['m']=_int(_k)
                _put(v,'lnum',None,_get(v['numptr_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:751
                _put(v,'lnumcw',None,_get(v['numptrcw_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:752
                _put(v,'srcn',(_slice(1,v['pver'] - 1,None),),_get(v['srcn'], (_slice(1, v['pver'] - 1, None),)) + _get(v['nact'], (_slice(1, v['pver'] - 1, None), v['m'])) * _get(v['raercol'], (_slice(2, v['pver'], None), v['lnum'], v['nsav'])))  # WRF:module_cam_mp_ndrop.F:756
                _put(v,'tmpa',None,_get(v['raercol'], (v['pver'], v['lnum'], v['nsav'])) * _get(v['nact'], (v['pver'], v['m'])) + _get(v['raercol'], (v['pver'], v['lnumcw'], v['nsav'])) * (_get(v['nact'], (v['pver'], v['m'])) - v['taumix_internal_pver_inv']))  # WRF:module_cam_mp_ndrop.F:759
                _put(v,'srcn',(v['pver'],),_get(v['srcn'], (v['pver'],)) + xp.maximum(_d(0.0), v['tmpa']))  # WRF:module_cam_mp_ndrop.F:761
                return tuple(v[_n] for _n in ('_jump', 'lnum', 'lnumcw', 'm', 'srcn', 'tmpa'))
            _result=_loop(1,v['ntot_amode'],1,_body48,tuple(v[_n] for _n in ('_jump', 'lnum', 'lnumcw', 'm', 'srcn', 'tmpa')),xp,jump_index=0)
            v.update(zip(('_jump', 'lnum', 'lnumcw', 'm', 'srcn', 'tmpa'),_result))
            _call=explmix(v['qcld'],v['srcn'],v['ekkp'],v['ekkm'],v['overlapp'],v['overlapm'],v['qncld'],_get(v['surfrate'], (v['ixndrop'],)),v['zero'],v['pver'],v['dtmix'],False,_d(0.0),_globals=_globals,xp=xp)
            _put(v,'qcld',None,_call[0])  # WRF:module_cam_mp_ndrop.F:763
            _outer49=dict(v)
            def _body49(_k,_carry):
                v=dict(_outer49);v.update(zip(('_jump', 'flxconv', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw', 'source', 'tmpa'),_carry));v['m']=_int(_k)
                _put(v,'lnum',None,_get(v['numptr_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:774
                _put(v,'lnumcw',None,_get(v['numptrcw_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:775
                _outer50=dict(v)
                def _yes50(_carry):
                    v=dict(_outer50);v.update(zip(('flxconv', 'raercol', 'raercol_cw', 'source', 'tmpa'),_carry))
                    _put(v,'source',(_slice(1,v['pver'] - 1,None),),_get(v['nact'], (_slice(1, v['pver'] - 1, None), v['m'])) * _get(v['raercol'], (_slice(2, v['pver'], None), v['lnum'], v['nsav'])))  # WRF:module_cam_mp_ndrop.F:779
                    _put(v,'tmpa',None,_get(v['raercol'], (v['pver'], v['lnum'], v['nsav'])) * _get(v['nact'], (v['pver'], v['m'])) + _get(v['raercol'], (v['pver'], v['lnumcw'], v['nsav'])) * (_get(v['nact'], (v['pver'], v['m'])) - v['taumix_internal_pver_inv']))  # WRF:module_cam_mp_ndrop.F:782
                    _put(v,'source',(v['pver'],),xp.maximum(_d(0.0), v['tmpa']))  # WRF:module_cam_mp_ndrop.F:784
                    _put(v,'flxconv',None,_r(0.0))  # WRF:module_cam_mp_ndrop.F:786
                    _call=explmix(_get(v['raercol_cw'], (_slice(1, v['pver'], None), v['lnumcw'], v['nnew'])),v['source'],v['ekkp'],v['ekkm'],v['overlapp'],v['overlapm'],_get(v['raercol_cw'], (_slice(1, v['pver'], None), v['lnumcw'], v['nsav'])),_get(v['surfrate_cw'], (v['lnumcw'],)),v['zero'],v['pver'],v['dtmix'],False,_d(0.0),_globals=_globals,xp=xp)
                    _put(v,'raercol_cw',(_slice(1,v['pver'],None),v['lnumcw'],v['nnew'],),_call[0])  # WRF:module_cam_mp_ndrop.F:787
                    _call=explmix(_get(v['raercol'], (_slice(1, v['pver'], None), v['lnum'], v['nnew'])),v['source'],v['ekkp'],v['ekkm'],v['overlapp'],v['overlapm'],_get(v['raercol'], (_slice(1, v['pver'], None), v['lnum'], v['nsav'])),_get(v['surfrate'], (v['lnum'],)),v['flxconv'],v['pver'],v['dtmix'],True,_get(v['raercol_cw'], (_slice(1, v['pver'], None), v['lnumcw'], v['nsav'])),_globals=_globals,xp=xp)
                    _put(v,'raercol',(_slice(1,v['pver'],None),v['lnum'],v['nnew'],),_call[0])  # WRF:module_cam_mp_ndrop.F:790
                    return tuple(v[_n] for _n in ('flxconv', 'raercol', 'raercol_cw', 'source', 'tmpa'))
                def _no50(_carry):
                    v=dict(_outer50);v.update(zip(('flxconv', 'raercol', 'raercol_cw', 'source', 'tmpa'),_carry))
                    return tuple(v[_n] for _n in ('flxconv', 'raercol', 'raercol_cw', 'source', 'tmpa'))
                _result=_cond(v['lnum'] > 0,_yes50,_no50,tuple(v[_n] for _n in ('flxconv', 'raercol', 'raercol_cw', 'source', 'tmpa')),xp)  # WRF:module_cam_mp_ndrop.F:776
                v.update(zip(('flxconv', 'raercol', 'raercol_cw', 'source', 'tmpa'),_result))
                _outer51=dict(v)
                def _body51(_k,_carry):
                    v=dict(_outer51);v.update(zip(('_jump', 'flxconv', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw', 'source', 'tmpa'),_carry));v['l']=_int(_k)
                    _put(v,'lmass',None,_get(v['lmassptr_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:796
                    _put(v,'lmasscw',None,_get(v['lmassptrcw_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:797
                    _put(v,'source',(_slice(1,v['pver'] - 1,None),),_get(v['mact'], (_slice(1, v['pver'] - 1, None), v['m'])) * _get(v['raercol'], (_slice(2, v['pver'], None), v['lmass'], v['nsav'])))  # WRF:module_cam_mp_ndrop.F:800
                    _put(v,'tmpa',None,_get(v['raercol'], (v['pver'], v['lmass'], v['nsav'])) * _get(v['mact'], (v['pver'], v['m'])) + _get(v['raercol'], (v['pver'], v['lmasscw'], v['nsav'])) * (_get(v['mact'], (v['pver'], v['m'])) - v['taumix_internal_pver_inv']))  # WRF:module_cam_mp_ndrop.F:803
                    _put(v,'source',(v['pver'],),xp.maximum(_d(0.0), v['tmpa']))  # WRF:module_cam_mp_ndrop.F:805
                    _put(v,'flxconv',None,_r(0.0))  # WRF:module_cam_mp_ndrop.F:807
                    _call=explmix(_get(v['raercol_cw'], (_slice(1, v['pver'], None), v['lmasscw'], v['nnew'])),v['source'],v['ekkp'],v['ekkm'],v['overlapp'],v['overlapm'],_get(v['raercol_cw'], (_slice(1, v['pver'], None), v['lmasscw'], v['nsav'])),_get(v['surfrate_cw'], (v['lmasscw'],)),v['zero'],v['pver'],v['dtmix'],False,_d(0.0),_globals=_globals,xp=xp)
                    _put(v,'raercol_cw',(_slice(1,v['pver'],None),v['lmasscw'],v['nnew'],),_call[0])  # WRF:module_cam_mp_ndrop.F:809
                    _call=explmix(_get(v['raercol'], (_slice(1, v['pver'], None), v['lmass'], v['nnew'])),v['source'],v['ekkp'],v['ekkm'],v['overlapp'],v['overlapm'],_get(v['raercol'], (_slice(1, v['pver'], None), v['lmass'], v['nsav'])),_get(v['surfrate'], (v['lmass'],)),v['flxconv'],v['pver'],v['dtmix'],True,_get(v['raercol_cw'], (_slice(1, v['pver'], None), v['lmasscw'], v['nsav'])),_globals=_globals,xp=xp)
                    _put(v,'raercol',(_slice(1,v['pver'],None),v['lmass'],v['nnew'],),_call[0])  # WRF:module_cam_mp_ndrop.F:812
                    return tuple(v[_n] for _n in ('_jump', 'flxconv', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw', 'source', 'tmpa'))
                _result=_loop(1,_get(v['nspec_amode'], (v['m'],)),1,_body51,tuple(v[_n] for _n in ('_jump', 'flxconv', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw', 'source', 'tmpa')),xp,jump_index=0)
                v.update(zip(('_jump', 'flxconv', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw', 'source', 'tmpa'),_result))
                return tuple(v[_n] for _n in ('_jump', 'flxconv', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw', 'source', 'tmpa'))
            _result=_loop(1,v['ntot_amode'],1,_body49,tuple(v[_n] for _n in ('_jump', 'flxconv', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw', 'source', 'tmpa')),xp,jump_index=0)
            v.update(zip(('_jump', 'flxconv', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw', 'source', 'tmpa'),_result))
            return tuple(v[_n] for _n in ('_jump', 'flxconv', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'n', 'nnew', 'nsav', 'ntemp', 'qcld', 'qncld', 'raercol', 'raercol_cw', 'source', 'srcn', 'tmpa'))
        _result=_loop(1,v['nsubmix'],1,_body47,tuple(v[_n] for _n in ('_jump', 'flxconv', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'n', 'nnew', 'nsav', 'ntemp', 'qcld', 'qncld', 'raercol', 'raercol_cw', 'source', 'srcn', 'tmpa')),xp,jump_index=0)
        v.update(zip(('_jump', 'flxconv', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'n', 'nnew', 'nsav', 'ntemp', 'qcld', 'qncld', 'raercol', 'raercol_cw', 'source', 'srcn', 'tmpa'),_result))
        _outer52=dict(v)
        def _body52(_k,_carry):
            v=dict(_outer52);v.update(zip(('_jump', 'k', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'qcld', 'raercol', 'raercol_cw'),_carry));v['k']=_int(_k)
            _outer53=dict(v)
            def _yes53(_carry):
                v=dict(_outer53);v.update(zip(('l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'qcld', 'raercol', 'raercol_cw'),_carry))
                _put(v,'qcld',(v['k'],),_r(0.0))  # WRF:module_cam_mp_ndrop.F:831
                _outer54=dict(v)
                def _body54(_k,_carry):
                    v=dict(_outer54);v.update(zip(('_jump', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw'),_carry));v['m']=_int(_k)
                    _put(v,'lnum',None,_get(v['numptr_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:834
                    _put(v,'lnumcw',None,_get(v['numptrcw_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:835
                    _outer55=dict(v)
                    def _yes55(_carry):
                        v=dict(_outer55);v.update(zip(('raercol', 'raercol_cw'),_carry))
                        _put(v,'raercol',(v['k'],v['lnum'],v['nnew'],),_get(v['raercol'], (v['k'], v['lnum'], v['nnew'])) + _get(v['raercol_cw'], (v['k'], v['lnumcw'], v['nnew'])))  # WRF:module_cam_mp_ndrop.F:837
                        _put(v,'raercol_cw',(v['k'],v['lnumcw'],v['nnew'],),_r(0.0))  # WRF:module_cam_mp_ndrop.F:838
                        return tuple(v[_n] for _n in ('raercol', 'raercol_cw'))
                    def _no55(_carry):
                        v=dict(_outer55);v.update(zip(('raercol', 'raercol_cw'),_carry))
                        return tuple(v[_n] for _n in ('raercol', 'raercol_cw'))
                    _result=_cond(v['lnum'] > 0,_yes55,_no55,tuple(v[_n] for _n in ('raercol', 'raercol_cw')),xp)  # WRF:module_cam_mp_ndrop.F:836
                    v.update(zip(('raercol', 'raercol_cw'),_result))
                    _outer56=dict(v)
                    def _body56(_k,_carry):
                        v=dict(_outer56);v.update(zip(('_jump', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw'),_carry));v['l']=_int(_k)
                        _put(v,'lmass',None,_get(v['lmassptr_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:842
                        _put(v,'lmasscw',None,_get(v['lmassptrcw_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:843
                        _put(v,'raercol',(v['k'],v['lmass'],v['nnew'],),_get(v['raercol'], (v['k'], v['lmass'], v['nnew'])) + _get(v['raercol_cw'], (v['k'], v['lmasscw'], v['nnew'])))  # WRF:module_cam_mp_ndrop.F:846
                        _put(v,'raercol_cw',(v['k'],v['lmasscw'],v['nnew'],),_r(0.0))  # WRF:module_cam_mp_ndrop.F:847
                        return tuple(v[_n] for _n in ('_jump', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw'))
                    _result=_loop(1,_get(v['nspec_amode'], (v['m'],)),1,_body56,tuple(v[_n] for _n in ('_jump', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw')),xp,jump_index=0)
                    v.update(zip(('_jump', 'l', 'lmass', 'lmasscw', 'raercol', 'raercol_cw'),_result))
                    return tuple(v[_n] for _n in ('_jump', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw'))
                _result=_loop(1,v['ntot_amode'],1,_body54,tuple(v[_n] for _n in ('_jump', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw')),xp,jump_index=0)
                v.update(zip(('_jump', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'raercol', 'raercol_cw'),_result))
                return tuple(v[_n] for _n in ('l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'qcld', 'raercol', 'raercol_cw'))
            def _no53(_carry):
                v=dict(_outer53);v.update(zip(('l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'qcld', 'raercol', 'raercol_cw'),_carry))
                return tuple(v[_n] for _n in ('l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'qcld', 'raercol', 'raercol_cw'))
            _result=_cond(_get(v['cldn'], (v['i'], v['k'])) == _r(0.0),_yes53,_no53,tuple(v[_n] for _n in ('l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'qcld', 'raercol', 'raercol_cw')),xp)  # WRF:module_cam_mp_ndrop.F:829
            v.update(zip(('l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'qcld', 'raercol', 'raercol_cw'),_result))
            return tuple(v[_n] for _n in ('_jump', 'k', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'qcld', 'raercol', 'raercol_cw'))
        _result=_loop(1,v['pver'],1,_body52,tuple(v[_n] for _n in ('_jump', 'k', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'qcld', 'raercol', 'raercol_cw')),xp,jump_index=0)
        v.update(zip(('_jump', 'k', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'qcld', 'raercol', 'raercol_cw'),_result))
        _put(v,'ndropcol',(v['i'],),_r(0.0))  # WRF:module_cam_mp_ndrop.F:855
        _outer57=dict(v)
        def _body57(_k,_carry):
            v=dict(_outer57);v.update(zip(('_jump', 'k', 'ndropcol', 'ndropmix', 'tendnd'),_carry));v['k']=_int(_k)
            _put(v,'ndropmix',(v['i'],v['k'],),(_get(v['qcld'], (v['k'],)) - _get(v['ncldwtr'], (v['i'], v['k']))) * v['dtinv'] - _get(v['nsource'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:857
            _put(v,'tendnd',(v['i'],v['k'],),(xp.maximum(_get(v['qcld'], (v['k'],)), _d(1e-06)) - _get(v['ncldwtr'], (v['i'], v['k']))) * v['dtinv'])  # WRF:module_cam_mp_ndrop.F:858
            _put(v,'ndropcol',(v['i'],),_get(v['ndropcol'], (v['i'],)) + _get(v['ncldwtr'], (v['i'], v['k'])) * _get(v['pdel'], (v['i'], v['k'])))  # WRF:module_cam_mp_ndrop.F:859
            return tuple(v[_n] for _n in ('_jump', 'k', 'ndropcol', 'ndropmix', 'tendnd'))
        _result=_loop(1,v['pver'],1,_body57,tuple(v[_n] for _n in ('_jump', 'k', 'ndropcol', 'ndropmix', 'tendnd')),xp,jump_index=0)
        v.update(zip(('_jump', 'k', 'ndropcol', 'ndropmix', 'tendnd'),_result))
        _put(v,'ndropcol',(v['i'],),_get(v['ndropcol'], (v['i'],)) / v['gravit'])  # WRF:module_cam_mp_ndrop.F:861
        _outer58=dict(v)
        def _body58(_k,_carry):
            v=dict(_outer58);v.update(zip(('_jump', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'qqcw_inout', 'qqcwtend', 'raertend'),_carry));v['m']=_int(_k)
            _put(v,'lnum',None,_get(v['numptr_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:866
            _put(v,'lnumcw',None,_get(v['numptrcw_amode'], (v['m'],)))  # WRF:module_cam_mp_ndrop.F:867
            _outer59=dict(v)
            def _yes59(_carry):
                v=dict(_outer59);v.update(zip(('qqcw_inout', 'qqcwtend', 'raertend'),_carry))
                _put(v,'qqcwtend',(v['i'],_slice(None,None,None),v['lnumcw'],),(_get(v['raercol_cw'], (_slice(1, None, None), v['lnumcw'], v['nnew'])) - _get(v['qqcw_inout'], (v['i'], _slice(1, None, None), v['lnumcw']))) * v['dtinv'])  # WRF:module_cam_mp_ndrop.F:869
                _put(v,'qqcw_inout',(v['i'],_slice(None,None,None),v['lnumcw'],),_get(v['raercol_cw'], (_slice(1, None, None), v['lnumcw'], v['nnew'])))  # WRF:module_cam_mp_ndrop.F:870
                _put(v,'raertend',(v['i'],_slice(None,None,None),v['lnum'],),(_get(v['raercol'], (_slice(1, None, None), v['lnum'], v['nnew'])) - _get(v['raer'], (v['i'], _slice(1, None, None), v['lnum']))) * v['dtinv'])  # WRF:module_cam_mp_ndrop.F:871
                return tuple(v[_n] for _n in ('qqcw_inout', 'qqcwtend', 'raertend'))
            def _no59(_carry):
                v=dict(_outer59);v.update(zip(('qqcw_inout', 'qqcwtend', 'raertend'),_carry))
                return tuple(v[_n] for _n in ('qqcw_inout', 'qqcwtend', 'raertend'))
            _result=_cond(v['lnum'] > 0,_yes59,_no59,tuple(v[_n] for _n in ('qqcw_inout', 'qqcwtend', 'raertend')),xp)  # WRF:module_cam_mp_ndrop.F:868
            v.update(zip(('qqcw_inout', 'qqcwtend', 'raertend'),_result))
            _outer60=dict(v)
            def _body60(_k,_carry):
                v=dict(_outer60);v.update(zip(('_jump', 'l', 'lmass', 'lmasscw', 'qqcw_inout', 'qqcwtend', 'raertend'),_carry));v['l']=_int(_k)
                _put(v,'lmass',None,_get(v['lmassptr_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:874
                _put(v,'lmasscw',None,_get(v['lmassptrcw_amode'], (v['l'], v['m'])))  # WRF:module_cam_mp_ndrop.F:875
                _put(v,'qqcwtend',(v['i'],_slice(None,None,None),v['lmasscw'],),(_get(v['raercol_cw'], (_slice(1, None, None), v['lmasscw'], v['nnew'])) - _get(v['qqcw_inout'], (v['i'], _slice(1, None, None), v['lmasscw']))) * v['dtinv'])  # WRF:module_cam_mp_ndrop.F:876
                _put(v,'qqcw_inout',(v['i'],_slice(None,None,None),v['lmasscw'],),_get(v['raercol_cw'], (_slice(1, None, None), v['lmasscw'], v['nnew'])))  # WRF:module_cam_mp_ndrop.F:877
                _put(v,'raertend',(v['i'],_slice(None,None,None),v['lmass'],),(_get(v['raercol'], (_slice(1, None, None), v['lmass'], v['nnew'])) - _get(v['raer'], (v['i'], _slice(1, None, None), v['lmass']))) * v['dtinv'])  # WRF:module_cam_mp_ndrop.F:878
                return tuple(v[_n] for _n in ('_jump', 'l', 'lmass', 'lmasscw', 'qqcw_inout', 'qqcwtend', 'raertend'))
            _result=_loop(1,_get(v['nspec_amode'], (v['m'],)),1,_body60,tuple(v[_n] for _n in ('_jump', 'l', 'lmass', 'lmasscw', 'qqcw_inout', 'qqcwtend', 'raertend')),xp,jump_index=0)
            v.update(zip(('_jump', 'l', 'lmass', 'lmasscw', 'qqcw_inout', 'qqcwtend', 'raertend'),_result))
            return tuple(v[_n] for _n in ('_jump', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'qqcw_inout', 'qqcwtend', 'raertend'))
        _result=_loop(1,v['ntot_amode'],1,_body58,tuple(v[_n] for _n in ('_jump', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'qqcw_inout', 'qqcwtend', 'raertend')),xp,jump_index=0)
        v.update(zip(('_jump', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'qqcw_inout', 'qqcwtend', 'raertend'),_result))
        return tuple(v[_n] for _n in ('_fatal', '_jump', 'alogarg', 'cldn_tmp', 'cldo_tmp', 'count_submix', 'cs', 'csbot', 'csbot_cscen', 'dact', 'dtmin', 'dtmix', 'dtt', 'dum', 'dumc', 'dz', 'ekd', 'ekk', 'ekkm', 'ekkp', 'fac_srflx', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'flxconv', 'fm', 'fn', 'hy', 'hygro', 'i', 'k', 'km1', 'kp1', 'kvhmax', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'mact', 'n', 'na', 'nact', 'naermod', 'ndropcol', 'ndropmix', 'nnew', 'nsav', 'nsource', 'nsubmix', 'nsubmix_bnd', 'ntemp', 'overlapm', 'overlapp', 'phase', 'qcld', 'qncld', 'qqcw_inout', 'qqcwtend', 'raercol', 'raercol_cw', 'raertend', 'source', 'srcn', 'surfrate', 'surfrate_cw', 'surfratemax', 'tau_cld_regenerate', 'taumix_internal_pver_inv', 'tendnd', 'tinv', 'tmpa', 'va', 'vaerosol', 'w', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix', 'wtke', 'wtke_cen', 'zn', 'zs'))
    _result=_loop(1,v['ncol'],1,_body5,tuple(v[_n] for _n in ('_fatal', '_jump', 'alogarg', 'cldn_tmp', 'cldo_tmp', 'count_submix', 'cs', 'csbot', 'csbot_cscen', 'dact', 'dtmin', 'dtmix', 'dtt', 'dum', 'dumc', 'dz', 'ekd', 'ekk', 'ekkm', 'ekkp', 'fac_srflx', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'flxconv', 'fm', 'fn', 'hy', 'hygro', 'i', 'k', 'km1', 'kp1', 'kvhmax', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'mact', 'n', 'na', 'nact', 'naermod', 'ndropcol', 'ndropmix', 'nnew', 'nsav', 'nsource', 'nsubmix', 'nsubmix_bnd', 'ntemp', 'overlapm', 'overlapp', 'phase', 'qcld', 'qncld', 'qqcw_inout', 'qqcwtend', 'raercol', 'raercol_cw', 'raertend', 'source', 'srcn', 'surfrate', 'surfrate_cw', 'surfratemax', 'tau_cld_regenerate', 'taumix_internal_pver_inv', 'tendnd', 'tinv', 'tmpa', 'va', 'vaerosol', 'w', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix', 'wtke', 'wtke_cen', 'zn', 'zs')),xp,jump_index=1)
    v.update(zip(('_fatal', '_jump', 'alogarg', 'cldn_tmp', 'cldo_tmp', 'count_submix', 'cs', 'csbot', 'csbot_cscen', 'dact', 'dtmin', 'dtmix', 'dtt', 'dum', 'dumc', 'dz', 'ekd', 'ekk', 'ekkm', 'ekkp', 'fac_srflx', 'flux_fullact', 'fluxm', 'fluxn', 'fluxntot', 'flxconv', 'fm', 'fn', 'hy', 'hygro', 'i', 'k', 'km1', 'kp1', 'kvhmax', 'l', 'lmass', 'lmasscw', 'lnum', 'lnumcw', 'm', 'mact', 'n', 'na', 'nact', 'naermod', 'ndropcol', 'ndropmix', 'nnew', 'nsav', 'nsource', 'nsubmix', 'nsubmix_bnd', 'ntemp', 'overlapm', 'overlapp', 'phase', 'qcld', 'qncld', 'qqcw_inout', 'qqcwtend', 'raercol', 'raercol_cw', 'raertend', 'source', 'srcn', 'surfrate', 'surfrate_cw', 'surfratemax', 'tau_cld_regenerate', 'taumix_internal_pver_inv', 'tendnd', 'tinv', 'tmpa', 'va', 'vaerosol', 'w', 'wbar', 'wdiab', 'wmax', 'wmin', 'wmix', 'wtke', 'wtke_cen', 'zn', 'zs'),_result))
    return (v['ncldwtr'],v['tendnd'],v['raertend'],v['qqcw_inout'],v['_fatal'],)

def microp_aero_ts(lchnk,ncol,deltatin,t,ttend,qn,qc,qi,nc,ni,p,pdel,cldn,liqcldf,icecldf,cldo,pint,rpdel,zm,omega,qaer,cflx,qaertend,dgnumwet,dgnum,kkvh,tke,turbtype,smaw,wsub,wsubi,naai,naai_hom,npccn,rndst,nacon,qqcw,*,_globals=None,xp=jnp):
    v=dict(_globals or {})
    if '__modules' in v:v.update(v['__modules'].get('microp_aero',{}))
    v['lchnk']=lchnk
    v['ncol']=ncol
    v['deltatin']=deltatin
    v['t']=t
    v['ttend']=ttend
    v['qn']=qn
    v['qc']=qc
    v['qi']=qi
    v['nc']=nc
    v['ni']=ni
    v['p']=p
    v['pdel']=pdel
    v['cldn']=cldn
    v['liqcldf']=liqcldf
    v['icecldf']=icecldf
    v['cldo']=cldo
    v['pint']=pint
    v['rpdel']=rpdel
    v['zm']=zm
    v['omega']=omega
    v['qaer']=qaer
    v['cflx']=cflx
    v['qaertend']=qaertend
    v['dgnumwet']=dgnumwet
    v['dgnum']=dgnum
    v['kkvh']=kkvh
    v['tke']=tke
    v['turbtype']=turbtype
    v['smaw']=smaw
    v['wsub']=wsub
    v['wsubi']=wsubi
    v['naai']=naai
    v['naai_hom']=naai_hom
    v['npccn']=npccn
    v['rndst']=rndst
    v['nacon']=nacon
    v['qqcw']=qqcw
    v['_jump']=_int(0)
    v['_fatal']=_int(0)
    v['tkem']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['smm']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['relhum']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['cldm']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['icldm']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['lcldm']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['nfice']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['dumfice']=_zeros((),'f64',xp)
    v['qcld']=_zeros((),'f64',xp)
    v['lcldn']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['lcldo']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['nctend_mixnuc']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['deltat']=_zeros((),'f64',xp)
    v['nihf2']=_zeros((),'f64',xp)
    v['niimm2']=_zeros((),'f64',xp)
    v['nidep2']=_zeros((),'f64',xp)
    v['nimey2']=_zeros((),'f64',xp)
    v['dum2']=_zeros((),'f64',xp)
    v['ftrue']=_zeros((),'i32',xp)
    v['dum']=_zeros((),'f64',xp)
    v['q']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['ncloc']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['rho']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['mincld']=_zeros((),'f64',xp)
    v['dmc']=_zeros((),'f64',xp)
    v['ssmc']=_zeros((),'f64',xp)
    v['qs']=_zeros((v['pcols'],),'f64',xp)
    v['es']=_zeros((v['pcols'],),'f64',xp)
    v['esl']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['esi']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['gammas']=_zeros((v['pcols'],),'f64',xp)
    v['naer2']=_zeros((v['pcols'],v['pver'],v['naer_all'],),'f64',xp)
    v['nimey']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['nihf']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['nidep']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['niimm']=_zeros((v['pcols'],v['pver'],),'f64',xp)
    v['i']=_zeros((),'i32',xp)
    v['k']=_zeros((),'i32',xp)
    v['l']=_zeros((),'i32',xp)
    _put(v,'nimey',(_slice(1,v['ncol'],None),_slice(1,v['pver'],None),),_d(0.0))  # WRF:module_cam_mp_microp_aero.F:538
    _put(v,'nihf',(_slice(1,v['ncol'],None),_slice(1,v['pver'],None),),_d(0.0))  # WRF:module_cam_mp_microp_aero.F:539
    _put(v,'nidep',(_slice(1,v['ncol'],None),_slice(1,v['pver'],None),),_d(0.0))  # WRF:module_cam_mp_microp_aero.F:540
    _put(v,'niimm',(_slice(1,v['ncol'],None),_slice(1,v['pver'],None),),_d(0.0))  # WRF:module_cam_mp_microp_aero.F:541
    _put(v,'naai',(_slice(1,v['ncol'],None),_slice(1,v['pver'],None),),_d(0.0))  # WRF:module_cam_mp_microp_aero.F:544
    _put(v,'naai_hom',(_slice(1,v['ncol'],None),_slice(1,v['pver'],None),),_d(0.0))  # WRF:module_cam_mp_microp_aero.F:545
    _put(v,'npccn',(_slice(1,v['ncol'],None),_slice(1,v['pver'],None),),_d(0.0))  # WRF:module_cam_mp_microp_aero.F:546
    _put(v,'nacon',(_slice(1,v['ncol'],None),_slice(1,v['pver'],None),_slice(None,None,None),),_d(0.0))  # WRF:module_cam_mp_microp_aero.F:547
    _put(v,'rndst',(_slice(1,v['ncol'],None),_slice(1,v['pver'],None),1,),v['rn_dst1'])  # WRF:module_cam_mp_microp_aero.F:550
    _put(v,'rndst',(_slice(1,v['ncol'],None),_slice(1,v['pver'],None),2,),v['rn_dst2'])  # WRF:module_cam_mp_microp_aero.F:551
    _put(v,'rndst',(_slice(1,v['ncol'],None),_slice(1,v['pver'],None),3,),v['rn_dst3'])  # WRF:module_cam_mp_microp_aero.F:552
    _put(v,'rndst',(_slice(1,v['ncol'],None),_slice(1,v['pver'],None),4,),v['rn_dst4'])  # WRF:module_cam_mp_microp_aero.F:553
    _put(v,'tkem',(_slice(1,v['ncol'],None),_slice(1,v['pver'],None),),_d(0.0))  # WRF:module_cam_mp_microp_aero.F:556
    _put(v,'smm',(_slice(1,v['ncol'],None),_slice(1,v['pver'],None),),_d(0.0))  # WRF:module_cam_mp_microp_aero.F:557
    _put(v,'deltat',None,v['deltatin'])  # WRF:module_cam_mp_microp_aero.F:565
    _put(v,'q',(_slice(1,v['ncol'],None),_slice(1,v['pver'],None),),_get(v['qn'], (_slice(1, v['ncol'], None), _slice(1, v['pver'], None))))  # WRF:module_cam_mp_microp_aero.F:568
    _put(v,'ncloc',(_slice(1,v['ncol'],None),_slice(1,v['pver'],None),),_get(v['nc'], (_slice(1, v['ncol'], None), _slice(1, v['pver'], None))))  # WRF:module_cam_mp_microp_aero.F:569
    _put(v,'mincld',None,_d(0.0001))  # WRF:module_cam_mp_microp_aero.F:572
    _outer1=dict(v)
    def _body1(_k,_carry):
        v=dict(_outer1);v.update(zip(('_jump', 'i', 'k', 'rho'),_carry));v['k']=_int(_k)
        _outer2=dict(v)
        def _body2(_k,_carry):
            v=dict(_outer2);v.update(zip(('_jump', 'i', 'rho'),_carry));v['i']=_int(_k)
            _put(v,'rho',(v['i'],v['k'],),_get(v['p'], (v['i'], v['k'])) / (v['r'] * _get(v['t'], (v['i'], v['k']))))  # WRF:module_cam_mp_microp_aero.F:578
            return tuple(v[_n] for _n in ('_jump', 'i', 'rho'))
        _result=_loop(1,v['ncol'],1,_body2,tuple(v[_n] for _n in ('_jump', 'i', 'rho')),xp,jump_index=0)
        v.update(zip(('_jump', 'i', 'rho'),_result))
        return tuple(v[_n] for _n in ('_jump', 'i', 'k', 'rho'))
    _result=_loop(1,v['pver'],1,_body1,tuple(v[_n] for _n in ('_jump', 'i', 'k', 'rho')),xp,jump_index=0)
    v.update(zip(('_jump', 'i', 'k', 'rho'),_result))
    _outer3=dict(v)
    def _yes3(_carry):
        v=dict(_outer3);v.update(zip(('i', 'k', 'smm', 'tkem'),_carry))
        _outer4=dict(v)
        def _body4(_k,_carry):
            v=dict(_outer4);v.update(zip(('_jump', 'i', 'k', 'smm', 'tkem'),_carry));v['i']=_int(_k)
            _outer5=dict(v)
            def _body5(_k,_carry):
                v=dict(_outer5);v.update(zip(('_jump', 'k', 'smm', 'tkem'),_carry));v['k']=_int(_k)
                _put(v,'tkem',(v['i'],v['k'],),_d(0.5) * (_get(v['tke'], (v['i'], v['k'])) + _get(v['tke'], (v['i'], v['k'] + 1))))  # WRF:module_cam_mp_microp_aero.F:590
                _put(v,'smm',(v['i'],v['k'],),_d(0.5) * (_get(v['smaw'], (v['i'], v['k'])) + _get(v['smaw'], (v['i'], v['k'] + 1))))  # WRF:module_cam_mp_microp_aero.F:591
                _outer6=dict(v)
                def _yes6(_carry):
                    v=dict(_outer6);v.update(zip(('smm', 'tkem'),_carry))
                    _put(v,'tkem',(v['i'],v['k'],),_d(0.5) * _get(v['tke'], (v['i'], v['k'] + 1)))  # WRF:module_cam_mp_microp_aero.F:593
                    _put(v,'smm',(v['i'],v['k'],),_d(0.5) * _get(v['smaw'], (v['i'], v['k'] + 1)))  # WRF:module_cam_mp_microp_aero.F:594
                    return tuple(v[_n] for _n in ('smm', 'tkem'))
                def _no6(_carry):
                    v=dict(_outer6);v.update(zip(('smm', 'tkem'),_carry))
                    _outer7=dict(v)
                    def _yes7(_carry):
                        v=dict(_outer7);v.update(zip(('smm', 'tkem'),_carry))
                        _put(v,'tkem',(v['i'],v['k'],),_d(0.5) * _get(v['tke'], (v['i'], v['k'])))  # WRF:module_cam_mp_microp_aero.F:596
                        _put(v,'smm',(v['i'],v['k'],),_d(0.5) * _get(v['smaw'], (v['i'], v['k'])))  # WRF:module_cam_mp_microp_aero.F:597
                        return tuple(v[_n] for _n in ('smm', 'tkem'))
                    def _no7(_carry):
                        v=dict(_outer7);v.update(zip(('smm', 'tkem'),_carry))
                        return tuple(v[_n] for _n in ('smm', 'tkem'))
                    _result=_cond(_get(v['turbtype'], (v['i'], v['k'] + 1)) == _d(4.0),_yes7,_no7,tuple(v[_n] for _n in ('smm', 'tkem')),xp)  # WRF:module_cam_mp_microp_aero.F:595
                    v.update(zip(('smm', 'tkem'),_result))
                    return tuple(v[_n] for _n in ('smm', 'tkem'))
                _result=_cond(_get(v['turbtype'], (v['i'], v['k'])) == _d(3.0),_yes6,_no6,tuple(v[_n] for _n in ('smm', 'tkem')),xp)  # WRF:module_cam_mp_microp_aero.F:592
                v.update(zip(('smm', 'tkem'),_result))
                _put(v,'smm',(v['i'],v['k'],),_d(0.259) * _get(v['smm'], (v['i'], v['k'])))  # WRF:module_cam_mp_microp_aero.F:599
                _put(v,'smm',(v['i'],v['k'],),xp.maximum(_get(v['smm'], (v['i'], v['k'])), _d(0.4743)))  # WRF:module_cam_mp_microp_aero.F:600
                return tuple(v[_n] for _n in ('_jump', 'k', 'smm', 'tkem'))
            _result=_loop(1,v['pver'],1,_body5,tuple(v[_n] for _n in ('_jump', 'k', 'smm', 'tkem')),xp,jump_index=0)
            v.update(zip(('_jump', 'k', 'smm', 'tkem'),_result))
            return tuple(v[_n] for _n in ('_jump', 'i', 'k', 'smm', 'tkem'))
        _result=_loop(1,v['ncol'],1,_body4,tuple(v[_n] for _n in ('_jump', 'i', 'k', 'smm', 'tkem')),xp,jump_index=0)
        v.update(zip(('_jump', 'i', 'k', 'smm', 'tkem'),_result))
        return tuple(v[_n] for _n in ('i', 'k', 'smm', 'tkem'))
    def _no3(_carry):
        v=dict(_outer3);v.update(zip(('i', 'k', 'smm', 'tkem'),_carry))
        return tuple(v[_n] for _n in ('i', 'k', 'smm', 'tkem'))
    _result=_cond(v['wsubtke'],_yes3,_no3,tuple(v[_n] for _n in ('i', 'k', 'smm', 'tkem')),xp)  # WRF:module_cam_mp_microp_aero.F:586
    v.update(zip(('i', 'k', 'smm', 'tkem'),_result))
    _outer8=dict(v)
    def _body8(_k,_carry):
        v=dict(_outer8);v.update(zip(('_jump', 'dum', 'ftrue', 'i', 'k', 'wsub', 'wsubi'),_carry));v['i']=_int(_k)
        _put(v,'ftrue',None,0)  # WRF:module_cam_mp_microp_aero.F:607
        _outer9=dict(v)
        def _body9(_k,_carry):
            v=dict(_outer9);v.update(zip(('_jump', 'dum', 'k', 'wsub', 'wsubi'),_carry));v['k']=_int(_k)
            _put(v,'dum',None,(_get(v['kkvh'], (v['i'], v['k'])) + _get(v['kkvh'], (v['i'], v['k'] + 1))) / _d(2.0) / _d(30.0))  # WRF:module_cam_mp_microp_aero.F:613
            _put(v,'dum',None,xp.minimum(v['dum'], _d(10.0)))  # WRF:module_cam_mp_microp_aero.F:615
            _put(v,'wsub',(v['i'],v['k'],),v['dum'])  # WRF:module_cam_mp_microp_aero.F:617
            _outer10=dict(v)
            def _yes10(_carry):
                v=dict(_outer10);v.update(zip(('wsub',),_carry))
                _put(v,'wsub',(v['i'],v['k'],),xp.sqrt(_d(0.5) * (_get(v['tke'], (v['i'], v['k'])) + _get(v['tke'], (v['i'], v['k'] + 1))) * (_d(2.0) / _d(3.0))))  # WRF:module_cam_mp_microp_aero.F:620
                _put(v,'wsub',(v['i'],v['k'],),xp.minimum(_get(v['wsub'], (v['i'], v['k'])), _d(10.0)))  # WRF:module_cam_mp_microp_aero.F:621
                return tuple(v[_n] for _n in ('wsub',))
            def _no10(_carry):
                v=dict(_outer10);v.update(zip(('wsub',),_carry))
                return tuple(v[_n] for _n in ('wsub',))
            _result=_cond(v['wsubtke'],_yes10,_no10,tuple(v[_n] for _n in ('wsub',)),xp)  # WRF:module_cam_mp_microp_aero.F:619
            v.update(zip(('wsub',),_result))
            _put(v,'wsubi',(v['i'],v['k'],),xp.maximum(_d(0.001), _get(v['wsub'], (v['i'], v['k']))))  # WRF:module_cam_mp_microp_aero.F:623
            _put(v,'wsubi',(v['i'],v['k'],),xp.minimum(_get(v['wsubi'], (v['i'], v['k'])), _d(0.2)))  # WRF:module_cam_mp_microp_aero.F:624
            _put(v,'wsub',(v['i'],v['k'],),xp.maximum(_d(0.2), _get(v['wsub'], (v['i'], v['k']))))  # WRF:module_cam_mp_microp_aero.F:625
            return tuple(v[_n] for _n in ('_jump', 'dum', 'k', 'wsub', 'wsubi'))
        _result=_loop(1,v['pver'],1,_body9,tuple(v[_n] for _n in ('_jump', 'dum', 'k', 'wsub', 'wsubi')),xp,jump_index=0)
        v.update(zip(('_jump', 'dum', 'k', 'wsub', 'wsubi'),_result))
        return tuple(v[_n] for _n in ('_jump', 'dum', 'ftrue', 'i', 'k', 'wsub', 'wsubi'))
    _result=_loop(1,v['ncol'],1,_body8,tuple(v[_n] for _n in ('_jump', 'dum', 'ftrue', 'i', 'k', 'wsub', 'wsubi')),xp,jump_index=0)
    v.update(zip(('_jump', 'dum', 'ftrue', 'i', 'k', 'wsub', 'wsubi'),_result))
    _outer11=dict(v)
    def _body11(_k,_carry):
        v=dict(_outer11);v.update(zip(('_jump', 'cldm', 'dum2', 'dumfice', 'es', 'esi', 'esl', 'gammas', 'i', 'icldm', 'k', 'lcldm', 'naai', 'naai_hom', 'nfice', 'nidep', 'nidep2', 'nihf', 'nihf2', 'niimm', 'niimm2', 'nimey', 'nimey2', 'qs', 'relhum'),_carry));v['k']=_int(_k)
        _call=vqsatd_water(_get(v['t'], (1, v['k'])),_get(v['p'], (1, v['k'])),v['es'],v['qs'],v['gammas'],v['ncol'],_globals=_globals,xp=xp)
        _put(v,'es',None,_call[0])  # WRF:module_cam_mp_microp_aero.F:637
        _put(v,'qs',None,_call[1])  # WRF:module_cam_mp_microp_aero.F:637
        _put(v,'gammas',None,_call[2])  # WRF:module_cam_mp_microp_aero.F:637
        _outer12=dict(v)
        def _body12(_k,_carry):
            v=dict(_outer12);v.update(zip(('_jump', 'cldm', 'dum2', 'dumfice', 'esi', 'esl', 'i', 'icldm', 'lcldm', 'naai', 'naai_hom', 'nfice', 'nidep', 'nidep2', 'nihf', 'nihf2', 'niimm', 'niimm2', 'nimey', 'nimey2', 'relhum'),_carry));v['i']=_int(_k)
            _put(v,'esl',(v['i'],v['k'],),polysvp(_get(v['t'], (v['i'], v['k'])), 0, xp=xp, _globals=_globals))  # WRF:module_cam_mp_microp_aero.F:641
            _put(v,'esi',(v['i'],v['k'],),polysvp(_get(v['t'], (v['i'], v['k'])), 1, xp=xp, _globals=_globals))  # WRF:module_cam_mp_microp_aero.F:642
            _outer13=dict(v)
            def _yes13(_carry):
                v=dict(_outer13);v.update(zip(('esi',),_carry))
                _put(v,'esi',(v['i'],v['k'],),_get(v['esl'], (v['i'], v['k'])))  # WRF:module_cam_mp_microp_aero.F:645
                return tuple(v[_n] for _n in ('esi',))
            def _no13(_carry):
                v=dict(_outer13);v.update(zip(('esi',),_carry))
                return tuple(v[_n] for _n in ('esi',))
            _result=_cond(_get(v['t'], (v['i'], v['k'])) > v['tmelt'],_yes13,_no13,tuple(v[_n] for _n in ('esi',)),xp)  # WRF:module_cam_mp_microp_aero.F:645
            v.update(zip(('esi',),_result))
            _put(v,'relhum',(v['i'],v['k'],),_get(v['q'], (v['i'], v['k'])) / _get(v['qs'], (v['i'],)))  # WRF:module_cam_mp_microp_aero.F:647
            _put(v,'cldm',(v['i'],v['k'],),xp.maximum(_get(v['cldn'], (v['i'], v['k'])), v['mincld']))  # WRF:module_cam_mp_microp_aero.F:651
            _put(v,'icldm',(v['i'],v['k'],),xp.maximum(_get(v['icecldf'], (v['i'], v['k'])), v['mincld']))  # WRF:module_cam_mp_microp_aero.F:653
            _put(v,'lcldm',(v['i'],v['k'],),xp.maximum(_get(v['liqcldf'], (v['i'], v['k'])), v['mincld']))  # WRF:module_cam_mp_microp_aero.F:654
            _put(v,'nfice',(v['i'],v['k'],),_d(0.0))  # WRF:module_cam_mp_microp_aero.F:659
            _put(v,'dumfice',None,_get(v['qc'], (v['i'], v['k'])) + _get(v['qi'], (v['i'], v['k'])))  # WRF:module_cam_mp_microp_aero.F:660
            _outer14=dict(v)
            def _yes14(_carry):
                v=dict(_outer14);v.update(zip(('nfice',),_carry))
                _put(v,'nfice',(v['i'],v['k'],),_get(v['qi'], (v['i'], v['k'])) / v['dumfice'])  # WRF:module_cam_mp_microp_aero.F:662
                return tuple(v[_n] for _n in ('nfice',))
            def _no14(_carry):
                v=dict(_outer14);v.update(zip(('nfice',),_carry))
                return tuple(v[_n] for _n in ('nfice',))
            _result=_cond(xp.logical_and(v['dumfice'] > v['qsmall'], _get(v['qi'], (v['i'], v['k'])) > v['qsmall']),_yes14,_no14,tuple(v[_n] for _n in ('nfice',)),xp)  # WRF:module_cam_mp_microp_aero.F:661
            v.update(zip(('nfice',),_result))
            _outer15=dict(v)
            def _yes15(_carry):
                v=dict(_outer15);v.update(zip(('dum2', 'naai', 'naai_hom', 'nidep', 'nidep2', 'nihf', 'nihf2', 'niimm', 'niimm2', 'nimey', 'nimey2'),_carry))
                _call=nucleati(_get(v['wsubi'], (v['i'], v['k'])),_get(v['t'], (v['i'], v['k'])),_get(v['relhum'], (v['i'], v['k'])),_get(v['icldm'], (v['i'], v['k'])),_get(v['qc'], (v['i'], v['k'])),_get(v['nfice'], (v['i'], v['k'])),_get(v['rho'], (v['i'], v['k'])),_get(v['qaer'], (v['i'], v['k'], _slice(1, None, None))) * _get(v['rho'], (v['i'], v['k'])),_get(v['dgnum'], (v['i'], v['k'], _slice(1, None, None))),1,v['naer_all'],v['dum2'],v['nihf2'],v['niimm2'],v['nidep2'],v['nimey2'],_globals=_globals,xp=xp)
                _put(v,'dum2',None,_call[0])  # WRF:module_cam_mp_microp_aero.F:695
                _put(v,'nihf2',None,_call[1])  # WRF:module_cam_mp_microp_aero.F:695
                _put(v,'niimm2',None,_call[2])  # WRF:module_cam_mp_microp_aero.F:695
                _put(v,'nidep2',None,_call[3])  # WRF:module_cam_mp_microp_aero.F:695
                _put(v,'nimey2',None,_call[4])  # WRF:module_cam_mp_microp_aero.F:695
                _put(v,'naai',(v['i'],v['k'],),v['dum2'])  # WRF:module_cam_mp_microp_aero.F:702
                _put(v,'naai_hom',(v['i'],v['k'],),v['nihf2'])  # WRF:module_cam_mp_microp_aero.F:703
                _put(v,'nihf',(v['i'],v['k'],),v['nihf2'])  # WRF:module_cam_mp_microp_aero.F:704
                _put(v,'niimm',(v['i'],v['k'],),v['niimm2'])  # WRF:module_cam_mp_microp_aero.F:705
                _put(v,'nidep',(v['i'],v['k'],),v['nidep2'])  # WRF:module_cam_mp_microp_aero.F:706
                _put(v,'nimey',(v['i'],v['k'],),v['nimey2'])  # WRF:module_cam_mp_microp_aero.F:707
                return tuple(v[_n] for _n in ('dum2', 'naai', 'naai_hom', 'nidep', 'nidep2', 'nihf', 'nihf2', 'niimm', 'niimm2', 'nimey', 'nimey2'))
            def _no15(_carry):
                v=dict(_outer15);v.update(zip(('dum2', 'naai', 'naai_hom', 'nidep', 'nidep2', 'nihf', 'nihf2', 'niimm', 'niimm2', 'nimey', 'nimey2'),_carry))
                return tuple(v[_n] for _n in ('dum2', 'naai', 'naai_hom', 'nidep', 'nidep2', 'nihf', 'nihf2', 'niimm', 'niimm2', 'nimey', 'nimey2'))
            _result=_cond(_get(v['t'], (v['i'], v['k'])) < v['tmelt'] - _d(5.0),_yes15,_no15,tuple(v[_n] for _n in ('dum2', 'naai', 'naai_hom', 'nidep', 'nidep2', 'nihf', 'nihf2', 'niimm', 'niimm2', 'nimey', 'nimey2')),xp)  # WRF:module_cam_mp_microp_aero.F:690
            v.update(zip(('dum2', 'naai', 'naai_hom', 'nidep', 'nidep2', 'nihf', 'nihf2', 'niimm', 'niimm2', 'nimey', 'nimey2'),_result))
            return tuple(v[_n] for _n in ('_jump', 'cldm', 'dum2', 'dumfice', 'esi', 'esl', 'i', 'icldm', 'lcldm', 'naai', 'naai_hom', 'nfice', 'nidep', 'nidep2', 'nihf', 'nihf2', 'niimm', 'niimm2', 'nimey', 'nimey2', 'relhum'))
        _result=_loop(1,v['ncol'],1,_body12,tuple(v[_n] for _n in ('_jump', 'cldm', 'dum2', 'dumfice', 'esi', 'esl', 'i', 'icldm', 'lcldm', 'naai', 'naai_hom', 'nfice', 'nidep', 'nidep2', 'nihf', 'nihf2', 'niimm', 'niimm2', 'nimey', 'nimey2', 'relhum')),xp,jump_index=0)
        v.update(zip(('_jump', 'cldm', 'dum2', 'dumfice', 'esi', 'esl', 'i', 'icldm', 'lcldm', 'naai', 'naai_hom', 'nfice', 'nidep', 'nidep2', 'nihf', 'nihf2', 'niimm', 'niimm2', 'nimey', 'nimey2', 'relhum'),_result))
        return tuple(v[_n] for _n in ('_jump', 'cldm', 'dum2', 'dumfice', 'es', 'esi', 'esl', 'gammas', 'i', 'icldm', 'k', 'lcldm', 'naai', 'naai_hom', 'nfice', 'nidep', 'nidep2', 'nihf', 'nihf2', 'niimm', 'niimm2', 'nimey', 'nimey2', 'qs', 'relhum'))
    _result=_loop(1,v['pver'],1,_body11,tuple(v[_n] for _n in ('_jump', 'cldm', 'dum2', 'dumfice', 'es', 'esi', 'esl', 'gammas', 'i', 'icldm', 'k', 'lcldm', 'naai', 'naai_hom', 'nfice', 'nidep', 'nidep2', 'nihf', 'nihf2', 'niimm', 'niimm2', 'nimey', 'nimey2', 'qs', 'relhum')),xp,jump_index=0)
    v.update(zip(('_jump', 'cldm', 'dum2', 'dumfice', 'es', 'esi', 'esl', 'gammas', 'i', 'icldm', 'k', 'lcldm', 'naai', 'naai_hom', 'nfice', 'nidep', 'nidep2', 'nihf', 'nihf2', 'niimm', 'niimm2', 'nimey', 'nimey2', 'qs', 'relhum'),_result))
    _outer16=dict(v)
    def _body16(_k,_carry):
        v=dict(_outer16);v.update(zip(('_jump', 'i', 'k', 'lcldn', 'lcldo', 'qcld'),_carry));v['k']=_int(_k)
        _outer17=dict(v)
        def _body17(_k,_carry):
            v=dict(_outer17);v.update(zip(('_jump', 'i', 'lcldn', 'lcldo', 'qcld'),_carry));v['i']=_int(_k)
            _put(v,'qcld',None,_get(v['qc'], (v['i'], v['k'])) + _get(v['qi'], (v['i'], v['k'])))  # WRF:module_cam_mp_microp_aero.F:743
            _outer18=dict(v)
            def _yes18(_carry):
                v=dict(_outer18);v.update(zip(('lcldn', 'lcldo'),_carry))
                _put(v,'lcldn',(v['i'],v['k'],),_get(v['cldn'], (v['i'], v['k'])) * _get(v['qc'], (v['i'], v['k'])) / v['qcld'])  # WRF:module_cam_mp_microp_aero.F:745
                _put(v,'lcldo',(v['i'],v['k'],),_get(v['cldo'], (v['i'], v['k'])) * _get(v['qc'], (v['i'], v['k'])) / v['qcld'])  # WRF:module_cam_mp_microp_aero.F:746
                return tuple(v[_n] for _n in ('lcldn', 'lcldo'))
            def _no18(_carry):
                v=dict(_outer18);v.update(zip(('lcldn', 'lcldo'),_carry))
                _put(v,'lcldn',(v['i'],v['k'],),_d(0.0))  # WRF:module_cam_mp_microp_aero.F:748
                _put(v,'lcldo',(v['i'],v['k'],),_d(0.0))  # WRF:module_cam_mp_microp_aero.F:749
                return tuple(v[_n] for _n in ('lcldn', 'lcldo'))
            _result=_cond(v['qcld'] > v['qsmall'],_yes18,_no18,tuple(v[_n] for _n in ('lcldn', 'lcldo')),xp)  # WRF:module_cam_mp_microp_aero.F:744
            v.update(zip(('lcldn', 'lcldo'),_result))
            return tuple(v[_n] for _n in ('_jump', 'i', 'lcldn', 'lcldo', 'qcld'))
        _result=_loop(1,v['ncol'],1,_body17,tuple(v[_n] for _n in ('_jump', 'i', 'lcldn', 'lcldo', 'qcld')),xp,jump_index=0)
        v.update(zip(('_jump', 'i', 'lcldn', 'lcldo', 'qcld'),_result))
        return tuple(v[_n] for _n in ('_jump', 'i', 'k', 'lcldn', 'lcldo', 'qcld'))
    _result=_loop(1,v['pver'],1,_body16,tuple(v[_n] for _n in ('_jump', 'i', 'k', 'lcldn', 'lcldo', 'qcld')),xp,jump_index=0)
    v.update(zip(('_jump', 'i', 'k', 'lcldn', 'lcldo', 'qcld'),_result))
    _call=dropmixnuc(v['lchnk'],v['ncol'],v['ncloc'],v['nctend_mixnuc'],v['t'],v['omega'],v['p'],v['pint'],v['pdel'],v['rpdel'],v['zm'],v['kkvh'],v['wsub'],v['lcldn'],v['lcldo'],v['qaer'],v['cflx'],v['qaertend'],v['deltat'],v['qqcw'],_globals=_globals,xp=xp)
    _put(v,'ncloc',None,_call[0])  # WRF:module_cam_mp_microp_aero.F:754
    _put(v,'nctend_mixnuc',None,_call[1])  # WRF:module_cam_mp_microp_aero.F:754
    _put(v,'qaertend',None,_call[2])  # WRF:module_cam_mp_microp_aero.F:754
    _put(v,'qqcw',None,_call[3])  # WRF:module_cam_mp_microp_aero.F:754
    v['_fatal']=v['_fatal']+_call[4]
    _put(v,'npccn',(_slice(None,v['ncol'],None),_slice(None,None,None),),_get(v['nctend_mixnuc'], (_slice(1, v['ncol'], None), _slice(1, None, None))))  # WRF:module_cam_mp_microp_aero.F:762
    _outer19=dict(v)
    def _body19(_k,_carry):
        v=dict(_outer19);v.update(zip(('_jump', 'dmc', 'i', 'k', 'nacon', 'rndst', 'ssmc'),_carry));v['k']=_int(_k)
        _outer20=dict(v)
        def _body20(_k,_carry):
            v=dict(_outer20);v.update(zip(('_jump', 'dmc', 'i', 'nacon', 'rndst', 'ssmc'),_carry));v['i']=_int(_k)
            _outer21=dict(v)
            def _yes21(_carry):
                v=dict(_outer21);v.update(zip(('dmc', 'nacon', 'rndst', 'ssmc'),_carry))
                _put(v,'dmc',None,_get(v['qaer'], (v['i'], v['k'], _get(v['lptr_dust_a_amode'], (v['modeptr_coarse'],)))))  # WRF:module_cam_mp_microp_aero.F:785
                _put(v,'ssmc',None,_get(v['qaer'], (v['i'], v['k'], _get(v['lptr_nacl_a_amode'], (v['modeptr_coarse'],)))))  # WRF:module_cam_mp_microp_aero.F:786
                _outer22=dict(v)
                def _yes22(_carry):
                    v=dict(_outer22);v.update(zip(('nacon',),_carry))
                    _put(v,'nacon',(v['i'],v['k'],3,),v['dmc'] / (v['ssmc'] + v['dmc']) * _get(v['qaer'], (v['i'], v['k'], _get(v['numptr_amode'], (v['modeptr_coarse'],)))) * _get(v['rho'], (v['i'], v['k'])))  # WRF:module_cam_mp_microp_aero.F:788
                    return tuple(v[_n] for _n in ('nacon',))
                def _no22(_carry):
                    v=dict(_outer22);v.update(zip(('nacon',),_carry))
                    _put(v,'nacon',(v['i'],v['k'],3,),_d(0.0))  # WRF:module_cam_mp_microp_aero.F:790
                    return tuple(v[_n] for _n in ('nacon',))
                _result=_cond(v['dmc'] > _d(0.0),_yes22,_no22,tuple(v[_n] for _n in ('nacon',)),xp)  # WRF:module_cam_mp_microp_aero.F:787
                v.update(zip(('nacon',),_result))
                _put(v,'rndst',(v['i'],v['k'],3,),_d(0.5) * _get(v['dgnumwet'], (v['i'], v['k'], v['modeptr_coarse'])))  # WRF:module_cam_mp_microp_aero.F:795
                _outer23=dict(v)
                def _yes23(_carry):
                    v=dict(_outer23);v.update(zip(('rndst',),_carry))
                    _put(v,'rndst',(v['i'],v['k'],3,),v['rn_dst3'])  # WRF:module_cam_mp_microp_aero.F:800
                    return tuple(v[_n] for _n in ('rndst',))
                def _no23(_carry):
                    v=dict(_outer23);v.update(zip(('rndst',),_carry))
                    return tuple(v[_n] for _n in ('rndst',))
                _result=_cond(_get(v['rndst'], (v['i'], v['k'], 3)) <= _d(0.0),_yes23,_no23,tuple(v[_n] for _n in ('rndst',)),xp)  # WRF:module_cam_mp_microp_aero.F:799
                v.update(zip(('rndst',),_result))
                return tuple(v[_n] for _n in ('dmc', 'nacon', 'rndst', 'ssmc'))
            def _no21(_carry):
                v=dict(_outer21);v.update(zip(('dmc', 'nacon', 'rndst', 'ssmc'),_carry))
                return tuple(v[_n] for _n in ('dmc', 'nacon', 'rndst', 'ssmc'))
            _result=_cond(_get(v['t'], (v['i'], v['k'])) < _d(269.15),_yes21,_no21,tuple(v[_n] for _n in ('dmc', 'nacon', 'rndst', 'ssmc')),xp)  # WRF:module_cam_mp_microp_aero.F:774
            v.update(zip(('dmc', 'nacon', 'rndst', 'ssmc'),_result))
            return tuple(v[_n] for _n in ('_jump', 'dmc', 'i', 'nacon', 'rndst', 'ssmc'))
        _result=_loop(1,v['ncol'],1,_body20,tuple(v[_n] for _n in ('_jump', 'dmc', 'i', 'nacon', 'rndst', 'ssmc')),xp,jump_index=0)
        v.update(zip(('_jump', 'dmc', 'i', 'nacon', 'rndst', 'ssmc'),_result))
        return tuple(v[_n] for _n in ('_jump', 'dmc', 'i', 'k', 'nacon', 'rndst', 'ssmc'))
    _result=_loop(1,v['pver'],1,_body19,tuple(v[_n] for _n in ('_jump', 'dmc', 'i', 'k', 'nacon', 'rndst', 'ssmc')),xp,jump_index=0)
    v.update(zip(('_jump', 'dmc', 'i', 'k', 'nacon', 'rndst', 'ssmc'),_result))
    _outer24=dict(v)
    def _body24(_k,_carry):
        v=dict(_outer24);v.update(zip(('_jump', 'l'),_carry));v['l']=_int(_k)
        return tuple(v[_n] for _n in ('_jump', 'l'))
    _result=_loop(1,v['naer_all'],1,_body24,tuple(v[_n] for _n in ('_jump', 'l')),xp,jump_index=0)
    v.update(zip(('_jump', 'l'),_result))
    _outer25=dict(v)
    def _body25(_k,_carry):
        v=dict(_outer25);v.update(zip(('_jump', 'i', 'k', 'nidep', 'nihf', 'niimm', 'nimey'),_carry));v['i']=_int(_k)
        _outer26=dict(v)
        def _body26(_k,_carry):
            v=dict(_outer26);v.update(zip(('_jump', 'k', 'nidep', 'nihf', 'niimm', 'nimey'),_carry));v['k']=_int(_k)
            _put(v,'nihf',(v['i'],v['k'],),_get(v['nihf'], (v['i'], v['k'])) * _get(v['rho'], (v['i'], v['k'])))  # WRF:module_cam_mp_microp_aero.F:875
            _put(v,'niimm',(v['i'],v['k'],),_get(v['niimm'], (v['i'], v['k'])) * _get(v['rho'], (v['i'], v['k'])))  # WRF:module_cam_mp_microp_aero.F:876
            _put(v,'nidep',(v['i'],v['k'],),_get(v['nidep'], (v['i'], v['k'])) * _get(v['rho'], (v['i'], v['k'])))  # WRF:module_cam_mp_microp_aero.F:877
            _put(v,'nimey',(v['i'],v['k'],),_get(v['nimey'], (v['i'], v['k'])) * _get(v['rho'], (v['i'], v['k'])))  # WRF:module_cam_mp_microp_aero.F:878
            return tuple(v[_n] for _n in ('_jump', 'k', 'nidep', 'nihf', 'niimm', 'nimey'))
        _result=_loop(1,v['pver'],1,_body26,tuple(v[_n] for _n in ('_jump', 'k', 'nidep', 'nihf', 'niimm', 'nimey')),xp,jump_index=0)
        v.update(zip(('_jump', 'k', 'nidep', 'nihf', 'niimm', 'nimey'),_result))
        return tuple(v[_n] for _n in ('_jump', 'i', 'k', 'nidep', 'nihf', 'niimm', 'nimey'))
    _result=_loop(1,v['ncol'],1,_body25,tuple(v[_n] for _n in ('_jump', 'i', 'k', 'nidep', 'nihf', 'niimm', 'nimey')),xp,jump_index=0)
    v.update(zip(('_jump', 'i', 'k', 'nidep', 'nihf', 'niimm', 'nimey'),_result))
    return (v['qaertend'],v['wsub'],v['wsubi'],v['naai'],v['naai_hom'],v['npccn'],v['rndst'],v['nacon'],v['qqcw'],v['_fatal'],)

ARGUMENTS={'explmix': ['q', 'src', 'ekkp', 'ekkm', 'overlapp', 'overlapm', 'qold', 'surfrate', 'flxconv', 'pver', 'dt', 'is_unact', 'qactold'], 'maxsat': ['zeta', 'eta', 'nmode', 'smc', 'smax'], 'activate_modal': ['wbar', 'sigw', 'wdiab', 'wminf', 'wmaxf', 'tair', 'rhoair', 'na', 'pmode', 'nmode', 'volume', 'sigman', 'hygro', 'fn', 'fm', 'fluxn', 'fluxm', 'flux_fullact'], 'loadaer': ['raer', 'qqcw', 'istart', 'istop', 'k', 'm', 'cs', 'npv1', 'phase', 'naerosol', 'vaerosol', 'hygro'], 'dropmixnuc': ['lchnk', 'ncol', 'ncldwtr', 'tendnd', 'temp', 'omega', 'pmid', 'pint', 'pdel', 'rpdel', 'zm', 'kvh', 'wsub', 'cldn', 'cldo', 'raer', 'cflx', 'raertend', 'dtmicro', 'qqcw_inout'], 'microp_aero_ts': ['lchnk', 'ncol', 'deltatin', 't', 'ttend', 'qn', 'qc', 'qi', 'nc', 'ni', 'p', 'pdel', 'cldn', 'liqcldf', 'icecldf', 'cldo', 'pint', 'rpdel', 'zm', 'omega', 'qaer', 'cflx', 'qaertend', 'dgnumwet', 'dgnum', 'kkvh', 'tke', 'turbtype', 'smaw', 'wsub', 'wsubi', 'naai', 'naai_hom', 'npccn', 'rndst', 'nacon', 'qqcw']}
OUTPUTS={'explmix': ['q'], 'maxsat': ['smax'], 'activate_modal': ['fn', 'fm', 'fluxn', 'fluxm', 'flux_fullact', '_fatal'], 'loadaer': ['naerosol', 'vaerosol', 'hygro', '_fatal'], 'dropmixnuc': ['ncldwtr', 'tendnd', 'raertend', 'qqcw_inout', '_fatal'], 'microp_aero_ts': ['qaertend', 'wsub', 'wsubi', 'naai', 'naai_hom', 'npccn', 'rndst', 'nacon', 'qqcw', '_fatal']}
VARIABLES={'explmix': {'pver': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'q': {'kind': 'f64', 'dims': ['pver'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'qold': {'kind': 'f64', 'dims': ['pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'src': {'kind': 'f64', 'dims': ['pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ekkp': {'kind': 'f64', 'dims': ['pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ekkm': {'kind': 'f64', 'dims': ['pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'overlapp': {'kind': 'f64', 'dims': ['pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'overlapm': {'kind': 'f64', 'dims': ['pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'surfrate': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'flxconv': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'dt': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'is_unact': {'kind': 'bool', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'qactold': {'kind': 'f64', 'dims': ['pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'k': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'kp1': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'km1': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}}, 'maxsat': {'nmode': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'smc': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'zeta': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'eta': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'smax': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'm': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'sum': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'g1': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'g2': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'g1sqrt': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'g2sqrt': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}}, 'activate_modal': {'pmode': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wbar': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'sigw': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wdiab': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wminf': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wmaxf': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'tair': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'rhoair': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'na': {'kind': 'f64', 'dims': ['pmode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nmode': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'volume': {'kind': 'f64', 'dims': ['pmode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'sigman': {'kind': 'f64', 'dims': ['pmode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'hygro': {'kind': 'f64', 'dims': ['pmode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fn': {'kind': 'f64', 'dims': ['pmode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fm': {'kind': 'f64', 'dims': ['pmode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fluxn': {'kind': 'f64', 'dims': ['pmode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fluxm': {'kind': 'f64', 'dims': ['pmode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'flux_fullact': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nx': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': '200', 'parameter': True, 'allocatable': False}, 'iquasisect_option': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'isectional': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'integ': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'integf': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'p0': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '1013.25e2_r8', 'parameter': True, 'allocatable': False}, 'xmin': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'xmax': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'volmin': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'volmax': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'tmass': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'sign': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'rm': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'pres': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'path': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'diff': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'conduct': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'diff0': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'conduct0': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'es': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'qs': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dqsdt': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dqsdp': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'g': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'zeta': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'eta': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lnsmax': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'alpha': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'gamma': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'beta': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'sqrtg': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'amcube': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'smcrit': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lnsm': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'smc': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'sumflx_fullact': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'sumflxn': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'sumflxm': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'sumfn': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'sumfm': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fnold': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fmold': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wold': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'gold': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'alogam': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'rlo': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'rhi': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'xint1': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'xint2': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'xint3': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'xint4': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wmin': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wmax': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'w': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dw': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dwmax': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dwmin': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wnuc': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dwnew': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wb': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dfmin': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dfmax': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fnew': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fold': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fnmin': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fnbar': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fsbar': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fmbar': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'alw': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'sqrtalw': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'smax': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'x': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'arg': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'xmincoeff': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'xcut': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'volcut': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'surfcut': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'z': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'z1': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'z2': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wf1': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wf2': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'zf1': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'zf2': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'gf1': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'gf2': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'gf': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'etafactor1': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'etafactor2': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'etafactor2max': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'm': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'n': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'eps': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '0.3_r8', 'parameter': True, 'allocatable': False}, 'fmax': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '0.99_r8', 'parameter': True, 'allocatable': False}, 'sds': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '3._r8', 'parameter': True, 'allocatable': False}, 'namin': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '1.e6_r8', 'parameter': True, 'allocatable': False}, 'ndist': {'kind': 'i32', 'dims': ['nx'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}}, 'loadaer': {'qqcw': {'kind': 'f64', 'dims': ['pcols', 'pver', 'pcnst'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'raer': {'kind': 'f64', 'dims': ['pcols', 'pver', 'pcnst'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'istart': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'istop': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'm': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'k': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'cs': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'phase': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'npv1': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'naerosol': {'kind': 'f64', 'dims': ['pcols'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'vaerosol': {'kind': 'f64', 'dims': ['pcols'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'hygro': {'kind': 'f64', 'dims': ['pcols'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'vol': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'i': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lnum': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lnumcw': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'l': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ltype': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lmass': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lmasscw': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}}, 'dropmixnuc': {'lchnk': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ncol': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'dtmicro': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'temp': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ncldwtr': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'tendnd': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'omega': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'pmid': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'pint': {'kind': 'f64', 'dims': ['pcols', 'pverp'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'pdel': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'rpdel': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'zm': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'kvh': {'kind': 'f64', 'dims': ['pcols', 'pverp'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'wsub': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'cldo': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'cldn': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'raer': {'kind': 'f64', 'dims': ['pcols', 'pver', 'pcnst'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'cflx': {'kind': 'f64', 'dims': ['pcols', 'pcnst'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'raertend': {'kind': 'f64', 'dims': ['pcols', 'pver', 'pcnst'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'qqcw_inout': {'kind': 'f64', 'dims': ['pcols', 'pver', 'pcnst'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'depvel': {'kind': 'f64', 'dims': ['pcols', 'pcnst'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'qcld': {'kind': 'f64', 'dims': ['pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wtke': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wtke_cen': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'zn': {'kind': 'f64', 'dims': ['pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'zs': {'kind': 'f64', 'dims': ['pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'zkmin': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '0.01_r8', 'parameter': True, 'allocatable': False}, 'zkmax': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '100._r8', 'parameter': True, 'allocatable': False}, 'w': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'cs': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dz': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'zero': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'flxconv': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wdiab': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wmixmin': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '0.1', 'parameter': True, 'allocatable': False}, 'qncld': {'kind': 'f64', 'dims': ['pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ekd': {'kind': 'f64', 'dims': ['pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ekk': {'kind': 'f64', 'dims': ['0:pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'srcn': {'kind': 'f64', 'dims': ['pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'sq2pi': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': '2.5066283_r8', 'parameter': True, 'allocatable': False}, 'dtinv': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'top': {'kind': 'bool', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'km1': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'kp1': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wbar': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wmix': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wmin': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'wmax': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dum': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dumc': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dact': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fluxntot': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fac_srflx': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'surfrate': {'kind': 'f64', 'dims': ['pcnst'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'surfratemax': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dtmin': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'tinv': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dtt': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nsubmix': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nsubmix_bnd': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'i': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'k': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'm': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'n': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'tempr4': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dtmix': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'alogarg': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nstep': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'pi': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nnew': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nsav': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ntemp': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'overlapp': {'kind': 'f64', 'dims': ['pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'overlapm': {'kind': 'f64', 'dims': ['pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ekkp': {'kind': 'f64', 'dims': ['pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ekkm': {'kind': 'f64', 'dims': ['pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'count_submix': {'kind': 'i32', 'dims': ['100'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'kvhmax': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nsource': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ndropmix': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ndropcol': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lnum': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lnumcw': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lmass': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lmasscw': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lphase': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lsfc': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lsfccw': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lsig': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lspec': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ltype': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lwater': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ntype': {'kind': 'i32', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'na': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'va': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'hy': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'naermod': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'sigmag': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'hygro': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'vaerosol': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'raercol': {'kind': 'f64', 'dims': ['pver', 'pcnst', '2'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'raercol_cw': {'kind': 'f64', 'dims': ['pver', 'pcnst', '2'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'surfrate_cw': {'kind': 'f64', 'dims': ['pcnst'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'source': {'kind': 'f64', 'dims': ['pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fn': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fm': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fluxn': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fluxm': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'sum': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'sumcw': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'flux': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fluxcw': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nact': {'kind': 'f64', 'dims': ['pver', 'ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'mact': {'kind': 'f64', 'dims': ['pver', 'ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'sigmag_amode_cur': {'kind': 'f64', 'dims': ['ntot_amode'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'qqcwtend': {'kind': 'f64', 'dims': ['pcols', 'pver', 'pcnst'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'coltend': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'tmpa': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lc': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'tmpname': {'kind': 'str', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'fieldname': {'kind': 'str', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'csbot': {'kind': 'f64', 'dims': ['pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'csbot_cscen': {'kind': 'f64', 'dims': ['pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'flux_fullact': {'kind': 'f64', 'dims': ['pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'taumix_internal_pver_inv': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dactn': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'taunuc': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'damp': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'cldo_tmp': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'cldn_tmp': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'tau_cld_regenerate': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'cldinterior': {'kind': 'bool', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ixndrop': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'l': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'naer_tot': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'naer': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'psat': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': '6', 'parameter': True, 'allocatable': False}, 'supersat': {'kind': 'f64', 'dims': ['psat'], 'intent': '', 'initial': '(/0.02,0.05,0.1,0.2,0.5,1.0/)', 'parameter': False, 'allocatable': False}, 'ccn': {'kind': 'f64', 'dims': ['pcols', 'pver', 'psat'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ccn_name': {'kind': 'str', 'dims': ['psat'], 'intent': '', 'initial': "(/'ccn1','ccn2','ccn3','ccn4','ccn5','ccn6'/)", 'parameter': False, 'allocatable': False}, 'arg': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'phase': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}}, 'microp_aero_ts': {'lchnk': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ncol': {'kind': 'i32', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'deltatin': {'kind': 'f64', 'dims': [], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 't': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ttend': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'qn': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'qc': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'qi': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'nc': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'ni': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'p': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'pdel': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'cldn': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'icecldf': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'liqcldf': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'cldo': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'pint': {'kind': 'f64', 'dims': ['pcols', 'pverp'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'rpdel': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'zm': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'omega': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'qaer': {'kind': 'f64', 'dims': ['pcols', 'pver', 'pcnst'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'cflx': {'kind': 'f64', 'dims': ['pcols', 'pcnst'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'qaertend': {'kind': 'f64', 'dims': ['pcols', 'pver', 'pcnst'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'qqcw': {'kind': 'f64', 'dims': ['pcols', 'pver', 'pcnst'], 'intent': 'inout', 'initial': None, 'parameter': False, 'allocatable': False}, 'dgnumwet': {'kind': 'f64', 'dims': ['pcols', 'pver', 'ntot_amode'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'dgnum': {'kind': 'f64', 'dims': ['pcols', 'pver', 'ntot_amode'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'kkvh': {'kind': 'f64', 'dims': ['pcols', 'pver+1'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'tke': {'kind': 'f64', 'dims': ['pcols', 'pver+1'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'turbtype': {'kind': 'f64', 'dims': ['pcols', 'pver+1'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'smaw': {'kind': 'f64', 'dims': ['pcols', 'pver+1'], 'intent': 'in', 'initial': None, 'parameter': False, 'allocatable': False}, 'wsub': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'wsubi': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'naai': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'naai_hom': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'npccn': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'rndst': {'kind': 'f64', 'dims': ['pcols', 'pver', '4'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'nacon': {'kind': 'f64', 'dims': ['pcols', 'pver', '4'], 'intent': 'out', 'initial': None, 'parameter': False, 'allocatable': False}, 'tkem': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'smm': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'relhum': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'cldm': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'icldm': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lcldm': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nfice': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dumfice': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'qcld': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lcldn': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'lcldo': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nctend_mixnuc': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'deltat': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nihf2': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'niimm2': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nidep2': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nimey2': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dum2': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'arg': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ftrue': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dum': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'q': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ncloc': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'rho': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'mincld': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'tcnt': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'viscosity': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'mfp': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'slip1': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'slip2': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'slip3': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'slip4': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dfaer1': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dfaer2': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dfaer3': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dfaer4': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nacon1': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nacon2': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nacon3': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nacon4': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dmc': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ssmc': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'dstrn': {'kind': 'f64', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'tsp': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'qsp': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'qsphy': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'qs': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'es': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'esl': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'esi': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'gammas': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'naermod': {'kind': 'f64', 'dims': ['naer_all'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': True}, 'naer2': {'kind': 'f64', 'dims': ['pcols', 'pver', 'naer_all'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': True}, 'maerosol': {'kind': 'f64', 'dims': ['1', 'naer_all'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': True}, 'naer': {'kind': 'f64', 'dims': ['pcols'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ccn': {'kind': 'f64', 'dims': ['pcols', 'pver', 'psat'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ccn_name': {'kind': 'f64', 'dims': ['psat'], 'intent': '', 'initial': "(/'ccn1','ccn2','ccn3','ccn4','ccn5','ccn6'/)", 'parameter': True, 'allocatable': False}, 'nimey': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nihf': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nidep': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'niimm': {'kind': 'f64', 'dims': ['pcols', 'pver'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'i': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'k': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'nstep': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'n': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'l': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ii': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'kk': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'm': {'kind': 'i32', 'dims': [], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': False}, 'ntype': {'kind': 'i32', 'dims': ['naer_all'], 'intent': '', 'initial': None, 'parameter': False, 'allocatable': True}}}
