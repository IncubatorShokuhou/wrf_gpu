"""Generated original Jensen final_checks; see lane port_levels proof. Not yet qualified."""
from .level_support import *  # noqa: F403


def final_checks(fields,tables,*,xp=jnp,observer=None):
    v=dict(fields)
    _,_r=_real(v["qv"],xp)
    cat=3;ice1=1;ice2=2;ice3=3
    riming=rain_ice=freeze_qc=splinters=wet_grow=True
    _nint=lambda a: _fortran_nint(a,xp)
    _int=lambda a: xp.asarray(a).astype("int32")
    _max=lambda *a: _maximum(a,xp)
    _min=lambda *a: _minimum(a,xp)
    _expand=lambda a,n: _trailing(a,n,xp)
    v['nu']=_r(4.0)
    v['qsmall']=_r(1e-12)
    v['qnsmall']=_r(1.25e-07)
    v['qasmall']=_r(1e-24)
    v['ao']=_r(1e-07)
    v['ar']=_r(149.1)
    v['br']=_r(0.5)
    v['f1r']=_r(0.78)
    v['f2r']=_r(0.308)
    v['lammaxr']=_r(1.0) / _r(2e-05)
    v['lamminr']=_r(1.0) / _r(0.0028)
    v['pi']=_r(3.14159265)
    v['g_home']=_r(9.8)
    v['rd']=_r(287.15)
    v['rv']=_r(461.5)
    v['cp']=_r(1004.0)
    v['cpw']=_r(4218.0)
    v['rcp']=_r(0.285856574)
    v['rhow']=_r(1000.0)
    v['t0']=_r(273.15)
    v['p0']=_r(101325.0)
    v['rhoi']=_r(920.0)
    v['r0']=_r(1.27494)
    v.setdefault('ai',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('alphstr',_initial(v["qv"],'real',(),xp))
    v.setdefault('alphv',_initial(v["qv"],'real',(),xp))
    v.setdefault('ani',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('betam',_initial(v["qv"],'real',(),xp))
    v.setdefault('ci',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('cni',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('deltastr',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('gamma_arg',_initial(v["qv"],'real',(),xp))
    v.setdefault('gi',_initial(v["qv"],'integer',(),xp))
    v.setdefault('i_temp',_initial(v["qv"],'real',(),xp))
    v.setdefault('lamr',_initial(v["qv"],'real',(),xp))
    v.setdefault('n0rr',_initial(v["qv"],'real',(),xp))
    v.setdefault('ni',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('nr',_initial(v["qv"],'real',(),xp))
    v.setdefault('phiagg3',_initial(v["qv"],'real',(),xp))
    v.setdefault('qi',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('qr',_initial(v["qv"],'real',(),xp))
    v.setdefault('qv',_initial(v["qv"],'real',(),xp))
    v.setdefault('rhobar',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('rni',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('temp',_initial(v["qv"],'real',(),xp))
    v.setdefault('theta',_initial(v["qv"],'real',(),xp))
    v.setdefault('xxls',_initial(v["qv"],'real',(),xp))
    v.setdefault('xxlv',_initial(v["qv"],'real',(),xp))
    _put(v,'temp',None,v['theta'] / xp.power(_r(100000.0) / v['pres_e'], v['rcp']),True,xp)  # WRF:2842
    _put(v,'i_temp',None,_r(1.0) / v['temp'],True,xp)  # WRF:2843
    _put(v,'xxls',None,_r(3150000.0) - _r(2370.0) * v['temp'] + _r(333700.0),True,xp)  # WRF:2844
    _put(v,'xxlv',None,_r(3148400.0) - _r(2370.0) * v['temp'],True,xp)  # WRF:2845
    for cc in range(1,cat+1):
        mask_1=xp.logical_and(True,_get(v['qi'], (cc - 1,)) > v['qsmall'])
        taken_1=mask_1
        _put(v,'ni',((cc)-1,),_max(_get(v['ni'], (cc - 1,)), v['qnsmall']),mask_1,xp)  # WRF:2851
        _put(v,'ai',((cc)-1,),_max(_get(v['ai'], (cc - 1,)), v['qasmall']),mask_1,xp)  # WRF:2852
        _put(v,'ci',((cc)-1,),_max(_get(v['ci'], (cc - 1,)), v['qasmall']),mask_1,xp)  # WRF:2853
        _put(v,'ani',((cc)-1,),_max(xp.power(_get(v['ai'], (cc - 1,)) * _get(v['ai'], (cc - 1,)) / (_get(v['ci'], (cc - 1,)) * _get(v['ni'], (cc - 1,))), _r(0.333333333333)), _r(2e-06)),mask_1,xp)  # WRF:2854
        _put(v,'cni',((cc)-1,),_max(xp.power(_get(v['ci'], (cc - 1,)) * _get(v['ci'], (cc - 1,)) / (_get(v['ai'], (cc - 1,)) * _get(v['ni'], (cc - 1,))), _r(0.333333333333)), _r(2e-06)),mask_1,xp)  # WRF:2855
        mask_2=xp.logical_and(mask_1,xp.logical_or(_get(v['ai'], (cc - 1,)) < _r(1e-12), _get(v['ci'], (cc - 1,)) < _r(1e-12)))
        taken_2=mask_2
        _put(v,'ai',((cc)-1,),_min(_get(v['ai'], (cc - 1,)), _get(v['ci'], (cc - 1,))),mask_2,xp)  # WRF:2860
        _put(v,'ci',((cc)-1,),_get(v['ai'], (cc - 1,)),mask_2,xp)  # WRF:2861
        _put(v,'ani',((cc)-1,),xp.power(_get(v['ai'], (cc - 1,)) / _get(v['ni'], (cc - 1,)), _r(0.3333333333)),mask_2,xp)  # WRF:2862
        _put(v,'cni',((cc)-1,),xp.power(_get(v['ci'], (cc - 1,)) / _get(v['ni'], (cc - 1,)), _r(0.3333333333)),mask_2,xp)  # WRF:2863
        mask_3=xp.logical_and(mask_1,xp.logical_or(_get(v['ani'], (cc - 1,)) < _r(2e-06), _get(v['cni'], (cc - 1,)) < _r(2e-06)))
        taken_3=mask_3
        _put(v,'ani',((cc)-1,),_r(2e-06),mask_3,xp)  # WRF:2867
        _put(v,'cni',((cc)-1,),_r(2e-06),mask_3,xp)  # WRF:2868
        _put(v,'ci',((cc)-1,),_get(v['cni'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_1,xp)  # WRF:2871
        _put(v,'ai',((cc)-1,),_get(v['ani'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_1,xp)  # WRF:2872
        _put(v,'deltastr',((cc)-1,),(xp.log(_get(v['cni'], (cc - 1,))) - xp.log(v['ao'])) / (xp.log(_get(v['ani'], (cc - 1,))) - xp.log(v['ao'])),mask_1,xp)  # WRF:2874
        mask_4=xp.logical_and(mask_1,cc == ice3)
        taken_4=mask_4
        mask_5=xp.logical_and(mask_4,v['temp'] <= v['t0'])
        taken_5=mask_5
        _put(v,'rhobar',((cc)-1,),_r(50.0),mask_5,xp)  # WRF:2879
        _put(v,'phiagg3',None,_r(0.2),mask_5,xp)  # WRF:2880
        _put(v,'gamma_arg',None,v['nu'] - _r(1.0) + _get(v['deltastr'], (cc - 1,)),mask_5,xp)  # WRF:2881
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_5,xp)  # WRF:2882
        _put(v,'cni',((cc)-1,),v['phiagg3'] * _get(v['ani'], (cc - 1,)) * v['gammnu'] / tables['gamma_tab'][v['gi'] - 1,],mask_5,xp)  # WRF:2883
        _put(v,'ai',((cc)-1,),_get(v['ani'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_4,xp)  # WRF:2886
        _put(v,'ci',((cc)-1,),_get(v['cni'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_4,xp)  # WRF:2887
        mask_6=xp.logical_and(mask_4,xp.logical_and(_get(v['ani'], (cc - 1,)) > _r(1.1) * v['ao'], _get(v['cni'], (cc - 1,)) > _r(1.1) * v['ao']))
        taken_6=mask_6
        _put(v,'deltastr',((cc)-1,),(xp.log(_get(v['cni'], (cc - 1,))) - xp.log(v['ao'])) / (xp.log(_get(v['ani'], (cc - 1,))) - xp.log(v['ao'])),mask_6,xp)  # WRF:2890
        mask_7=xp.logical_and(mask_4,xp.logical_not(taken_6))
        _put(v,'deltastr',((cc)-1,),_r(1.0),mask_7,xp)  # WRF:2892
        _put(v,'deltastr',((cc)-1,),_min(_get(v['deltastr'], (cc - 1,)), _r(1.0)),mask_4,xp)  # WRF:2895
        _put(v,'alphstr',None,xp.power(v['ao'], _r(1.0) - _get(v['deltastr'], (cc - 1,))),mask_4,xp)  # WRF:2897
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_4,xp)  # WRF:2898
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_4,xp)  # WRF:2899
        _put(v,'ani',((cc)-1,),xp.power(_get(v['qi'], (cc - 1,)) / (_get(v['ni'], (cc - 1,)) * _get(v['rhobar'], (cc - 1,)) * v['fourthirdspi'] * v['alphstr'] * tables['gamma_tab'][v['gi'] - 1,] / v['gammnu']), _r(1.0) / (_r(2.0) + _get(v['deltastr'], (cc - 1,)))),mask_4,xp)  # WRF:2900
        _put(v,'ani',((cc)-1,),_max(_get(v['ani'], (cc - 1,)), _r(2e-06)),mask_4,xp)  # WRF:2903
        _put(v,'cni',((cc)-1,),v['alphstr'] * xp.power(_get(v['ani'], (cc - 1,)), _get(v['deltastr'], (cc - 1,))),mask_4,xp)  # WRF:2904
        mask_8=xp.logical_and(mask_4,_get(v['ani'], (cc - 1,)) > _r(0.0005))
        taken_8=mask_8
        _put(v,'ani',((cc)-1,),_r(0.0005),mask_8,xp)  # WRF:2908
        _put(v,'cni',((cc)-1,),xp.power(v['ao'], _r(1.0) - _get(v['deltastr'], (cc - 1,))) * xp.power(_get(v['ani'], (cc - 1,)), _get(v['deltastr'], (cc - 1,))),mask_8,xp)  # WRF:2909
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_8,xp)  # WRF:2910
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_8,xp)  # WRF:2911
        _put(v,'ni',((cc)-1,),_get(v['qi'], (cc - 1,)) / (_get(v['rhobar'], (cc - 1,)) * v['fourthirdspi'] * xp.power(v['ao'], _r(1.0) - _get(v['deltastr'], (cc - 1,))) * xp.power(_get(v['ani'], (cc - 1,)), _r(2.0) + _get(v['deltastr'], (cc - 1,))) * tables['gamma_tab'][v['gi'] - 1,] / v['gammnu']),mask_8,xp)  # WRF:2912
        _put(v,'ni',((cc)-1,),_max(_get(v['ni'], (cc - 1,)), v['qnsmall']),mask_8,xp)  # WRF:2914
        _put(v,'ai',((cc)-1,),_get(v['ani'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_4,xp)  # WRF:2917
        _put(v,'ci',((cc)-1,),_get(v['cni'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_4,xp)  # WRF:2918
        call_out=_call('var_check',(v['nu'],v['ao'],v['fourthirdspi'],v['gammnu'],_get(v['qi'], (cc - 1,)),_get(v['deltastr'], (cc - 1,)),_get(v['ani'], (cc - 1,)),_get(v['cni'], (cc - 1,)),_get(v['rni'], (cc - 1,)),_get(v['rhobar'], (cc - 1,)),_get(v['ni'], (cc - 1,)),_get(v['ai'], (cc - 1,)),_get(v['ci'], (cc - 1,)),v['alphstr'],v['alphv'],v['betam'],),tables,xp)  # WRF:2920
        _put(v,'deltastr',((cc)-1,),call_out[0],mask_4,xp)  # WRF:2920
        _put(v,'ani',((cc)-1,),call_out[1],mask_4,xp)  # WRF:2920
        _put(v,'cni',((cc)-1,),call_out[2],mask_4,xp)  # WRF:2920
        _put(v,'rni',((cc)-1,),call_out[3],mask_4,xp)  # WRF:2920
        _put(v,'rhobar',((cc)-1,),call_out[4],mask_4,xp)  # WRF:2920
        _put(v,'ni',((cc)-1,),call_out[5],mask_4,xp)  # WRF:2920
        _put(v,'ai',((cc)-1,),call_out[6],mask_4,xp)  # WRF:2920
        _put(v,'ci',((cc)-1,),call_out[7],mask_4,xp)  # WRF:2920
        _put(v,'alphstr',None,call_out[8],mask_4,xp)  # WRF:2920
        _put(v,'alphv',None,call_out[9],mask_4,xp)  # WRF:2920
        _put(v,'betam',None,call_out[10],mask_4,xp)  # WRF:2920
        call_out=_call('var_check',(v['nu'],v['ao'],v['fourthirdspi'],v['gammnu'],_get(v['qi'], (cc - 1,)),_get(v['deltastr'], (cc - 1,)),_get(v['ani'], (cc - 1,)),_get(v['cni'], (cc - 1,)),_get(v['rni'], (cc - 1,)),_get(v['rhobar'], (cc - 1,)),_get(v['ni'], (cc - 1,)),_get(v['ai'], (cc - 1,)),_get(v['ci'], (cc - 1,)),v['alphstr'],v['alphv'],v['betam'],),tables,xp)  # WRF:2926
        _put(v,'deltastr',((cc)-1,),call_out[0],mask_1,xp)  # WRF:2926
        _put(v,'ani',((cc)-1,),call_out[1],mask_1,xp)  # WRF:2926
        _put(v,'cni',((cc)-1,),call_out[2],mask_1,xp)  # WRF:2926
        _put(v,'rni',((cc)-1,),call_out[3],mask_1,xp)  # WRF:2926
        _put(v,'rhobar',((cc)-1,),call_out[4],mask_1,xp)  # WRF:2926
        _put(v,'ni',((cc)-1,),call_out[5],mask_1,xp)  # WRF:2926
        _put(v,'ai',((cc)-1,),call_out[6],mask_1,xp)  # WRF:2926
        _put(v,'ci',((cc)-1,),call_out[7],mask_1,xp)  # WRF:2926
        _put(v,'alphstr',None,call_out[8],mask_1,xp)  # WRF:2926
        _put(v,'alphv',None,call_out[9],mask_1,xp)  # WRF:2926
        _put(v,'betam',None,call_out[10],mask_1,xp)  # WRF:2926
        mask_9=xp.logical_and(True,xp.logical_not(taken_1))
        _put(v,'qv',None,v['qv'] + _get(v['qi'], (cc - 1,)),mask_9,xp)  # WRF:2931
        _put(v,'theta',None,v['theta'] + v['theta'] * v['i_temp'] * (_get(v['qi'], (cc - 1,)) * v['xxls']) * v['i_cp'],mask_9,xp)  # WRF:2932
        _put(v,'qi',((cc)-1,),_r(0.0),mask_9,xp)  # WRF:2933
        _put(v,'ni',((cc)-1,),_r(0.0),mask_9,xp)  # WRF:2934
        _put(v,'ai',((cc)-1,),_r(0.0),mask_9,xp)  # WRF:2935
        _put(v,'ci',((cc)-1,),_r(0.0),mask_9,xp)  # WRF:2936
    mask_10=xp.logical_and(True,v['qr'] > v['qsmall'])
    taken_10=mask_10
    _put(v,'nr',None,_max(v['nr'], v['qnsmall']),mask_10,xp)  # WRF:2941
    _put(v,'lamr',None,xp.power(v['pi'] * v['rhow'] * v['nr'] / v['qr'], _r(0.333333333)),mask_10,xp)  # WRF:2942
    _put(v,'n0rr',None,v['nr'] * v['lamr'],mask_10,xp)  # WRF:2943
    mask_11=xp.logical_and(mask_10,v['lamr'] < v['lamminr'])
    taken_11=mask_11
    _put(v,'lamr',None,v['lamminr'],mask_11,xp)  # WRF:2945
    _put(v,'n0rr',None,_r(16269267968.0 if v['qv'].dtype == 'float32' else 16269262807.163687) * v['qr'] / (v['pi'] * v['rhow']),mask_11,xp)  # WRF:2946
    _put(v,'nr',None,v['n0rr'] / v['lamr'],mask_11,xp)  # WRF:2947
    mask_12=xp.logical_and(xp.logical_and(mask_10,xp.logical_not(taken_11)),v['lamr'] > v['lammaxr'])
    taken_11=xp.logical_or(taken_11,mask_12)
    _put(v,'lamr',None,v['lammaxr'],mask_12,xp)  # WRF:2949
    _put(v,'n0rr',None,_r(6.250000125255483e+18 if v['qv'].dtype == 'float32' else 6.249999999999996e+18) * v['qr'] / (v['pi'] * v['rhow']),mask_12,xp)  # WRF:2950
    _put(v,'nr',None,v['n0rr'] / v['lamr'],mask_12,xp)  # WRF:2951
    mask_13=xp.logical_and(True,xp.logical_not(taken_10))
    _put(v,'qv',None,v['qv'] + v['qr'],mask_13,xp)  # WRF:2954
    _put(v,'theta',None,v['theta'] + v['theta'] * v['i_temp'] * (v['qr'] * v['xxlv']) * v['i_cp'],mask_13,xp)  # WRF:2955
    _put(v,'qr',None,_r(0.0),mask_13,xp)  # WRF:2956
    _put(v,'nr',None,_r(0.0),mask_13,xp)  # WRF:2957
    return v
