"""Generated original Jensen processes; see lane port_levels proof. Not yet qualified."""
from .level_support import *  # noqa: F403


def processes(fields,tables,*,xp=jnp,observer=None):
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
    v.setdefault('a_demott',_initial(v["qv"],'real',(),xp))
    v.setdefault('aa',_initial(v["qv"],'real',(),xp))
    v.setdefault('ab',_initial(v["qv"],'real',(),xp))
    v.setdefault('abi',_initial(v["qv"],'real',(),xp))
    v.setdefault('ai',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('al',_initial(v["qv"],'real',(),xp))
    v.setdefault('alphstr',_initial(v["qv"],'real',(),xp))
    v.setdefault('alphv',_initial(v["qv"],'real',(),xp))
    v.setdefault('am',_initial(v["qv"],'real',(),xp))
    v.setdefault('amass',_initial(v["qv"],'real',(),xp))
    v.setdefault('amlt',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('anf',_initial(v["qv"],'real',(),xp))
    v.setdefault('anfr',_initial(v["qv"],'real',(),xp))
    v.setdefault('ani',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('aniold',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('anuc',_initial(v["qv"],'real',(),xp))
    v.setdefault('ard',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('ardr',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('arn',_initial(v["qv"],'real',(),xp))
    v.setdefault('b_demott',_initial(v["qv"],'real',(),xp))
    v.setdefault('ba',_initial(v["qv"],'real',(),xp))
    v.setdefault('betam',_initial(v["qv"],'real',(),xp))
    v.setdefault('bl',_initial(v["qv"],'real',(),xp))
    v.setdefault('bm',_initial(v["qv"],'real',(),xp))
    v.setdefault('bx',_initial(v["qv"],'real',(),xp))
    v.setdefault('c_demott',_initial(v["qv"],'real',(),xp))
    v.setdefault('capgam',_initial(v["qv"],'real',(),xp))
    v.setdefault('ci',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('cmass',_initial(v["qv"],'real',(),xp))
    v.setdefault('cmlt',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('cnf',_initial(v["qv"],'real',(),xp))
    v.setdefault('cnfr',_initial(v["qv"],'real',(),xp))
    v.setdefault('cni',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('cniold',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('crd',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('crdr',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('curnum',_initial(v["qv"],'real',(),xp))
    v.setdefault('d_demott',_initial(v["qv"],'real',(),xp))
    v.setdefault('dbz1d',_initial(v["qv"],'real',(),xp))
    v.setdefault('dbzr',_initial(v["qv"],'real',(),xp))
    v.setdefault('dbzsum',_initial(v["qv"],'real',(),xp))
    v.setdefault('deltastr',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('di1d',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('dn1',_initial(v["qv"],'real',(),xp))
    v.setdefault('dn2',_initial(v["qv"],'real',(),xp))
    v.setdefault('dn3',_initial(v["qv"],'real',(),xp))
    v.setdefault('dnfzri',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('dnmltri',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('domicro',_initial(v["qv"],'logical',(),xp))
    v.setdefault('dqifzri',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('dqimltri',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('dqrfzri',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('dry_growth',_initial(v["qv"],'logical',(3,),xp))
    v.setdefault('dsdepout',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('dsold',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('dum',_initial(v["qv"],'real',(),xp))
    v.setdefault('dum1',_initial(v["qv"],'real',(),xp))
    v.setdefault('dumqv',_initial(v["qv"],'real',(),xp))
    v.setdefault('dumqvs',_initial(v["qv"],'real',(),xp))
    v.setdefault('dumt',_initial(v["qv"],'real',(),xp))
    v.setdefault('dv',_initial(v["qv"],'real',(),xp))
    v.setdefault('effc1d',_initial(v["qv"],'real',(),xp))
    v.setdefault('effi',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('effi1d',_initial(v["qv"],'real',(),xp))
    v.setdefault('epsr',_initial(v["qv"],'real',(),xp))
    v.setdefault('f_c1',_initial(v["qv"],'real',(),xp))
    v.setdefault('f_c2',_initial(v["qv"],'real',(),xp))
    v.setdefault('falltndai',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('falltndci',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('falltndni',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('falltndnr',_initial(v["qv"],'real',(),xp))
    v.setdefault('falltndqi',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('falltndqr',_initial(v["qv"],'real',(),xp))
    v.setdefault('fh',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('fluxai',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('fluxci',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('fluxni',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('fluxnr',_initial(v["qv"],'real',(),xp))
    v.setdefault('fluxqi',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('fluxqr',_initial(v["qv"],'real',(),xp))
    v.setdefault('fmult',_initial(v["qv"],'real',(),xp))
    v.setdefault('fourthirdspi',_initial(v["qv"],'real',(),xp))
    v.setdefault('fv',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('gamma_arg',_initial(v["qv"],'real',(),xp))
    v.setdefault('gammnu',_initial(v["qv"],'real',(),xp))
    v.setdefault('gdenavg',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('gdenavgr',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('gdentotal',_initial(v["qv"],'real',(),xp))
    v.setdefault('gi',_initial(v["qv"],'integer',(),xp))
    v.setdefault('gi2',_initial(v["qv"],'integer',(),xp))
    v.setdefault('gi3',_initial(v["qv"],'integer',(),xp))
    v.setdefault('has_ice',_initial(v["qv"],'logical',(3,),xp))
    v.setdefault('i_cp',_initial(v["qv"],'real',(),xp))
    v.setdefault('i_dt',_initial(v["qv"],'real',(),xp))
    v.setdefault('i_gammnu',_initial(v["qv"],'real',(),xp))
    v.setdefault('i_rhoair',_initial(v["qv"],'real',(),xp))
    v.setdefault('i_temp',_initial(v["qv"],'real',(),xp))
    v.setdefault('icerateri',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('icetype1d',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('idsi',_initial(v["qv"],'integer',(),xp))
    v.setdefault('igr',_initial(v["qv"],'real',(),xp))
    v.setdefault('inrate',_initial(v["qv"],'real',(),xp))
    v.setdefault('iqci',_initial(v["qv"],'integer',(),xp))
    v.setdefault('irho',_initial(v["qv"],'integer',(),xp))
    v.setdefault('irni',_initial(v["qv"],'integer',(),xp))
    v.setdefault('irri',_initial(v["qv"],'integer',(),xp))
    v.setdefault('iwcf',_initial(v["qv"],'real',(),xp))
    v.setdefault('iwcfr',_initial(v["qv"],'real',(),xp))
    v.setdefault('iwci',_initial(v["qv"],'real',(),xp))
    v.setdefault('kt',_initial(v["qv"],'real',(),xp))
    v.setdefault('lamr',_initial(v["qv"],'real',(),xp))
    v.setdefault('lrsig',_initial(v["qv"],'real',(),xp))
    v.setdefault('lwc',_initial(v["qv"],'real',(),xp))
    v.setdefault('mbiggr',_initial(v["qv"],'real',(),xp))
    v.setdefault('mim',_initial(v["qv"],'real',(),xp))
    v.setdefault('mimr',_initial(v["qv"],'real',(),xp))
    v.setdefault('mnuccd',_initial(v["qv"],'real',(),xp))
    v.setdefault('mu',_initial(v["qv"],'real',(),xp))
    v.setdefault('n0rr',_initial(v["qv"],'real',(),xp))
    v.setdefault('nagg',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('nbiggr',_initial(v["qv"],'real',(),xp))
    v.setdefault('nc',_initial(v["qv"],'real',(),xp))
    v.setdefault('ncm3dum',_initial(v["qv"],'real',(),xp))
    v.setdefault('ni',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('nibnuc',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('nim',_initial(v["qv"],'real',(),xp))
    v.setdefault('nim3dum',_initial(v["qv"],'real',(),xp))
    v.setdefault('nimr',_initial(v["qv"],'real',(),xp))
    v.setdefault('nmlt',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('nmult',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('nnuccd',_initial(v["qv"],'real',(),xp))
    v.setdefault('npr',_initial(v["qv"],'real',(),xp))
    v.setdefault('npra',_initial(v["qv"],'real',(),xp))
    v.setdefault('nprc',_initial(v["qv"],'real',(),xp))
    v.setdefault('nprc1',_initial(v["qv"],'real',(),xp))
    v.setdefault('npre',_initial(v["qv"],'real',(),xp))
    v.setdefault('nr',_initial(v["qv"],'real',(),xp))
    v.setdefault('nragg',_initial(v["qv"],'real',(),xp))
    v.setdefault('nrd',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('nre',_initial(v["qv"],'real',(),xp))
    v.setdefault('nsch',_initial(v["qv"],'real',(),xp))
    v.setdefault('nucfrac',_initial(v["qv"],'real',(),xp))
    v.setdefault('numice',_initial(v["qv"],'integer',(),xp))
    v.setdefault('numrateri',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('pcc',_initial(v["qv"],'real',(),xp))
    v.setdefault('phiagg1',_initial(v["qv"],'real',(),xp))
    v.setdefault('phiagg2',_initial(v["qv"],'real',(),xp))
    v.setdefault('phiagg3',_initial(v["qv"],'real',(),xp))
    v.setdefault('phibr',_initial(v["qv"],'real',(),xp))
    v.setdefault('phifr',_initial(v["qv"],'real',(),xp))
    v.setdefault('phii1d',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('phiivt',_initial(v["qv"],'real',(),xp))
    v.setdefault('phirad',_initial(v["qv"],'real',(),xp))
    v.setdefault('pra',_initial(v["qv"],'real',(),xp))
    v.setdefault('prc',_initial(v["qv"],'real',(),xp))
    v.setdefault('prd',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('prdr',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('prdsum',_initial(v["qv"],'real',(),xp))
    v.setdefault('pre',_initial(v["qv"],'real',(),xp))
    v.setdefault('proc',_initial(v["qv"],'real',(3, 2),xp))
    v.setdefault('procr',_initial(v["qv"],'real',(3, 6),xp))
    v.setdefault('qagg',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('qc',_initial(v["qv"],'real',(),xp))
    v.setdefault('qcrimefrac',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('qcrimesum',_initial(v["qv"],'real',(),xp))
    v.setdefault('qcrimesumr',_initial(v["qv"],'real',(),xp))
    v.setdefault('qe',_initial(v["qv"],'real',(),xp))
    v.setdefault('qi',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('qi_qc_nrd',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('qi_qc_nrm',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('qi_qr_nrd',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('qi_qr_nrm',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('qi_qr_nrn',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('qipre1d',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('qmlt',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('qmult',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('qr',_initial(v["qv"],'real',(),xp))
    v.setdefault('qrpre1d',_initial(v["qv"],'real',(),xp))
    v.setdefault('qs0',_initial(v["qv"],'real',(),xp))
    v.setdefault('qv',_initial(v["qv"],'real',(),xp))
    v.setdefault('qvi',_initial(v["qv"],'real',(),xp))
    v.setdefault('qvs',_initial(v["qv"],'real',(),xp))
    v.setdefault('r_n',_initial(v["qv"],'real',(),xp))
    v.setdefault('radw1',_initial(v["qv"],'real',(),xp))
    v.setdefault('radw2',_initial(v["qv"],'real',(),xp))
    v.setdefault('radw3',_initial(v["qv"],'real',(),xp))
    v.setdefault('rainrateri',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('ratekg',_initial(v["qv"],'real',(),xp))
    v.setdefault('ratel',_initial(v["qv"],'real',(),xp))
    v.setdefault('ratio',_initial(v["qv"],'real',(),xp))
    v.setdefault('rdsi',_initial(v["qv"],'real',(),xp))
    v.setdefault('rhoair',_initial(v["qv"],'real',(),xp))
    v.setdefault('rhobar',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('rhobarrime',_initial(v["qv"],'real',(),xp))
    v.setdefault('rhodepout',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('rhopo1d',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('rhorimeout',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('rimec1',_initial(v["qv"],'real',(),xp))
    v.setdefault('rimedr',_initial(v["qv"],'real',(),xp))
    v.setdefault('rimedrr',_initial(v["qv"],'real',(),xp))
    v.setdefault('rimesum',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('rimesumr',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('rimetotal',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('rnf',_initial(v["qv"],'real',(),xp))
    v.setdefault('rnfr',_initial(v["qv"],'real',(),xp))
    v.setdefault('rni',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('rqci',_initial(v["qv"],'real',(),xp))
    v.setdefault('rrho',_initial(v["qv"],'real',(),xp))
    v.setdefault('rrni',_initial(v["qv"],'real',(),xp))
    v.setdefault('rrr',_initial(v["qv"],'real',(),xp))
    v.setdefault('rrri',_initial(v["qv"],'real',(),xp))
    v.setdefault('sedi',_initial(v["qv"],'logical',(),xp))
    v.setdefault('sig',_initial(v["qv"],'real',(),xp))
    v.setdefault('sink',_initial(v["qv"],'real',(),xp))
    v.setdefault('source',_initial(v["qv"],'real',(),xp))
    v.setdefault('sui',_initial(v["qv"],'real',(),xp))
    v.setdefault('sup',_initial(v["qv"],'real',(),xp))
    v.setdefault('svpi',_initial(v["qv"],'real',(),xp))
    v.setdefault('svpl',_initial(v["qv"],'real',(),xp))
    v.setdefault('temp',_initial(v["qv"],'real',(),xp))
    v.setdefault('theta',_initial(v["qv"],'real',(),xp))
    v.setdefault('tmpmelt',_initial(v["qv"],'real',(),xp))
    v.setdefault('tmpsum',_initial(v["qv"],'real',(),xp))
    v.setdefault('tmpsumr',_initial(v["qv"],'real',(),xp))
    v.setdefault('totaliwc',_initial(v["qv"],'real',(),xp))
    v.setdefault('totalnuc',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('totalnucn',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('vfr',_initial(v["qv"],'real',(),xp))
    v.setdefault('vgflag',_initial(v["qv"],'logical',(),xp))
    v.setdefault('vi',_initial(v["qv"],'real',(),xp))
    v.setdefault('vir',_initial(v["qv"],'real',(),xp))
    v.setdefault('vmi1d',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('vtrm',_initial(v["qv"],'real',(),xp))
    v.setdefault('vtrmc',_initial(v["qv"],'real',(),xp))
    v.setdefault('vtrmi1',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('vtrn',_initial(v["qv"],'real',(),xp))
    v.setdefault('vtrnc',_initial(v["qv"],'real',(),xp))
    v.setdefault('vtrni1',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('vtrzi1',_initial(v["qv"],'real',(3,),xp))
    v.setdefault('xm',_initial(v["qv"],'real',(),xp))
    v.setdefault('xn',_initial(v["qv"],'real',(),xp))
    v.setdefault('xxlf',_initial(v["qv"],'real',(),xp))
    v.setdefault('xxls',_initial(v["qv"],'real',(),xp))
    v.setdefault('xxlv',_initial(v["qv"],'real',(),xp))
    _put(v,'gammnu',None,gamma(v['nu'], xp=xp),True,xp)  # WRF:684
    _put(v,'i_gammnu',None,_r(1.0) / v['gammnu'],True,xp)  # WRF:685
    _put(v,'i_dt',None,_r(1.0) / v['dt'],True,xp)  # WRF:686
    _put(v,'i_cp',None,_r(1.0) / v['cp'],True,xp)  # WRF:687
    _put(v,'fourthirdspi',None,_r(4.0) / _r(3.0) * v['pi'],True,xp)  # WRF:688
    _put(v,'sedi',None,False,True,xp)  # WRF:692
    _put(v,'domicro',None,False,True,xp)  # WRF:700
    _put(v,'temp',None,v['theta'] / xp.power(_r(100000.0) / v['pres_e'], v['rcp']),True,xp)  # WRF:701
    _put(v,'svpl',None,polysvp(v['temp'], 0, xp=xp),True,xp)  # WRF:702
    _put(v,'qvs',None,_r(0.622) * v['svpl'] / (v['pres_e'] - v['svpl']),True,xp)  # WRF:703
    _put(v,'sup',None,v['qv'] / v['qvs'] - _r(1.0),True,xp)  # WRF:704
    _put(v,'svpi',None,polysvp(v['temp'], 1, xp=xp),True,xp)  # WRF:706
    _put(v,'qvi',None,_r(0.622) * v['svpi'] / (v['pres_e'] - v['svpi']),True,xp)  # WRF:707
    _put(v,'qvi',None,v['qvs'],xp.logical_and(True,v['temp'] > v['t0']),xp)  # WRF:708
    _put(v,'sui',None,v['qv'] / v['qvi'] - _r(1.0),True,xp)  # WRF:709
    mask_1=xp.logical_and(True,xp.logical_or(xp.logical_or(xp.logical_or(xp.logical_or(xp.logical_or(v['qc'] > v['qsmall'], v['qr'] > v['qsmall']), _get(v['qi'], (ice1 - 1,)) > v['qsmall']), _get(v['qi'], (ice2 - 1,)) > v['qsmall']), _get(v['qi'], (ice3 - 1,)) > v['qsmall']), v['sup'] >= _r(0.0)))
    taken_1=mask_1
    _put(v,'domicro',None,True,mask_1,xp)  # WRF:713
    mask_2=xp.logical_and(True,v['domicro'])
    taken_2=mask_2
    _put(v,'numice',None,0,mask_2,xp)  # WRF:721
    for cc in range(1,cat+1):
        _put(v,'has_ice',((cc)-1,),False,mask_2,xp)  # WRF:723
        mask_3=xp.logical_and(mask_2,_get(v['qi'], (cc - 1,)) > v['qsmall'])
        taken_3=mask_3
        _put(v,'has_ice',((cc)-1,),True,mask_3,xp)  # WRF:727
        _put(v,'numice',None,v['numice'] + 1,mask_3,xp)  # WRF:728
    for cc in range(1,cat+1):
        _put(v,'dry_growth',((cc)-1,),True,mask_2,xp)  # WRF:747
        _put(v,'qcrimefrac',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:748
        _put(v,'rimetotal',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:749
        _put(v,'nmult',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:750
        _put(v,'qmult',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:751
        _put(v,'ani',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:752
        _put(v,'cni',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:753
        _put(v,'rni',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:754
        _put(v,'qagg',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:755
        _put(v,'nagg',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:756
        _put(v,'prd',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:757
        _put(v,'nrd',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:758
        _put(v,'ard',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:759
        _put(v,'crd',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:760
        _put(v,'prdr',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:761
        _put(v,'ardr',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:762
        _put(v,'crdr',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:763
        _put(v,'qmlt',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:764
        _put(v,'nmlt',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:765
        _put(v,'amlt',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:766
        _put(v,'cmlt',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:767
        _put(v,'deltastr',((cc)-1,),_r(1.0),mask_2,xp)  # WRF:768
        _put(v,'dsold',((cc)-1,),_r(1.0),mask_2,xp)  # WRF:769
        _put(v,'dsdepout',((cc)-1,),_r(1.0),mask_2,xp)  # WRF:770
        _put(v,'rhobar',((cc)-1,),v['rhoi'],mask_2,xp)  # WRF:771
        _put(v,'rhodepout',((cc)-1,),v['rhoi'],mask_2,xp)  # WRF:772
        _put(v,'rhorimeout',((cc)-1,),v['rhoi'],mask_2,xp)  # WRF:773
        _put(v,'rhobar',((cc)-1,),_r(50.0),xp.logical_and(mask_2,cc == 3),xp)  # WRF:774
        _put(v,'fv',((cc)-1,),_r(1.0),mask_2,xp)  # WRF:775
        _put(v,'fh',((cc)-1,),_r(1.0),mask_2,xp)  # WRF:776
        _put(v,'rimesum',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:777
        _put(v,'rimesumr',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:778
        _put(v,'numrateri',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:779
        _put(v,'rainrateri',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:780
        _put(v,'icerateri',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:781
        _put(v,'dqrfzri',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:782
        _put(v,'dqifzri',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:783
        _put(v,'dnfzri',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:784
        _put(v,'dqimltri',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:785
        _put(v,'dnmltri',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:786
        _put(v,'gdenavg',((cc)-1,),_r(400.0),mask_2,xp)  # WRF:787
        _put(v,'gdenavgr',((cc)-1,),_r(400.0),mask_2,xp)  # WRF:788
        _put(v,'totalnuc',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:789
        _put(v,'totalnucn',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:790
        _put(v,'nibnuc',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:791
        _put(v,'aniold',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:792
        _put(v,'cniold',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:793
        _put(v,'proc',((cc)-1,(1)-1,),_r(0.0),mask_2,xp)  # WRF:794
        _put(v,'proc',((cc)-1,(2)-1,),_r(0.0),mask_2,xp)  # WRF:795
        _put(v,'qi_qc_nrm',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:796
        _put(v,'qi_qc_nrd',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:797
        _put(v,'qi_qr_nrm',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:798
        _put(v,'qi_qr_nrn',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:799
        _put(v,'qi_qr_nrd',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:800
        _put(v,'procr',((cc)-1,(1)-1,),_r(0.0),mask_2,xp)  # WRF:801
        _put(v,'procr',((cc)-1,(2)-1,),_r(0.0),mask_2,xp)  # WRF:802
        _put(v,'procr',((cc)-1,(3)-1,),_r(0.0),mask_2,xp)  # WRF:803
        _put(v,'procr',((cc)-1,(4)-1,),_r(0.0),mask_2,xp)  # WRF:804
        _put(v,'procr',((cc)-1,(5)-1,),_r(0.0),mask_2,xp)  # WRF:805
        _put(v,'procr',((cc)-1,(6)-1,),_r(0.0),mask_2,xp)  # WRF:806
        _put(v,'fluxqi',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:807
        _put(v,'fluxni',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:808
        _put(v,'fluxai',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:809
        _put(v,'fluxci',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:810
        _put(v,'falltndqi',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:811
        _put(v,'falltndni',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:812
        _put(v,'falltndai',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:813
        _put(v,'falltndci',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:814
        _put(v,'di1d',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:815
        _put(v,'vmi1d',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:816
        _put(v,'vtrmi1',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:817
        _put(v,'vtrni1',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:818
        _put(v,'vtrzi1',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:819
        _put(v,'rhopo1d',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:820
        _put(v,'phii1d',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:821
        _put(v,'qipre1d',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:822
        _put(v,'effi',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:823
        _put(v,'icetype1d',((cc)-1,),_r(0.0),mask_2,xp)  # WRF:824
    _put(v,'dbz1d',None,_r(-35.0),mask_2,xp)  # WRF:826
    _put(v,'qrpre1d',None,_r(0.0),mask_2,xp)  # WRF:827
    _put(v,'fluxqr',None,_r(0.0),mask_2,xp)  # WRF:828
    _put(v,'fluxnr',None,_r(0.0),mask_2,xp)  # WRF:829
    _put(v,'falltndqr',None,_r(0.0),mask_2,xp)  # WRF:830
    _put(v,'falltndnr',None,_r(0.0),mask_2,xp)  # WRF:830
    _put(v,'vtrm',None,_r(0.0),mask_2,xp)  # WRF:831
    _put(v,'vtrn',None,_r(0.0),mask_2,xp)  # WRF:831
    _put(v,'effc1d',None,_r(0.0),mask_2,xp)  # WRF:832
    _put(v,'effi1d',None,_r(0.0),mask_2,xp)  # WRF:833
    _put(v,'nprc',None,_r(0.0),mask_2,xp)  # WRF:834
    _put(v,'npre',None,_r(0.0),mask_2,xp)  # WRF:834
    _put(v,'npra',None,_r(0.0),mask_2,xp)  # WRF:834
    _put(v,'pra',None,_r(0.0),mask_2,xp)  # WRF:834
    _put(v,'prc',None,_r(0.0),mask_2,xp)  # WRF:834
    _put(v,'nprc1',None,_r(0.0),mask_2,xp)  # WRF:835
    _put(v,'nragg',None,_r(0.0),mask_2,xp)  # WRF:835
    _put(v,'nim',None,_r(0.0),mask_2,xp)  # WRF:836
    _put(v,'mim',None,_r(0.0),mask_2,xp)  # WRF:836
    _put(v,'nimr',None,_r(0.0),mask_2,xp)  # WRF:837
    _put(v,'mimr',None,_r(0.0),mask_2,xp)  # WRF:837
    _put(v,'mbiggr',None,_r(0.0),mask_2,xp)  # WRF:838
    _put(v,'nbiggr',None,_r(0.0),mask_2,xp)  # WRF:838
    _put(v,'mnuccd',None,_r(0.0),mask_2,xp)  # WRF:839
    _put(v,'nnuccd',None,_r(0.0),mask_2,xp)  # WRF:839
    _put(v,'pcc',None,_r(0.0),mask_2,xp)  # WRF:839
    _put(v,'fmult',None,_r(0.0),mask_2,xp)  # WRF:839
    _put(v,'gdentotal',None,_r(400.0),mask_2,xp)  # WRF:840
    _put(v,'rimedr',None,_r(0.0),mask_2,xp)  # WRF:841
    _put(v,'rimedrr',None,_r(0.0),mask_2,xp)  # WRF:841
    _put(v,'dbzr',None,_r(0.0),mask_2,xp)  # WRF:842
    _put(v,'dbzsum',None,_r(0.0),mask_2,xp)  # WRF:842
    _put(v,'i_temp',None,_r(1.0) / v['temp'],mask_2,xp)  # WRF:847
    _put(v,'rhoair',None,v['pres_e'] / (v['rd'] * v['temp']),mask_2,xp)  # WRF:848
    _put(v,'i_rhoair',None,_r(1.0) / v['rhoair'],mask_2,xp)  # WRF:849
    _put(v,'svpi',None,polysvp(v['temp'], 1, xp=xp),mask_2,xp)  # WRF:850
    _put(v,'qvi',None,_r(0.622) * v['svpi'] / (v['pres_e'] - v['svpi']),mask_2,xp)  # WRF:851
    _put(v,'qvi',None,v['qvs'],xp.logical_and(mask_2,v['temp'] > v['t0']),xp)  # WRF:852
    _put(v,'qs0',None,_r(0.622) * polysvp(v['t0'], 0, xp=xp) / (v['pres_e'] - polysvp(v['t0'], 0, xp=xp)),mask_2,xp)  # WRF:853
    _put(v,'sui',None,v['qv'] / v['qvi'] - _r(1.0),mask_2,xp)  # WRF:854
    _put(v,'xxls',None,_r(3150000.0) - _r(2370.0) * v['temp'] + _r(333700.0),mask_2,xp)  # WRF:855
    _put(v,'xxlv',None,_r(3148400.0) - _r(2370.0) * v['temp'],mask_2,xp)  # WRF:856
    _put(v,'xxlf',None,v['xxls'] - v['xxlv'],mask_2,xp)  # WRF:857
    _put(v,'ab',None,_r(1.0) + v['xxlv'] * v['qvs'] / (v['rv'] * (v['temp'] * v['temp'])) * v['xxlv'] * v['i_cp'],mask_2,xp)  # WRF:858
    _put(v,'abi',None,_r(1.0) + v['xxls'] * v['qvi'] / (v['rv'] * (v['temp'] * v['temp'])) * v['xxls'] * v['i_cp'],mask_2,xp)  # WRF:859
    _put(v,'mu',None,_r(1.496e-06) * xp.power(v['temp'], _r(1.5)) / (v['temp'] + _r(120.0)),mask_2,xp)  # WRF:860
    _put(v,'arn',None,v['ar'] * xp.power(v['r0'] * v['i_rhoair'], _r(0.5)),mask_2,xp)  # WRF:861
    _put(v,'dv',None,_r(8.794e-05) * xp.power(v['temp'], _r(1.81)) / v['pres_e'],mask_2,xp)  # WRF:862
    _put(v,'kt',None,_r(0.023823) + _r(7.1177e-05) * (v['temp'] - v['t0']),mask_2,xp)  # WRF:863
    _put(v,'nsch',None,v['mu'] * v['i_rhoair'] / v['dv'],mask_2,xp)  # WRF:864
    _put(v,'npr',None,v['mu'] * v['i_rhoair'] / v['kt'],mask_2,xp)  # WRF:865
    _put(v,'igr',None,get_igr(v['temp'], data=tables['igrdata'], xp=xp),mask_2,xp)  # WRF:870
    mask_4=xp.logical_and(mask_2,v['temp'] - v['t0'] < _r(-20.0))
    taken_4=mask_4
    _put(v,'igr',None,_r(0.7),mask_4,xp)  # WRF:872
    _put(v,'nc',None,_r(200000000.0) * v['i_rhoair'],mask_2,xp)  # WRF:876
    if observer is not None: observer(0,v,mask_2)  # Original checkpoint WRF:882
    mask_5=xp.logical_and(mask_2,v['numice'] > 0)
    taken_5=mask_5
    _put(v,'tmpsum',None,_r(0.0),mask_5,xp)  # WRF:883
    _put(v,'tmpsumr',None,_r(0.0),mask_5,xp)  # WRF:884
    v["ccvar"]=xp.zeros_like(v["numice"])
    for cc in range(1,4):
        mask_6=xp.logical_and(mask_5,_get(v["has_ice"],(cc-1,)))
        v["ccvar"]=xp.where(mask_6,v["ccvar"]+1,v["ccvar"])  # Original active-list ordinal
        _put(v,'ni',((cc)-1,),_max(_get(v['ni'], (cc - 1,)), v['qnsmall']),mask_6,xp)  # WRF:889
        _put(v,'ai',((cc)-1,),_max(_get(v['ai'], (cc - 1,)), v['qasmall']),mask_6,xp)  # WRF:890
        _put(v,'ci',((cc)-1,),_max(_get(v['ci'], (cc - 1,)), v['qasmall']),mask_6,xp)  # WRF:891
        _put(v,'ani',((cc)-1,),_max(xp.power(_get(v['ai'], (cc - 1,)) * _get(v['ai'], (cc - 1,)) / (_get(v['ci'], (cc - 1,)) * _get(v['ni'], (cc - 1,))), _r(0.333333333333)), _r(2e-06)),mask_6,xp)  # WRF:892
        _put(v,'cni',((cc)-1,),_max(xp.power(_get(v['ci'], (cc - 1,)) * _get(v['ci'], (cc - 1,)) / (_get(v['ai'], (cc - 1,)) * _get(v['ni'], (cc - 1,))), _r(0.333333333333)), _r(2e-06)),mask_6,xp)  # WRF:893
        mask_7=xp.logical_and(mask_6,xp.logical_or(_get(v['ai'], (cc - 1,)) < _r(1e-12), _get(v['ci'], (cc - 1,)) < _r(1e-12)))
        taken_7=mask_7
        _put(v,'ai',((cc)-1,),_min(_get(v['ai'], (cc - 1,)), _get(v['ci'], (cc - 1,))),mask_7,xp)  # WRF:899
        _put(v,'ci',((cc)-1,),_get(v['ai'], (cc - 1,)),mask_7,xp)  # WRF:900
        _put(v,'ani',((cc)-1,),xp.power(_get(v['ai'], (cc - 1,)) / _get(v['ni'], (cc - 1,)), _r(0.3333333333)),mask_7,xp)  # WRF:901
        _put(v,'cni',((cc)-1,),xp.power(_get(v['ci'], (cc - 1,)) / _get(v['ni'], (cc - 1,)), _r(0.3333333333)),mask_7,xp)  # WRF:902
        mask_8=xp.logical_and(mask_6,xp.logical_or(_get(v['ani'], (cc - 1,)) < _r(2e-06), _get(v['cni'], (cc - 1,)) < _r(2e-06)))
        taken_8=mask_8
        _put(v,'ani',((cc)-1,),_r(2e-06),mask_8,xp)  # WRF:906
        _put(v,'cni',((cc)-1,),_r(2e-06),mask_8,xp)  # WRF:907
        _put(v,'ci',((cc)-1,),_get(v['cni'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_6,xp)  # WRF:910
        _put(v,'ai',((cc)-1,),_get(v['ani'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_6,xp)  # WRF:911
        _put(v,'deltastr',((cc)-1,),(xp.log(_get(v['cni'], (cc - 1,))) - xp.log(v['ao'])) / (xp.log(_get(v['ani'], (cc - 1,))) - xp.log(v['ao'])),mask_6,xp)  # WRF:913
        call_out=_call('var_check',(v['nu'],v['ao'],v['fourthirdspi'],v['gammnu'],_get(v['qi'], (cc - 1,)),_get(v['deltastr'], (cc - 1,)),_get(v['ani'], (cc - 1,)),_get(v['cni'], (cc - 1,)),_get(v['rni'], (cc - 1,)),_get(v['rhobar'], (cc - 1,)),_get(v['ni'], (cc - 1,)),_get(v['ai'], (cc - 1,)),_get(v['ci'], (cc - 1,)),v['alphstr'],v['alphv'],v['betam'],),tables,xp)  # WRF:916
        _put(v,'deltastr',((cc)-1,),call_out[0],mask_6,xp)  # WRF:916
        _put(v,'ani',((cc)-1,),call_out[1],mask_6,xp)  # WRF:916
        _put(v,'cni',((cc)-1,),call_out[2],mask_6,xp)  # WRF:916
        _put(v,'rni',((cc)-1,),call_out[3],mask_6,xp)  # WRF:916
        _put(v,'rhobar',((cc)-1,),call_out[4],mask_6,xp)  # WRF:916
        _put(v,'ni',((cc)-1,),call_out[5],mask_6,xp)  # WRF:916
        _put(v,'ai',((cc)-1,),call_out[6],mask_6,xp)  # WRF:916
        _put(v,'ci',((cc)-1,),call_out[7],mask_6,xp)  # WRF:916
        _put(v,'alphstr',None,call_out[8],mask_6,xp)  # WRF:916
        _put(v,'alphv',None,call_out[9],mask_6,xp)  # WRF:916
        _put(v,'betam',None,call_out[10],mask_6,xp)  # WRF:916
        _put(v,'dsold',((cc)-1,),_get(v['deltastr'], (cc - 1,)),mask_6,xp)  # WRF:920
        mask_9=xp.logical_and(mask_6,xp.logical_and(riming, v['qc'] > _r(1e-07)))
        taken_9=mask_9
        _put(v,'rrni',None,_r(13.498) * xp.log10(_r(500000.0) * _get(v['rni'], (cc - 1,))),mask_9,xp)  # WRF:929
        _put(v,'rqci',None,_r(8.776) * xp.log10(_r(10000000.0) * (xp.exp(v['qc']) - _r(1.0))),mask_9,xp)  # WRF:930
        _put(v,'rdsi',None,_r(50.0) * (_get(v['deltastr'], (cc - 1,)) - _r(0.5)),mask_9,xp)  # WRF:931
        _put(v,'rrho',None,_r(7.888) * xp.log10(_r(0.02) * _get(v['rhobar'], (cc - 1,))),mask_9,xp)  # WRF:932
        _put(v,'irni',None,_int(v['rrni']),mask_9,xp)  # WRF:933
        _put(v,'iqci',None,_int(v['rqci']),mask_9,xp)  # WRF:934
        _put(v,'idsi',None,_int(v['rdsi']),mask_9,xp)  # WRF:935
        _put(v,'irho',None,_int(v['rrho']),mask_9,xp)  # WRF:936
        _put(v,'rrni',None,_max(v['rrni'], _r(1.0)),mask_9,xp)  # WRF:939
        _put(v,'rqci',None,_max(v['rqci'], _r(1.0)),mask_9,xp)  # WRF:940
        _put(v,'rdsi',None,_max(v['rdsi'], _r(1.0)),mask_9,xp)  # WRF:941
        _put(v,'rrho',None,_max(v['rrho'], _r(1.0)),mask_9,xp)  # WRF:942
        _put(v,'irni',None,_max(v['irni'], 1),mask_9,xp)  # WRF:944
        _put(v,'iqci',None,_max(v['iqci'], 1),mask_9,xp)  # WRF:945
        _put(v,'idsi',None,_max(v['idsi'], 1),mask_9,xp)  # WRF:946
        _put(v,'irho',None,_max(v['irho'], 1),mask_9,xp)  # WRF:947
        _put(v,'rrni',None,_min(v['rrni'], _r(tables['itab'].shape[1 - 1]) - _r(1.0)),mask_9,xp)  # WRF:950
        _put(v,'rqci',None,_min(v['rqci'], _r(tables['itab'].shape[2 - 1]) - _r(1.0)),mask_9,xp)  # WRF:951
        _put(v,'rdsi',None,_min(v['rdsi'], _r(tables['itab'].shape[3 - 1]) - _r(1.0)),mask_9,xp)  # WRF:952
        _put(v,'rrho',None,_min(v['rrho'], _r(tables['itab'].shape[4 - 1]) - _r(1.0)),mask_9,xp)  # WRF:953
        _put(v,'irni',None,_min(v['irni'], tables['itab'].shape[1 - 1] - 1),mask_9,xp)  # WRF:955
        _put(v,'iqci',None,_min(v['iqci'], tables['itab'].shape[2 - 1] - 1),mask_9,xp)  # WRF:956
        _put(v,'idsi',None,_min(v['idsi'], tables['itab'].shape[3 - 1] - 1),mask_9,xp)  # WRF:957
        _put(v,'irho',None,_min(v['irho'], tables['itab'].shape[4 - 1] - 1),mask_9,xp)  # WRF:958
        for iti in range(1,2+1):
            call_out=_call('access_lookup_table',(tables['itab'],v['irni'],v['iqci'],v['idsi'],v['irho'],iti,v['rdsi'],v['rrho'],v['rqci'],v['rrni'],_get(v['proc'], (cc - 1, iti - 1)),),tables,xp)  # WRF:961
            _put(v,'proc',((cc)-1,(iti)-1,),call_out[0],mask_9,xp)  # WRF:961
        _put(v,'rimesum',((cc)-1,),_max(_get(v['proc'], (cc - 1, 1 - 1)) * _get(v['ni'], (cc - 1,)) * v['nc'] * (v['rhoair'] * v['rhoair']), _r(0.0)),mask_9,xp)  # WRF:968
        _put(v,'qi_qc_nrm',((cc)-1,),_get(v['proc'], (cc - 1, 1 - 1)),mask_9,xp)  # WRF:969
        _put(v,'qi_qc_nrd',((cc)-1,),_get(v['proc'], (cc - 1, 2 - 1)),mask_9,xp)  # WRF:970
        mask_10=xp.logical_and(mask_9,_get(v['rimesum'], (cc - 1,)) * v['i_rhoair'] < v['qsmall'])
        taken_10=mask_10
        _put(v,'rimesum',((cc)-1,),_r(0.0),mask_10,xp)  # WRF:974
        _put(v,'qi_qc_nrm',((cc)-1,),_r(0.0),mask_10,xp)  # WRF:975
        _put(v,'qi_qc_nrd',((cc)-1,),_r(0.0),mask_10,xp)  # WRF:976
        mask_11=xp.logical_and(mask_6,xp.logical_and(rain_ice, v['qr'] > v['qsmall']))
        taken_11=mask_11
        _put(v,'nr',None,_max(v['nr'], v['qnsmall']),mask_11,xp)  # WRF:985
        _put(v,'lamr',None,xp.power(v['pi'] * v['rhow'] * v['nr'] / v['qr'], _r(0.333333333)),mask_11,xp)  # WRF:987
        _put(v,'n0rr',None,v['nr'] * v['lamr'],mask_11,xp)  # WRF:988
        mask_12=xp.logical_and(mask_11,v['lamr'] < v['lamminr'])
        taken_12=mask_12
        _put(v,'lamr',None,v['lamminr'],mask_12,xp)  # WRF:990
        _put(v,'n0rr',None,_r(16269267968.0 if v['qv'].dtype == 'float32' else 16269262807.163687) * v['qr'] / (v['pi'] * v['rhow']),mask_12,xp)  # WRF:991
        _put(v,'nr',None,v['n0rr'] / v['lamr'],mask_12,xp)  # WRF:992
        mask_13=xp.logical_and(xp.logical_and(mask_11,xp.logical_not(taken_12)),v['lamr'] > v['lammaxr'])
        taken_12=xp.logical_or(taken_12,mask_13)
        _put(v,'lamr',None,v['lammaxr'],mask_13,xp)  # WRF:994
        _put(v,'n0rr',None,_r(6.250000125255483e+18 if v['qv'].dtype == 'float32' else 6.249999999999996e+18) * v['qr'] / (v['pi'] * v['rhow']),mask_13,xp)  # WRF:995
        _put(v,'nr',None,v['n0rr'] / v['lamr'],mask_13,xp)  # WRF:996
        _put(v,'rrr',None,_r(0.5) * (_r(1.0) / v['lamr']),mask_11,xp)  # WRF:998
        _put(v,'rrni',None,_r(13.498) * xp.log10(_r(500000.0) * _get(v['rni'], (cc - 1,))),mask_11,xp)  # WRF:1002
        _put(v,'rrri',None,_r(23.273) * xp.log10(_r(100000.0) * v['rrr']),mask_11,xp)  # WRF:1003
        _put(v,'rdsi',None,_r(50.0) * (_get(v['deltastr'], (cc - 1,)) - _r(0.5)),mask_11,xp)  # WRF:1004
        _put(v,'rrho',None,_r(7.888) * xp.log10(_r(0.02) * _get(v['rhobar'], (cc - 1,))),mask_11,xp)  # WRF:1005
        _put(v,'irni',None,_int(v['rrni']),mask_11,xp)  # WRF:1007
        _put(v,'irri',None,_int(v['rrri']),mask_11,xp)  # WRF:1008
        _put(v,'idsi',None,_int(v['rdsi']),mask_11,xp)  # WRF:1009
        _put(v,'irho',None,_int(v['rrho']),mask_11,xp)  # WRF:1010
        _put(v,'rrni',None,_max(v['rrni'], _r(1.0)),mask_11,xp)  # WRF:1013
        _put(v,'rrri',None,_max(v['rrri'], _r(1.0)),mask_11,xp)  # WRF:1014
        _put(v,'rdsi',None,_max(v['rdsi'], _r(1.0)),mask_11,xp)  # WRF:1015
        _put(v,'rrho',None,_max(v['rrho'], _r(1.0)),mask_11,xp)  # WRF:1016
        _put(v,'irni',None,_max(v['irni'], 1),mask_11,xp)  # WRF:1018
        _put(v,'irri',None,_max(v['irri'], 1),mask_11,xp)  # WRF:1019
        _put(v,'idsi',None,_max(v['idsi'], 1),mask_11,xp)  # WRF:1020
        _put(v,'irho',None,_max(v['irho'], 1),mask_11,xp)  # WRF:1021
        _put(v,'rrni',None,_min(v['rrni'], _r(tables['itabr'].shape[1 - 1]) - _r(1.0)),mask_11,xp)  # WRF:1024
        _put(v,'rrri',None,_min(v['rrri'], _r(tables['itabr'].shape[2 - 1]) - _r(1.0)),mask_11,xp)  # WRF:1025
        _put(v,'rdsi',None,_min(v['rdsi'], _r(tables['itabr'].shape[3 - 1]) - _r(1.0)),mask_11,xp)  # WRF:1026
        _put(v,'rrho',None,_min(v['rrho'], _r(tables['itabr'].shape[4 - 1]) - _r(1.0)),mask_11,xp)  # WRF:1027
        _put(v,'irni',None,_min(v['irni'], tables['itabr'].shape[1 - 1] - 1),mask_11,xp)  # WRF:1029
        _put(v,'irri',None,_min(v['irri'], tables['itabr'].shape[2 - 1] - 1),mask_11,xp)  # WRF:1030
        _put(v,'idsi',None,_min(v['idsi'], tables['itabr'].shape[3 - 1] - 1),mask_11,xp)  # WRF:1031
        _put(v,'irho',None,_min(v['irho'], tables['itabr'].shape[4 - 1] - 1),mask_11,xp)  # WRF:1032
        for iti in range(1,6+1):
            call_out=_call('access_lookup_table',(tables['itabr'],v['irni'],v['irri'],v['idsi'],v['irho'],iti,v['rdsi'],v['rrho'],v['rrri'],v['rrni'],_get(v['procr'], (cc - 1, iti - 1)),),tables,xp)  # WRF:1035
            _put(v,'procr',((cc)-1,(iti)-1,),call_out[0],mask_11,xp)  # WRF:1035
        _put(v,'numrateri',((cc)-1,),_max(_get(v['procr'], (cc - 1, 4 - 1)) * _get(v['ni'], (cc - 1,)) * v['nr'] * v['rhoair'], _r(0.0)),mask_11,xp)  # WRF:1042
        _put(v,'rainrateri',((cc)-1,),_max(_get(v['procr'], (cc - 1, 5 - 1)) * _get(v['ni'], (cc - 1,)) * v['nr'] * v['rhoair'], _r(0.0)),mask_11,xp)  # WRF:1043
        _put(v,'icerateri',((cc)-1,),_max(_get(v['procr'], (cc - 1, 6 - 1)) * _get(v['ni'], (cc - 1,)) * v['nr'] * v['rhoair'], _r(0.0)),mask_11,xp)  # WRF:1044
        mask_14=xp.logical_and(mask_11,xp.logical_or(_get(v['rainrateri'], (cc - 1,)) < v['qsmall'], _get(v['icerateri'], (cc - 1,)) < v['qsmall']))
        taken_14=mask_14
        _put(v,'rainrateri',((cc)-1,),_r(0.0),mask_14,xp)  # WRF:1048
        _put(v,'icerateri',((cc)-1,),_r(0.0),mask_14,xp)  # WRF:1049
        _put(v,'numrateri',((cc)-1,),_r(0.0),mask_14,xp)  # WRF:1050
        mask_15=xp.logical_and(mask_11,v['temp'] <= v['t0'])
        taken_15=mask_15
        mask_16=xp.logical_and(mask_15,xp.logical_and(v['qr'] > _r(0.0001), _get(v['qi'], (cc - 1,)) > _r(0.0001)))
        taken_16=mask_16
        _put(v,'dqrfzri',((cc)-1,),_get(v['rainrateri'], (cc - 1,)),mask_16,xp)  # WRF:1056
        _put(v,'dqifzri',((cc)-1,),_get(v['icerateri'], (cc - 1,)),mask_16,xp)  # WRF:1057
        _put(v,'dnfzri',((cc)-1,),_get(v['numrateri'], (cc - 1,)),mask_16,xp)  # WRF:1058
        mask_17=xp.logical_and(mask_11,xp.logical_not(taken_15))
        _put(v,'dqimltri',((cc)-1,),_get(v['icerateri'], (cc - 1,)),mask_17,xp)  # WRF:1062
        _put(v,'dnmltri',((cc)-1,),_get(v['numrateri'], (cc - 1,)),mask_17,xp)  # WRF:1063
        _put(v,'rimesumr',((cc)-1,),_max(_get(v['procr'], (cc - 1, 1 - 1)) * _get(v['ni'], (cc - 1,)) * v['nr'] * (v['rhoair'] * v['rhoair']), _r(0.0)),mask_11,xp)  # WRF:1073
        _put(v,'qi_qr_nrm',((cc)-1,),_get(v['procr'], (cc - 1, 1 - 1)),mask_11,xp)  # WRF:1074
        _put(v,'qi_qr_nrd',((cc)-1,),_get(v['procr'], (cc - 1, 2 - 1)),mask_11,xp)  # WRF:1075
        _put(v,'qi_qr_nrn',((cc)-1,),_get(v['procr'], (cc - 1, 3 - 1)),mask_11,xp)  # WRF:1076
        mask_18=xp.logical_and(mask_11,_get(v['rimesumr'], (cc - 1,)) * v['i_rhoair'] < v['qsmall'])
        taken_18=mask_18
        _put(v,'rimesumr',((cc)-1,),_r(0.0),mask_18,xp)  # WRF:1080
        _put(v,'qi_qr_nrm',((cc)-1,),_r(0.0),mask_18,xp)  # WRF:1081
        _put(v,'qi_qr_nrd',((cc)-1,),_r(0.0),mask_18,xp)  # WRF:1082
        _put(v,'qi_qr_nrn',((cc)-1,),_r(0.0),mask_18,xp)  # WRF:1083
        _put(v,'tmpsum',None,v['tmpsum'] + _get(v['rimesum'], (cc - 1,)),mask_6,xp)  # WRF:1089
        _put(v,'tmpsumr',None,v['tmpsumr'] + _get(v['rimesumr'], (cc - 1,)),mask_6,xp)  # WRF:1090
        mask_19=xp.logical_and(mask_6,v['ccvar'] == v['numice'])
        taken_19=mask_19
        _put(v,'sink',None,v['tmpsum'] * v['i_rhoair'],mask_19,xp)  # WRF:1095
        _put(v,'source',None,v['qc'] * v['i_dt'],mask_19,xp)  # WRF:1096
        mask_20=xp.logical_and(mask_19,xp.logical_and(v['sink'] > v['source'], v['qc'] > v['qsmall']))
        taken_20=mask_20
        _put(v,'ratio',None,v['source'] / v['sink'],mask_20,xp)  # WRF:1098
        _put(v,'rimesum',None,v['rimesum'] * _expand(v['ratio'], 1),mask_20,xp)  # WRF:1099
        _put(v,'qi_qc_nrm',None,v['qi_qc_nrm'] * _expand(v['ratio'], 1),mask_20,xp)  # WRF:1100
        _put(v,'qi_qc_nrd',None,v['qi_qc_nrd'] * _expand(v['ratio'], 1),mask_20,xp)  # WRF:1101
        _put(v,'sink',None,v['tmpsum'] * v['i_rhoair'],mask_19,xp)  # WRF:1105
        _put(v,'source',None,v['qr'] * v['i_dt'],mask_19,xp)  # WRF:1106
        mask_21=xp.logical_and(mask_19,xp.logical_and(v['sink'] > v['source'], v['qr'] > v['qsmall']))
        taken_21=mask_21
        _put(v,'ratio',None,v['source'] / v['sink'],mask_21,xp)  # WRF:1108
        _put(v,'rimesumr',None,v['rimesumr'] * _expand(v['ratio'], 1),mask_21,xp)  # WRF:1109
        _put(v,'qi_qr_nrm',None,v['qi_qr_nrm'] * _expand(v['ratio'], 1),mask_21,xp)  # WRF:1110
        _put(v,'qi_qr_nrd',None,v['qi_qr_nrd'] * _expand(v['ratio'], 1),mask_21,xp)  # WRF:1111
        _put(v,'qi_qr_nrn',None,v['qi_qr_nrn'] * _expand(v['ratio'], 1),mask_21,xp)  # WRF:1112
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_6,xp)  # WRF:1119
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_6,xp)  # WRF:1120
        _put(v,'vi',None,v['fourthirdspi'] * (_get(v['rni'], (cc - 1,)) * _get(v['rni'], (cc - 1,)) * _get(v['rni'], (cc - 1,))) * tables['gamma_tab'][v['gi'] - 1,] * v['i_gammnu'],mask_6,xp)  # WRF:1122
        _put(v,'iwci',None,_get(v['ni'], (cc - 1,)) * _get(v['rhobar'], (cc - 1,)) * v['vi'] * v['rhoair'],mask_6,xp)  # WRF:1123
        _put(v,'nim3dum',None,_get(v['ni'], (cc - 1,)) * v['rhoair'],mask_6,xp)  # WRF:1124
        _put(v,'rimetotal',((cc)-1,),_get(v['rimesum'], (cc - 1,)) + _get(v['rimesumr'], (cc - 1,)),mask_6,xp)  # WRF:1127
        _put(v,'alphstr',None,xp.power(v['ao'], _r(1.0) - _get(v['deltastr'], (cc - 1,))),mask_6,xp)  # WRF:1128
        _put(v,'capgam',None,capacitance_gamma(_get(v['ani'], (cc - 1,)), _get(v['deltastr'], (cc - 1,)), v['nu'], v['alphstr'], v['i_gammnu'], xp=xp),mask_6,xp)  # WRF:1130
        call_out=_call('vaporgrow',(v['dt'],_get(v['ani'], (cc - 1,)),_get(v['cni'], (cc - 1,)),_get(v['rni'], (cc - 1,)),v['igr'],v['nim3dum'],v['temp'],_get(v['rimetotal'], (cc - 1,)),v['pres_e'],v['nu'],v['alphstr'],v['sui'],v['sup'],v['qvs'],v['qvi'],v['mu'],v['iwci'],v['rhoair'],_get(v['qi'], (cc - 1,)),v['dv'],v['kt'],v['ao'],v['nsch'],v['npr'],v['gammnu'],v['i_gammnu'],v['fourthirdspi'],v['svpi'],v['xxls'],v['xxlv'],v['xxlf'],v['capgam'],_get(v['vtrni1'], (cc - 1,)),_get(v['vtrmi1'], (cc - 1,)),_get(v['vtrzi1'], (cc - 1,)),v['anf'],v['cnf'],v['rnf'],v['iwcf'],_get(v['fv'], (cc - 1,)),_get(v['fh'], (cc - 1,)),_get(v['rhobar'], (cc - 1,)),_get(v['deltastr'], (cc - 1,)),_get(v['rhodepout'], (cc - 1,)),_get(v['dsdepout'], (cc - 1,)),),tables,xp)  # WRF:1134
        _put(v,'vtrni1',((cc)-1,),call_out[0],mask_6,xp)  # WRF:1134
        _put(v,'vtrmi1',((cc)-1,),call_out[1],mask_6,xp)  # WRF:1134
        _put(v,'vtrzi1',((cc)-1,),call_out[2],mask_6,xp)  # WRF:1134
        _put(v,'anf',None,call_out[3],mask_6,xp)  # WRF:1134
        _put(v,'cnf',None,call_out[4],mask_6,xp)  # WRF:1134
        _put(v,'rnf',None,call_out[5],mask_6,xp)  # WRF:1134
        _put(v,'iwcf',None,call_out[6],mask_6,xp)  # WRF:1134
        _put(v,'fv',((cc)-1,),call_out[7],mask_6,xp)  # WRF:1134
        _put(v,'fh',((cc)-1,),call_out[8],mask_6,xp)  # WRF:1134
        _put(v,'rhobar',((cc)-1,),call_out[9],mask_6,xp)  # WRF:1134
        _put(v,'rhodepout',((cc)-1,),call_out[10],mask_6,xp)  # WRF:1134
        _put(v,'dsdepout',((cc)-1,),call_out[11],mask_6,xp)  # WRF:1134
        _put(v,'prd',((cc)-1,),(v['iwcf'] - v['iwci']) * v['i_rhoair'] * v['i_dt'],mask_6,xp)  # WRF:1141
        _put(v,'prd',((cc)-1,),_max(_get(v['prd'], (cc - 1,)), -_get(v['qi'], (cc - 1,)) * v['i_dt']),mask_6,xp)  # WRF:1142
        _put(v,'ard',((cc)-1,),(_r(2.0) * (v['anf'] - _get(v['ani'], (cc - 1,))) * _get(v['cni'], (cc - 1,)) + (v['cnf'] - _get(v['cni'], (cc - 1,))) * _get(v['ani'], (cc - 1,))) * _get(v['ani'], (cc - 1,)) * _get(v['ni'], (cc - 1,)) * v['i_dt'],mask_6,xp)  # WRF:1143
        _put(v,'crd',((cc)-1,),(_r(2.0) * (v['cnf'] - _get(v['cni'], (cc - 1,))) * _get(v['ani'], (cc - 1,)) + (v['anf'] - _get(v['ani'], (cc - 1,))) * _get(v['cni'], (cc - 1,))) * _get(v['cni'], (cc - 1,)) * _get(v['ni'], (cc - 1,)) * v['i_dt'],mask_6,xp)  # WRF:1144
        mask_22=xp.logical_and(mask_6,xp.logical_and(_get(v['prd'], (cc - 1,)) < _r(0.0), _get(v['qi'], (cc - 1,)) > v['qsmall']))
        taken_22=mask_22
        _put(v,'nrd',((cc)-1,),_get(v['prd'], (cc - 1,)) * _get(v['ni'], (cc - 1,)) / _get(v['qi'], (cc - 1,)),mask_22,xp)  # WRF:1148
        mask_23=xp.logical_and(mask_6,xp.abs(_get(v['prd'], (cc - 1,)) * v['dt']) < v['qsmall'] * _r(0.01))
        taken_23=mask_23
        _put(v,'prd',((cc)-1,),_r(0.0),mask_23,xp)  # WRF:1153
        _put(v,'nrd',((cc)-1,),_r(0.0),mask_23,xp)  # WRF:1154
        _put(v,'ard',((cc)-1,),_r(0.0),mask_23,xp)  # WRF:1155
        _put(v,'crd',((cc)-1,),_r(0.0),mask_23,xp)  # WRF:1156
        mask_24=xp.logical_and(mask_6,wet_grow)
        taken_24=mask_24
        _put(v,'rimetotal',((cc)-1,),_get(v['rimesum'], (cc - 1,)) + _get(v['rimesumr'], (cc - 1,)),mask_24,xp)  # WRF:1166
        call_out=_call('wet_growth_check',(v['nu'],v['temp'],v['rhoair'],v['xxlv'],v['xxlf'],v['qv'],v['dv'],v['kt'],v['qs0'],_get(v['fv'], (cc - 1,)),_get(v['fh'], (cc - 1,)),_get(v['rimetotal'], (cc - 1,)),_get(v['rni'], (cc - 1,)),_get(v['ni'], (cc - 1,)),_get(v['dry_growth'], (cc - 1,)),),tables,xp)  # WRF:1167
        _put(v,'dry_growth',((cc)-1,),call_out[0],mask_24,xp)  # WRF:1167
        mask_25=xp.logical_and(mask_6,v['temp'] > v['t0'])
        taken_25=mask_25
        _put(v,'dry_growth',((cc)-1,),False,mask_25,xp)  # WRF:1173
        _put(v,'vir',None,v['vi'],mask_6,xp)  # WRF:1177
        _put(v,'rnfr',None,_get(v['rni'], (cc - 1,)),mask_6,xp)  # WRF:1178
        mask_26=xp.logical_and(mask_6,xp.logical_and(xp.logical_and(riming, v['qc'] > v['qsmall']), _get(v['qi_qc_nrm'], (cc - 1,)) > _r(0.0)))
        taken_26=mask_26
        _put(v,'rimec1',None,_r(0.0066),mask_26,xp)  # WRF:1184
        mask_27=xp.logical_and(mask_26,v['temp'] - v['t0'] < _r(-30.0))
        taken_27=mask_27
        _put(v,'rimec1',None,_r(0.0036),mask_27,xp)  # WRF:1186
        mask_28=xp.logical_and(xp.logical_and(mask_26,xp.logical_not(taken_27)),xp.logical_and(v['temp'] - v['t0'] < -20, v['temp'] - v['t0'] >= _r(-30.0)))
        taken_27=xp.logical_or(taken_27,mask_28)
        _put(v,'dum',None,(xp.abs(v['temp'] - v['t0']) - _r(20.0)) / _r(10.0),mask_28,xp)  # WRF:1188
        _put(v,'rimec1',None,v['dum'] * _r(0.0036) + (_r(1.0) - v['dum']) * _r(0.004),mask_28,xp)  # WRF:1189
        mask_29=xp.logical_and(xp.logical_and(mask_26,xp.logical_not(taken_27)),xp.logical_and(v['temp'] - v['t0'] < -15, v['temp'] - v['t0'] >= _r(-20.0)))
        taken_27=xp.logical_or(taken_27,mask_29)
        _put(v,'dum',None,(xp.abs(v['temp'] - v['t0']) - _r(15.0)) / _r(5.0),mask_29,xp)  # WRF:1191
        _put(v,'rimec1',None,v['dum'] * _r(0.004) + (_r(1.0) - v['dum']) * _r(0.005),mask_29,xp)  # WRF:1192
        mask_30=xp.logical_and(xp.logical_and(mask_26,xp.logical_not(taken_27)),xp.logical_and(v['temp'] - v['t0'] < -10, v['temp'] - v['t0'] >= _r(-15.0)))
        taken_27=xp.logical_or(taken_27,mask_30)
        _put(v,'dum',None,(xp.abs(v['temp'] - v['t0']) - _r(10.0)) / _r(5.0),mask_30,xp)  # WRF:1194
        _put(v,'rimec1',None,v['dum'] * _r(0.005) + (_r(1.0) - v['dum']) * _r(0.0066),mask_30,xp)  # WRF:1195
        mask_31=xp.logical_and(xp.logical_and(mask_26,xp.logical_not(taken_27)),xp.logical_and(v['temp'] - v['t0'] < -5, v['temp'] - v['t0'] >= _r(-10.0)))
        taken_27=xp.logical_or(taken_27,mask_31)
        _put(v,'dum',None,(xp.abs(v['temp'] - v['t0']) - _r(5.0)) / _r(5.0),mask_31,xp)  # WRF:1197
        _put(v,'rimec1',None,v['dum'] * _r(0.0066) + (_r(1.0) - v['dum']) * _r(0.012),mask_31,xp)  # WRF:1198
        mask_32=xp.logical_and(xp.logical_and(mask_26,xp.logical_not(taken_27)),v['temp'] - v['t0'] >= _r(-5.0))
        taken_27=xp.logical_or(taken_27,mask_32)
        _put(v,'rimec1',None,_r(0.012),mask_32,xp)  # WRF:1200
        _put(v,'gdenavg',((cc)-1,),_r(1000.0) * (_r(0.8) * xp.tanh(v['rimec1'] * _get(v['qi_qc_nrd'], (cc - 1,)) / _get(v['qi_qc_nrm'], (cc - 1,))) + _r(0.1)),mask_26,xp)  # WRF:1202
        mask_33=xp.logical_and(mask_26,xp.logical_and(v['temp'] - v['t0'] > _r(-5.0), v['temp'] - v['t0'] <= _r(0.0)))
        taken_33=mask_33
        _put(v,'dum',None,(xp.abs(v['temp'] - v['t0']) - _r(0.0)) / _r(5.0),mask_33,xp)  # WRF:1205
        _put(v,'gdenavg',((cc)-1,),v['dum'] * _get(v['gdenavg'], (cc - 1,)) + (_r(1.0) - v['dum']) * _r(900.0),mask_33,xp)  # WRF:1206
        mask_34=xp.logical_and(mask_26,xp.logical_or(v['temp'] - v['t0'] > _r(0.0), xp.logical_not(_get(v['dry_growth'], (cc - 1,)))))
        taken_34=mask_34
        _put(v,'gdenavg',((cc)-1,),_r(900.0),mask_34,xp)  # WRF:1210
        _put(v,'gdenavg',((cc)-1,),_max(_get(v['gdenavg'], (cc - 1,)), _r(50.0)),mask_26,xp)  # WRF:1213
        _put(v,'gdenavg',((cc)-1,),_min(_get(v['gdenavg'], (cc - 1,)), _r(900.0)),mask_26,xp)  # WRF:1214
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_26,xp)  # WRF:1216
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_26,xp)  # WRF:1217
        _put(v,'rimedr',None,_max(_get(v['qi_qc_nrm'], (cc - 1,)) / _get(v['gdenavg'], (cc - 1,)) / (tables['gamma_tab'][v['gi'] - 1,] * v['i_gammnu'] * _r(4.0) * v['pi'] * (_get(v['rni'], (cc - 1,)) * _get(v['rni'], (cc - 1,)))), _r(0.0)),mask_26,xp)  # WRF:1220
        _put(v,'rnfr',None,v['rnfr'] + _max(v['rimedr'] * v['nc'] * v['rhoair'], _r(0.0)) * v['dt'],mask_26,xp)  # WRF:1222
        mask_35=xp.logical_and(mask_6,xp.logical_and(xp.logical_and(rain_ice, v['qr'] > v['qsmall']), _get(v['qi_qr_nrm'], (cc - 1,)) > _r(0.0)))
        taken_35=mask_35
        _put(v,'rimec1',None,_r(0.0066),mask_35,xp)  # WRF:1229
        mask_36=xp.logical_and(mask_35,v['temp'] - v['t0'] < _r(-30.0))
        taken_36=mask_36
        _put(v,'rimec1',None,_r(0.0036),mask_36,xp)  # WRF:1231
        mask_37=xp.logical_and(xp.logical_and(mask_35,xp.logical_not(taken_36)),xp.logical_and(v['temp'] - v['t0'] < -20, v['temp'] - v['t0'] >= _r(-30.0)))
        taken_36=xp.logical_or(taken_36,mask_37)
        _put(v,'dum',None,(xp.abs(v['temp'] - v['t0']) - _r(20.0)) / _r(10.0),mask_37,xp)  # WRF:1233
        _put(v,'rimec1',None,v['dum'] * _r(0.0036) + (_r(1.0) - v['dum']) * _r(0.004),mask_37,xp)  # WRF:1234
        mask_38=xp.logical_and(xp.logical_and(mask_35,xp.logical_not(taken_36)),xp.logical_and(v['temp'] - v['t0'] < -15, v['temp'] - v['t0'] >= _r(-20.0)))
        taken_36=xp.logical_or(taken_36,mask_38)
        _put(v,'dum',None,(xp.abs(v['temp'] - v['t0']) - _r(15.0)) / _r(5.0),mask_38,xp)  # WRF:1236
        _put(v,'rimec1',None,v['dum'] * _r(0.004) + (_r(1.0) - v['dum']) * _r(0.005),mask_38,xp)  # WRF:1237
        mask_39=xp.logical_and(xp.logical_and(mask_35,xp.logical_not(taken_36)),xp.logical_and(v['temp'] - v['t0'] < -10, v['temp'] - v['t0'] >= _r(-15.0)))
        taken_36=xp.logical_or(taken_36,mask_39)
        _put(v,'dum',None,(xp.abs(v['temp'] - v['t0']) - _r(10.0)) / _r(5.0),mask_39,xp)  # WRF:1239
        _put(v,'rimec1',None,v['dum'] * _r(0.005) + (_r(1.0) - v['dum']) * _r(0.0066),mask_39,xp)  # WRF:1240
        mask_40=xp.logical_and(xp.logical_and(mask_35,xp.logical_not(taken_36)),xp.logical_and(v['temp'] - v['t0'] < -5, v['temp'] - v['t0'] >= _r(-10.0)))
        taken_36=xp.logical_or(taken_36,mask_40)
        _put(v,'dum',None,(xp.abs(v['temp'] - v['t0']) - _r(5.0)) / _r(5.0),mask_40,xp)  # WRF:1242
        _put(v,'rimec1',None,v['dum'] * _r(0.0066) + (_r(1.0) - v['dum']) * _r(0.012),mask_40,xp)  # WRF:1243
        mask_41=xp.logical_and(xp.logical_and(mask_35,xp.logical_not(taken_36)),v['temp'] - v['t0'] >= _r(-5.0))
        taken_36=xp.logical_or(taken_36,mask_41)
        _put(v,'rimec1',None,_r(0.012),mask_41,xp)  # WRF:1245
        _put(v,'gdenavgr',((cc)-1,),_r(1000.0) * (_r(0.8) * xp.tanh(v['rimec1'] * _get(v['qi_qr_nrd'], (cc - 1,)) / _get(v['qi_qr_nrm'], (cc - 1,))) + _r(0.1)),mask_35,xp)  # WRF:1247
        mask_42=xp.logical_and(mask_35,xp.logical_and(v['temp'] - v['t0'] > _r(-5.0), v['temp'] - v['t0'] <= _r(0.0)))
        taken_42=mask_42
        _put(v,'dum',None,(xp.abs(v['temp'] - v['t0']) - _r(0.0)) / _r(5.0),mask_42,xp)  # WRF:1250
        _put(v,'gdenavgr',((cc)-1,),v['dum'] * _get(v['gdenavgr'], (cc - 1,)) + (_r(1.0) - v['dum']) * _r(900.0),mask_42,xp)  # WRF:1251
        mask_43=xp.logical_and(mask_35,xp.logical_or(v['temp'] - v['t0'] > _r(0.0), xp.logical_not(_get(v['dry_growth'], (cc - 1,)))))
        taken_43=mask_43
        _put(v,'gdenavgr',((cc)-1,),_r(900.0),mask_43,xp)  # WRF:1255
        _put(v,'gdenavgr',((cc)-1,),_max(_get(v['gdenavgr'], (cc - 1,)), _r(50.0)),mask_35,xp)  # WRF:1258
        _put(v,'gdenavgr',((cc)-1,),_min(_get(v['gdenavgr'], (cc - 1,)), _r(900.0)),mask_35,xp)  # WRF:1259
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_35,xp)  # WRF:1261
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_35,xp)  # WRF:1262
        _put(v,'rimedrr',None,_max(_get(v['qi_qr_nrm'], (cc - 1,)) / _get(v['gdenavgr'], (cc - 1,)) / (tables['gamma_tab'][v['gi'] - 1,] * v['i_gammnu'] * _r(4.0) * v['pi'] * (_get(v['rni'], (cc - 1,)) * _get(v['rni'], (cc - 1,)))), _r(0.0)),mask_35,xp)  # WRF:1265
        _put(v,'rnfr',None,v['rnfr'] + _max(v['rimedrr'] * v['nr'] * v['rhoair'], _r(0.0)) * v['dt'],mask_35,xp)  # WRF:1267
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_6,xp)  # WRF:1272
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_6,xp)  # WRF:1273
        _put(v,'vfr',None,v['fourthirdspi'] * (v['rnfr'] * v['rnfr'] * v['rnfr']) * tables['gamma_tab'][v['gi'] - 1,] * v['i_gammnu'],mask_6,xp)  # WRF:1274
        _put(v,'vfr',None,_max(v['vfr'], v['vi']),mask_6,xp)  # WRF:1275
        _put(v,'rnfr',None,_max(v['rnfr'], _get(v['rni'], (cc - 1,))),mask_6,xp)  # WRF:1276
        mask_44=xp.logical_and(mask_6,xp.logical_and(_get(v['rimetotal'], (cc - 1,)) > 0, v['vfr'] > v['vi']))
        taken_44=mask_44
        _put(v,'qcrimefrac',((cc)-1,),_get(v['rimesum'], (cc - 1,)) / _get(v['rimetotal'], (cc - 1,)),mask_44,xp)  # WRF:1281
        _put(v,'qcrimefrac',((cc)-1,),_max(_get(v['qcrimefrac'], (cc - 1,)), _r(0.0)),mask_44,xp)  # WRF:1282
        _put(v,'qcrimefrac',((cc)-1,),_min(_get(v['qcrimefrac'], (cc - 1,)), _r(1.0)),mask_44,xp)  # WRF:1283
        _put(v,'gdentotal',None,_get(v['qcrimefrac'], (cc - 1,)) * _get(v['gdenavg'], (cc - 1,)) + (_r(1.0) - _get(v['qcrimefrac'], (cc - 1,))) * _get(v['gdenavgr'], (cc - 1,)),mask_44,xp)  # WRF:1284
        _put(v,'rhobarrime',None,_get(v['rhobar'], (cc - 1,)),mask_44,xp)  # WRF:1285
        _put(v,'rhorimeout',((cc)-1,),_get(v['rhobar'], (cc - 1,)) * (v['vi'] / v['vfr']) + v['gdentotal'] * (_r(1.0) - v['vi'] / v['vfr']),mask_44,xp)  # WRF:1286
        _put(v,'rhorimeout',((cc)-1,),_min(_get(v['rhorimeout'], (cc - 1,)), v['rhoi']),mask_44,xp)  # WRF:1287
        _put(v,'iwcfr',None,_get(v['rhorimeout'], (cc - 1,)) * v['vfr'] * v['nim3dum'],mask_44,xp)  # WRF:1288
        _put(v,'rhorimeout',((cc)-1,),v['gdentotal'],mask_44,xp)  # WRF:1289
        _put(v,'rhorimeout',((cc)-1,),_min(_get(v['rhorimeout'], (cc - 1,)), v['rhoi']),mask_44,xp)  # WRF:1290
        mask_45=xp.logical_and(mask_6,xp.logical_not(taken_44))
        _put(v,'iwcfr',None,v['iwci'],mask_45,xp)  # WRF:1292
        _put(v,'rnfr',None,_get(v['rni'], (cc - 1,)),mask_45,xp)  # WRF:1293
        mask_46=xp.logical_and(mask_6,_get(v['dry_growth'], (cc - 1,)))
        taken_46=mask_46
        _put(v,'cnfr',None,_get(v['cni'], (cc - 1,)),mask_46,xp)  # WRF:1299
        _put(v,'anfr',None,_get(v['ani'], (cc - 1,)),mask_46,xp)  # WRF:1300
        _put(v,'phibr',None,_r(1.0),mask_46,xp)  # WRF:1301
        mask_47=xp.logical_and(mask_46,xp.logical_and(_get(v['rimetotal'], (cc - 1,)) > 0, v['vfr'] > v['vi']))
        taken_47=mask_47
        _put(v,'gamma_arg',None,v['nu'] - _r(1.0) + _get(v['deltastr'], (cc - 1,)),mask_47,xp)  # WRF:1303
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_47,xp)  # WRF:1304
        _put(v,'phibr',None,_get(v['cni'], (cc - 1,)) / _get(v['ani'], (cc - 1,)) * tables['gamma_tab'][v['gi'] - 1,] * v['i_gammnu'],mask_47,xp)  # WRF:1305
        mask_48=xp.logical_and(mask_47,v['phibr'] == _r(1.0))
        taken_48=mask_48
        _put(v,'phifr',None,_r(1.0),mask_48,xp)  # WRF:1307
        _put(v,'anfr',None,v['rnfr'],mask_48,xp)  # WRF:1308
        _put(v,'cnfr',None,v['rnfr'],mask_48,xp)  # WRF:1309
        mask_49=xp.logical_and(xp.logical_and(mask_47,xp.logical_not(taken_48)),v['phibr'] > _r(1.25))
        taken_48=xp.logical_or(taken_48,mask_49)
        _put(v,'phifr',None,v['phibr'] * xp.power(v['rnfr'] / _get(v['rni'], (cc - 1,)) * (v['rnfr'] / _get(v['rni'], (cc - 1,))) * (v['rnfr'] / _get(v['rni'], (cc - 1,))), _r(-0.5)),mask_49,xp)  # WRF:1311
        _put(v,'anfr',None,_get(v['ani'], (cc - 1,)) / xp.power(_get(v['rni'], (cc - 1,)), _r(1.5)) * xp.power(v['rnfr'], _r(1.5)),mask_49,xp)  # WRF:1312
        mask_50=xp.logical_and(xp.logical_and(mask_47,xp.logical_not(taken_48)),v['phibr'] < _r(0.8))
        taken_48=xp.logical_or(taken_48,mask_50)
        _put(v,'phifr',None,v['phibr'] * (v['rnfr'] / _get(v['rni'], (cc - 1,)) * (v['rnfr'] / _get(v['rni'], (cc - 1,))) * (v['rnfr'] / _get(v['rni'], (cc - 1,)))),mask_50,xp)  # WRF:1314
        _put(v,'gamma_arg',None,v['nu'] - _r(1.0) + _get(v['deltastr'], (cc - 1,)),mask_50,xp)  # WRF:1315
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_50,xp)  # WRF:1316
        _put(v,'cnfr',None,v['phifr'] * v['anfr'] * v['gammnu'] / tables['gamma_tab'][v['gi'] - 1,],mask_50,xp)  # WRF:1317
        mask_51=xp.logical_and(mask_47,xp.logical_not(taken_48))
        _put(v,'phifr',None,v['phibr'],mask_51,xp)  # WRF:1319
        _put(v,'gamma_arg',None,v['nu'] - _r(1.0) + _get(v['deltastr'], (cc - 1,)),mask_51,xp)  # WRF:1320
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_51,xp)  # WRF:1321
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_51,xp)  # WRF:1322
        _put(v,'gi2',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_51,xp)  # WRF:1323
        _put(v,'anfr',None,xp.power(v['vfr'] * tables['gamma_tab'][v['gi'] - 1,] / (v['fourthirdspi'] * v['phifr'] * tables['gamma_tab'][v['gi2'] - 1,]), _r(0.3333333333)),mask_51,xp)  # WRF:1324
        _put(v,'cnfr',None,v['phifr'] * v['anfr'] * v['gammnu'] / tables['gamma_tab'][v['gi'] - 1,],mask_51,xp)  # WRF:1325
        mask_52=xp.logical_and(mask_47,xp.logical_and(v['phibr'] <= _r(1.0), v['phifr'] > _r(1.0)))
        taken_52=mask_52
        _put(v,'phifr',None,_r(0.99),mask_52,xp)  # WRF:1331
        _put(v,'gamma_arg',None,v['nu'] - _r(1.0) + _get(v['deltastr'], (cc - 1,)),mask_52,xp)  # WRF:1332
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_52,xp)  # WRF:1333
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_52,xp)  # WRF:1334
        _put(v,'gi2',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_52,xp)  # WRF:1335
        _put(v,'anfr',None,xp.power(v['vfr'] * tables['gamma_tab'][v['gi'] - 1,] / (v['fourthirdspi'] * v['phifr'] * tables['gamma_tab'][v['gi2'] - 1,]), _r(0.3333333333)),mask_52,xp)  # WRF:1336
        _put(v,'cnfr',None,v['phifr'] * v['anfr'] * v['gammnu'] / tables['gamma_tab'][v['gi'] - 1,],mask_52,xp)  # WRF:1337
        mask_53=xp.logical_and(mask_47,xp.logical_and(v['phibr'] >= _r(1.0), v['phifr'] < _r(1.0)))
        taken_53=mask_53
        _put(v,'phifr',None,_r(1.01),mask_53,xp)  # WRF:1341
        _put(v,'gamma_arg',None,v['nu'] - _r(1.0) + _get(v['deltastr'], (cc - 1,)),mask_53,xp)  # WRF:1342
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_53,xp)  # WRF:1343
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_53,xp)  # WRF:1344
        _put(v,'gi2',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_53,xp)  # WRF:1345
        _put(v,'anfr',None,xp.power(v['vfr'] * tables['gamma_tab'][v['gi'] - 1,] / (v['fourthirdspi'] * v['phifr'] * tables['gamma_tab'][v['gi2'] - 1,]), _r(0.3333333333)),mask_53,xp)  # WRF:1346
        _put(v,'cnfr',None,v['phifr'] * v['anfr'] * v['gammnu'] / tables['gamma_tab'][v['gi'] - 1,],mask_53,xp)  # WRF:1347
        _put(v,'cnfr',None,_max(v['cnfr'], _get(v['cni'], (cc - 1,))),mask_46,xp)  # WRF:1352
        _put(v,'anfr',None,_max(v['anfr'], _get(v['ani'], (cc - 1,))),mask_46,xp)  # WRF:1353
        _put(v,'prdr',((cc)-1,),(v['iwcfr'] - v['iwci']) * v['i_rhoair'] * v['i_dt'],mask_46,xp)  # WRF:1356
        _put(v,'ardr',((cc)-1,),(_r(2.0) * (v['anfr'] - _get(v['ani'], (cc - 1,))) * _get(v['cni'], (cc - 1,)) + (v['cnfr'] - _get(v['cni'], (cc - 1,))) * _get(v['ani'], (cc - 1,))) * _get(v['ani'], (cc - 1,)) * _get(v['ni'], (cc - 1,)) * v['i_dt'],mask_46,xp)  # WRF:1357
        _put(v,'crdr',((cc)-1,),(_r(2.0) * (v['cnfr'] - _get(v['cni'], (cc - 1,))) * _get(v['ani'], (cc - 1,)) + (v['anfr'] - _get(v['ani'], (cc - 1,))) * _get(v['cni'], (cc - 1,))) * _get(v['cni'], (cc - 1,)) * _get(v['ni'], (cc - 1,)) * v['i_dt'],mask_46,xp)  # WRF:1358
        _put(v,'prdr',((cc)-1,),_max(_get(v['prdr'], (cc - 1,)), _r(0.0)),mask_46,xp)  # WRF:1361
        _put(v,'ardr',((cc)-1,),_max(_get(v['ardr'], (cc - 1,)), _r(0.0)),mask_46,xp)  # WRF:1362
        _put(v,'crdr',((cc)-1,),_max(_get(v['crdr'], (cc - 1,)), _r(0.0)),mask_46,xp)  # WRF:1363
        mask_54=xp.logical_and(mask_46,_get(v['prdr'], (cc - 1,)) == _r(0.0))
        taken_54=mask_54
        _put(v,'ardr',((cc)-1,),_r(0.0),mask_54,xp)  # WRF:1367
        _put(v,'crdr',((cc)-1,),_r(0.0),mask_54,xp)  # WRF:1368
        mask_55=xp.logical_and(mask_6,xp.logical_not(taken_46))
        _put(v,'prdr',((cc)-1,),(v['iwcfr'] - v['iwci']) * v['i_rhoair'] * v['i_dt'],mask_55,xp)  # WRF:1375
        _put(v,'ardr',((cc)-1,),_r(0.0),mask_55,xp)  # WRF:1376
        _put(v,'crdr',((cc)-1,),_r(0.0),mask_55,xp)  # WRF:1377
        mask_56=xp.logical_and(mask_6,v['temp'] > v['t0'])
        taken_56=mask_56
        _put(v,'qmlt',((cc)-1,),_r(2.0) * v['pi'] * (v['kt'] * _get(v['fh'], (cc - 1,)) * (v['t0'] - v['temp']) + v['rhoair'] * v['xxlv'] * v['dv'] * _get(v['fv'], (cc - 1,)) * (v['qs0'] - v['qv'])) / v['xxlf'] * (_get(v['ni'], (cc - 1,)) * v['nu'] * _max(_get(v['ani'], (cc - 1,)), _get(v['cni'], (cc - 1,)))) - v['cpw'] / v['xxlf'] * (v['temp'] - v['t0']) * (_get(v['rimetotal'], (cc - 1,)) / v['rhoair'] + _get(v['dqimltri'], (cc - 1,))),mask_56,xp)  # WRF:1384
        _put(v,'qmlt',((cc)-1,),_min(_get(v['qmlt'], (cc - 1,)), _r(0.0)),mask_56,xp)  # WRF:1388
        _put(v,'qmlt',((cc)-1,),_max(_get(v['qmlt'], (cc - 1,)), -_get(v['qi'], (cc - 1,)) * v['i_dt']),mask_56,xp)  # WRF:1389
        mask_57=xp.logical_and(mask_56,_get(v['qmlt'], (cc - 1,)) < _r(0.0))
        taken_57=mask_57
        mask_58=xp.logical_and(mask_57,xp.logical_or(_get(v['ai'], (cc - 1,)) < _r(1e-12), _get(v['ci'], (cc - 1,)) < _r(1e-12)))
        taken_58=mask_58
        _put(v,'qmlt',((cc)-1,),-_get(v['qi'], (cc - 1,)) * v['i_dt'],mask_58,xp)  # WRF:1394
        _put(v,'nmlt',((cc)-1,),_max(-_get(v['ni'], (cc - 1,)) * v['i_dt'], _get(v['ni'], (cc - 1,)) * _get(v['qmlt'], (cc - 1,)) / _get(v['qi'], (cc - 1,)) - _get(v['dnmltri'], (cc - 1,))),mask_56,xp)  # WRF:1398
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_56,xp)  # WRF:1400
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_56,xp)  # WRF:1401
        _put(v,'tmpmelt',None,v['fourthirdspi'] * v['alphstr'] * tables['gamma_tab'][v['gi'] - 1,] * v['i_gammnu'],mask_56,xp)  # WRF:1402
        _put(v,'amlt',((cc)-1,),_r(1.0) / v['tmpmelt'] * (_r(1.0) / _get(v['rhobar'], (cc - 1,))) * _get(v['qmlt'], (cc - 1,)) + _get(v['ai'], (cc - 1,)) * _get(v['nmlt'], (cc - 1,)) / _get(v['ni'], (cc - 1,)),mask_56,xp)  # WRF:1404
        _put(v,'cmlt',((cc)-1,),_get(v['amlt'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) / _get(v['ani'], (cc - 1,)) * (_r(1.0) + _r(2.0) * _get(v['deltastr'], (cc - 1,))) / (_r(2.0) + _get(v['deltastr'], (cc - 1,))),mask_56,xp)  # WRF:1405
    mask_59=xp.logical_and(mask_2,xp.logical_and(xp.logical_and(freeze_qc, v['qc'] > v['qsmall']), v['temp'] < v['t0'] - _r(35.0)))
    taken_59=mask_59
    _put(v,'mim',None,v['qc'] * v['i_dt'],mask_59,xp)  # WRF:1418
    _put(v,'nim',None,v['nc'] * v['i_dt'],mask_59,xp)  # WRF:1419
    mask_60=xp.logical_and(mask_2,xp.logical_and(xp.logical_and(freeze_qc, v['qr'] > v['qsmall']), v['temp'] < v['t0'] - _r(35.0)))
    taken_60=mask_60
    _put(v,'mimr',None,v['qr'] * v['i_dt'],mask_60,xp)  # WRF:1423
    _put(v,'nimr',None,v['nr'] * v['i_dt'],mask_60,xp)  # WRF:1424
    mask_61=xp.logical_and(mask_2,xp.logical_and(v['qr'] > v['qsmall'], v['temp'] < v['t0'] - _r(4.0)))
    taken_61=mask_61
    _put(v,'nr',None,_max(v['nr'], v['qnsmall']),mask_61,xp)  # WRF:1429
    _put(v,'lamr',None,xp.power(v['pi'] * v['rhow'] * v['nr'] / v['qr'], _r(0.333333333)),mask_61,xp)  # WRF:1430
    _put(v,'n0rr',None,v['nr'] * v['lamr'],mask_61,xp)  # WRF:1431
    mask_62=xp.logical_and(mask_61,v['lamr'] < v['lamminr'])
    taken_62=mask_62
    _put(v,'lamr',None,v['lamminr'],mask_62,xp)  # WRF:1433
    _put(v,'n0rr',None,_r(16269267968.0 if v['qv'].dtype == 'float32' else 16269262807.163687) * v['qr'] / (v['pi'] * v['rhow']),mask_62,xp)  # WRF:1434
    _put(v,'nr',None,v['n0rr'] / v['lamr'],mask_62,xp)  # WRF:1435
    mask_63=xp.logical_and(xp.logical_and(mask_61,xp.logical_not(taken_62)),v['lamr'] > v['lammaxr'])
    taken_62=xp.logical_or(taken_62,mask_63)
    _put(v,'lamr',None,v['lammaxr'],mask_63,xp)  # WRF:1437
    _put(v,'n0rr',None,_r(6.250000125255483e+18 if v['qv'].dtype == 'float32' else 6.249999999999996e+18) * v['qr'] / (v['pi'] * v['rhow']),mask_63,xp)  # WRF:1438
    _put(v,'nr',None,v['n0rr'] / v['lamr'],mask_63,xp)  # WRF:1439
    _put(v,'mbiggr',None,_r(20.0) * v['pi'] * v['pi'] * v['rhow'] * _r(100.0) * v['nr'] * (xp.exp(_r(0.66) * (v['t0'] - v['temp'])) - _r(1.0)) / (v['lamr'] * v['lamr'] * v['lamr']) / (v['lamr'] * v['lamr'] * v['lamr']),mask_61,xp)  # WRF:1442
    _put(v,'nbiggr',None,v['pi'] * _r(100.0) * v['nr'] * (xp.exp(_r(0.66) * (v['t0'] - v['temp'])) - _r(1.0)) / (v['lamr'] * v['lamr'] * v['lamr']),mask_61,xp)  # WRF:1443
    _put(v,'mbiggr',None,_min(v['mbiggr'], v['qr'] * v['i_dt']),mask_61,xp)  # WRF:1444
    _put(v,'nbiggr',None,_min(v['nbiggr'], v['nr'] * v['i_dt']),mask_61,xp)  # WRF:1445
    mask_64=xp.logical_and(mask_2,xp.logical_and(v['temp'] < v['t0'], v['sup'] >= _r(0.0)))
    taken_64=mask_64
    _put(v,'a_demott',None,_r(5.94e-05),mask_64,xp)  # WRF:1451
    _put(v,'b_demott',None,_r(3.33),mask_64,xp)  # WRF:1452
    _put(v,'c_demott',None,_r(0.0264),mask_64,xp)  # WRF:1453
    _put(v,'d_demott',None,_r(0.0033),mask_64,xp)  # WRF:1454
    _put(v,'inrate',None,v['a_demott'] * xp.power(_r(273.16) - v['temp'], v['b_demott']) * xp.power(_r(0.03), v['c_demott'] * (_r(273.16) - v['temp']) + v['d_demott']),mask_64,xp)  # WRF:1458
    _put(v,'inrate',None,v['inrate'] * _r(1000.0),mask_64,xp)  # WRF:1460
    _put(v,'inrate',None,v['inrate'] * v['i_rhoair'],mask_64,xp)  # WRF:1461
    mask_65=xp.logical_and(mask_64,(_get(v['ni'], (ice1 - 1,)) + _get(v['ni'], (ice2 - 1,)) + v['inrate']) * v['rhoair'] / _r(1000.0) <= _r(10000.0))
    taken_65=mask_65
    _put(v,'mnuccd',None,v['inrate'] * (v['fourthirdspi'] * v['rhoi'] * _r(7.999999539329027e-18 if v['qv'].dtype == 'float32' else 7.999999999999999e-18)) * v['i_dt'],mask_65,xp)  # WRF:1464
    _put(v,'nnuccd',None,v['inrate'] * v['i_dt'],mask_65,xp)  # WRF:1465
    mask_66=xp.logical_and(mask_64,xp.logical_not(taken_65))
    _put(v,'curnum',None,(_get(v['ni'], (ice1 - 1,)) + _get(v['ni'], (ice2 - 1,))) * v['rhoair'] / _r(1000.0),mask_66,xp)  # WRF:1467
    _put(v,'ratel',None,_max(_r(0.0), _r(10000.0) - v['curnum']),mask_66,xp)  # WRF:1468
    _put(v,'ratekg',None,v['ratel'] * _r(1000.0) * v['i_rhoair'],mask_66,xp)  # WRF:1469
    _put(v,'mnuccd',None,v['ratekg'] * (v['fourthirdspi'] * v['rhoi'] * _r(7.999999539329027e-18 if v['qv'].dtype == 'float32' else 7.999999999999999e-18)) * v['i_dt'],mask_66,xp)  # WRF:1470
    _put(v,'nnuccd',None,v['ratekg'] * v['i_dt'],mask_66,xp)  # WRF:1471
    mask_67=xp.logical_and(mask_2,splinters)
    taken_67=mask_67
    mask_68=xp.logical_and(mask_67,xp.logical_and(v['temp'] < _r(270.16), v['temp'] > _r(265.16)))
    taken_68=mask_68
    mask_69=xp.logical_and(mask_68,v['temp'] > _r(270.16))
    taken_69=mask_69
    _put(v,'fmult',None,_r(0.0),mask_69,xp)  # WRF:1479
    mask_70=xp.logical_and(xp.logical_and(mask_68,xp.logical_not(taken_69)),xp.logical_and(v['temp'] <= _r(270.16), v['temp'] > _r(268.16)))
    taken_69=xp.logical_or(taken_69,mask_70)
    _put(v,'fmult',None,(_r(270.16) - v['temp']) / _r(2.0),mask_70,xp)  # WRF:1481
    mask_71=xp.logical_and(xp.logical_and(mask_68,xp.logical_not(taken_69)),xp.logical_and(v['temp'] >= _r(265.16), v['temp'] <= _r(268.16)))
    taken_69=xp.logical_or(taken_69,mask_71)
    _put(v,'fmult',None,(v['temp'] - _r(265.16)) / _r(3.0),mask_71,xp)  # WRF:1483
    mask_72=xp.logical_and(xp.logical_and(mask_68,xp.logical_not(taken_69)),v['temp'] < _r(265.16))
    taken_69=xp.logical_or(taken_69,mask_72)
    _put(v,'fmult',None,_r(0.0),mask_72,xp)  # WRF:1485
    for cc in range(1,cat+1):
        mask_73=xp.logical_and(mask_67,_get(v['prdr'], (cc - 1,)) > _r(0.0))
        taken_73=mask_73
        _put(v,'nmult',((cc)-1,),_r(350000.0) * _get(v['prdr'], (cc - 1,)) * v['fmult'] * _r(1000.0),mask_73,xp)  # WRF:1491
        _put(v,'qmult',((cc)-1,),_get(v['nmult'], (cc - 1,)) * v['fourthirdspi'] * v['rhoi'] * _r(1.2499998721854691e-16 if v['qv'].dtype == 'float32' else 1.2500000000000003e-16),mask_73,xp)  # WRF:1492
        _put(v,'qmult',((cc)-1,),_min(_get(v['qmult'], (cc - 1,)), _get(v['prdr'], (cc - 1,))),mask_73,xp)  # WRF:1493
        _put(v,'prdr',((cc)-1,),_get(v['prdr'], (cc - 1,)) - _get(v['qmult'], (cc - 1,)),mask_73,xp)  # WRF:1494
    mask_74=xp.logical_and(mask_2,v['temp'] <= v['t0'])
    taken_74=mask_74
    for cc in range(1,cat+1):
        _put(v,'ni',((cc)-1,),_max(_get(v['ni'], (cc - 1,)), v['qnsmall']),mask_74,xp)  # WRF:1503
        _put(v,'ai',((cc)-1,),_max(_get(v['ai'], (cc - 1,)), v['qasmall']),mask_74,xp)  # WRF:1504
        _put(v,'ci',((cc)-1,),_max(_get(v['ci'], (cc - 1,)), v['qasmall']),mask_74,xp)  # WRF:1505
    _put(v,'dn1',None,_r(2.0) * xp.power(_get(v['ai'], (ice1 - 1,)) * _get(v['ai'], (ice1 - 1,)) / (_get(v['ci'], (ice1 - 1,)) * _get(v['ni'], (ice1 - 1,))), _r(0.333333333333)),mask_74,xp)  # WRF:1509
    _put(v,'dn2',None,_r(2.0) * xp.power(_get(v['ci'], (ice2 - 1,)) * _get(v['ci'], (ice2 - 1,)) / (_get(v['ai'], (ice2 - 1,)) * _get(v['ni'], (ice2 - 1,))), _r(0.333333333333)),mask_74,xp)  # WRF:1510
    _put(v,'dn3',None,_r(2.0) * xp.power(_get(v['ai'], (ice3 - 1,)) * _get(v['ai'], (ice3 - 1,)) / (_get(v['ci'], (ice3 - 1,)) * _get(v['ni'], (ice3 - 1,))), _r(0.333333333333)),mask_74,xp)  # WRF:1511
    _put(v,'dn1',None,_min(v['dn1'], _r(0.01)),mask_74,xp)  # WRF:1513
    _put(v,'dn2',None,_min(v['dn2'], _r(0.01)),mask_74,xp)  # WRF:1514
    _put(v,'dn3',None,_min(v['dn3'], _r(0.01)),mask_74,xp)  # WRF:1515
    _put(v,'dn1',None,_max(v['dn1'], _r(1e-06)),mask_74,xp)  # WRF:1517
    _put(v,'dn2',None,_max(v['dn2'], _r(1e-06)),mask_74,xp)  # WRF:1518
    _put(v,'dn3',None,_max(v['dn3'], _r(1e-06)),mask_74,xp)  # WRF:1519
    _put(v,'gamma_arg',None,v['nu'] - _r(1.0) + _get(v['deltastr'], (1 - 1,)),mask_74,xp)  # WRF:1523
    _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_74,xp)  # WRF:1524
    _put(v,'phiagg1',None,_get(v['ci'], (1 - 1,)) / _get(v['ai'], (1 - 1,)) * tables['gamma_tab'][v['gi'] - 1,] * v['i_gammnu'],mask_74,xp)  # WRF:1525
    _put(v,'gamma_arg',None,v['nu'] - _r(1.0) + _get(v['deltastr'], (2 - 1,)),mask_74,xp)  # WRF:1527
    _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_74,xp)  # WRF:1528
    _put(v,'phiagg2',None,_get(v['ci'], (2 - 1,)) / _get(v['ai'], (2 - 1,)) * tables['gamma_tab'][v['gi'] - 1,] * v['i_gammnu'],mask_74,xp)  # WRF:1529
    _put(v,'phiagg1',None,_min(v['phiagg1'], _r(100.0)),mask_74,xp)  # WRF:1531
    _put(v,'phiagg1',None,_max(v['phiagg1'], _r(0.01)),mask_74,xp)  # WRF:1532
    _put(v,'phiagg2',None,_min(v['phiagg2'], _r(100.0)),mask_74,xp)  # WRF:1534
    _put(v,'phiagg2',None,_max(v['phiagg2'], _r(0.01)),mask_74,xp)  # WRF:1535
    mask_75=xp.logical_and(mask_74,xp.logical_or(xp.logical_or(_get(v['qi'], (ice1 - 1,)) > _r(1e-08), _get(v['qi'], (ice2 - 1,)) > _r(1e-08)), _get(v['qi'], (ice3 - 1,)) > _r(1e-08)))
    taken_75=mask_75
    call_out=_call('aggregation',(v['dt'],v['rhoair'],v['temp'],_get(v['qi'], (ice1 - 1,)),_get(v['ni'], (ice1 - 1,)),v['dn1'],_get(v['qi'], (ice2 - 1,)),_get(v['ni'], (ice2 - 1,)),v['dn2'],_get(v['qi'], (ice3 - 1,)),_get(v['ni'], (ice3 - 1,)),v['dn3'],_get(v['rhobar'], (ice1 - 1,)),_get(v['rhobar'], (ice2 - 1,)),v['phiagg1'],v['phiagg2'],tables['coltab'],tables['coltabn'],_get(v['qagg'], (ice1 - 1,)),_get(v['qagg'], (ice2 - 1,)),_get(v['qagg'], (ice3 - 1,)),_get(v['nagg'], (ice1 - 1,)),_get(v['nagg'], (ice2 - 1,)),_get(v['nagg'], (ice3 - 1,)),),tables,xp)  # WRF:1542
    _put(v,'dn3',None,call_out[0],mask_75,xp)  # WRF:1542
    _put(v,'qagg',((ice1)-1,),call_out[1],mask_75,xp)  # WRF:1542
    _put(v,'qagg',((ice2)-1,),call_out[2],mask_75,xp)  # WRF:1542
    _put(v,'qagg',((ice3)-1,),call_out[3],mask_75,xp)  # WRF:1542
    _put(v,'nagg',((ice1)-1,),call_out[4],mask_75,xp)  # WRF:1542
    _put(v,'nagg',((ice2)-1,),call_out[5],mask_75,xp)  # WRF:1542
    _put(v,'nagg',((ice3)-1,),call_out[6],mask_75,xp)  # WRF:1542
    if observer is not None: observer(1,v,mask_2)  # Original checkpoint WRF:1553
    mask_76=xp.logical_and(mask_2,v['qr'] > v['qsmall'])
    taken_76=mask_76
    _put(v,'nr',None,_max(v['nr'], v['qnsmall']),mask_76,xp)  # WRF:1554
    _put(v,'lamr',None,xp.power(v['pi'] * v['rhow'] * v['nr'] / v['qr'], _r(0.333333333)),mask_76,xp)  # WRF:1555
    _put(v,'n0rr',None,v['nr'] * v['lamr'],mask_76,xp)  # WRF:1556
    mask_77=xp.logical_and(mask_76,v['lamr'] < v['lamminr'])
    taken_77=mask_77
    _put(v,'lamr',None,v['lamminr'],mask_77,xp)  # WRF:1558
    _put(v,'n0rr',None,_r(16269267968.0 if v['qv'].dtype == 'float32' else 16269262807.163687) * v['qr'] / (v['pi'] * v['rhow']),mask_77,xp)  # WRF:1559
    _put(v,'nr',None,v['n0rr'] / v['lamr'],mask_77,xp)  # WRF:1560
    mask_78=xp.logical_and(xp.logical_and(mask_76,xp.logical_not(taken_77)),v['lamr'] > v['lammaxr'])
    taken_77=xp.logical_or(taken_77,mask_78)
    _put(v,'lamr',None,v['lammaxr'],mask_78,xp)  # WRF:1562
    _put(v,'n0rr',None,_r(6.250000125255483e+18 if v['qv'].dtype == 'float32' else 6.249999999999996e+18) * v['qr'] / (v['pi'] * v['rhow']),mask_78,xp)  # WRF:1563
    _put(v,'nr',None,v['n0rr'] / v['lamr'],mask_78,xp)  # WRF:1564
    mask_79=xp.logical_and(mask_2,v['qc'] >= _r(1e-06))
    taken_79=mask_79
    _put(v,'prc',None,_r(1350.0) * xp.power(v['qc'], _r(2.47)) * xp.power(v['nc'] / _r(1000000.0) * v['rhoair'], _r(-1.79)),mask_79,xp)  # WRF:1573
    _put(v,'nprc1',None,v['prc'] / (_r(4.0) / _r(3.0) * v['pi'] * v['rhow'] * _r(1.5624998243499687e-14 if v['qv'].dtype == 'float32' else 1.5625000000000003e-14)),mask_79,xp)  # WRF:1575
    _put(v,'nprc',None,v['prc'] / (v['qc'] / v['nc']),mask_79,xp)  # WRF:1576
    _put(v,'nprc',None,_min(v['nprc'], v['nc'] * v['i_dt']),mask_79,xp)  # WRF:1577
    _put(v,'nprc1',None,_min(v['nprc1'], v['nprc']),mask_79,xp)  # WRF:1578
    mask_80=xp.logical_and(mask_2,xp.logical_and(v['qr'] >= _r(1e-08), v['qc'] >= _r(1e-08)))
    taken_80=mask_80
    _put(v,'dum',None,v['qc'] * v['qr'],mask_80,xp)  # WRF:1584
    _put(v,'pra',None,_r(67.0) * xp.power(v['dum'], _r(1.15)),mask_80,xp)  # WRF:1585
    _put(v,'npra',None,v['pra'] / (v['qc'] / v['nc']),mask_80,xp)  # WRF:1586
    mask_81=xp.logical_and(mask_2,v['qr'] >= _r(1e-08))
    taken_81=mask_81
    _put(v,'dum1',None,_r(0.0003),mask_81,xp)  # WRF:1591
    mask_82=xp.logical_and(mask_81,_r(1.0) / v['lamr'] < v['dum1'])
    taken_82=mask_82
    _put(v,'dum',None,_r(1.0),mask_82,xp)  # WRF:1593
    mask_83=xp.logical_and(xp.logical_and(mask_81,xp.logical_not(taken_82)),_r(1.0) / v['lamr'] >= v['dum1'])
    taken_82=xp.logical_or(taken_82,mask_83)
    _put(v,'dum',None,_r(2.0) - xp.exp(_r(2300.0) * (_r(1.0) / v['lamr'] - v['dum1'])),mask_83,xp)  # WRF:1595
    _put(v,'nragg',None,_r(-5.78) * v['dum'] * v['nr'] * v['qr'] * v['rhoair'],mask_81,xp)  # WRF:1597
    mask_84=xp.logical_and(mask_2,v['qr'] > v['qsmall'])
    taken_84=mask_84
    _put(v,'epsr',None,_r(2.0) * v['pi'] * v['n0rr'] * v['rhoair'] * v['dv'] * (v['f1r'] / (v['lamr'] * v['lamr']) + v['f2r'] * xp.power(v['arn'] * v['rhoair'] / v['mu'], _r(0.5)) * xp.power(v['nsch'], _r(0.333333333)) * gamma(_r(5.0) / _r(2.0) + v['br'] / _r(2.0), xp=xp) / xp.power(v['lamr'], _r(5.0) / _r(2.0) + v['br'] / _r(2.0))),mask_84,xp)  # WRF:1602
    mask_85=xp.logical_and(mask_2,xp.logical_not(taken_84))
    _put(v,'epsr',None,_r(0.0),mask_85,xp)  # WRF:1608
    mask_86=xp.logical_and(mask_2,v['qv'] < v['qvs'])
    taken_86=mask_86
    _put(v,'pre',None,v['epsr'] * (v['qv'] - v['qvs']) / v['ab'],mask_86,xp)  # WRF:1612
    _put(v,'pre',None,_min(v['pre'], _r(0.0)),mask_86,xp)  # WRF:1613
    _put(v,'dumt',None,v['temp'] + v['xxlv'] * v['i_cp'] * v['pre'] * v['dt'],mask_86,xp)  # WRF:1614
    _put(v,'dumqv',None,v['qv'] - v['pre'] * v['dt'],mask_86,xp)  # WRF:1615
    _put(v,'dumqvs',None,_r(0.622) * polysvp(v['dumt'], 0, xp=xp) / (v['pres_e'] - polysvp(v['dumt'], 0, xp=xp)),mask_86,xp)  # WRF:1616
    mask_87=xp.logical_and(mask_86,xp.logical_and(v['pre'] < _r(0.0), v['dumqv'] / v['dumqvs'] > _r(1.0)))
    taken_87=mask_87
    _put(v,'pre',None,(v['qv'] - v['qvs']) * v['i_dt'] / v['ab'],mask_87,xp)  # WRF:1619
    mask_88=xp.logical_and(mask_2,xp.logical_not(taken_86))
    _put(v,'pre',None,_r(0.0),mask_88,xp)  # WRF:1622
    mask_89=xp.logical_and(mask_2,v['pre'] < _r(0.0))
    taken_89=mask_89
    _put(v,'dum',None,v['pre'] * v['dt'] / v['qr'],mask_89,xp)  # WRF:1627
    _put(v,'dum',None,_max(_r(-1.0), v['dum']),mask_89,xp)  # WRF:1628
    _put(v,'npre',None,v['dum'] * v['nr'] / v['dt'],mask_89,xp)  # WRF:1629
    if observer is not None: observer(2,v,mask_2)  # Original checkpoint WRF:1647
    _put(v,'qcrimesum',None,_r(0.0),mask_2,xp)  # WRF:1647
    for cc in range(1,cat+1):
        _put(v,'qcrimesum',None,v['qcrimesum'] + _get(v['prdr'], (cc - 1,)) * _get(v['qcrimefrac'], (cc - 1,)),mask_2,xp)  # WRF:1649
    _put(v,'sink',None,(v['prc'] + v['pra'] + v['qcrimesum'] + v['mim']) * v['dt'],mask_2,xp)  # WRF:1652
    _put(v,'source',None,v['qc'],mask_2,xp)  # WRF:1653
    mask_90=xp.logical_and(mask_2,xp.logical_and(v['sink'] > v['source'], v['qc'] > v['qsmall']))
    taken_90=mask_90
    _put(v,'ratio',None,v['source'] / v['sink'],mask_90,xp)  # WRF:1655
    _put(v,'prc',None,v['prc'] * v['ratio'],mask_90,xp)  # WRF:1656
    _put(v,'pra',None,v['pra'] * v['ratio'],mask_90,xp)  # WRF:1657
    _put(v,'mim',None,v['mim'] * v['ratio'],mask_90,xp)  # WRF:1658
    for cc in range(1,cat+1):
        _put(v,'prdr',((cc)-1,),_get(v['prdr'], (cc - 1,)) * v['ratio'],mask_90,xp)  # WRF:1660
        _put(v,'ardr',((cc)-1,),_get(v['ardr'], (cc - 1,)) * v['ratio'],mask_90,xp)  # WRF:1661
        _put(v,'crdr',((cc)-1,),_get(v['crdr'], (cc - 1,)) * v['ratio'],mask_90,xp)  # WRF:1662
    _put(v,'qcrimesumr',None,_r(0.0),mask_2,xp)  # WRF:1668
    for cc in range(1,cat+1):
        _put(v,'qcrimesumr',None,v['qcrimesumr'] + _get(v['prdr'], (cc - 1,)) * (_r(1.0) - _get(v['qcrimefrac'], (cc - 1,))),mask_2,xp)  # WRF:1670
    _put(v,'sink',None,(-v['pre'] + v['qcrimesumr'] + v['mimr'] + v['mbiggr']) * v['dt'],mask_2,xp)  # WRF:1673
    _put(v,'source',None,v['qr'] + (v['pra'] + v['prc']) * v['dt'],mask_2,xp)  # WRF:1674
    for cc in range(1,cat+1):
        _put(v,'source',None,v['source'] + -_get(v['qmlt'], (cc - 1,)) * v['dt'],mask_2,xp)  # WRF:1676
    for cc in range(1,cat+1):
        _put(v,'sink',None,v['sink'] + _get(v['dqrfzri'], (cc - 1,)) * v['dt'],mask_2,xp)  # WRF:1679
    mask_91=xp.logical_and(mask_2,xp.logical_and(v['sink'] > v['source'], v['qr'] > v['qsmall']))
    taken_91=mask_91
    _put(v,'ratio',None,v['source'] / v['sink'],mask_91,xp)  # WRF:1682
    for cc in range(1,cat+1):
        _put(v,'prdr',((cc)-1,),_get(v['prdr'], (cc - 1,)) * v['ratio'],mask_91,xp)  # WRF:1684
        _put(v,'ardr',((cc)-1,),_get(v['ardr'], (cc - 1,)) * v['ratio'],mask_91,xp)  # WRF:1685
        _put(v,'crdr',((cc)-1,),_get(v['crdr'], (cc - 1,)) * v['ratio'],mask_91,xp)  # WRF:1686
        _put(v,'qi_qr_nrn',((cc)-1,),_get(v['qi_qr_nrn'], (cc - 1,)) * v['ratio'],mask_91,xp)  # WRF:1687
        _put(v,'dqrfzri',((cc)-1,),_get(v['dqrfzri'], (cc - 1,)) * v['ratio'],mask_91,xp)  # WRF:1688
        _put(v,'dqifzri',((cc)-1,),_get(v['dqifzri'], (cc - 1,)) * v['ratio'],mask_91,xp)  # WRF:1689
        _put(v,'dnfzri',((cc)-1,),_get(v['dnfzri'], (cc - 1,)) * v['ratio'],mask_91,xp)  # WRF:1690
    _put(v,'mimr',None,v['mimr'] * v['ratio'],mask_91,xp)  # WRF:1692
    _put(v,'nimr',None,v['nimr'] * v['ratio'],mask_91,xp)  # WRF:1693
    _put(v,'mbiggr',None,v['mbiggr'] * v['ratio'],mask_91,xp)  # WRF:1694
    _put(v,'nbiggr',None,v['nbiggr'] * v['ratio'],mask_91,xp)  # WRF:1695
    _put(v,'pre',None,v['pre'] * v['ratio'],mask_91,xp)  # WRF:1696
    mask_92=xp.logical_and(mask_2,v['igr'] <= _r(1.0))
    taken_92=mask_92
    _put(v,'sink',None,(-_get(v['qmlt'], (ice1 - 1,)) - _get(v['qagg'], (ice1 - 1,))) * v['dt'],mask_92,xp)  # WRF:1702
    _put(v,'source',None,_get(v['qi'], (ice1 - 1,)) + _get(v['prdr'], (ice1 - 1,)) * v['dt'],mask_92,xp)  # WRF:1703
    mask_93=xp.logical_and(mask_92,_get(v['prd'], (ice1 - 1,)) > _r(0.0))
    taken_93=mask_93
    _put(v,'source',None,v['source'] + _get(v['prd'], (ice1 - 1,)) * v['dt'],mask_93,xp)  # WRF:1705
    mask_94=xp.logical_and(mask_92,xp.logical_not(taken_93))
    _put(v,'sink',None,v['sink'] + -_get(v['prd'], (ice1 - 1,)) * v['dt'],mask_94,xp)  # WRF:1707
    _put(v,'source',None,v['source'] + (v['mnuccd'] + v['mim'] + v['mimr'] + v['mbiggr'] + _get(v['dqifzri'], (ice2 - 1,)) + _get(v['dqifzri'], (ice3 - 1,))) * v['dt'],mask_92,xp)  # WRF:1711
    for cc in range(1,cat+1):
        _put(v,'source',None,v['source'] + (_get(v['qmult'], (cc - 1,)) + _get(v['dqrfzri'], (cc - 1,))) * v['dt'],mask_92,xp)  # WRF:1713
    mask_95=xp.logical_and(mask_92,xp.logical_and(v['sink'] > v['source'], _get(v['qi'], (ice1 - 1,)) > v['qsmall']))
    taken_95=mask_95
    _put(v,'ratio',None,v['source'] / v['sink'],mask_95,xp)  # WRF:1717
    _put(v,'qmlt',((ice1)-1,),_get(v['qmlt'], (ice1 - 1,)) * v['ratio'],mask_95,xp)  # WRF:1718
    _put(v,'qagg',((ice1)-1,),_get(v['qagg'], (ice1 - 1,)) * v['ratio'],mask_95,xp)  # WRF:1719
    _put(v,'nagg',((ice1)-1,),_get(v['nagg'], (ice1 - 1,)) * v['ratio'],mask_95,xp)  # WRF:1720
    mask_96=xp.logical_and(mask_95,_get(v['prd'], (ice1 - 1,)) < _r(0.0))
    taken_96=mask_96
    _put(v,'prd',((ice1)-1,),_get(v['prd'], (ice1 - 1,)) * v['ratio'],mask_96,xp)  # WRF:1722
    _put(v,'sink',None,(_get(v['dqifzri'], (ice2 - 1,)) - _get(v['qmlt'], (ice2 - 1,)) - _get(v['qagg'], (ice2 - 1,))) * v['dt'],mask_92,xp)  # WRF:1727
    _put(v,'source',None,_get(v['qi'], (ice2 - 1,)) + _get(v['prdr'], (ice2 - 1,)) * v['dt'],mask_92,xp)  # WRF:1728
    mask_97=xp.logical_and(mask_92,_get(v['prd'], (ice2 - 1,)) > _r(0.0))
    taken_97=mask_97
    _put(v,'source',None,v['source'] + _get(v['prd'], (ice2 - 1,)) * v['dt'],mask_97,xp)  # WRF:1730
    mask_98=xp.logical_and(mask_92,xp.logical_not(taken_97))
    _put(v,'sink',None,v['sink'] + -_get(v['prd'], (ice2 - 1,)) * v['dt'],mask_98,xp)  # WRF:1732
    mask_99=xp.logical_and(mask_92,xp.logical_and(v['sink'] > v['source'], _get(v['qi'], (ice2 - 1,)) > v['qsmall']))
    taken_99=mask_99
    _put(v,'ratio',None,v['source'] / v['sink'],mask_99,xp)  # WRF:1736
    _put(v,'dqifzri',((ice2)-1,),_get(v['dqifzri'], (ice2 - 1,)) * v['ratio'],mask_99,xp)  # WRF:1737
    _put(v,'dqrfzri',((ice2)-1,),_get(v['dqrfzri'], (ice2 - 1,)) * v['ratio'],mask_99,xp)  # WRF:1738
    _put(v,'dnfzri',((ice2)-1,),_get(v['dnfzri'], (ice2 - 1,)) * v['ratio'],mask_99,xp)  # WRF:1739
    _put(v,'qmlt',((ice2)-1,),_get(v['qmlt'], (ice2 - 1,)) * v['ratio'],mask_99,xp)  # WRF:1740
    _put(v,'qagg',((ice2)-1,),_get(v['qagg'], (ice2 - 1,)) * v['ratio'],mask_99,xp)  # WRF:1741
    _put(v,'nagg',((ice2)-1,),_get(v['nagg'], (ice2 - 1,)) * v['ratio'],mask_99,xp)  # WRF:1742
    mask_100=xp.logical_and(mask_99,_get(v['prd'], (ice2 - 1,)) < _r(0.0))
    taken_100=mask_100
    _put(v,'prd',((ice2)-1,),_get(v['prd'], (ice2 - 1,)) * v['ratio'],mask_100,xp)  # WRF:1744
    _put(v,'sink',None,(_get(v['dqifzri'], (ice3 - 1,)) - _get(v['qmlt'], (ice3 - 1,))) * v['dt'],mask_92,xp)  # WRF:1748
    _put(v,'source',None,_get(v['qi'], (ice3 - 1,)) + (_get(v['prdr'], (ice3 - 1,)) + _get(v['qagg'], (ice3 - 1,))) * v['dt'],mask_92,xp)  # WRF:1749
    mask_101=xp.logical_and(mask_92,_get(v['prd'], (ice3 - 1,)) > _r(0.0))
    taken_101=mask_101
    _put(v,'source',None,v['source'] + _get(v['prd'], (ice3 - 1,)) * v['dt'],mask_101,xp)  # WRF:1751
    mask_102=xp.logical_and(mask_92,xp.logical_not(taken_101))
    _put(v,'sink',None,v['sink'] + -_get(v['prd'], (ice3 - 1,)) * v['dt'],mask_102,xp)  # WRF:1753
    mask_103=xp.logical_and(mask_92,xp.logical_and(v['sink'] > v['source'], _get(v['qi'], (ice3 - 1,)) > v['qsmall']))
    taken_103=mask_103
    _put(v,'ratio',None,v['source'] / v['sink'],mask_103,xp)  # WRF:1757
    _put(v,'dqifzri',((ice3)-1,),_get(v['dqifzri'], (ice3 - 1,)) * v['ratio'],mask_103,xp)  # WRF:1758
    _put(v,'dqrfzri',((ice3)-1,),_get(v['dqrfzri'], (ice3 - 1,)) * v['ratio'],mask_103,xp)  # WRF:1759
    _put(v,'dnfzri',((ice3)-1,),_get(v['dnfzri'], (ice3 - 1,)) * v['ratio'],mask_103,xp)  # WRF:1760
    _put(v,'qmlt',((ice3)-1,),_get(v['qmlt'], (ice3 - 1,)) * v['ratio'],mask_103,xp)  # WRF:1761
    mask_104=xp.logical_and(mask_103,_get(v['prd'], (ice3 - 1,)) < _r(0.0))
    taken_104=mask_104
    _put(v,'prd',((ice3)-1,),_get(v['prd'], (ice3 - 1,)) * v['ratio'],mask_104,xp)  # WRF:1763
    mask_105=xp.logical_and(mask_2,xp.logical_not(taken_92))
    _put(v,'sink',None,(-_get(v['qmlt'], (ice2 - 1,)) - _get(v['qagg'], (ice2 - 1,))) * v['dt'],mask_105,xp)  # WRF:1768
    _put(v,'source',None,_get(v['qi'], (ice2 - 1,)) + _get(v['prdr'], (ice2 - 1,)) * v['dt'],mask_105,xp)  # WRF:1769
    mask_106=xp.logical_and(mask_105,_get(v['prd'], (ice2 - 1,)) > _r(0.0))
    taken_106=mask_106
    _put(v,'source',None,v['source'] + _get(v['prd'], (ice2 - 1,)) * v['dt'],mask_106,xp)  # WRF:1771
    mask_107=xp.logical_and(mask_105,xp.logical_not(taken_106))
    _put(v,'sink',None,v['sink'] + -_get(v['prd'], (ice2 - 1,)) * v['dt'],mask_107,xp)  # WRF:1773
    _put(v,'source',None,v['source'] + (v['mnuccd'] + v['mim'] + v['mimr'] + v['mbiggr'] + _get(v['dqifzri'], (ice1 - 1,)) + _get(v['dqifzri'], (ice3 - 1,))) * v['dt'],mask_105,xp)  # WRF:1777
    for cc in range(1,cat+1):
        _put(v,'source',None,v['source'] + (_get(v['qmult'], (cc - 1,)) + _get(v['dqrfzri'], (cc - 1,))) * v['dt'],mask_105,xp)  # WRF:1779
    mask_108=xp.logical_and(mask_105,xp.logical_and(v['sink'] > v['source'], _get(v['qi'], (ice2 - 1,)) > v['qsmall']))
    taken_108=mask_108
    _put(v,'ratio',None,v['source'] / v['sink'],mask_108,xp)  # WRF:1783
    _put(v,'qmlt',((ice2)-1,),_get(v['qmlt'], (ice2 - 1,)) * v['ratio'],mask_108,xp)  # WRF:1784
    _put(v,'qagg',((ice2)-1,),_get(v['qagg'], (ice2 - 1,)) * v['ratio'],mask_108,xp)  # WRF:1785
    _put(v,'nagg',((ice2)-1,),_get(v['nagg'], (ice2 - 1,)) * v['ratio'],mask_108,xp)  # WRF:1786
    mask_109=xp.logical_and(mask_108,_get(v['prd'], (ice2 - 1,)) < _r(0.0))
    taken_109=mask_109
    _put(v,'prd',((ice2)-1,),_get(v['prd'], (ice2 - 1,)) * v['ratio'],mask_109,xp)  # WRF:1788
    _put(v,'sink',None,(_get(v['dqifzri'], (ice1 - 1,)) - _get(v['qmlt'], (ice1 - 1,)) - _get(v['qagg'], (ice1 - 1,))) * v['dt'],mask_105,xp)  # WRF:1793
    _put(v,'source',None,_get(v['qi'], (ice1 - 1,)) + _get(v['prdr'], (ice1 - 1,)) * v['dt'],mask_105,xp)  # WRF:1794
    mask_110=xp.logical_and(mask_105,_get(v['prd'], (ice1 - 1,)) > _r(0.0))
    taken_110=mask_110
    _put(v,'source',None,v['source'] + _get(v['prd'], (ice1 - 1,)) * v['dt'],mask_110,xp)  # WRF:1796
    mask_111=xp.logical_and(mask_105,xp.logical_not(taken_110))
    _put(v,'sink',None,v['sink'] + -_get(v['prd'], (ice1 - 1,)) * v['dt'],mask_111,xp)  # WRF:1798
    mask_112=xp.logical_and(mask_105,xp.logical_and(v['sink'] > v['source'], _get(v['qi'], (ice1 - 1,)) > v['qsmall']))
    taken_112=mask_112
    _put(v,'ratio',None,v['source'] / v['sink'],mask_112,xp)  # WRF:1802
    _put(v,'dqifzri',((ice1)-1,),_get(v['dqifzri'], (ice1 - 1,)) * v['ratio'],mask_112,xp)  # WRF:1803
    _put(v,'dqrfzri',((ice1)-1,),_get(v['dqrfzri'], (ice1 - 1,)) * v['ratio'],mask_112,xp)  # WRF:1804
    _put(v,'dnfzri',((ice1)-1,),_get(v['dnfzri'], (ice1 - 1,)) * v['ratio'],mask_112,xp)  # WRF:1805
    _put(v,'qmlt',((ice1)-1,),_get(v['qmlt'], (ice1 - 1,)) * v['ratio'],mask_112,xp)  # WRF:1806
    _put(v,'qagg',((ice1)-1,),_get(v['qagg'], (ice1 - 1,)) * v['ratio'],mask_112,xp)  # WRF:1807
    _put(v,'nagg',((ice1)-1,),_get(v['nagg'], (ice1 - 1,)) * v['ratio'],mask_112,xp)  # WRF:1808
    mask_113=xp.logical_and(mask_112,_get(v['prd'], (ice1 - 1,)) < _r(0.0))
    taken_113=mask_113
    _put(v,'prd',((ice1)-1,),_get(v['prd'], (ice1 - 1,)) * v['ratio'],mask_113,xp)  # WRF:1810
    _put(v,'sink',None,(_get(v['dqifzri'], (ice3 - 1,)) - _get(v['qmlt'], (ice3 - 1,))) * v['dt'],mask_105,xp)  # WRF:1814
    _put(v,'source',None,_get(v['qi'], (ice3 - 1,)) + (_get(v['prdr'], (ice3 - 1,)) + _get(v['qagg'], (ice3 - 1,))) * v['dt'],mask_105,xp)  # WRF:1815
    mask_114=xp.logical_and(mask_105,_get(v['prd'], (ice3 - 1,)) > _r(0.0))
    taken_114=mask_114
    _put(v,'source',None,v['source'] + _get(v['prd'], (ice3 - 1,)) * v['dt'],mask_114,xp)  # WRF:1817
    mask_115=xp.logical_and(mask_105,xp.logical_not(taken_114))
    _put(v,'sink',None,v['sink'] + -_get(v['prd'], (ice3 - 1,)) * v['dt'],mask_115,xp)  # WRF:1819
    mask_116=xp.logical_and(mask_105,xp.logical_and(v['sink'] > v['source'], _get(v['qi'], (ice3 - 1,)) > v['qsmall']))
    taken_116=mask_116
    _put(v,'ratio',None,v['source'] / v['sink'],mask_116,xp)  # WRF:1823
    _put(v,'dqifzri',((ice3)-1,),_get(v['dqifzri'], (ice3 - 1,)) * v['ratio'],mask_116,xp)  # WRF:1824
    _put(v,'dqrfzri',((ice3)-1,),_get(v['dqrfzri'], (ice3 - 1,)) * v['ratio'],mask_116,xp)  # WRF:1825
    _put(v,'dnfzri',((ice3)-1,),_get(v['dnfzri'], (ice3 - 1,)) * v['ratio'],mask_116,xp)  # WRF:1826
    _put(v,'qmlt',((ice3)-1,),_get(v['qmlt'], (ice3 - 1,)) * v['ratio'],mask_116,xp)  # WRF:1827
    mask_117=xp.logical_and(mask_116,_get(v['prd'], (ice3 - 1,)) < _r(0.0))
    taken_117=mask_117
    _put(v,'prd',((ice3)-1,),_get(v['prd'], (ice3 - 1,)) * v['ratio'],mask_117,xp)  # WRF:1829
    _put(v,'prdsum',None,_r(0.0),mask_2,xp)  # WRF:1835
    for cc in range(1,cat+1):
        _put(v,'prdsum',None,v['prdsum'] + _get(v['prd'], (cc - 1,)),mask_2,xp)  # WRF:1837
    _put(v,'vgflag',None,False,mask_2,xp)  # WRF:1839
    _put(v,'sink',None,v['prdsum'],mask_2,xp)  # WRF:1840
    _put(v,'source',None,_r(0.99) * v['sui'] * v['qvi'] / v['abi'] * v['i_dt'],mask_2,xp)  # WRF:1841
    mask_118=xp.logical_and(mask_2,xp.logical_and(v['sink'] > v['source'], v['sui'] > _r(0.0)))
    taken_118=mask_118
    _put(v,'vgflag',None,True,mask_118,xp)  # WRF:1843
    _put(v,'ratio',None,v['source'] / v['sink'],mask_118,xp)  # WRF:1844
    for cc in range(1,cat+1):
        _put(v,'prd',((cc)-1,),_get(v['prd'], (cc - 1,)) * v['ratio'],mask_118,xp)  # WRF:1846
    for cc in range(1,cat+1):
        mask_119=xp.logical_and(mask_2,xp.logical_and(_get(v['qi'], (cc - 1,)) + _get(v['prd'], (cc - 1,)) * v['dt'] > v['qsmall'], _get(v['qi'], (cc - 1,)) > v['qsmall']))
        taken_119=mask_119
        _put(v,'rhobar',((cc)-1,),_get(v['rhobar'], (cc - 1,)) * (_get(v['qi'], (cc - 1,)) / (_get(v['qi'], (cc - 1,)) + _get(v['prd'], (cc - 1,)) * v['dt'])) + _get(v['rhodepout'], (cc - 1,)) * (_r(1.0) - _get(v['qi'], (cc - 1,)) / (_get(v['qi'], (cc - 1,)) + _get(v['prd'], (cc - 1,)) * v['dt'])),mask_119,xp)  # WRF:1855
        mask_120=xp.logical_and(mask_2,xp.logical_not(taken_119))
        _put(v,'rhobar',((cc)-1,),v['rhoi'],mask_120,xp)  # WRF:1858
        _put(v,'qi',((cc)-1,),_get(v['qi'], (cc - 1,)) + _get(v['prd'], (cc - 1,)) * v['dt'],mask_2,xp)  # WRF:1860
        _put(v,'ni',((cc)-1,),_get(v['ni'], (cc - 1,)) + _get(v['nrd'], (cc - 1,)) * v['dt'],mask_2,xp)  # WRF:1861
        _put(v,'ai',((cc)-1,),_get(v['ai'], (cc - 1,)) + _get(v['ard'], (cc - 1,)) * v['dt'],mask_2,xp)  # WRF:1862
        _put(v,'ci',((cc)-1,),_get(v['ci'], (cc - 1,)) + _get(v['crd'], (cc - 1,)) * v['dt'],mask_2,xp)  # WRF:1863
        _put(v,'ni',((cc)-1,),_max(_get(v['ni'], (cc - 1,)), v['qnsmall']),mask_2,xp)  # WRF:1865
        _put(v,'ai',((cc)-1,),_max(_get(v['ai'], (cc - 1,)), v['qasmall']),mask_2,xp)  # WRF:1866
        _put(v,'ci',((cc)-1,),_max(_get(v['ci'], (cc - 1,)), v['qasmall']),mask_2,xp)  # WRF:1867
        mask_121=xp.logical_and(mask_2,xp.logical_and(v['vgflag'], _get(v['qi'], (cc - 1,)) > v['qsmall']))
        taken_121=mask_121
        _put(v,'alphstr',None,xp.power(v['ao'], _r(1.0) - _get(v['deltastr'], (cc - 1,))),mask_121,xp)  # WRF:1873
        _put(v,'alphv',None,v['fourthirdspi'] * v['alphstr'],mask_121,xp)  # WRF:1874
        _put(v,'betam',None,_r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_121,xp)  # WRF:1875
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_121,xp)  # WRF:1876
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_121,xp)  # WRF:1877
        _put(v,'ani',((cc)-1,),xp.power(_get(v['qi'], (cc - 1,)) * v['gammnu'] / (_get(v['rhobar'], (cc - 1,)) * _get(v['ni'], (cc - 1,)) * v['alphv'] * tables['gamma_tab'][v['gi'] - 1,]), _r(1.0) / v['betam']),mask_121,xp)  # WRF:1878
        _put(v,'cni',((cc)-1,),xp.power(v['ao'], _r(1.0) - _get(v['deltastr'], (cc - 1,))) * xp.power(_get(v['ani'], (cc - 1,)), _get(v['deltastr'], (cc - 1,))),mask_121,xp)  # WRF:1880
        _put(v,'ai',((cc)-1,),_get(v['ani'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_121,xp)  # WRF:1881
        _put(v,'ci',((cc)-1,),_get(v['cni'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_121,xp)  # WRF:1882
    for cc in range(1,cat+1):
        mask_122=xp.logical_and(mask_2,_get(v['qi'], (cc - 1,)) + _get(v['prdr'], (cc - 1,)) * v['dt'] > v['qsmall'])
        taken_122=mask_122
        mask_123=xp.logical_and(mask_122,_get(v['dry_growth'], (cc - 1,)))
        taken_123=mask_123
        _put(v,'rhobar',((cc)-1,),_get(v['rhobar'], (cc - 1,)) * (_get(v['qi'], (cc - 1,)) / (_get(v['qi'], (cc - 1,)) + _get(v['prdr'], (cc - 1,)) * v['dt'])) + _get(v['rhorimeout'], (cc - 1,)) * (_r(1.0) - _get(v['qi'], (cc - 1,)) / (_get(v['qi'], (cc - 1,)) + _get(v['prdr'], (cc - 1,)) * v['dt'])),mask_123,xp)  # WRF:1891
        mask_124=xp.logical_and(mask_122,xp.logical_not(taken_123))
        _put(v,'rhobar',((cc)-1,),_get(v['rhobar'], (cc - 1,)) * (_get(v['qi'], (cc - 1,)) / (_get(v['qi'], (cc - 1,)) + _get(v['prdr'], (cc - 1,)) * v['dt'])) + v['rhow'] * (_r(1.0) - _get(v['qi'], (cc - 1,)) / (_get(v['qi'], (cc - 1,)) + _get(v['prdr'], (cc - 1,)) * v['dt'])),mask_124,xp)  # WRF:1894
        _put(v,'rhobar',((cc)-1,),_max(_get(v['rhobar'], (cc - 1,)), v['rhoi']),mask_124,xp)  # WRF:1896
        mask_125=xp.logical_and(mask_2,xp.logical_not(taken_122))
        _put(v,'rhobar',((cc)-1,),v['rhoi'],mask_125,xp)  # WRF:1899
        _put(v,'qi',((cc)-1,),_get(v['qi'], (cc - 1,)) + _get(v['prdr'], (cc - 1,)) * v['dt'],mask_2,xp)  # WRF:1902
        _put(v,'ai',((cc)-1,),_get(v['ai'], (cc - 1,)) + _get(v['ardr'], (cc - 1,)) * v['dt'],mask_2,xp)  # WRF:1903
        _put(v,'ci',((cc)-1,),_get(v['ci'], (cc - 1,)) + _get(v['crdr'], (cc - 1,)) * v['dt'],mask_2,xp)  # WRF:1904
    for cc in range(1,cat+1):
        mask_126=xp.logical_and(mask_2,_get(v['qi'], (cc - 1,)) > v['qsmall'])
        taken_126=mask_126
        _put(v,'ni',((cc)-1,),_max(_get(v['ni'], (cc - 1,)), v['qnsmall']),mask_126,xp)  # WRF:1911
        _put(v,'ai',((cc)-1,),_max(_get(v['ai'], (cc - 1,)), v['qasmall']),mask_126,xp)  # WRF:1912
        _put(v,'ci',((cc)-1,),_max(_get(v['ci'], (cc - 1,)), v['qasmall']),mask_126,xp)  # WRF:1913
        _put(v,'ani',((cc)-1,),_max(xp.power(_get(v['ai'], (cc - 1,)) * _get(v['ai'], (cc - 1,)) / (_get(v['ci'], (cc - 1,)) * _get(v['ni'], (cc - 1,))), _r(0.333333333333)), _r(2e-06)),mask_126,xp)  # WRF:1914
        _put(v,'cni',((cc)-1,),_max(xp.power(_get(v['ci'], (cc - 1,)) * _get(v['ci'], (cc - 1,)) / (_get(v['ai'], (cc - 1,)) * _get(v['ni'], (cc - 1,))), _r(0.333333333333)), _r(2e-06)),mask_126,xp)  # WRF:1915
        _put(v,'ci',((cc)-1,),_get(v['cni'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_126,xp)  # WRF:1917
        _put(v,'ai',((cc)-1,),_get(v['ani'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_126,xp)  # WRF:1918
        _put(v,'deltastr',((cc)-1,),(xp.log(_get(v['cni'], (cc - 1,))) - xp.log(v['ao'])) / (xp.log(_get(v['ani'], (cc - 1,))) - xp.log(v['ao'])),mask_126,xp)  # WRF:1920
        call_out=_call('var_check',(v['nu'],v['ao'],v['fourthirdspi'],v['gammnu'],_get(v['qi'], (cc - 1,)),_get(v['deltastr'], (cc - 1,)),_get(v['ani'], (cc - 1,)),_get(v['cni'], (cc - 1,)),_get(v['rni'], (cc - 1,)),_get(v['rhobar'], (cc - 1,)),_get(v['ni'], (cc - 1,)),_get(v['ai'], (cc - 1,)),_get(v['ci'], (cc - 1,)),v['alphstr'],v['alphv'],v['betam'],),tables,xp)  # WRF:1922
        _put(v,'deltastr',((cc)-1,),call_out[0],mask_126,xp)  # WRF:1922
        _put(v,'ani',((cc)-1,),call_out[1],mask_126,xp)  # WRF:1922
        _put(v,'cni',((cc)-1,),call_out[2],mask_126,xp)  # WRF:1922
        _put(v,'rni',((cc)-1,),call_out[3],mask_126,xp)  # WRF:1922
        _put(v,'rhobar',((cc)-1,),call_out[4],mask_126,xp)  # WRF:1922
        _put(v,'ni',((cc)-1,),call_out[5],mask_126,xp)  # WRF:1922
        _put(v,'ai',((cc)-1,),call_out[6],mask_126,xp)  # WRF:1922
        _put(v,'ci',((cc)-1,),call_out[7],mask_126,xp)  # WRF:1922
        _put(v,'alphstr',None,call_out[8],mask_126,xp)  # WRF:1922
        _put(v,'alphv',None,call_out[9],mask_126,xp)  # WRF:1922
        _put(v,'betam',None,call_out[10],mask_126,xp)  # WRF:1922
        _put(v,'dsold',((cc)-1,),_get(v['deltastr'], (cc - 1,)),mask_126,xp)  # WRF:1926
    mask_127=xp.logical_and(mask_2,v['temp'] > v['t0'])
    taken_127=mask_127
    for cc in range(1,cat+1):
        mask_128=xp.logical_and(mask_127,_get(v['qi'], (cc - 1,)) > v['qsmall'])
        taken_128=mask_128
        _put(v,'rhobar',((cc)-1,),_get(v['rhobar'], (cc - 1,)) * ((_get(v['qi'], (cc - 1,)) + _get(v['qmlt'], (cc - 1,)) * v['dt']) / _get(v['qi'], (cc - 1,))) + v['rhow'] * (-_get(v['qmlt'], (cc - 1,)) * v['dt'] / _get(v['qi'], (cc - 1,))),mask_128,xp)  # WRF:1937
        _put(v,'rhobar',((cc)-1,),_min(_get(v['rhobar'], (cc - 1,)), v['rhoi']),mask_128,xp)  # WRF:1939
        mask_129=xp.logical_and(mask_127,xp.logical_not(taken_128))
        _put(v,'rhobar',((cc)-1,),v['rhoi'],mask_129,xp)  # WRF:1941
        _put(v,'qi',((cc)-1,),_get(v['qi'], (cc - 1,)) + _get(v['qmlt'], (cc - 1,)) * v['dt'],mask_127,xp)  # WRF:1944
        _put(v,'ni',((cc)-1,),_get(v['ni'], (cc - 1,)) + _get(v['nmlt'], (cc - 1,)) * v['dt'],mask_127,xp)  # WRF:1945
        _put(v,'ai',((cc)-1,),_get(v['ai'], (cc - 1,)) + _get(v['amlt'], (cc - 1,)) * v['dt'],mask_127,xp)  # WRF:1946
        _put(v,'ci',((cc)-1,),_get(v['ci'], (cc - 1,)) + _get(v['cmlt'], (cc - 1,)) * v['dt'],mask_127,xp)  # WRF:1947
        mask_130=xp.logical_and(mask_127,_get(v['qi'], (cc - 1,)) <= v['qsmall'])
        taken_130=mask_130
        _put(v,'qr',None,v['qr'] + _get(v['qi'], (cc - 1,)),mask_130,xp)  # WRF:1951
        _put(v,'qi',((cc)-1,),_r(0.0),mask_130,xp)  # WRF:1952
        mask_131=xp.logical_and(mask_127,_get(v['qi'], (cc - 1,)) > v['qsmall'])
        taken_131=mask_131
        _put(v,'ni',((cc)-1,),_max(_get(v['ni'], (cc - 1,)), v['qnsmall']),mask_131,xp)  # WRF:1957
        _put(v,'ai',((cc)-1,),_max(_get(v['ai'], (cc - 1,)), v['qasmall']),mask_131,xp)  # WRF:1958
        _put(v,'ci',((cc)-1,),_max(_get(v['ci'], (cc - 1,)), v['qasmall']),mask_131,xp)  # WRF:1959
        _put(v,'ani',((cc)-1,),_max(xp.power(_get(v['ai'], (cc - 1,)) * _get(v['ai'], (cc - 1,)) / (_get(v['ci'], (cc - 1,)) * _get(v['ni'], (cc - 1,))), _r(0.333333333333)), _r(2e-06)),mask_131,xp)  # WRF:1960
        _put(v,'cni',((cc)-1,),_max(xp.power(_get(v['ci'], (cc - 1,)) * _get(v['ci'], (cc - 1,)) / (_get(v['ai'], (cc - 1,)) * _get(v['ni'], (cc - 1,))), _r(0.333333333333)), _r(2e-06)),mask_131,xp)  # WRF:1961
        _put(v,'ci',((cc)-1,),_get(v['cni'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_131,xp)  # WRF:1963
        _put(v,'ai',((cc)-1,),_get(v['ani'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_131,xp)  # WRF:1964
        _put(v,'deltastr',((cc)-1,),(xp.log(_get(v['cni'], (cc - 1,))) - xp.log(v['ao'])) / (xp.log(_get(v['ani'], (cc - 1,))) - xp.log(v['ao'])),mask_131,xp)  # WRF:1966
        mask_132=xp.logical_and(mask_131,_get(v['dsold'], (cc - 1,)) <= _r(1.0))
        taken_132=mask_132
        mask_133=xp.logical_and(mask_132,_get(v['deltastr'], (cc - 1,)) < _get(v['dsold'], (cc - 1,)))
        taken_133=mask_133
        _put(v,'deltastr',((cc)-1,),_get(v['dsold'], (cc - 1,)),mask_133,xp)  # WRF:1971
        mask_134=xp.logical_and(mask_132,_get(v['deltastr'], (cc - 1,)) > _r(1.0))
        taken_134=mask_134
        _put(v,'deltastr',((cc)-1,),_r(1.0),mask_134,xp)  # WRF:1974
        mask_135=xp.logical_and(mask_131,xp.logical_not(taken_132))
        mask_136=xp.logical_and(mask_135,_get(v['deltastr'], (cc - 1,)) > _get(v['dsold'], (cc - 1,)))
        taken_136=mask_136
        _put(v,'deltastr',((cc)-1,),_get(v['dsold'], (cc - 1,)),mask_136,xp)  # WRF:1978
        mask_137=xp.logical_and(mask_135,_get(v['deltastr'], (cc - 1,)) < _r(1.0))
        taken_137=mask_137
        _put(v,'deltastr',((cc)-1,),_r(1.0),mask_137,xp)  # WRF:1981
        _put(v,'cni',((cc)-1,),xp.power(v['ao'], _r(1.0) - _get(v['deltastr'], (cc - 1,))) * xp.power(_get(v['ani'], (cc - 1,)), _get(v['deltastr'], (cc - 1,))),mask_131,xp)  # WRF:1984
        _put(v,'ai',((cc)-1,),_get(v['ani'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_131,xp)  # WRF:1985
        _put(v,'ci',((cc)-1,),_get(v['cni'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_131,xp)  # WRF:1986
        call_out=_call('var_check',(v['nu'],v['ao'],v['fourthirdspi'],v['gammnu'],_get(v['qi'], (cc - 1,)),_get(v['deltastr'], (cc - 1,)),_get(v['ani'], (cc - 1,)),_get(v['cni'], (cc - 1,)),_get(v['rni'], (cc - 1,)),_get(v['rhobar'], (cc - 1,)),_get(v['ni'], (cc - 1,)),_get(v['ai'], (cc - 1,)),_get(v['ci'], (cc - 1,)),v['alphstr'],v['alphv'],v['betam'],),tables,xp)  # WRF:1988
        _put(v,'deltastr',((cc)-1,),call_out[0],mask_131,xp)  # WRF:1988
        _put(v,'ani',((cc)-1,),call_out[1],mask_131,xp)  # WRF:1988
        _put(v,'cni',((cc)-1,),call_out[2],mask_131,xp)  # WRF:1988
        _put(v,'rni',((cc)-1,),call_out[3],mask_131,xp)  # WRF:1988
        _put(v,'rhobar',((cc)-1,),call_out[4],mask_131,xp)  # WRF:1988
        _put(v,'ni',((cc)-1,),call_out[5],mask_131,xp)  # WRF:1988
        _put(v,'ai',((cc)-1,),call_out[6],mask_131,xp)  # WRF:1988
        _put(v,'ci',((cc)-1,),call_out[7],mask_131,xp)  # WRF:1988
        _put(v,'alphstr',None,call_out[8],mask_131,xp)  # WRF:1988
        _put(v,'alphv',None,call_out[9],mask_131,xp)  # WRF:1988
        _put(v,'betam',None,call_out[10],mask_131,xp)  # WRF:1988
        _put(v,'dsold',((cc)-1,),_get(v['deltastr'], (cc - 1,)),mask_131,xp)  # WRF:1992
    _put(v,'qv',None,v['qv'] + (-v['pre'] - v['mnuccd']) * v['dt'],mask_2,xp)  # WRF:2000
    for cc in range(1,cat+1):
        _put(v,'qv',None,v['qv'] + -_get(v['prd'], (cc - 1,)) * v['dt'],mask_2,xp)  # WRF:2002
    _put(v,'qc',None,v['qc'] + (-v['prc'] - v['pra'] - v['mim']) * v['dt'],mask_2,xp)  # WRF:2006
    for cc in range(1,cat+1):
        _put(v,'qc',None,v['qc'] - _get(v['prdr'], (cc - 1,)) * _get(v['qcrimefrac'], (cc - 1,)) * v['dt'],mask_2,xp)  # WRF:2008
    _put(v,'qr',None,v['qr'] + (v['pre'] + v['prc'] + v['pra'] - v['mimr'] - v['mbiggr']) * v['dt'],mask_2,xp)  # WRF:2012
    _put(v,'nr',None,v['nr'] + (v['nprc1'] + v['nragg'] + v['npre'] - v['nimr'] - v['nbiggr']) * v['dt'],mask_2,xp)  # WRF:2013
    for cc in range(1,cat+1):
        _put(v,'qr',None,v['qr'] + (-_get(v['prdr'], (cc - 1,)) * (_r(1.0) - _get(v['qcrimefrac'], (cc - 1,))) - _get(v['qmlt'], (cc - 1,)) - _get(v['dqrfzri'], (cc - 1,))) * v['dt'],mask_2,xp)  # WRF:2015
        _put(v,'nr',None,_max(v['nr'] + (-_get(v['qi_qr_nrn'], (cc - 1,)) * _get(v['ni'], (cc - 1,)) * v['nr'] * v['rhoair'] - _get(v['nmlt'], (cc - 1,)) - _get(v['dnfzri'], (cc - 1,))) * v['dt'], _r(0.0)),mask_2,xp)  # WRF:2016
    for cc in range(1,cat+1):
        _put(v,'nibnuc',((cc)-1,),_get(v['ni'], (cc - 1,)),mask_2,xp)  # WRF:2023
    mask_138=xp.logical_and(mask_2,v['igr'] <= _r(1.0))
    taken_138=mask_138
    _put(v,'totalnuc',((ice1)-1,),(v['mnuccd'] + v['mim'] + v['mimr'] + v['mbiggr'] + _get(v['dqifzri'], (ice2 - 1,)) + _get(v['dqifzri'], (ice3 - 1,))) * v['dt'],mask_138,xp)  # WRF:2028
    _put(v,'totalnucn',((ice1)-1,),(v['nnuccd'] + v['nim'] + v['nimr'] + v['nbiggr'] + _get(v['dnfzri'], (ice2 - 1,)) + _get(v['dnfzri'], (ice3 - 1,))) * v['dt'],mask_138,xp)  # WRF:2029
    for cc in range(1,cat+1):
        _put(v,'totalnuc',((ice1)-1,),_get(v['totalnuc'], (ice1 - 1,)) + (_get(v['qmult'], (cc - 1,)) + _get(v['dqrfzri'], (cc - 1,))) * v['dt'],mask_138,xp)  # WRF:2032
        _put(v,'totalnucn',((ice1)-1,),_get(v['totalnucn'], (ice1 - 1,)) + _get(v['nmult'], (cc - 1,)) * v['dt'],mask_138,xp)  # WRF:2033
    _put(v,'qi',((ice1)-1,),_get(v['qi'], (ice1 - 1,)) + _get(v['totalnuc'], (ice1 - 1,)),mask_138,xp)  # WRF:2036
    _put(v,'ni',((ice1)-1,),_get(v['ni'], (ice1 - 1,)) + _get(v['totalnucn'], (ice1 - 1,)),mask_138,xp)  # WRF:2037
    _put(v,'qi',((ice2)-1,),_get(v['qi'], (ice2 - 1,)) - _get(v['dqifzri'], (ice2 - 1,)) * v['dt'],mask_138,xp)  # WRF:2040
    _put(v,'ni',((ice2)-1,),_get(v['ni'], (ice2 - 1,)) - _get(v['dnfzri'], (ice2 - 1,)) * v['dt'],mask_138,xp)  # WRF:2041
    _put(v,'qi',((ice3)-1,),_get(v['qi'], (ice3 - 1,)) - _get(v['dqifzri'], (ice3 - 1,)) * v['dt'],mask_138,xp)  # WRF:2044
    _put(v,'ni',((ice3)-1,),_get(v['ni'], (ice3 - 1,)) - _get(v['dnfzri'], (ice3 - 1,)) * v['dt'],mask_138,xp)  # WRF:2045
    mask_139=xp.logical_and(mask_2,xp.logical_not(taken_138))
    _put(v,'totalnuc',((ice2)-1,),(v['mnuccd'] + v['mim'] + v['mimr'] + v['mbiggr'] + _get(v['dqifzri'], (ice1 - 1,)) + _get(v['dqifzri'], (ice3 - 1,))) * v['dt'],mask_139,xp)  # WRF:2049
    _put(v,'totalnucn',((ice2)-1,),(v['nnuccd'] + v['nim'] + v['nimr'] + v['nbiggr'] + _get(v['dnfzri'], (ice1 - 1,)) + _get(v['dnfzri'], (ice3 - 1,))) * v['dt'],mask_139,xp)  # WRF:2050
    for cc in range(1,cat+1):
        _put(v,'totalnuc',((ice2)-1,),_get(v['totalnuc'], (ice2 - 1,)) + (_get(v['qmult'], (cc - 1,)) + _get(v['dqrfzri'], (cc - 1,))) * v['dt'],mask_139,xp)  # WRF:2053
        _put(v,'totalnucn',((ice2)-1,),_get(v['totalnucn'], (ice2 - 1,)) + _get(v['nmult'], (cc - 1,)) * v['dt'],mask_139,xp)  # WRF:2054
    _put(v,'qi',((ice2)-1,),_get(v['qi'], (ice2 - 1,)) + _get(v['totalnuc'], (ice2 - 1,)),mask_139,xp)  # WRF:2057
    _put(v,'ni',((ice2)-1,),_get(v['ni'], (ice2 - 1,)) + _get(v['totalnucn'], (ice2 - 1,)),mask_139,xp)  # WRF:2058
    _put(v,'qi',((ice1)-1,),_get(v['qi'], (ice1 - 1,)) - _get(v['dqifzri'], (ice1 - 1,)) * v['dt'],mask_139,xp)  # WRF:2061
    _put(v,'ni',((ice1)-1,),_get(v['ni'], (ice1 - 1,)) - _get(v['dnfzri'], (ice1 - 1,)) * v['dt'],mask_139,xp)  # WRF:2062
    _put(v,'qi',((ice3)-1,),_get(v['qi'], (ice3 - 1,)) - _get(v['dqifzri'], (ice3 - 1,)) * v['dt'],mask_139,xp)  # WRF:2065
    _put(v,'ni',((ice3)-1,),_get(v['ni'], (ice3 - 1,)) - _get(v['dnfzri'], (ice3 - 1,)) * v['dt'],mask_139,xp)  # WRF:2066
    for cc in range(1,cat+1):
        _put(v,'ni',((cc)-1,),_min(_get(v['ni'], (cc - 1,)), _r(1000.0) * _r(1000.0) * v['i_rhoair']),mask_2,xp)  # WRF:2072
    mask_140=xp.logical_and(mask_2,v['qr'] > v['qsmall'])
    taken_140=mask_140
    _put(v,'nr',None,_max(v['nr'], v['qnsmall']),mask_140,xp)  # WRF:2077
    _put(v,'lamr',None,xp.power(v['pi'] * v['rhow'] * v['nr'] / v['qr'], _r(0.333333333)),mask_140,xp)  # WRF:2078
    _put(v,'n0rr',None,v['nr'] * v['lamr'],mask_140,xp)  # WRF:2079
    mask_141=xp.logical_and(mask_140,v['lamr'] < v['lamminr'])
    taken_141=mask_141
    _put(v,'lamr',None,v['lamminr'],mask_141,xp)  # WRF:2081
    _put(v,'n0rr',None,_r(16269267968.0 if v['qv'].dtype == 'float32' else 16269262807.163687) * v['qr'] / (v['pi'] * v['rhow']),mask_141,xp)  # WRF:2082
    _put(v,'nr',None,v['n0rr'] / v['lamr'],mask_141,xp)  # WRF:2083
    mask_142=xp.logical_and(xp.logical_and(mask_140,xp.logical_not(taken_141)),v['lamr'] > v['lammaxr'])
    taken_141=xp.logical_or(taken_141,mask_142)
    _put(v,'lamr',None,v['lammaxr'],mask_142,xp)  # WRF:2085
    _put(v,'n0rr',None,_r(6.250000125255483e+18 if v['qv'].dtype == 'float32' else 6.249999999999996e+18) * v['qr'] / (v['pi'] * v['rhow']),mask_142,xp)  # WRF:2086
    _put(v,'nr',None,v['n0rr'] / v['lamr'],mask_142,xp)  # WRF:2087
    _put(v,'vtrn',None,v['arn'] * gamma(_r(1.0) + v['br'], xp=xp) / xp.power(v['lamr'], v['br']),mask_140,xp)  # WRF:2089
    _put(v,'vtrm',None,v['arn'] * gamma(_r(4.0) + v['br'], xp=xp) / _r(6.0) / xp.power(v['lamr'], v['br']),mask_140,xp)  # WRF:2090
    _put(v,'vtrn',None,_min(v['vtrn'], _r(9.1)),mask_140,xp)  # WRF:2091
    _put(v,'vtrm',None,_min(v['vtrm'], _r(9.1)),mask_140,xp)  # WRF:2092
    _put(v,'dbzr',None,v['rhoair'] * v['nr'] / (v['lamr'] * v['lamr'] * v['lamr']) / (v['lamr'] * v['lamr'] * v['lamr']) * _r(720.0),mask_140,xp)  # WRF:2094
    _put(v,'dbzr',None,_r(1e+18) * v['dbzr'],mask_140,xp)  # WRF:2095
    mask_143=xp.logical_and(mask_2,xp.logical_not(taken_140))
    _put(v,'qv',None,v['qv'] + v['qr'],mask_143,xp)  # WRF:2099
    _put(v,'theta',None,v['theta'] + v['theta'] * v['i_temp'] * (v['qr'] * v['xxlv']) * v['i_cp'],mask_143,xp)  # WRF:2100
    _put(v,'qr',None,_r(0.0),mask_143,xp)  # WRF:2101
    _put(v,'nr',None,_r(0.0),mask_143,xp)  # WRF:2102
    _put(v,'vtrm',None,_r(0.0),mask_143,xp)  # WRF:2103
    _put(v,'vtrn',None,_r(0.0),mask_143,xp)  # WRF:2104
    for cc in range(1,cat+1):
        mask_144=xp.logical_and(mask_2,_get(v['qi'], (cc - 1,)) >= v['qsmall'])
        taken_144=mask_144
        _put(v,'ni',((cc)-1,),_max(_get(v['ni'], (cc - 1,)), v['qnsmall']),mask_144,xp)  # WRF:2112
        _put(v,'nucfrac',None,_get(v['totalnuc'], (cc - 1,)) / _get(v['qi'], (cc - 1,)),mask_144,xp)  # WRF:2113
        _put(v,'nucfrac',None,_min(v['nucfrac'], _r(1.0)),mask_144,xp)  # WRF:2114
        _put(v,'nucfrac',None,_max(v['nucfrac'], _r(0.0)),mask_144,xp)  # WRF:2115
        mask_145=xp.logical_and(mask_144,cc == ice3)
        taken_145=mask_145
        _put(v,'nucfrac',None,_r(0.0),mask_145,xp)  # WRF:2119
        _put(v,'rhobar',((cc)-1,),v['rhoi'] * v['nucfrac'] + (_r(1.0) - v['nucfrac']) * _get(v['rhobar'], (cc - 1,)),mask_144,xp)  # WRF:2123
        _put(v,'ai',((cc)-1,),_max(_get(v['ai'], (cc - 1,)), v['qasmall']),mask_144,xp)  # WRF:2126
        _put(v,'ci',((cc)-1,),_max(_get(v['ci'], (cc - 1,)), v['qasmall']),mask_144,xp)  # WRF:2127
        _put(v,'nibnuc',((cc)-1,),_max(_get(v['nibnuc'], (cc - 1,)), v['qnsmall']),mask_144,xp)  # WRF:2128
        _put(v,'ani',((cc)-1,),_max(xp.power(_get(v['ai'], (cc - 1,)) * _get(v['ai'], (cc - 1,)) / (_get(v['ci'], (cc - 1,)) * _get(v['nibnuc'], (cc - 1,))), _r(0.333333333333)), _r(2e-06)),mask_144,xp)  # WRF:2130
        _put(v,'cni',((cc)-1,),_max(xp.power(_get(v['ci'], (cc - 1,)) * _get(v['ci'], (cc - 1,)) / (_get(v['ai'], (cc - 1,)) * _get(v['nibnuc'], (cc - 1,))), _r(0.333333333333)), _r(2e-06)),mask_144,xp)  # WRF:2131
        _put(v,'ci',((cc)-1,),_get(v['cni'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_144,xp)  # WRF:2133
        _put(v,'ai',((cc)-1,),_get(v['ani'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_144,xp)  # WRF:2134
        _put(v,'aniold',((cc)-1,),_get(v['ani'], (cc - 1,)),mask_144,xp)  # WRF:2136
        _put(v,'cniold',((cc)-1,),xp.power(v['ao'], _r(1.0) - _get(v['dsold'], (cc - 1,))) * xp.power(_get(v['aniold'], (cc - 1,)), _get(v['dsold'], (cc - 1,))),mask_144,xp)  # WRF:2137
        mask_146=xp.logical_and(mask_144,xp.logical_and(v['nucfrac'] > _r(1e-05), _get(v['totalnucn'], (cc - 1,)) > _r(0.0)))
        taken_146=mask_146
        _put(v,'gamma_arg',None,v['nu'] + _r(3.0),mask_146,xp)  # WRF:2141
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_146,xp)  # WRF:2142
        _put(v,'gamma_arg',None,v['nu'] + _r(4.0),mask_146,xp)  # WRF:2143
        _put(v,'gi2',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_146,xp)  # WRF:2144
        _put(v,'anuc',None,xp.power(_get(v['totalnuc'], (cc - 1,)) * v['gammnu'] / (v['rhoi'] * _get(v['totalnucn'], (cc - 1,)) * v['fourthirdspi'] * tables['gamma_tab'][v['gi'] - 1,]), _r(0.3333333)) * tables['gamma_tab'][v['gi2'] - 1,] / tables['gamma_tab'][v['gi'] - 1,],mask_146,xp)  # WRF:2145
        _put(v,'anuc',None,_max(v['anuc'], _r(2e-06)),mask_146,xp)  # WRF:2147
        _put(v,'anuc',None,_min(v['anuc'], _r(0.003)),mask_146,xp)  # WRF:2148
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_146,xp)  # WRF:2150
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_146,xp)  # WRF:2151
        _put(v,'gamma_arg',None,v['nu'] + _r(3.0) + _get(v['deltastr'], (cc - 1,)),mask_146,xp)  # WRF:2152
        _put(v,'gi2',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_146,xp)  # WRF:2153
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _r(2.0) * _get(v['deltastr'], (cc - 1,)),mask_146,xp)  # WRF:2154
        _put(v,'gi3',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_146,xp)  # WRF:2155
        _put(v,'amass',None,_get(v['ani'], (cc - 1,)) * tables['gamma_tab'][v['gi2'] - 1,] / tables['gamma_tab'][v['gi'] - 1,],mask_146,xp)  # WRF:2157
        _put(v,'cmass',None,_get(v['cni'], (cc - 1,)) * tables['gamma_tab'][v['gi3'] - 1,] / tables['gamma_tab'][v['gi'] - 1,],mask_146,xp)  # WRF:2158
        _put(v,'amass',None,v['amass'] * (_r(1.0) - v['nucfrac']) + v['anuc'] * v['nucfrac'],mask_146,xp)  # WRF:2161
        _put(v,'cmass',None,v['cmass'] * (_r(1.0) - v['nucfrac']) + v['anuc'] * v['nucfrac'],mask_146,xp)  # WRF:2162
        _put(v,'ani',((cc)-1,),_max(v['amass'] * tables['gamma_tab'][v['gi'] - 1,] / tables['gamma_tab'][v['gi2'] - 1,], _r(2e-06)),mask_146,xp)  # WRF:2165
        _put(v,'cni',((cc)-1,),_max(v['cmass'] * tables['gamma_tab'][v['gi'] - 1,] / tables['gamma_tab'][v['gi3'] - 1,], _r(2e-06)),mask_146,xp)  # WRF:2166
        mask_147=xp.logical_and(mask_146,xp.logical_and(_get(v['ani'], (cc - 1,)) > _r(1.1) * v['ao'], _get(v['cni'], (cc - 1,)) > _r(1.1) * v['ao']))
        taken_147=mask_147
        _put(v,'deltastr',((cc)-1,),(xp.log(_get(v['cni'], (cc - 1,))) - xp.log(v['ao'])) / (xp.log(_get(v['ani'], (cc - 1,))) - xp.log(v['ao'])),mask_147,xp)  # WRF:2170
        mask_148=xp.logical_and(mask_146,xp.logical_not(taken_147))
        _put(v,'deltastr',((cc)-1,),_r(1.0),mask_148,xp)  # WRF:2172
        mask_149=xp.logical_and(mask_146,_get(v['dsold'], (cc - 1,)) <= _r(1.0))
        taken_149=mask_149
        mask_150=xp.logical_and(mask_149,_get(v['deltastr'], (cc - 1,)) < _get(v['dsold'], (cc - 1,)))
        taken_150=mask_150
        _put(v,'deltastr',((cc)-1,),_get(v['dsold'], (cc - 1,)),mask_150,xp)  # WRF:2178
        mask_151=xp.logical_and(mask_149,_get(v['deltastr'], (cc - 1,)) > _r(1.0))
        taken_151=mask_151
        _put(v,'deltastr',((cc)-1,),_r(1.0),mask_151,xp)  # WRF:2181
        mask_152=xp.logical_and(mask_146,xp.logical_not(taken_149))
        mask_153=xp.logical_and(mask_152,_get(v['deltastr'], (cc - 1,)) > _get(v['dsold'], (cc - 1,)))
        taken_153=mask_153
        _put(v,'deltastr',((cc)-1,),_get(v['dsold'], (cc - 1,)),mask_153,xp)  # WRF:2185
        mask_154=xp.logical_and(mask_152,_get(v['deltastr'], (cc - 1,)) < _r(1.0))
        taken_154=mask_154
        _put(v,'deltastr',((cc)-1,),_r(1.0),mask_154,xp)  # WRF:2188
        _put(v,'alphstr',None,xp.power(v['ao'], _r(1.0) - _get(v['deltastr'], (cc - 1,))),mask_144,xp)  # WRF:2202
        _put(v,'alphv',None,v['fourthirdspi'] * v['alphstr'],mask_144,xp)  # WRF:2203
        _put(v,'betam',None,_r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_144,xp)  # WRF:2204
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_144,xp)  # WRF:2205
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_144,xp)  # WRF:2206
        _put(v,'ani',((cc)-1,),xp.power(_get(v['qi'], (cc - 1,)) * v['gammnu'] / (_get(v['rhobar'], (cc - 1,)) * _get(v['ni'], (cc - 1,)) * v['alphv'] * tables['gamma_tab'][v['gi'] - 1,]), _r(1.0) / v['betam']),mask_144,xp)  # WRF:2207
        _put(v,'cni',((cc)-1,),xp.power(v['ao'], _r(1.0) - _get(v['deltastr'], (cc - 1,))) * xp.power(_get(v['ani'], (cc - 1,)), _get(v['deltastr'], (cc - 1,))),mask_144,xp)  # WRF:2209
        _put(v,'ai',((cc)-1,),_get(v['ai'], (cc - 1,)) + _get(v['cniold'], (cc - 1,)) * _get(v['nibnuc'], (cc - 1,)) * _r(2.0) * _get(v['aniold'], (cc - 1,)) * (_get(v['ani'], (cc - 1,)) - _get(v['aniold'], (cc - 1,))) + _get(v['nibnuc'], (cc - 1,)) * (_get(v['aniold'], (cc - 1,)) * _get(v['aniold'], (cc - 1,))) * (_get(v['cni'], (cc - 1,)) - _get(v['cniold'], (cc - 1,))) + _get(v['aniold'], (cc - 1,)) * _get(v['aniold'], (cc - 1,)) * _get(v['cniold'], (cc - 1,)) * (_get(v['ni'], (cc - 1,)) - _get(v['nibnuc'], (cc - 1,))),mask_144,xp)  # WRF:2212
        _put(v,'ci',((cc)-1,),_get(v['ci'], (cc - 1,)) + _get(v['aniold'], (cc - 1,)) * _get(v['nibnuc'], (cc - 1,)) * _r(2.0) * _get(v['cniold'], (cc - 1,)) * (_get(v['cni'], (cc - 1,)) - _get(v['cniold'], (cc - 1,))) + _get(v['nibnuc'], (cc - 1,)) * (_get(v['cniold'], (cc - 1,)) * _get(v['cniold'], (cc - 1,))) * (_get(v['ani'], (cc - 1,)) - _get(v['aniold'], (cc - 1,))) + _get(v['cniold'], (cc - 1,)) * _get(v['cniold'], (cc - 1,)) * _get(v['aniold'], (cc - 1,)) * (_get(v['ni'], (cc - 1,)) - _get(v['nibnuc'], (cc - 1,))),mask_144,xp)  # WRF:2216
        _put(v,'ni',((cc)-1,),_max(_get(v['ni'], (cc - 1,)), v['qnsmall']),mask_144,xp)  # WRF:2221
        _put(v,'ai',((cc)-1,),_max(_get(v['ai'], (cc - 1,)), v['qasmall']),mask_144,xp)  # WRF:2222
        _put(v,'ci',((cc)-1,),_max(_get(v['ci'], (cc - 1,)), v['qasmall']),mask_144,xp)  # WRF:2223
        mask_155=xp.logical_and(mask_144,xp.logical_and(_get(v['ani'], (cc - 1,)) > _r(1.1) * v['ao'], _get(v['cni'], (cc - 1,)) > _r(1.1) * v['ao']))
        taken_155=mask_155
        _put(v,'deltastr',((cc)-1,),(xp.log(_get(v['cni'], (cc - 1,))) - xp.log(v['ao'])) / (xp.log(_get(v['ani'], (cc - 1,))) - xp.log(v['ao'])),mask_155,xp)  # WRF:2226
        mask_156=xp.logical_and(mask_144,xp.logical_not(taken_155))
        _put(v,'deltastr',((cc)-1,),_r(1.0),mask_156,xp)  # WRF:2228
        call_out=_call('var_check',(v['nu'],v['ao'],v['fourthirdspi'],v['gammnu'],_get(v['qi'], (cc - 1,)),_get(v['deltastr'], (cc - 1,)),_get(v['ani'], (cc - 1,)),_get(v['cni'], (cc - 1,)),_get(v['rni'], (cc - 1,)),_get(v['rhobar'], (cc - 1,)),_get(v['ni'], (cc - 1,)),_get(v['ai'], (cc - 1,)),_get(v['ci'], (cc - 1,)),v['alphstr'],v['alphv'],v['betam'],),tables,xp)  # WRF:2231
        _put(v,'deltastr',((cc)-1,),call_out[0],mask_144,xp)  # WRF:2231
        _put(v,'ani',((cc)-1,),call_out[1],mask_144,xp)  # WRF:2231
        _put(v,'cni',((cc)-1,),call_out[2],mask_144,xp)  # WRF:2231
        _put(v,'rni',((cc)-1,),call_out[3],mask_144,xp)  # WRF:2231
        _put(v,'rhobar',((cc)-1,),call_out[4],mask_144,xp)  # WRF:2231
        _put(v,'ni',((cc)-1,),call_out[5],mask_144,xp)  # WRF:2231
        _put(v,'ai',((cc)-1,),call_out[6],mask_144,xp)  # WRF:2231
        _put(v,'ci',((cc)-1,),call_out[7],mask_144,xp)  # WRF:2231
        _put(v,'alphstr',None,call_out[8],mask_144,xp)  # WRF:2231
        _put(v,'alphv',None,call_out[9],mask_144,xp)  # WRF:2231
        _put(v,'betam',None,call_out[10],mask_144,xp)  # WRF:2231
        _put(v,'dsold',((cc)-1,),_get(v['deltastr'], (cc - 1,)),mask_144,xp)  # WRF:2235
        _put(v,'vtrzi1',((cc)-1,),_min(_get(v['vtrzi1'], (cc - 1,)), _r(25.0)),mask_144,xp)  # WRF:2238
        _put(v,'vtrmi1',((cc)-1,),_min(_get(v['vtrmi1'], (cc - 1,)), _r(25.0)),mask_144,xp)  # WRF:2239
        _put(v,'vtrni1',((cc)-1,),_min(_get(v['vtrni1'], (cc - 1,)), _r(25.0)),mask_144,xp)  # WRF:2240
        mask_157=xp.logical_and(mask_2,xp.logical_not(taken_144))
        _put(v,'qv',None,v['qv'] + _get(v['qi'], (cc - 1,)),mask_157,xp)  # WRF:2245
        _put(v,'theta',None,v['theta'] + v['theta'] * v['i_temp'] * (_get(v['qi'], (cc - 1,)) * v['xxls']) * v['i_cp'],mask_157,xp)  # WRF:2246
        _put(v,'qi',((cc)-1,),_r(0.0),mask_157,xp)  # WRF:2247
        _put(v,'ai',((cc)-1,),_r(0.0),mask_157,xp)  # WRF:2248
        _put(v,'ni',((cc)-1,),_r(0.0),mask_157,xp)  # WRF:2249
        _put(v,'ci',((cc)-1,),_r(0.0),mask_157,xp)  # WRF:2250
        _put(v,'vtrzi1',((cc)-1,),_r(0.0),mask_157,xp)  # WRF:2251
        _put(v,'vtrmi1',((cc)-1,),_r(0.0),mask_157,xp)  # WRF:2252
        _put(v,'vtrni1',((cc)-1,),_r(0.0),mask_157,xp)  # WRF:2253
    mask_158=xp.logical_and(mask_2,v['temp'] <= v['t0'])
    taken_158=mask_158
    for cc in range(1,cat+1):
        _put(v,'dsold',((cc)-1,),_get(v['deltastr'], (cc - 1,)),mask_158,xp)  # WRF:2267
        _put(v,'nibnuc',((cc)-1,),_max(_get(v['ni'], (cc - 1,)), v['qnsmall']),mask_158,xp)  # WRF:2268
        _put(v,'qi',((cc)-1,),_get(v['qi'], (cc - 1,)) + _get(v['qagg'], (cc - 1,)),mask_158,xp)  # WRF:2269
        _put(v,'ni',((cc)-1,),_get(v['ni'], (cc - 1,)) + _get(v['nagg'], (cc - 1,)),mask_158,xp)  # WRF:2270
        mask_159=xp.logical_and(mask_158,_get(v['qi'], (cc - 1,)) > v['qsmall'])
        taken_159=mask_159
        mask_160=xp.logical_and(mask_159,cc == ice3)
        taken_160=mask_160
        _put(v,'rhobar',((cc)-1,),_r(50.0),mask_160,xp)  # WRF:2276
        _put(v,'phiagg3',None,_r(0.2),mask_160,xp)  # WRF:2277
        _put(v,'ani',((cc)-1,),_r(0.5) * v['dn3'],mask_160,xp)  # WRF:2278
        _put(v,'gamma_arg',None,v['nu'] - _r(1.0) + _get(v['deltastr'], (cc - 1,)),mask_160,xp)  # WRF:2279
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_160,xp)  # WRF:2280
        _put(v,'cni',((cc)-1,),v['phiagg3'] * _get(v['ani'], (cc - 1,)) * v['gammnu'] / tables['gamma_tab'][v['gi'] - 1,],mask_160,xp)  # WRF:2281
        _put(v,'ai',((cc)-1,),_get(v['ani'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_160,xp)  # WRF:2283
        _put(v,'ci',((cc)-1,),_get(v['cni'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_160,xp)  # WRF:2284
        _put(v,'ni',((cc)-1,),_max(_get(v['ni'], (cc - 1,)), v['qnsmall']),mask_160,xp)  # WRF:2286
        _put(v,'ai',((cc)-1,),_max(_get(v['ai'], (cc - 1,)), v['qasmall']),mask_160,xp)  # WRF:2287
        _put(v,'ci',((cc)-1,),_max(_get(v['ci'], (cc - 1,)), v['qasmall']),mask_160,xp)  # WRF:2288
        mask_161=xp.logical_and(mask_160,xp.logical_and(_get(v['ani'], (cc - 1,)) > _r(1.1) * v['ao'], _get(v['cni'], (cc - 1,)) > _r(1.1) * v['ao']))
        taken_161=mask_161
        _put(v,'deltastr',((cc)-1,),(xp.log(_get(v['cni'], (cc - 1,))) - xp.log(v['ao'])) / (xp.log(_get(v['ani'], (cc - 1,))) - xp.log(v['ao'])),mask_161,xp)  # WRF:2291
        mask_162=xp.logical_and(mask_160,xp.logical_not(taken_161))
        _put(v,'deltastr',((cc)-1,),_r(1.0),mask_162,xp)  # WRF:2293
        call_out=_call('var_check',(v['nu'],v['ao'],v['fourthirdspi'],v['gammnu'],_get(v['qi'], (cc - 1,)),_get(v['deltastr'], (cc - 1,)),_get(v['ani'], (cc - 1,)),_get(v['cni'], (cc - 1,)),_get(v['rni'], (cc - 1,)),_get(v['rhobar'], (cc - 1,)),_get(v['ni'], (cc - 1,)),_get(v['ai'], (cc - 1,)),_get(v['ci'], (cc - 1,)),v['alphstr'],v['alphv'],v['betam'],),tables,xp)  # WRF:2296
        _put(v,'deltastr',((cc)-1,),call_out[0],mask_160,xp)  # WRF:2296
        _put(v,'ani',((cc)-1,),call_out[1],mask_160,xp)  # WRF:2296
        _put(v,'cni',((cc)-1,),call_out[2],mask_160,xp)  # WRF:2296
        _put(v,'rni',((cc)-1,),call_out[3],mask_160,xp)  # WRF:2296
        _put(v,'rhobar',((cc)-1,),call_out[4],mask_160,xp)  # WRF:2296
        _put(v,'ni',((cc)-1,),call_out[5],mask_160,xp)  # WRF:2296
        _put(v,'ai',((cc)-1,),call_out[6],mask_160,xp)  # WRF:2296
        _put(v,'ci',((cc)-1,),call_out[7],mask_160,xp)  # WRF:2296
        _put(v,'alphstr',None,call_out[8],mask_160,xp)  # WRF:2296
        _put(v,'alphv',None,call_out[9],mask_160,xp)  # WRF:2296
        _put(v,'betam',None,call_out[10],mask_160,xp)  # WRF:2296
        mask_163=xp.logical_and(mask_159,xp.logical_not(taken_160))
        _put(v,'ni',((cc)-1,),_max(_get(v['ni'], (cc - 1,)), v['qnsmall']),mask_163,xp)  # WRF:2303
        _put(v,'ai',((cc)-1,),_max(_get(v['ai'], (cc - 1,)), v['qasmall']),mask_163,xp)  # WRF:2304
        _put(v,'ci',((cc)-1,),_max(_get(v['ci'], (cc - 1,)), v['qasmall']),mask_163,xp)  # WRF:2305
        _put(v,'ani',((cc)-1,),_max(xp.power(_get(v['ai'], (cc - 1,)) * _get(v['ai'], (cc - 1,)) / (_get(v['ci'], (cc - 1,)) * _get(v['nibnuc'], (cc - 1,))), _r(0.333333333333)), _r(2e-06)),mask_163,xp)  # WRF:2306
        _put(v,'cni',((cc)-1,),_max(xp.power(_get(v['ci'], (cc - 1,)) * _get(v['ci'], (cc - 1,)) / (_get(v['ai'], (cc - 1,)) * _get(v['nibnuc'], (cc - 1,))), _r(0.333333333333)), _r(2e-06)),mask_163,xp)  # WRF:2307
        _put(v,'aniold',((cc)-1,),_get(v['ani'], (cc - 1,)),mask_163,xp)  # WRF:2309
        _put(v,'cniold',((cc)-1,),_get(v['cni'], (cc - 1,)),mask_163,xp)  # WRF:2310
        _put(v,'deltastr',((cc)-1,),_get(v['dsold'], (cc - 1,)),mask_163,xp)  # WRF:2311
        _put(v,'alphstr',None,xp.power(v['ao'], _r(1.0) - _get(v['deltastr'], (cc - 1,))),mask_163,xp)  # WRF:2316
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_163,xp)  # WRF:2317
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_163,xp)  # WRF:2318
        _put(v,'ani',((cc)-1,),xp.power(_get(v['qi'], (cc - 1,)) / (_get(v['ni'], (cc - 1,)) * _get(v['rhobar'], (cc - 1,)) * v['fourthirdspi'] * v['alphstr'] * tables['gamma_tab'][v['gi'] - 1,] * v['i_gammnu']), _r(1.0) / (_r(2.0) + _get(v['deltastr'], (cc - 1,)))),mask_163,xp)  # WRF:2319
        _put(v,'ani',((cc)-1,),_max(_get(v['ani'], (cc - 1,)), _r(2e-06)),mask_163,xp)  # WRF:2321
        _put(v,'cni',((cc)-1,),v['alphstr'] * xp.power(_get(v['ani'], (cc - 1,)), _get(v['deltastr'], (cc - 1,))),mask_163,xp)  # WRF:2322
        _put(v,'ai',((cc)-1,),_get(v['ai'], (cc - 1,)) + _get(v['cniold'], (cc - 1,)) * _get(v['nibnuc'], (cc - 1,)) * _r(2.0) * _get(v['aniold'], (cc - 1,)) * (_get(v['ani'], (cc - 1,)) - _get(v['aniold'], (cc - 1,))) + _get(v['nibnuc'], (cc - 1,)) * (_get(v['aniold'], (cc - 1,)) * _get(v['aniold'], (cc - 1,))) * (_get(v['cni'], (cc - 1,)) - _get(v['cniold'], (cc - 1,))) + _get(v['aniold'], (cc - 1,)) * _get(v['aniold'], (cc - 1,)) * _get(v['cniold'], (cc - 1,)) * (_get(v['ni'], (cc - 1,)) - _get(v['nibnuc'], (cc - 1,))),mask_163,xp)  # WRF:2325
        _put(v,'ci',((cc)-1,),_get(v['ci'], (cc - 1,)) + _get(v['aniold'], (cc - 1,)) * _get(v['nibnuc'], (cc - 1,)) * _r(2.0) * _get(v['cniold'], (cc - 1,)) * (_get(v['cni'], (cc - 1,)) - _get(v['cniold'], (cc - 1,))) + _get(v['nibnuc'], (cc - 1,)) * (_get(v['cniold'], (cc - 1,)) * _get(v['cniold'], (cc - 1,))) * (_get(v['ani'], (cc - 1,)) - _get(v['aniold'], (cc - 1,))) + _get(v['cniold'], (cc - 1,)) * _get(v['cniold'], (cc - 1,)) * _get(v['aniold'], (cc - 1,)) * (_get(v['ni'], (cc - 1,)) - _get(v['nibnuc'], (cc - 1,))),mask_163,xp)  # WRF:2329
        _put(v,'ni',((cc)-1,),_max(_get(v['ni'], (cc - 1,)), v['qnsmall']),mask_163,xp)  # WRF:2334
        _put(v,'ai',((cc)-1,),_max(_get(v['ai'], (cc - 1,)), v['qasmall']),mask_163,xp)  # WRF:2335
        _put(v,'ci',((cc)-1,),_max(_get(v['ci'], (cc - 1,)), v['qasmall']),mask_163,xp)  # WRF:2336
        mask_164=xp.logical_and(mask_163,xp.logical_and(_get(v['ani'], (cc - 1,)) > _r(1.1) * v['ao'], _get(v['cni'], (cc - 1,)) > _r(1.1) * v['ao']))
        taken_164=mask_164
        _put(v,'deltastr',((cc)-1,),(xp.log(_get(v['cni'], (cc - 1,))) - xp.log(v['ao'])) / (xp.log(_get(v['ani'], (cc - 1,))) - xp.log(v['ao'])),mask_164,xp)  # WRF:2339
        mask_165=xp.logical_and(mask_163,xp.logical_not(taken_164))
        _put(v,'deltastr',((cc)-1,),_r(1.0),mask_165,xp)  # WRF:2341
        call_out=_call('var_check',(v['nu'],v['ao'],v['fourthirdspi'],v['gammnu'],_get(v['qi'], (cc - 1,)),_get(v['deltastr'], (cc - 1,)),_get(v['ani'], (cc - 1,)),_get(v['cni'], (cc - 1,)),_get(v['rni'], (cc - 1,)),_get(v['rhobar'], (cc - 1,)),_get(v['ni'], (cc - 1,)),_get(v['ai'], (cc - 1,)),_get(v['ci'], (cc - 1,)),v['alphstr'],v['alphv'],v['betam'],),tables,xp)  # WRF:2344
        _put(v,'deltastr',((cc)-1,),call_out[0],mask_163,xp)  # WRF:2344
        _put(v,'ani',((cc)-1,),call_out[1],mask_163,xp)  # WRF:2344
        _put(v,'cni',((cc)-1,),call_out[2],mask_163,xp)  # WRF:2344
        _put(v,'rni',((cc)-1,),call_out[3],mask_163,xp)  # WRF:2344
        _put(v,'rhobar',((cc)-1,),call_out[4],mask_163,xp)  # WRF:2344
        _put(v,'ni',((cc)-1,),call_out[5],mask_163,xp)  # WRF:2344
        _put(v,'ai',((cc)-1,),call_out[6],mask_163,xp)  # WRF:2344
        _put(v,'ci',((cc)-1,),call_out[7],mask_163,xp)  # WRF:2344
        _put(v,'alphstr',None,call_out[8],mask_163,xp)  # WRF:2344
        _put(v,'alphv',None,call_out[9],mask_163,xp)  # WRF:2344
        _put(v,'betam',None,call_out[10],mask_163,xp)  # WRF:2344
    _put(v,'theta',None,v['theta'] + v['theta'] * v['i_temp'] * (v['mnuccd'] * v['xxls'] * v['i_cp'] * v['dt']),mask_2,xp)  # WRF:2356
    for cc in range(1,cat+1):
        _put(v,'theta',None,v['theta'] + v['theta'] * v['i_temp'] * (_get(v['prd'], (cc - 1,)) * v['xxls'] * v['i_cp'] * v['dt']),mask_2,xp)  # WRF:2358
        _put(v,'theta',None,v['theta'] + v['theta'] * v['i_temp'] * (_get(v['qmlt'], (cc - 1,)) * v['xxlf'] * v['i_cp'] * v['dt']),mask_2,xp)  # WRF:2359
    _put(v,'theta',None,v['theta'] + v['theta'] * v['i_temp'] * (v['pre'] * v['xxlv']) * v['i_cp'] * v['dt'],mask_2,xp)  # WRF:2362
    mask_166=xp.logical_and(mask_2,v['temp'] <= v['t0'])
    taken_166=mask_166
    _put(v,'theta',None,v['theta'] + v['theta'] * v['i_temp'] * (v['mim'] + v['mimr'] + v['mbiggr']) * v['xxlf'] * v['i_cp'] * v['dt'],mask_166,xp)  # WRF:2365
    for cc in range(1,cat+1):
        _put(v,'theta',None,v['theta'] + v['theta'] * v['i_temp'] * (_get(v['prdr'], (cc - 1,)) + _get(v['dqrfzri'], (cc - 1,))) * v['xxlf'] * v['i_cp'] * v['dt'],mask_166,xp)  # WRF:2368
    _put(v,'temp',None,v['theta'] / xp.power(_r(100000.0) / v['pres_e'], v['rcp']),mask_2,xp)  # WRF:2372
    _put(v,'i_temp',None,_r(1.0) / v['temp'],mask_2,xp)  # WRF:2373
    _put(v,'qvs',None,_r(0.622) * polysvp(v['temp'], 0, xp=xp) / (v['pres_e'] - polysvp(v['temp'], 0, xp=xp)),mask_2,xp)  # WRF:2374
    _put(v,'xxlv',None,_r(3148400.0) - _r(2370.0) * v['temp'],mask_2,xp)  # WRF:2375
    _put(v,'rhoair',None,v['pres_e'] / (v['rd'] * v['temp']),mask_2,xp)  # WRF:2376
    _put(v,'i_rhoair',None,_r(1.0) / v['rhoair'],mask_2,xp)  # WRF:2377
    _put(v,'pcc',None,(v['qv'] - v['qvs']) / (_r(1.0) + v['xxlv'] * v['xxlv'] * v['qvs'] / (v['cp'] * v['rv'] * (v['temp'] * v['temp']))) * v['i_dt'],mask_2,xp)  # WRF:2382
    mask_167=xp.logical_and(mask_2,v['pcc'] * v['dt'] + v['qc'] < _r(0.0))
    taken_167=mask_167
    _put(v,'pcc',None,-v['qc'] / v['dt'],mask_167,xp)  # WRF:2384
    _put(v,'theta',None,v['theta'] + v['theta'] * v['i_temp'] * v['pcc'] * v['xxlv'] * v['i_cp'] * v['dt'],mask_2,xp)  # WRF:2386
    _put(v,'qv',None,v['qv'] - v['pcc'] * v['dt'],mask_2,xp)  # WRF:2387
    _put(v,'qc',None,v['qc'] + v['pcc'] * v['dt'],mask_2,xp)  # WRF:2388
    mask_168=xp.logical_and(mask_2,v['qc'] <= v['qsmall'])
    taken_168=mask_168
    _put(v,'qv',None,v['qv'] + v['qc'],mask_168,xp)  # WRF:2391
    _put(v,'theta',None,v['theta'] + v['theta'] * v['i_temp'] * (v['qc'] * v['xxlv']) * v['i_cp'],mask_168,xp)  # WRF:2392
    _put(v,'qc',None,_r(0.0),mask_168,xp)  # WRF:2393
    _put(v,'nc',None,_r(0.0),mask_168,xp)  # WRF:2394
    _put(v,'lrsig',None,_r(1.3),mask_2,xp)  # WRF:2400
    _put(v,'sig',None,xp.log(v['lrsig']),mask_2,xp)  # WRF:2401
    _put(v,'lwc',None,v['qc'] * v['rhoair'],mask_2,xp)  # WRF:2402
    _put(v,'ncm3dum',None,v['nc'] * v['rhoair'],mask_2,xp)  # WRF:2403
    mask_169=xp.logical_and(mask_2,v['ncm3dum'] > _r(0.0))
    taken_169=mask_169
    _put(v,'r_n',None,xp.power(v['lwc'] / (v['fourthirdspi'] * v['ncm3dum'] * v['rhow'] * xp.exp(_r(4.5) * (v['sig'] * v['sig']))), _r(0.33333333333)),mask_169,xp)  # WRF:2406
    mask_170=xp.logical_and(mask_2,xp.logical_not(taken_169))
    _put(v,'r_n',None,_r(0.0),mask_170,xp)  # WRF:2408
    _put(v,'effc1d',None,v['r_n'] * (_r(5.0) * xp.exp(v['sig'] * v['sig'] / _r(2.0))),mask_2,xp)  # WRF:2411
    _put(v,'effc1d',None,_min(v['effc1d'], _r(5e-05)),mask_2,xp)  # WRF:2412
    _put(v,'effc1d',None,_max(v['effc1d'], _r(2.51e-06)),mask_2,xp)  # WRF:2413
    _put(v,'vtrmc',None,_r(0.0),mask_2,xp)  # WRF:2416
    _put(v,'vtrnc',None,_r(0.0),mask_2,xp)  # WRF:2417
    if observer is not None: observer(3,v,mask_2)  # Original checkpoint WRF:2422
    for cc in range(1,cat+1):
        mask_171=xp.logical_and(mask_2,_get(v['qi'], (cc - 1,)) > v['qsmall'])
        taken_171=mask_171
        _put(v,'ni',((cc)-1,),_max(_get(v['ni'], (cc - 1,)), v['qnsmall']),mask_171,xp)  # WRF:2425
        _put(v,'ai',((cc)-1,),_max(_get(v['ai'], (cc - 1,)), v['qasmall']),mask_171,xp)  # WRF:2426
        _put(v,'ci',((cc)-1,),_max(_get(v['ci'], (cc - 1,)), v['qasmall']),mask_171,xp)  # WRF:2427
        _put(v,'ani',((cc)-1,),_max(xp.power(_get(v['ai'], (cc - 1,)) * _get(v['ai'], (cc - 1,)) / (_get(v['ci'], (cc - 1,)) * _get(v['ni'], (cc - 1,))), _r(0.333333333333)), _r(2e-06)),mask_171,xp)  # WRF:2428
        _put(v,'cni',((cc)-1,),_max(xp.power(_get(v['ci'], (cc - 1,)) * _get(v['ci'], (cc - 1,)) / (_get(v['ai'], (cc - 1,)) * _get(v['ni'], (cc - 1,))), _r(0.333333333333)), _r(2e-06)),mask_171,xp)  # WRF:2429
        _put(v,'ci',((cc)-1,),_get(v['cni'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_171,xp)  # WRF:2431
        _put(v,'ai',((cc)-1,),_get(v['ani'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_171,xp)  # WRF:2432
        _put(v,'deltastr',((cc)-1,),(xp.log(_get(v['cni'], (cc - 1,))) - xp.log(v['ao'])) / (xp.log(_get(v['ani'], (cc - 1,))) - xp.log(v['ao'])),mask_171,xp)  # WRF:2434
        call_out=_call('var_check',(v['nu'],v['ao'],v['fourthirdspi'],v['gammnu'],_get(v['qi'], (cc - 1,)),_get(v['deltastr'], (cc - 1,)),_get(v['ani'], (cc - 1,)),_get(v['cni'], (cc - 1,)),_get(v['rni'], (cc - 1,)),_get(v['rhobar'], (cc - 1,)),_get(v['ni'], (cc - 1,)),_get(v['ai'], (cc - 1,)),_get(v['ci'], (cc - 1,)),v['alphstr'],v['alphv'],v['betam'],),tables,xp)  # WRF:2436
        _put(v,'deltastr',((cc)-1,),call_out[0],mask_171,xp)  # WRF:2436
        _put(v,'ani',((cc)-1,),call_out[1],mask_171,xp)  # WRF:2436
        _put(v,'cni',((cc)-1,),call_out[2],mask_171,xp)  # WRF:2436
        _put(v,'rni',((cc)-1,),call_out[3],mask_171,xp)  # WRF:2436
        _put(v,'rhobar',((cc)-1,),call_out[4],mask_171,xp)  # WRF:2436
        _put(v,'ni',((cc)-1,),call_out[5],mask_171,xp)  # WRF:2436
        _put(v,'ai',((cc)-1,),call_out[6],mask_171,xp)  # WRF:2436
        _put(v,'ci',((cc)-1,),call_out[7],mask_171,xp)  # WRF:2436
        _put(v,'alphstr',None,call_out[8],mask_171,xp)  # WRF:2436
        _put(v,'alphv',None,call_out[9],mask_171,xp)  # WRF:2436
        _put(v,'betam',None,call_out[10],mask_171,xp)  # WRF:2436
        mask_172=xp.logical_and(mask_171,cc == ice3)
        taken_172=mask_172
        mask_173=xp.logical_and(mask_172,v['temp'] <= v['t0'])
        taken_173=mask_173
        _put(v,'rhobar',((cc)-1,),_r(50.0),mask_173,xp)  # WRF:2443
        _put(v,'phiagg3',None,_r(0.2),mask_173,xp)  # WRF:2444
        _put(v,'gamma_arg',None,v['nu'] - _r(1.0) + _get(v['deltastr'], (cc - 1,)),mask_173,xp)  # WRF:2445
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_173,xp)  # WRF:2446
        _put(v,'cni',((cc)-1,),v['phiagg3'] * _get(v['ani'], (cc - 1,)) * v['gammnu'] / tables['gamma_tab'][v['gi'] - 1,],mask_173,xp)  # WRF:2447
        _put(v,'ai',((cc)-1,),_get(v['ani'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_173,xp)  # WRF:2449
        _put(v,'ci',((cc)-1,),_get(v['cni'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_173,xp)  # WRF:2450
        mask_174=xp.logical_and(mask_172,xp.logical_and(_get(v['ani'], (cc - 1,)) > _r(1.1) * v['ao'], _get(v['cni'], (cc - 1,)) > _r(1.1) * v['ao']))
        taken_174=mask_174
        _put(v,'deltastr',((cc)-1,),(xp.log(_get(v['cni'], (cc - 1,))) - xp.log(v['ao'])) / (xp.log(_get(v['ani'], (cc - 1,))) - xp.log(v['ao'])),mask_174,xp)  # WRF:2454
        mask_175=xp.logical_and(mask_172,xp.logical_not(taken_174))
        _put(v,'deltastr',((cc)-1,),_r(1.0),mask_175,xp)  # WRF:2456
        _put(v,'deltastr',((cc)-1,),_min(_get(v['deltastr'], (cc - 1,)), _r(1.0)),mask_172,xp)  # WRF:2459
        _put(v,'alphstr',None,xp.power(v['ao'], _r(1.0) - _get(v['deltastr'], (cc - 1,))),mask_172,xp)  # WRF:2461
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_172,xp)  # WRF:2462
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_172,xp)  # WRF:2463
        _put(v,'ani',((cc)-1,),xp.power(_get(v['qi'], (cc - 1,)) / (_get(v['ni'], (cc - 1,)) * _get(v['rhobar'], (cc - 1,)) * v['fourthirdspi'] * v['alphstr'] * tables['gamma_tab'][v['gi'] - 1,] / v['gammnu']), _r(1.0) / (_r(2.0) + _get(v['deltastr'], (cc - 1,)))),mask_172,xp)  # WRF:2464
        _put(v,'ani',((cc)-1,),_max(_get(v['ani'], (cc - 1,)), _r(2e-06)),mask_172,xp)  # WRF:2467
        _put(v,'cni',((cc)-1,),v['alphstr'] * xp.power(_get(v['ani'], (cc - 1,)), _get(v['deltastr'], (cc - 1,))),mask_172,xp)  # WRF:2468
        mask_176=xp.logical_and(mask_172,_get(v['ani'], (cc - 1,)) > _r(0.0005))
        taken_176=mask_176
        _put(v,'ani',((cc)-1,),_r(0.0005),mask_176,xp)  # WRF:2472
        _put(v,'cni',((cc)-1,),xp.power(v['ao'], _r(1.0) - _get(v['deltastr'], (cc - 1,))) * xp.power(_get(v['ani'], (cc - 1,)), _get(v['deltastr'], (cc - 1,))),mask_176,xp)  # WRF:2473
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_176,xp)  # WRF:2474
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_176,xp)  # WRF:2475
        _put(v,'ni',((cc)-1,),_get(v['qi'], (cc - 1,)) / (_get(v['rhobar'], (cc - 1,)) * v['fourthirdspi'] * xp.power(v['ao'], _r(1.0) - _get(v['deltastr'], (cc - 1,))) * xp.power(_get(v['ani'], (cc - 1,)), _r(2.0) + _get(v['deltastr'], (cc - 1,))) * tables['gamma_tab'][v['gi'] - 1,] / v['gammnu']),mask_176,xp)  # WRF:2476
        _put(v,'ni',((cc)-1,),_max(_get(v['ni'], (cc - 1,)), v['qnsmall']),mask_176,xp)  # WRF:2478
        _put(v,'ai',((cc)-1,),_get(v['ani'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_172,xp)  # WRF:2481
        _put(v,'ci',((cc)-1,),_get(v['cni'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_172,xp)  # WRF:2482
        call_out=_call('var_check',(v['nu'],v['ao'],v['fourthirdspi'],v['gammnu'],_get(v['qi'], (cc - 1,)),_get(v['deltastr'], (cc - 1,)),_get(v['ani'], (cc - 1,)),_get(v['cni'], (cc - 1,)),_get(v['rni'], (cc - 1,)),_get(v['rhobar'], (cc - 1,)),_get(v['ni'], (cc - 1,)),_get(v['ai'], (cc - 1,)),_get(v['ci'], (cc - 1,)),v['alphstr'],v['alphv'],v['betam'],),tables,xp)  # WRF:2484
        _put(v,'deltastr',((cc)-1,),call_out[0],mask_172,xp)  # WRF:2484
        _put(v,'ani',((cc)-1,),call_out[1],mask_172,xp)  # WRF:2484
        _put(v,'cni',((cc)-1,),call_out[2],mask_172,xp)  # WRF:2484
        _put(v,'rni',((cc)-1,),call_out[3],mask_172,xp)  # WRF:2484
        _put(v,'rhobar',((cc)-1,),call_out[4],mask_172,xp)  # WRF:2484
        _put(v,'ni',((cc)-1,),call_out[5],mask_172,xp)  # WRF:2484
        _put(v,'ai',((cc)-1,),call_out[6],mask_172,xp)  # WRF:2484
        _put(v,'ci',((cc)-1,),call_out[7],mask_172,xp)  # WRF:2484
        _put(v,'alphstr',None,call_out[8],mask_172,xp)  # WRF:2484
        _put(v,'alphv',None,call_out[9],mask_172,xp)  # WRF:2484
        _put(v,'betam',None,call_out[10],mask_172,xp)  # WRF:2484
        _put(v,'fv',((cc)-1,),_r(1.0),mask_171,xp)  # WRF:2492
        _put(v,'fh',((cc)-1,),_r(1.0),mask_171,xp)  # WRF:2493
        _put(v,'gamma_arg',None,v['nu'] - _r(1.0) + _get(v['deltastr'], (cc - 1,)),mask_171,xp)  # WRF:2495
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_171,xp)  # WRF:2496
        _put(v,'phiivt',None,_get(v['cni'], (cc - 1,)) / _get(v['ani'], (cc - 1,)) * tables['gamma_tab'][v['gi'] - 1,] * v['i_gammnu'],mask_171,xp)  # WRF:2497
        _put(v,'alphstr',None,xp.power(v['ao'], _r(1.0) - _get(v['deltastr'], (cc - 1,))),mask_171,xp)  # WRF:2498
        mask_177=xp.logical_and(mask_171,v['phiivt'] < _r(1.0))
        taken_177=mask_177
        _put(v,'bl',None,_r(1.0),mask_177,xp)  # WRF:2502
        _put(v,'al',None,_r(2.0),mask_177,xp)  # WRF:2503
        _put(v,'aa',None,v['pi'],mask_177,xp)  # WRF:2504
        _put(v,'ba',None,_r(2.0),mask_177,xp)  # WRF:2505
        _put(v,'qe',None,(_r(1.0) - v['phiivt']) * (_get(v['rhobar'], (cc - 1,)) / v['rhoi']) + v['phiivt'],mask_177,xp)  # WRF:2506
        mask_178=xp.logical_and(xp.logical_and(mask_171,xp.logical_not(taken_177)),v['phiivt'] > _r(1.0))
        taken_177=xp.logical_or(taken_177,mask_178)
        _put(v,'al',None,_r(2.0),mask_178,xp)  # WRF:2508
        _put(v,'bl',None,_r(1.0),mask_178,xp)  # WRF:2509
        _put(v,'aa',None,v['pi'] * v['alphstr'],mask_178,xp)  # WRF:2510
        _put(v,'ba',None,_get(v['deltastr'], (cc - 1,)) + _r(1.0),mask_178,xp)  # WRF:2511
        _put(v,'qe',None,_r(1.0),mask_178,xp)  # WRF:2512
        mask_179=xp.logical_and(xp.logical_and(mask_171,xp.logical_not(taken_177)),v['phiivt'] == _r(1.0))
        taken_177=xp.logical_or(taken_177,mask_179)
        _put(v,'bl',None,_r(1.0),mask_179,xp)  # WRF:2514
        _put(v,'al',None,_r(2.0),mask_179,xp)  # WRF:2515
        _put(v,'aa',None,v['pi'],mask_179,xp)  # WRF:2516
        _put(v,'ba',None,_r(2.0),mask_179,xp)  # WRF:2517
        _put(v,'qe',None,_r(1.0),mask_179,xp)  # WRF:2518
        _put(v,'qe',None,_min(v['qe'], _r(1.0)),mask_171,xp)  # WRF:2520
        _put(v,'xn',None,_r(2.0) / _get(v['rhobar'], (cc - 1,)) * (_get(v['rhobar'], (cc - 1,)) - v['rhoair']) * v['g_home'] * v['rhoair'] / (v['mu'] * v['mu']) * (v['fourthirdspi'] * _get(v['rhobar'], (cc - 1,))) * v['alphstr'] * (v['al'] * v['al']) / (v['aa'] * v['qe']) * xp.power(v['qe'], _r(3.0) / _r(4.0)),mask_171,xp)  # WRF:2523
        _put(v,'bx',None,_get(v['deltastr'], (cc - 1,)) + _r(2.0) + _r(2.0) * v['bl'] - v['ba'],mask_171,xp)  # WRF:2525
        _put(v,'xm',None,v['xn'] * xp.power(_get(v['ani'], (cc - 1,)), v['bx']) * gamma(v['nu'] + v['bx'], xp=xp) * v['i_gammnu'],mask_171,xp)  # WRF:2528
        _put(v,'f_c1',None,_r(4.0) / (_r(5.83) * _r(5.83) * xp.sqrt(_r(0.6))),mask_171,xp)  # WRF:2532
        _put(v,'f_c2',None,_r(5.83) * _r(5.83) / _r(4.0),mask_171,xp)  # WRF:2533
        _put(v,'bm',None,v['f_c1'] * xp.sqrt(v['xm']) / (_r(2.0) * (xp.sqrt(_r(1.0) + v['f_c1'] * xp.sqrt(v['xm'])) - _r(1.0)) * xp.sqrt(_r(1.0) + v['f_c1'] * xp.sqrt(v['xm']))) - _r(1e-05) * v['xm'] / (v['f_c2'] * ((xp.sqrt(_r(1.0) + v['f_c1'] * xp.sqrt(v['xm'])) - _r(1.0)) * (xp.sqrt(_r(1.0) + v['f_c1'] * xp.sqrt(v['xm'])) - _r(1.0)))),mask_171,xp)  # WRF:2534
        _put(v,'am',None,(v['f_c2'] * ((xp.sqrt(_r(1.0) + v['f_c1'] * xp.sqrt(v['xm'])) - _r(1.0)) * (xp.sqrt(_r(1.0) + v['f_c1'] * xp.sqrt(v['xm'])) - _r(1.0))) - _r(1e-05) * v['xm']) / xp.power(v['xm'], v['bm']),mask_171,xp)  # WRF:2539
        mask_180=xp.logical_and(mask_171,v['xm'] > _r(100000000.0))
        taken_180=mask_180
        _put(v,'am',None,_r(1.0865),mask_180,xp)  # WRF:2543
        _put(v,'bm',None,_r(0.499),mask_180,xp)  # WRF:2544
        _put(v,'nre',None,v['am'] * xp.power(v['xm'], v['bm']),mask_171,xp)  # WRF:2548
        _put(v,'vtrni1',((cc)-1,),_min(v['mu'] / v['rhoair'] * _r(0.5) * v['am'] * xp.power(v['xn'], v['bm']) * xp.power(_get(v['ani'], (cc - 1,)), v['bx'] * v['bm'] - _r(1.0)) * gamma(v['nu'] + v['bx'] * v['bm'] - _r(1.0), xp=xp) * v['i_gammnu'], _r(25.0)),mask_171,xp)  # WRF:2551
        _put(v,'vtrmi1',((cc)-1,),_min(v['mu'] / v['rhoair'] * _r(0.5) * v['am'] * xp.power(v['xn'], v['bm']) * xp.power(_get(v['ani'], (cc - 1,)), v['bx'] * v['bm'] - _r(1.0)) * gamma(v['nu'] + v['bx'] * v['bm'] - _r(1.0) + _r(2.0) + _get(v['deltastr'], (cc - 1,)), xp=xp) / gamma(v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)), xp=xp), _r(25.0)),mask_171,xp)  # WRF:2555
        mask_181=xp.logical_and(mask_171,xp.logical_and(v['temp'] > v['t0'], _get(v['qmlt'], (cc - 1,)) < _r(0.0)))
        taken_181=mask_181
        _put(v,'vtrni1',((cc)-1,),_get(v['vtrmi1'], (cc - 1,)),mask_181,xp)  # WRF:2560
        _put(v,'gamma_arg',None,v['nu'] - _r(1.0) + _get(v['deltastr'], (cc - 1,)),mask_171,xp)  # WRF:2570
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_171,xp)  # WRF:2571
        _put(v,'phirad',None,_get(v['cni'], (cc - 1,)) / _get(v['ani'], (cc - 1,)) * tables['gamma_tab'][v['gi'] - 1,] * v['i_gammnu'],mask_171,xp)  # WRF:2572
        mask_182=xp.logical_and(mask_171,v['phirad'] <= _r(1.0))
        taken_182=mask_182
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_182,xp)  # WRF:2574
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_182,xp)  # WRF:2575
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0),mask_182,xp)  # WRF:2576
        _put(v,'gi2',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_182,xp)  # WRF:2577
        _put(v,'effi',((cc)-1,),_get(v['cni'], (cc - 1,)) * tables['gamma_tab'][v['gi'] - 1,] / tables['gamma_tab'][v['gi2'] - 1,],mask_182,xp)  # WRF:2578
        mask_183=xp.logical_and(mask_171,xp.logical_not(taken_182))
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_183,xp)  # WRF:2580
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_183,xp)  # WRF:2581
        _put(v,'gamma_arg',None,v['nu'] + _r(1.0) + _get(v['deltastr'], (cc - 1,)),mask_183,xp)  # WRF:2582
        _put(v,'gi2',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_183,xp)  # WRF:2583
        _put(v,'effi',((cc)-1,),_get(v['ani'], (cc - 1,)) * tables['gamma_tab'][v['gi'] - 1,] / tables['gamma_tab'][v['gi2'] - 1,],mask_183,xp)  # WRF:2584
        _put(v,'ci',((cc)-1,),_get(v['cni'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_171,xp)  # WRF:2588
        _put(v,'ai',((cc)-1,),_get(v['ani'], (cc - 1,)) * _get(v['ani'], (cc - 1,)) * _get(v['cni'], (cc - 1,)) * _get(v['ni'], (cc - 1,)),mask_171,xp)  # WRF:2589
        _put(v,'ai',((cc)-1,),_max(_get(v['ai'], (cc - 1,)), v['qasmall']),mask_171,xp)  # WRF:2591
        _put(v,'ci',((cc)-1,),_max(_get(v['ci'], (cc - 1,)), v['qasmall']),mask_171,xp)  # WRF:2592
        _put(v,'ni',((cc)-1,),_max(_get(v['ni'], (cc - 1,)), v['qnsmall']),mask_171,xp)  # WRF:2593
        mask_184=xp.logical_and(mask_2,xp.logical_not(taken_171))
        _put(v,'qv',None,v['qv'] + _get(v['qi'], (cc - 1,)),mask_184,xp)  # WRF:2597
        _put(v,'theta',None,v['theta'] + v['theta'] * v['i_temp'] * (_get(v['qi'], (cc - 1,)) * v['xxls']) * v['i_cp'],mask_184,xp)  # WRF:2598
        _put(v,'qi',((cc)-1,),_r(0.0),mask_184,xp)  # WRF:2599
        _put(v,'ai',((cc)-1,),_r(0.0),mask_184,xp)  # WRF:2600
        _put(v,'ni',((cc)-1,),_r(0.0),mask_184,xp)  # WRF:2601
        _put(v,'ci',((cc)-1,),_r(0.0),mask_184,xp)  # WRF:2602
        _put(v,'vtrzi1',((cc)-1,),_r(0.0),mask_184,xp)  # WRF:2603
        _put(v,'vtrmi1',((cc)-1,),_r(0.0),mask_184,xp)  # WRF:2604
        _put(v,'vtrni1',((cc)-1,),_r(0.0),mask_184,xp)  # WRF:2605
        _put(v,'gamma_arg',None,v['nu'] - _r(1.0) + _get(v['deltastr'], (cc - 1,)),mask_2,xp)  # WRF:2611
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_2,xp)  # WRF:2612
        mask_185=xp.logical_and(mask_2,xp.logical_and(xp.logical_and(xp.logical_and(xp.logical_and(_get(v['ai'], (cc - 1,)) > _r(1e-12), _get(v['ci'], (cc - 1,)) > _r(1e-12)), _get(v['ani'], (cc - 1,)) > _r(2e-06)), _get(v['cni'], (cc - 1,)) > _r(2e-06)), _get(v['qi'], (cc - 1,)) > _r(1e-09)))
        taken_185=mask_185
        _put(v,'phii1d',((cc)-1,),_get(v['cni'], (cc - 1,)) / _get(v['ani'], (cc - 1,)) * tables['gamma_tab'][v['gi'] - 1,] * v['i_gammnu'],mask_185,xp)  # WRF:2617
        _put(v,'rhopo1d',((cc)-1,),_get(v['rhobar'], (cc - 1,)),mask_185,xp)  # WRF:2618
        _put(v,'gamma_arg',None,v['nu'] + _r(3.0) + _get(v['deltastr'], (cc - 1,)),mask_2,xp)  # WRF:2621
        _put(v,'gi',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_2,xp)  # WRF:2622
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _get(v['deltastr'], (cc - 1,)),mask_2,xp)  # WRF:2623
        _put(v,'gi2',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_2,xp)  # WRF:2624
        _put(v,'gamma_arg',None,v['nu'] + _r(2.0) + _r(2.0) * _get(v['deltastr'], (cc - 1,)),mask_2,xp)  # WRF:2625
        _put(v,'gi3',None,_min(_max(_nint(v['gamma_arg'] * _r(100000.0) - 355000 + 1), 1), 505001),mask_2,xp)  # WRF:2626
        mask_186=xp.logical_and(mask_2,xp.logical_and(xp.logical_and(xp.logical_and(xp.logical_and(_get(v['ai'], (cc - 1,)) > _r(1e-12), _get(v['ci'], (cc - 1,)) > _r(1e-12)), _get(v['ani'], (cc - 1,)) > _r(2e-06)), _get(v['cni'], (cc - 1,)) > _r(2e-06)), _get(v['qi'], (cc - 1,)) > _r(1e-09)))
        taken_186=mask_186
        _put(v,'di1d',((cc)-1,),_max(_r(2.0) * _get(v['ani'], (cc - 1,)) * tables['gamma_tab'][v['gi'] - 1,] / tables['gamma_tab'][v['gi2'] - 1,], _r(2.0) * _get(v['cni'], (cc - 1,)) * tables['gamma_tab'][v['gi3'] - 1,] / tables['gamma_tab'][v['gi2'] - 1,]),mask_186,xp)  # WRF:2630
        _put(v,'vmi1d',((cc)-1,),_get(v['vtrmi1'], (cc - 1,)),mask_186,xp)  # WRF:2632
        mask_187=xp.logical_and(mask_2,xp.logical_and(_get(v['ni'], (cc - 1,)) > _r(0.0), _get(v['ani'], (cc - 1,)) > _r(0.0)))
        taken_187=mask_187
        _put(v,'vtrzi1',((cc)-1,),_r(0.176) / _r(0.93) * (_r(6.0) / v['pi'] * (_r(6.0) / v['pi'])) * (v['fourthirdspi'] * _get(v['rhobar'], (cc - 1,)) * xp.power(v['ao'], _r(1.0) - _get(v['deltastr'], (cc - 1,))) / xp.power(_r(2.0), _r(2.0) + _get(v['deltastr'], (cc - 1,))) * (v['fourthirdspi'] * _get(v['rhobar'], (cc - 1,)) * xp.power(v['ao'], _r(1.0) - _get(v['deltastr'], (cc - 1,))) / xp.power(_r(2.0), _r(2.0) + _get(v['deltastr'], (cc - 1,))))) * v['rhoair'] / _r(900.0) / _r(900.0) * _get(v['ni'], (cc - 1,)) * xp.power(_r(2.0) * _get(v['ani'], (cc - 1,)), _r(2.0) * (_r(2.0) + _get(v['deltastr'], (cc - 1,)))) * gamma(v['nu'] + _r(2.0) * (_r(2.0) + _get(v['deltastr'], (cc - 1,))), xp=xp) * v['i_gammnu'],mask_187,xp)  # WRF:2636
        _put(v,'vtrzi1',((cc)-1,),_r(1e+18) * _get(v['vtrzi1'], (cc - 1,)),mask_187,xp)  # WRF:2640
        mask_188=xp.logical_and(mask_2,xp.logical_not(taken_187))
        _put(v,'vtrzi1',((cc)-1,),_r(0.0),mask_188,xp)  # WRF:2642
    _put(v,'dbzsum',None,v['dbzr'] + _get(v['vtrzi1'], (ice1 - 1,)) + _get(v['vtrzi1'], (ice2 - 1,)) + _get(v['vtrzi1'], (ice3 - 1,)),mask_2,xp)  # WRF:2646
    mask_189=xp.logical_and(mask_2,v['dbzsum'] != _r(0.0))
    taken_189=mask_189
    _put(v,'dbz1d',None,_r(10.0) * xp.log10(v['dbzsum']),mask_189,xp)  # WRF:2648
    _put(v,'dbz1d',None,_max(_r(-35.0), v['dbz1d']),mask_189,xp)  # WRF:2649
    _put(v,'totaliwc',None,_r(0.0),mask_2,xp)  # WRF:2655
    for cc in range(1,cat+1):
        _put(v,'totaliwc',None,v['totaliwc'] + _get(v['qi'], (cc - 1,)),mask_2,xp)  # WRF:2657
    mask_190=xp.logical_and(mask_2,v['totaliwc'] > _r(0.0))
    taken_190=mask_190
    _put(v,'radw1',None,_max(_min(_get(v['qi'], (ice1 - 1,)) / v['totaliwc'], _r(1.0)), _r(0.0)),mask_190,xp)  # WRF:2660
    _put(v,'radw2',None,_max(_min(_get(v['qi'], (ice2 - 1,)) / v['totaliwc'], _r(1.0)), _r(0.0)),mask_190,xp)  # WRF:2661
    _put(v,'radw3',None,_max(_min(_get(v['qi'], (ice3 - 1,)) / v['totaliwc'], _r(1.0)), _r(0.0)),mask_190,xp)  # WRF:2662
    _put(v,'effi1d',None,_max(_min(_get(v['effi'], (ice1 - 1,)) * v['radw1'] + _get(v['effi'], (ice2 - 1,)) * v['radw2'] + _get(v['effi'], (ice3 - 1,)) * v['radw3'], _r(0.000999)), _r(5e-06)),mask_190,xp)  # WRF:2663
    mask_191=xp.logical_and(mask_2,xp.logical_not(taken_190))
    _put(v,'effi1d',None,_r(0.0),mask_191,xp)  # WRF:2666
    mask_192=xp.logical_and(mask_2,v['qr'] > v['qsmall'])
    taken_192=mask_192
    _put(v,'sedi',None,True,mask_192,xp)  # WRF:2673
    for cc in range(1,cat+1):
        mask_193=xp.logical_and(mask_2,_get(v['qi'], (cc - 1,)) > v['qsmall'])
        taken_193=mask_193
        _put(v,'sedi',None,True,mask_193,xp)  # WRF:2677
    mask_194=xp.logical_and(True,xp.logical_not(taken_2))
    _put(v,'xxls',None,_r(3150000.0) - _r(2370.0) * v['temp'] + _r(333700.0),mask_194,xp)  # WRF:2689
    _put(v,'xxlv',None,_r(3148400.0) - _r(2370.0) * v['temp'],mask_194,xp)  # WRF:2690
    _put(v,'i_temp',None,_r(1.0) / v['temp'],mask_194,xp)  # WRF:2692
    _put(v,'rhoair',None,v['pres_e'] / (v['rd'] * v['temp']),mask_194,xp)  # WRF:2693
    _put(v,'i_rhoair',None,_r(1.0) / v['rhoair'],mask_194,xp)  # WRF:2694
    _put(v,'qv',None,v['qv'] + v['qc'] + v['qr'],mask_194,xp)  # WRF:2696
    _put(v,'theta',None,v['theta'] + v['theta'] * v['i_temp'] * ((v['qc'] + v['qr']) * v['xxlv']) * v['i_cp'],mask_194,xp)  # WRF:2697
    for cc in range(1,cat+1):
        _put(v,'qv',None,v['qv'] + _get(v['qi'], (cc - 1,)),mask_194,xp)  # WRF:2699
        _put(v,'theta',None,v['theta'] + v['theta'] * v['i_temp'] * (_get(v['qi'], (cc - 1,)) * v['xxls']) * v['i_cp'],mask_194,xp)  # WRF:2700
        _put(v,'qi',((cc)-1,),_r(0.0),mask_194,xp)  # WRF:2701
        _put(v,'ni',((cc)-1,),_r(0.0),mask_194,xp)  # WRF:2702
        _put(v,'ai',((cc)-1,),_r(0.0),mask_194,xp)  # WRF:2703
        _put(v,'ci',((cc)-1,),_r(0.0),mask_194,xp)  # WRF:2704
        _put(v,'vtrmi1',((cc)-1,),_r(0.0),mask_194,xp)  # WRF:2705
        _put(v,'vtrni1',((cc)-1,),_r(0.0),mask_194,xp)  # WRF:2706
        _put(v,'vtrzi1',((cc)-1,),_r(0.0),mask_194,xp)  # WRF:2707
        _put(v,'vmi1d',((cc)-1,),_r(0.0),mask_194,xp)  # WRF:2708
        _put(v,'di1d',((cc)-1,),_r(0.0),mask_194,xp)  # WRF:2709
        _put(v,'phii1d',((cc)-1,),_r(0.0),mask_194,xp)  # WRF:2710
        _put(v,'rhopo1d',((cc)-1,),_r(0.0),mask_194,xp)  # WRF:2711
        _put(v,'qipre1d',((cc)-1,),_r(0.0),mask_194,xp)  # WRF:2712
        _put(v,'icetype1d',((cc)-1,),_r(0.0),mask_194,xp)  # WRF:2713
    _put(v,'effc1d',None,_r(0.0),mask_194,xp)  # WRF:2716
    _put(v,'effi1d',None,_r(0.0),mask_194,xp)  # WRF:2717
    _put(v,'dbz1d',None,_r(-35.0),mask_194,xp)  # WRF:2718
    _put(v,'qc',None,_r(0.0),mask_194,xp)  # WRF:2719
    _put(v,'qr',None,_r(0.0),mask_194,xp)  # WRF:2720
    _put(v,'nr',None,_r(0.0),mask_194,xp)  # WRF:2721
    _put(v,'vtrm',None,_r(0.0),mask_194,xp)  # WRF:2722
    _put(v,'vtrn',None,_r(0.0),mask_194,xp)  # WRF:2723
    _put(v,'qrpre1d',None,_r(0.0),mask_194,xp)  # WRF:2724
    if observer is not None: observer(4,v,True)  # Original checkpoint WRF:2727
    return v
