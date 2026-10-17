"""Generated literal Milbrandt-Yau local processes; regenerate with proofs/s2_milbrandt/generate_point.py.
Pristine source SHA256: 76d11c00976f3c7dac30ff6ebb3efac1698bf43c13131b3aae43e464f22d23e5
"""
import numpy as jnp
from functools import lru_cache
from .math import _gamma

def _op(op,a,b):
    return {'add':lambda:a+b,'sub':lambda:a-b,'mul':lambda:a*b,'div':lambda:a/b}[op]()
def _pow(a,b): return a**b
_exp, _log, _log10 = jnp.exp, jnp.log, jnp.log10
_not = jnp.logical_not

@lru_cache(maxsize=2)
def constants(dtype='float32'):
    _r = jnp.dtype(dtype).type
    ccntype = 2
    dt = _r(1.)  # idt is NOT cached; computed from actual dt in local_processes
    muc = _r(_r(3.0))  # WRF:1026
    alpha_c = _r(_r(1.0))  # WRF:1027
    alpha_r = _r(_r(0.0))  # WRF:1028
    alpha_i = _r(_r(0.0))  # WRF:1029
    alpha_s = _r(_r(0.0))  # WRF:1030
    alpha_g = _r(_r(0.0))  # WRF:1031
    alpha_h = _r(_r(0.0))  # WRF:1032
    no_s_max = _r(_r(100000000.0))  # WRF:1033
    lamdas_min = _r(_r(500.0))  # WRF:1034
    no_r_sm = _r(_r(10000000.0))  # WRF:1037
    no_g_sm = _r(_r(4000000.0))  # WRF:1038
    no_h_sm = _r(_r(100000.0))  # WRF:1039
    afr = _r(_r(149.1))  # WRF:1053
    bfr = _r(_r(0.5))  # WRF:1053
    afi = _r(_r(71.34))  # WRF:1054
    bfi = _r(_r(0.6635))  # WRF:1054
    afs = _r(_r(11.72))  # WRF:1055
    bfs = _r(_r(0.41))  # WRF:1055
    afg = _r(_r(19.3))  # WRF:1056
    bfg = _r(_r(0.37))  # WRF:1056
    afh = _r(_r(206.89))  # WRF:1057
    bfh = _r(_r(0.6384))  # WRF:1057
    epsq = _r(_r(1e-14))  # WRF:1062
    epsn = _r(_r(0.001))  # WRF:1063
    epsq2 = _r(_r(1e-06))  # WRF:1064
    epsvis = _r(_r(1.0))  # WRF:1065
    ilammin1 = _r(_r(1e-06))  # WRF:1067
    ilammin2 = _r(_r(1e-10))  # WRF:1068
    eps = _r(_r(1e-32))  # WRF:1069
    deg = _r(_r(400.0))  # WRF:1071
    mgo = _r(_r(1.6e-10))  # WRF:1071
    deh = _r(_r(900.0))  # WRF:1072
    dei = _r(_r(500.0))  # WRF:1073
    mio = _r(_r(1e-12))  # WRF:1073
    nti0 = _r(_r(1000.0))  # WRF:1073
    dew = _r(_r(1000.0))  # WRF:1074
    desfix = _r(_r(100.0))  # WRF:1075
    desmax = _r(_r(500.0))  # WRF:1076
    dso = _r(_r(0.000125))  # WRF:1077
    dmr = _r(_r(3.0))  # WRF:1078
    dmi = _r(_r(3.0))  # WRF:1078
    dmg = _r(_r(3.0))  # WRF:1078
    dmh = _r(_r(3.0))  # WRF:1078
    drmax = _r(_r(0.005))  # WRF:1083
    vrmax = _r(_r(16.0))  # WRF:1083
    epsqr_sedi = _r(_r(1e-08))  # WRF:1083
    dimax = _r(_r(0.005))  # WRF:1084
    vimax = _r(_r(2.0))  # WRF:1084
    epsqi_sedi = _r(_r(1e-10))  # WRF:1084
    dsmax = _r(_r(0.005))  # WRF:1085
    vsmax = _r(_r(4.0))  # WRF:1085
    epsqs_sedi = _r(_r(1e-08))  # WRF:1085
    dgmax = _r(_r(0.002))  # WRF:1086
    vgmax = _r(_r(6.0))  # WRF:1086
    epsqg_sedi = _r(_r(1e-08))  # WRF:1086
    dhmax = _r(_r(0.08))  # WRF:1087
    vhmax = _r(_r(25.0))  # WRF:1087
    epsqh_sedi = _r(_r(1e-10))  # WRF:1087
    cpw = _r(_r(4218.0))  # WRF:1089
    deo = _r(_r(1.225))  # WRF:1090
    thrd = _r(_op('div', _r(1.0), _r(3.0)))  # WRF:1091
    sixth = _r(_op('mul', _r(0.5), thrd))  # WRF:1092
    ers = _r(_r(1.0))  # WRF:1093
    eci = _r(_r(1.0))  # WRF:1093
    eri = _r(_r(1.0))  # WRF:1094
    erh = _r(_r(1.0))  # WRF:1094
    xdisp = _r(_r(0.25))  # WRF:1095
    aa11 = _r(_r(9440000000000000.0))  # WRF:1096
    aa22 = _r(_r(5780.0))  # WRF:1096
    rh = _r(_r(4.1e-05))  # WRF:1096
    avx = _r(_r(0.78))  # WRF:1097
    bvx = _r(_r(0.3))  # WRF:1097
    abigg = _r(_r(0.66))  # WRF:1098
    bbigg = _r(_r(100.0))  # WRF:1098
    fdielec = _r(_r(4.464))  # WRF:1099
    zfact = _r(_r(1e+18))  # WRF:1100
    minzet = _r(-_r(99.0))  # WRF:1101
    maxvis = _r(_r(99000.0))  # WRF:1102
    drshed = _r(_r(0.001))  # WRF:1103
    sigcthrs = _r(_r(1.5e-05))  # WRF:1104
    kk1 = _r(_r(3030.0))  # WRF:1105
    kk2 = _r(_r(2590000000000000.0))  # WRF:1106
    dhh = _r(_r(8.2e-05))  # WRF:1107
    zmax_sedi = _r(_r(20000.0))  # WRF:1108
    dr_large = _r(_r(0.0002))  # WRF:1109
    ds_large = _r(_r(0.0002))  # WRF:1110
    dh_large = _r(_r(0.01))  # WRF:1111
    dh_min = _r(_r(0.001))  # WRF:1112
    dr_3cmpthrs = _r(_r(0.0025))  # WRF:1113
    w_cngh = _r(_r(3.0))  # WRF:1114
    ngh_crit = _r(_r(1.0))  # WRF:1115
    tc_fzrh = _r(-_r(10.0))  # WRF:1116
    cnsgthres = _r(_r(1.0))  # WRF:1117
    capfact_i = _r(_r(0.5))  # WRF:1118
    capfact_s = _r(_r(0.5))  # WRF:1119
    fv_dsmin = _r(_r(0.000125))  # WRF:1120
    fv_dsmax = _r(_r(0.008))  # WRF:1121
    ni_max = _r(_r(10000000.0))  # WRF:1122
    satw_peak = _r(_r(1.01))  # WRF:1123
    cpi = _r(_r(2115.3))  # WRF:1130
    trpl = _r(_r(273.16))  # WRF:1131
    tcdk = _r(_r(273.15))  # WRF:1132
    rauw = _r(_r(1000.0))  # WRF:1133
    eps2 = _r(_r(0.3780199778986))  # WRF:1134
    tgl = _r(_r(273.16))  # WRF:1135
    consol = _r(_r(1367.0))  # WRF:1136
    rayt = _r(_r(6371220.0))  # WRF:1137
    stefan = _r(_r(5.66948e-08))  # WRF:1138
    pi = _r(_r(3.14159265359))  # WRF:1139
    omega = _r(_r(7.292e-05))  # WRF:1140
    knams = _r(_r(0.514791))  # WRF:1141
    stlo = _r(_r(0.0006628486583943))  # WRF:1142
    karman = _r(_r(0.35))  # WRF:1143
    ric = _r(_r(0.2))  # WRF:1144
    chlc = _r(_r(2501000.0))  # WRF:1146
    chlf = _r(_r(334000.0))  # WRF:1147
    cpd = _r(_r(1005.46))  # WRF:1148
    cpv = _r(_r(1869.46))  # WRF:1149
    rgasd = _r(_r(287.05))  # WRF:1150
    rgasv = _r(_r(461.51))  # WRF:1151
    eps1 = _r(_r(0.62194800221014))  # WRF:1152
    delta = _r(_r(0.6077686814144))  # WRF:1153
    cappa = _r(_r(0.28549121795))  # WRF:1154
    grav = _r(_r(9.80616))  # WRF:1155
    lama0 = _r(_r(6.6e-08))  # WRF:1160
    t0 = _r(_r(293.15))  # WRF:1161
    p0 = _r(_r(101325.0))  # WRF:1162
    ra = _r(_r(1e-06))  # WRF:1163
    kboltz = _r(_r(1.381e-23))  # WRF:1164
    kapa = _r(_r(539000.0))  # WRF:1165
    debug_on = bool(False)  # WRF:1168
    debug_abort = bool(True)  # WRF:1169
    icedep_on = bool(True)  # WRF:1170
    grpl_on = bool(True)  # WRF:1171
    hail_on = bool(True)  # WRF:1172
    rainaccr_on = bool(True)  # WRF:1173
    snowspherical = bool(False)  # WRF:1174
    primicenucl = _r(1)  # WRF:1175
    outfreq = _r(_r(60.0))  # WRF:1176
    cexr1 = _r(0.)
    cexr2 = _r(0.)
    cexr3 = _r(0.)
    cexr4 = _r(0.)
    cexr5 = _r(0.)
    cexr6 = _r(0.)
    cexr9 = _r(0.)
    cexs1 = _r(0.)
    cexs2 = _r(0.)
    chls = _r(0.)
    ck5 = _r(0.)
    ck6 = _r(0.)
    ckqg1 = _r(0.)
    ckqg2 = _r(0.)
    ckqg4 = _r(0.)
    ckqh1 = _r(0.)
    ckqh2 = _r(0.)
    ckqh4 = _r(0.)
    ckqi1 = _r(0.)
    ckqi2 = _r(0.)
    ckqi4 = _r(0.)
    ckqr1 = _r(0.)
    ckqr2 = _r(0.)
    ckqr3 = _r(0.)
    ckqs1 = _r(0.)
    ckqs2 = _r(0.)
    cmg = _r(0.)
    cmh = _r(0.)
    cmi = _r(0.)
    cmr = _r(0.)
    cms = _r(0.)
    cms_d3 = _r(0.)
    dms = _r(0.)
    eds = _r(0.)
    fds = _r(0.)
    gc1 = _r(0.)
    gc11 = _r(0.)
    gc12 = _r(0.)
    gc13 = _r(0.)
    gc14 = _r(0.)
    gc15 = _r(0.)
    gc2 = _r(0.)
    gc3 = _r(0.)
    gc4 = _r(0.)
    gc5 = _r(0.)
    gc6 = _r(0.)
    gc7 = _r(0.)
    gc8 = _r(0.)
    gg09 = _r(0.)
    gg11 = _r(0.)
    gg12 = _r(0.)
    gg13 = _r(0.)
    gg31 = _r(0.)
    gg32 = _r(0.)
    gg33 = _r(0.)
    gg34 = _r(0.)
    gg35 = _r(0.)
    gg36 = _r(0.)
    gg40 = _r(0.)
    gg50 = _r(0.)
    gh09 = _r(0.)
    gh11 = _r(0.)
    gh12 = _r(0.)
    gh13 = _r(0.)
    gh31 = _r(0.)
    gh32 = _r(0.)
    gh33 = _r(0.)
    gh40 = _r(0.)
    gh50 = _r(0.)
    gi11 = _r(0.)
    gi20 = _r(0.)
    gi21 = _r(0.)
    gi22 = _r(0.)
    gi31 = _r(0.)
    gi32 = _r(0.)
    gi33 = _r(0.)
    gi34 = _r(0.)
    gi35 = _r(0.)
    gi36 = _r(0.)
    gi4 = _r(0.)
    gi40 = _r(0.)
    gi6 = _r(0.)
    gr17 = _r(0.)
    gr31 = _r(0.)
    gr32 = _r(0.)
    gr33 = _r(0.)
    gr34 = _r(0.)
    gr35 = _r(0.)
    gr36 = _r(0.)
    gr37 = _r(0.)
    gr50 = _r(0.)
    gs09 = _r(0.)
    gs11 = _r(0.)
    gs12 = _r(0.)
    gs13 = _r(0.)
    gs31 = _r(0.)
    gs32 = _r(0.)
    gs33 = _r(0.)
    gs34 = _r(0.)
    gs35 = _r(0.)
    gs36 = _r(0.)
    gs40 = _r(0.)
    gs40_d3 = _r(0.)
    gs50 = _r(0.)
    icexc9 = _r(0.)
    icexi9 = _r(0.)
    icexr9 = _r(0.)
    icexs2 = _r(0.)
    ichlf = _r(0.)
    icmg = _r(0.)
    icmh = _r(0.)
    icmi = _r(0.)
    icmr = _r(0.)
    icms = _r(0.)
    ideg = _r(0.)
    ideh = _r(0.)
    idei = _r(0.)
    idew = _r(0.)
    idms = _r(0.)
    idt = _r(0.)
    igc1 = _r(0.)
    igc5 = _r(0.)
    igg31 = _r(0.)
    igg34 = _r(0.)
    igg99 = _r(0.)
    igh31 = _r(0.)
    igh34 = _r(0.)
    igh99 = _r(0.)
    igi31 = _r(0.)
    igr31 = _r(0.)
    igr34 = _r(0.)
    igs20 = _r(0.)
    igs20_d3 = _r(0.)
    igs31 = _r(0.)
    igs34 = _r(0.)
    igs40 = _r(0.)
    imgo = _r(0.)
    imso = _r(0.)
    imuc = _r(0.)
    lcp = _r(0.)
    lfp = _r(0.)
    lsp = _r(0.)
    mso = _r(0.)
    n_c_sm = _r(0.)
    pi2 = _r(0.)
    piov4 = _r(0.)
    piov6 = _r(0.)
    rfact_fvfm = _r(0.)
    _m1 = (True) & (True)
    # WRF:1259
    pi2 = jnp.where(_m1, _r(_op('mul', pi, _r(2.0))), pi2)
    # WRF:1260
    piov4 = jnp.where(_m1, _r(_op('mul', _r(0.25), pi)), piov4)
    # WRF:1261
    piov6 = jnp.where(_m1, _r(_op('mul', pi, sixth)), piov6)
    # WRF:1262
    chls = jnp.where(_m1, _r(_op('add', chlc, chlf)), chls)
    # WRF:1263
    lcp = jnp.where(_m1, _r(_op('div', chlc, cpd)), lcp)
    # WRF:1264
    lfp = jnp.where(_m1, _r(_op('div', chlf, cpd)), lfp)
    # WRF:1265
    ichlf = jnp.where(_m1, _r(_op('div', _r(1.0), chlf)), ichlf)
    # WRF:1266
    lsp = jnp.where(_m1, _r(_op('add', lcp, lfp)), lsp)
    # WRF:1267
    ck5 = jnp.where(_m1, _r(_op('mul', _r(4098.17), lcp)), ck5)
    # WRF:1268
    ck6 = jnp.where(_m1, _r(_op('mul', _r(5806.485), lsp)), ck6)
    # WRF:1269
    idt = jnp.where(_m1, _r(_op('div', _r(1.0), dt)), idt)
    # WRF:1270
    imgo = jnp.where(_m1, _r(_op('div', _r(1.0), mgo)), imgo)
    # WRF:1271
    idew = jnp.where(_m1, _r(_op('div', _r(1.0), dew)), idew)
    # WRF:1272
    idei = jnp.where(_m1, _r(_op('div', _r(1.0), dei)), idei)
    # WRF:1273
    ideg = jnp.where(_m1, _r(_op('div', _r(1.0), deg)), ideg)
    # WRF:1274
    ideh = jnp.where(_m1, _r(_op('div', _r(1.0), deh)), ideh)
    # WRF:1279
    cmr = jnp.where(_m1, _r(_op('mul', piov6, dew)), cmr)
    # WRF:1279
    icmr = jnp.where(_m1, _r(_op('div', _r(1.0), cmr)), icmr)
    # WRF:1280
    cmi = jnp.where(_m1, _r(_r(440.0)), cmi)
    # WRF:1280
    icmi = jnp.where(_m1, _r(_op('div', _r(1.0), cmi)), icmi)
    # WRF:1281
    cmg = jnp.where(_m1, _r(_op('mul', piov6, deg)), cmg)
    # WRF:1281
    icmg = jnp.where(_m1, _r(_op('div', _r(1.0), cmg)), icmg)
    # WRF:1282
    cmh = jnp.where(_m1, _r(_op('mul', piov6, deh)), cmh)
    # WRF:1282
    icmh = jnp.where(_m1, _r(_op('div', _r(1.0), cmh)), icmh)
    # WRF:1284
    cms_d3 = jnp.where(_m1, _r(_op('mul', piov6, desfix)), cms_d3)
    _m2 = (_m1) & (snowspherical)
    # WRF:1286
    cms = jnp.where(_m2, _r(cms_d3), cms)
    # WRF:1287
    dms = jnp.where(_m2, _r(_r(3.0)), dms)
    _m3 = (_m1) & _not(_m2)
    # WRF:1290
    cms = jnp.where(_m3, _r(_r(0.1597)), cms)
    # WRF:1290
    dms = jnp.where(_m3, _r(_r(2.078)), dms)
    # WRF:1292
    icms = jnp.where(_m1, _r(_op('div', _r(1.0), cms)), icms)
    # WRF:1293
    idms = jnp.where(_m1, _r(_op('div', _r(1.0), dms)), idms)
    # WRF:1294
    mso = jnp.where(_m1, _r(_op('mul', cms, _pow(dso, dms))), mso)
    # WRF:1295
    imso = jnp.where(_m1, _r(_op('div', _r(1.0), mso)), imso)
    # WRF:1300
    eds = jnp.where(_m1, _r(_op('div', cms, piov6)), eds)
    # WRF:1301
    fds = jnp.where(_m1, _r(_op('sub', dms, _r(3.0))), fds)
    _m4 = (_m1) & ((fds != -_r(1.0)) & _not(snowspherical))
    # WRF:1302
    gs50 = jnp.where(_m4, _r(_gamma(_op('add', _op('add', _r(1.0), fds), alpha_s))), gs50)
    # WRF:1305
    imuc = jnp.where(_m1, _r(_op('div', _r(1.0), muc)), imuc)
    # WRF:1306
    gc1 = jnp.where(_m1, _r(_gamma(_op('add', alpha_c, _r(1.0)))), gc1)
    # WRF:1307
    igc1 = jnp.where(_m1, _r(_op('div', _r(1.0), gc1)), igc1)
    # WRF:1308
    gc2 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', alpha_c, _r(1.0)), _op('mul', _r(3.0), imuc)))), gc2)
    # WRF:1309
    gc3 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', alpha_c, _r(1.0)), _op('mul', _r(6.0), imuc)))), gc3)
    # WRF:1310
    gc4 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', alpha_c, _r(1.0)), _op('mul', _r(9.0), imuc)))), gc4)
    # WRF:1311
    gc11 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _op('mul', _r(1.0), imuc), _r(1.0)), alpha_c))), gc11)
    # WRF:1312
    gc12 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _op('mul', _r(2.0), imuc), _r(1.0)), alpha_c))), gc12)
    # WRF:1313
    gc5 = jnp.where(_m1, _r(_gamma(_op('add', _r(1.0), alpha_c))), gc5)
    # WRF:1314
    igc5 = jnp.where(_m1, _r(_op('div', _r(1.0), gc5)), igc5)
    # WRF:1315
    gc6 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(1.0), alpha_c), _op('mul', _r(1.0), imuc)))), gc6)
    # WRF:1316
    gc7 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(1.0), alpha_c), _op('mul', _r(2.0), imuc)))), gc7)
    # WRF:1317
    gc8 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(1.0), alpha_c), _op('mul', _r(3.0), imuc)))), gc8)
    # WRF:1318
    gc13 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _op('mul', _r(3.0), imuc), _r(1.0)), alpha_c))), gc13)
    # WRF:1319
    gc14 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _op('mul', _r(4.0), imuc), _r(1.0)), alpha_c))), gc14)
    # WRF:1320
    gc15 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _op('mul', _r(5.0), imuc), _r(1.0)), alpha_c))), gc15)
    # WRF:1321
    icexc9 = jnp.where(_m1, _r(_op('div', _r(1.0), _op('mul', _op('mul', _op('mul', gc2, igc1), piov6), dew))), icexc9)
    _m5 = (_m1) & (ccntype == 1)
    # WRF:1324
    n_c_sm = jnp.where(_m5, _r(_r(80000000.0)), n_c_sm)
    _m6 = (_m1) & _not(_m5) & (ccntype == 2)
    # WRF:1326
    n_c_sm = jnp.where(_m6, _r(_r(200000000.0)), n_c_sm)
    _m7 = (_m1) & _not((_m5) | (_m6)) & (ccntype == 3)
    # WRF:1328
    n_c_sm = jnp.where(_m7, _r(_r(500000000.0)), n_c_sm)
    _m8 = (_m1) & _not(((_m5) | (_m6)) | (_m7))
    # WRF:1330
    n_c_sm = jnp.where(_m8, _r(_r(200000000.0)), n_c_sm)
    # WRF:1334
    cexr1 = jnp.where(_m1, _r(_op('add', _op('add', _op('add', _r(1.0), alpha_r), dmr), bfr)), cexr1)
    # WRF:1335
    cexr2 = jnp.where(_m1, _r(_op('add', _op('add', _r(1.0), alpha_r), dmr)), cexr2)
    # WRF:1336
    gr17 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(2.5), alpha_r), _op('mul', _r(0.5), bfr)))), gr17)
    # WRF:1337
    gr31 = jnp.where(_m1, _r(_gamma(_op('add', _r(1.0), alpha_r))), gr31)
    # WRF:1338
    igr31 = jnp.where(_m1, _r(_op('div', _r(1.0), gr31)), igr31)
    # WRF:1339
    gr32 = jnp.where(_m1, _r(_gamma(_op('add', _r(2.0), alpha_r))), gr32)
    # WRF:1340
    gr33 = jnp.where(_m1, _r(_gamma(_op('add', _r(3.0), alpha_r))), gr33)
    # WRF:1341
    gr34 = jnp.where(_m1, _r(_gamma(_op('add', _r(4.0), alpha_r))), gr34)
    # WRF:1342
    igr34 = jnp.where(_m1, _r(_op('div', _r(1.0), gr34)), igr34)
    # WRF:1343
    gr35 = jnp.where(_m1, _r(_gamma(_op('add', _r(5.0), alpha_r))), gr35)
    # WRF:1344
    gr36 = jnp.where(_m1, _r(_gamma(_op('add', _r(6.0), alpha_r))), gr36)
    # WRF:1345
    gr37 = jnp.where(_m1, _r(_gamma(_op('add', _r(7.0), alpha_r))), gr37)
    # WRF:1346
    gr50 = jnp.where(_m1, _r(_pow(_op('mul', no_r_sm, gr31), _r(0.75))), gr50)
    # WRF:1347
    cexr5 = jnp.where(_m1, _r(_op('add', _r(2.0), alpha_r)), cexr5)
    # WRF:1348
    cexr6 = jnp.where(_m1, _r(_op('add', _op('add', _r(2.5), alpha_r), _op('mul', _r(0.5), bfr))), cexr6)
    # WRF:1349
    cexr9 = jnp.where(_m1, _r(_op('mul', _op('mul', cmr, gr34), igr31)), cexr9)
    # WRF:1349
    icexr9 = jnp.where(_m1, _r(_op('div', _r(1.0), cexr9)), icexr9)
    # WRF:1350
    cexr3 = jnp.where(_m1, _r(_op('add', _op('add', _r(1.0), bfr), alpha_r)), cexr3)
    # WRF:1351
    cexr4 = jnp.where(_m1, _r(_op('add', _r(1.0), alpha_r)), cexr4)
    # WRF:1352
    ckqr1 = jnp.where(_m1, _r(_op('div', _op('mul', afr, _gamma(_op('add', _op('add', _op('add', _r(1.0), alpha_r), dmr), bfr))), _gamma(_op('add', _op('add', _r(1.0), alpha_r), dmr)))), ckqr1)
    # WRF:1353
    ckqr2 = jnp.where(_m1, _r(_op('mul', _op('mul', afr, _gamma(_op('add', _op('add', _r(1.0), alpha_r), bfr))), gr31)), ckqr2)
    # WRF:1354
    ckqr3 = jnp.where(_m1, _r(_op('div', _op('mul', afr, _gamma(_op('add', _op('add', _r(7.0), alpha_r), bfr))), gr37)), ckqr3)
    # WRF:1357
    gi4 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', alpha_i, dmi), bfi))), gi4)
    # WRF:1358
    gi6 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(2.5), _op('mul', bfi, _r(0.5))), alpha_i))), gi6)
    # WRF:1359
    gi11 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(1.0), bfi), alpha_i))), gi11)
    # WRF:1360
    gi20 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _op('add', _r(0.0), bfi), _r(1.0)), alpha_i))), gi20)
    # WRF:1361
    gi21 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _op('add', _r(1.0), bfi), _r(1.0)), alpha_i))), gi21)
    # WRF:1362
    gi22 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _op('add', _r(2.0), bfi), _r(1.0)), alpha_i))), gi22)
    # WRF:1363
    gi31 = jnp.where(_m1, _r(_gamma(_op('add', _r(1.0), alpha_i))), gi31)
    # WRF:1364
    igi31 = jnp.where(_m1, _r(_op('div', _r(1.0), gi31)), igi31)
    # WRF:1365
    gi32 = jnp.where(_m1, _r(_gamma(_op('add', _r(2.0), alpha_i))), gi32)
    # WRF:1366
    gi33 = jnp.where(_m1, _r(_gamma(_op('add', _r(3.0), alpha_i))), gi33)
    # WRF:1367
    gi34 = jnp.where(_m1, _r(_gamma(_op('add', _r(4.0), alpha_i))), gi34)
    # WRF:1368
    gi35 = jnp.where(_m1, _r(_gamma(_op('add', _r(5.0), alpha_i))), gi35)
    # WRF:1369
    gi36 = jnp.where(_m1, _r(_gamma(_op('add', _r(6.0), alpha_i))), gi36)
    # WRF:1370
    gi40 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(1.0), alpha_i), dmi))), gi40)
    # WRF:1371
    icexi9 = jnp.where(_m1, _r(_op('div', _r(1.0), _op('mul', _op('mul', cmi, _gamma(_op('add', _op('add', _r(1.0), alpha_i), dmi))), igi31))), icexi9)
    # WRF:1372
    ckqi1 = jnp.where(_m1, _r(_op('div', _op('mul', afi, _gamma(_op('add', _op('add', _op('add', _r(1.0), alpha_i), dmi), bfi))), gi40)), ckqi1)
    # WRF:1373
    ckqi2 = jnp.where(_m1, _r(_op('mul', _op('mul', afi, gi11), igi31)), ckqi2)
    # WRF:1374
    ckqi4 = jnp.where(_m1, _r(_op('div', _r(1.0), _op('mul', _op('mul', cmi, gi40), igi31))), ckqi4)
    # WRF:1377
    cexs1 = jnp.where(_m1, _r(_op('add', _op('add', _r(2.5), _op('mul', _r(0.5), bfs)), alpha_s)), cexs1)
    # WRF:1378
    cexs2 = jnp.where(_m1, _r(_op('add', _op('add', _r(1.0), alpha_s), dms)), cexs2)
    # WRF:1379
    icexs2 = jnp.where(_m1, _r(_op('div', _r(1.0), cexs2)), icexs2)
    # WRF:1380
    gs09 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(2.5), _op('mul', bfs, _r(0.5))), alpha_s))), gs09)
    # WRF:1381
    gs11 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(1.0), bfs), alpha_s))), gs11)
    # WRF:1382
    gs12 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(2.0), bfs), alpha_s))), gs12)
    # WRF:1383
    gs13 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(3.0), bfs), alpha_s))), gs13)
    # WRF:1384
    gs31 = jnp.where(_m1, _r(_gamma(_op('add', _r(1.0), alpha_s))), gs31)
    # WRF:1385
    igs31 = jnp.where(_m1, _r(_op('div', _r(1.0), gs31)), igs31)
    # WRF:1386
    gs32 = jnp.where(_m1, _r(_gamma(_op('add', _r(2.0), alpha_s))), gs32)
    # WRF:1387
    gs33 = jnp.where(_m1, _r(_gamma(_op('add', _r(3.0), alpha_s))), gs33)
    # WRF:1388
    gs34 = jnp.where(_m1, _r(_gamma(_op('add', _r(4.0), alpha_s))), gs34)
    # WRF:1389
    igs34 = jnp.where(_m1, _r(_op('div', _r(1.0), gs34)), igs34)
    # WRF:1390
    gs35 = jnp.where(_m1, _r(_gamma(_op('add', _r(5.0), alpha_s))), gs35)
    # WRF:1391
    gs36 = jnp.where(_m1, _r(_gamma(_op('add', _r(6.0), alpha_s))), gs36)
    # WRF:1392
    gs40 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(1.0), alpha_s), dms))), gs40)
    # WRF:1393
    igs40 = jnp.where(_m1, _r(_op('div', _r(1.0), gs40)), igs40)
    # WRF:1394
    igs20 = jnp.where(_m1, _r(_op('div', _r(1.0), _op('mul', _op('mul', gs40, igs31), cms))), igs20)
    # WRF:1395
    ckqs1 = jnp.where(_m1, _r(_op('mul', _op('mul', afs, _gamma(_op('add', _op('add', _op('add', _r(1.0), alpha_s), dms), bfs))), igs40)), ckqs1)
    # WRF:1396
    ckqs2 = jnp.where(_m1, _r(_op('mul', _op('mul', afs, gs11), igs31)), ckqs2)
    # WRF:1397
    gs40_d3 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(1.0), alpha_s), _r(3.0)))), gs40_d3)
    # WRF:1398
    igs20_d3 = jnp.where(_m1, _r(_op('div', _r(1.0), _op('mul', _op('mul', gs40_d3, igs31), cms_d3))), igs20_d3)
    # WRF:1399
    rfact_fvfm = jnp.where(_m1, _r(_op('div', _op('mul', _op('mul', piov6, icms), _gamma(_op('add', _op('add', _r(4.0), bfs), alpha_s))), _gamma(_op('add', _op('add', _op('add', _r(1.0), dms), bfs), alpha_s)))), rfact_fvfm)
    # WRF:1402
    gg09 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(2.5), _op('mul', _r(0.5), bfg)), alpha_g))), gg09)
    # WRF:1403
    gg11 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(1.0), bfg), alpha_g))), gg11)
    # WRF:1404
    gg12 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(2.0), bfg), alpha_g))), gg12)
    # WRF:1405
    gg13 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(3.0), bfg), alpha_g))), gg13)
    # WRF:1406
    gg31 = jnp.where(_m1, _r(_gamma(_op('add', _r(1.0), alpha_g))), gg31)
    # WRF:1407
    igg31 = jnp.where(_m1, _r(_op('div', _r(1.0), gg31)), igg31)
    # WRF:1408
    gg32 = jnp.where(_m1, _r(_gamma(_op('add', _r(2.0), alpha_g))), gg32)
    # WRF:1409
    gg33 = jnp.where(_m1, _r(_gamma(_op('add', _r(3.0), alpha_g))), gg33)
    # WRF:1410
    gg34 = jnp.where(_m1, _r(_gamma(_op('add', _r(4.0), alpha_g))), gg34)
    # WRF:1411
    igg34 = jnp.where(_m1, _r(_op('div', _r(1.0), gg34)), igg34)
    # WRF:1412
    gg35 = jnp.where(_m1, _r(_gamma(_op('add', _r(5.0), alpha_g))), gg35)
    # WRF:1413
    gg36 = jnp.where(_m1, _r(_gamma(_op('add', _r(6.0), alpha_g))), gg36)
    # WRF:1414
    gg40 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(1.0), alpha_g), dmg))), gg40)
    # WRF:1415
    igg99 = jnp.where(_m1, _r(_op('div', _r(1.0), _op('mul', _op('mul', gg40, igg31), cmg))), igg99)
    # WRF:1416
    gg50 = jnp.where(_m1, _r(_pow(_op('mul', no_g_sm, gg31), _r(0.75))), gg50)
    # WRF:1417
    ckqg1 = jnp.where(_m1, _r(_op('div', _op('mul', afg, _gamma(_op('add', _op('add', _op('add', _r(1.0), alpha_g), dmg), bfg))), gg40)), ckqg1)
    # WRF:1418
    ckqg2 = jnp.where(_m1, _r(_op('mul', _op('mul', afg, gg11), igg31)), ckqg2)
    # WRF:1419
    ckqg4 = jnp.where(_m1, _r(_op('div', _r(1.0), _op('mul', _op('mul', cmg, gg40), igg31))), ckqg4)
    # WRF:1422
    gh09 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(2.5), _op('mul', bfh, _r(0.5))), alpha_h))), gh09)
    # WRF:1423
    gh11 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(1.0), bfh), alpha_h))), gh11)
    # WRF:1424
    gh12 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(2.0), bfh), alpha_h))), gh12)
    # WRF:1425
    gh13 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(3.0), bfh), alpha_h))), gh13)
    # WRF:1426
    gh31 = jnp.where(_m1, _r(_gamma(_op('add', _r(1.0), alpha_h))), gh31)
    # WRF:1427
    igh31 = jnp.where(_m1, _r(_op('div', _r(1.0), gh31)), igh31)
    # WRF:1428
    gh32 = jnp.where(_m1, _r(_gamma(_op('add', _r(2.0), alpha_h))), gh32)
    # WRF:1429
    gh33 = jnp.where(_m1, _r(_gamma(_op('add', _r(3.0), alpha_h))), gh33)
    # WRF:1430
    igh34 = jnp.where(_m1, _r(_op('div', _r(1.0), _gamma(_op('add', _r(4.0), alpha_h)))), igh34)
    # WRF:1431
    gh40 = jnp.where(_m1, _r(_gamma(_op('add', _op('add', _r(1.0), alpha_h), dmh))), gh40)
    # WRF:1432
    igh99 = jnp.where(_m1, _r(_op('div', _r(1.0), _op('mul', _op('mul', gh40, igh31), cmh))), igh99)
    # WRF:1433
    gh50 = jnp.where(_m1, _r(_pow(_op('mul', no_h_sm, gh31), _r(0.75))), gh50)
    # WRF:1434
    ckqh1 = jnp.where(_m1, _r(_op('div', _op('mul', afh, _gamma(_op('add', _op('add', _op('add', _r(1.0), alpha_h), dmh), bfh))), gh40)), ckqh1)
    # WRF:1435
    ckqh2 = jnp.where(_m1, _r(_op('mul', _op('mul', afh, gh11), igh31)), ckqh2)
    # WRF:1436
    ckqh4 = jnp.where(_m1, _r(_op('div', _r(1.0), _op('mul', _op('mul', cmh, gh40), igh31))), ckqh4)
    return {
        'aa11': aa11,
        'aa22': aa22,
        'abigg': abigg,
        'afg': afg,
        'afh': afh,
        'afi': afi,
        'afr': afr,
        'afs': afs,
        'alpha_c': alpha_c,
        'alpha_g': alpha_g,
        'alpha_h': alpha_h,
        'alpha_i': alpha_i,
        'alpha_r': alpha_r,
        'alpha_s': alpha_s,
        'avx': avx,
        'bbigg': bbigg,
        'bfg': bfg,
        'bfh': bfh,
        'bfi': bfi,
        'bfr': bfr,
        'bfs': bfs,
        'bvx': bvx,
        'capfact_i': capfact_i,
        'capfact_s': capfact_s,
        'cappa': cappa,
        'cexr1': cexr1,
        'cexr2': cexr2,
        'cexr3': cexr3,
        'cexr4': cexr4,
        'cexr5': cexr5,
        'cexr6': cexr6,
        'cexr9': cexr9,
        'cexs1': cexs1,
        'cexs2': cexs2,
        'chlc': chlc,
        'chlf': chlf,
        'chls': chls,
        'ck5': ck5,
        'ck6': ck6,
        'ckqg1': ckqg1,
        'ckqg2': ckqg2,
        'ckqg4': ckqg4,
        'ckqh1': ckqh1,
        'ckqh2': ckqh2,
        'ckqh4': ckqh4,
        'ckqi1': ckqi1,
        'ckqi2': ckqi2,
        'ckqi4': ckqi4,
        'ckqr1': ckqr1,
        'ckqr2': ckqr2,
        'ckqr3': ckqr3,
        'ckqs1': ckqs1,
        'ckqs2': ckqs2,
        'cmg': cmg,
        'cmh': cmh,
        'cmi': cmi,
        'cmr': cmr,
        'cms': cms,
        'cms_d3': cms_d3,
        'cnsgthres': cnsgthres,
        'consol': consol,
        'cpd': cpd,
        'cpi': cpi,
        'cpv': cpv,
        'cpw': cpw,
        'debug_abort': debug_abort,
        'debug_on': debug_on,
        'deg': deg,
        'deh': deh,
        'dei': dei,
        'delta': delta,
        'deo': deo,
        'desfix': desfix,
        'desmax': desmax,
        'dew': dew,
        'dgmax': dgmax,
        'dh_large': dh_large,
        'dh_min': dh_min,
        'dhh': dhh,
        'dhmax': dhmax,
        'dimax': dimax,
        'dmg': dmg,
        'dmh': dmh,
        'dmi': dmi,
        'dmr': dmr,
        'dms': dms,
        'dr_3cmpthrs': dr_3cmpthrs,
        'dr_large': dr_large,
        'drmax': drmax,
        'drshed': drshed,
        'ds_large': ds_large,
        'dsmax': dsmax,
        'dso': dso,
        'eci': eci,
        'eds': eds,
        'eps': eps,
        'eps1': eps1,
        'eps2': eps2,
        'epsn': epsn,
        'epsq': epsq,
        'epsq2': epsq2,
        'epsqg_sedi': epsqg_sedi,
        'epsqh_sedi': epsqh_sedi,
        'epsqi_sedi': epsqi_sedi,
        'epsqr_sedi': epsqr_sedi,
        'epsqs_sedi': epsqs_sedi,
        'epsvis': epsvis,
        'erh': erh,
        'eri': eri,
        'ers': ers,
        'fdielec': fdielec,
        'fds': fds,
        'fv_dsmax': fv_dsmax,
        'fv_dsmin': fv_dsmin,
        'gc1': gc1,
        'gc11': gc11,
        'gc12': gc12,
        'gc13': gc13,
        'gc14': gc14,
        'gc15': gc15,
        'gc2': gc2,
        'gc3': gc3,
        'gc4': gc4,
        'gc5': gc5,
        'gc6': gc6,
        'gc7': gc7,
        'gc8': gc8,
        'gg09': gg09,
        'gg11': gg11,
        'gg12': gg12,
        'gg13': gg13,
        'gg31': gg31,
        'gg32': gg32,
        'gg33': gg33,
        'gg34': gg34,
        'gg35': gg35,
        'gg36': gg36,
        'gg40': gg40,
        'gg50': gg50,
        'gh09': gh09,
        'gh11': gh11,
        'gh12': gh12,
        'gh13': gh13,
        'gh31': gh31,
        'gh32': gh32,
        'gh33': gh33,
        'gh40': gh40,
        'gh50': gh50,
        'gi11': gi11,
        'gi20': gi20,
        'gi21': gi21,
        'gi22': gi22,
        'gi31': gi31,
        'gi32': gi32,
        'gi33': gi33,
        'gi34': gi34,
        'gi35': gi35,
        'gi36': gi36,
        'gi4': gi4,
        'gi40': gi40,
        'gi6': gi6,
        'gr17': gr17,
        'gr31': gr31,
        'gr32': gr32,
        'gr33': gr33,
        'gr34': gr34,
        'gr35': gr35,
        'gr36': gr36,
        'gr37': gr37,
        'gr50': gr50,
        'grav': grav,
        'grpl_on': grpl_on,
        'gs09': gs09,
        'gs11': gs11,
        'gs12': gs12,
        'gs13': gs13,
        'gs31': gs31,
        'gs32': gs32,
        'gs33': gs33,
        'gs34': gs34,
        'gs35': gs35,
        'gs36': gs36,
        'gs40': gs40,
        'gs40_d3': gs40_d3,
        'gs50': gs50,
        'hail_on': hail_on,
        'icedep_on': icedep_on,
        'icexc9': icexc9,
        'icexi9': icexi9,
        'icexr9': icexr9,
        'icexs2': icexs2,
        'ichlf': ichlf,
        'icmg': icmg,
        'icmh': icmh,
        'icmi': icmi,
        'icmr': icmr,
        'icms': icms,
        'ideg': ideg,
        'ideh': ideh,
        'idei': idei,
        'idew': idew,
        'idms': idms,
        'igc1': igc1,
        'igc5': igc5,
        'igg31': igg31,
        'igg34': igg34,
        'igg99': igg99,
        'igh31': igh31,
        'igh34': igh34,
        'igh99': igh99,
        'igi31': igi31,
        'igr31': igr31,
        'igr34': igr34,
        'igs20': igs20,
        'igs20_d3': igs20_d3,
        'igs31': igs31,
        'igs34': igs34,
        'igs40': igs40,
        'ilammin1': ilammin1,
        'ilammin2': ilammin2,
        'imgo': imgo,
        'imso': imso,
        'imuc': imuc,
        'kapa': kapa,
        'karman': karman,
        'kboltz': kboltz,
        'kk1': kk1,
        'kk2': kk2,
        'knams': knams,
        'lama0': lama0,
        'lamdas_min': lamdas_min,
        'lcp': lcp,
        'lfp': lfp,
        'lsp': lsp,
        'maxvis': maxvis,
        'mgo': mgo,
        'minzet': minzet,
        'mio': mio,
        'mso': mso,
        'muc': muc,
        'n_c_sm': n_c_sm,
        'ngh_crit': ngh_crit,
        'ni_max': ni_max,
        'no_g_sm': no_g_sm,
        'no_h_sm': no_h_sm,
        'no_r_sm': no_r_sm,
        'no_s_max': no_s_max,
        'nti0': nti0,
        'omega': omega,
        'outfreq': outfreq,
        'p0': p0,
        'pi': pi,
        'pi2': pi2,
        'piov4': piov4,
        'piov6': piov6,
        'primicenucl': primicenucl,
        'ra': ra,
        'rainaccr_on': rainaccr_on,
        'rauw': rauw,
        'rayt': rayt,
        'rfact_fvfm': rfact_fvfm,
        'rgasd': rgasd,
        'rgasv': rgasv,
        'rh': rh,
        'ric': ric,
        'satw_peak': satw_peak,
        'sigcthrs': sigcthrs,
        'sixth': sixth,
        'snowspherical': snowspherical,
        'stefan': stefan,
        'stlo': stlo,
        't0': t0,
        'tc_fzrh': tc_fzrh,
        'tcdk': tcdk,
        'tgl': tgl,
        'thrd': thrd,
        'trpl': trpl,
        'vgmax': vgmax,
        'vhmax': vhmax,
        'vimax': vimax,
        'vrmax': vrmax,
        'vsmax': vsmax,
        'w_cngh': w_cngh,
        'xdisp': xdisp,
        'zfact': zfact,
        'zmax_sedi': zmax_sedi,
    }
