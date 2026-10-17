"""WRF GFDL radiation scalar support, preserving native REAL operation order.

This module does not admit ra=99. The radiation solver and its call/carry
adapter must be qualified separately before operational admission.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import lax


def gaussian_area(xsd):
    """module_ra_gfdleta.F:GAUSIN; used once to build AXSD, not per step."""
    pi = jnp.asarray(np.float32(np.arccos(np.float32(-1.))), xsd.dtype)
    rsqr = 1. / jnp.sqrt(2. * pi)
    a1 = xsd * rsqr
    a2 = jnp.exp(-.5 * xsd ** 2)
    one = jnp.ones_like(xsd)

    def advance(carry):
        rn, b1, _, total = carry
        rn = rn + 1.
        b2 = xsd ** 2 / (2. * rn - 1.)
        b3 = b1 * b2
        return rn, b3, b2, total + b3

    _, _, _, total = lax.while_loop(lambda s: s[2] > 1.e-3, advance,
                                   (one, one, one, one))
    return a1 * a2 * total


def zenith(times, longitude, latitude, *, year, month, day, start_hour):
    """Native ZENITH. Coordinates are radians; time is forecast seconds.

    Date integers describe the initial date and are static for this call.
    WRF's divisible-by-four leap rule and REAL Julian-date rounding are kept.
    Return (nonnegative cosine, day-of-year, hour).
    """
    dtype = times.dtype
    real = lambda x: jnp.asarray(x, dtype)
    f = np.float32
    pi = real(f(3.1415926))
    pi2 = real(f(f(2.) * f(3.1415926)))
    degrad = real(f(f(3.1415926) / f(180.)))
    obliq = real(f(f(23.440) * f(f(3.1415926) / f(180.))))
    months = [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30,
              31, 31, 30, 31, 30, 31]
    elapsed = real(sum(months[:month-1]))
    elapsed = elapsed + real(day-1) + (real(start_hour) + times / 3600.) / 24.
    # WRF stores DAY and DATJUL as separate REAL assignments. Reassociation
    # across the large Julian-date sum/subtraction removes that rounding:
    # native stage capture FG22 measured DATJUL .25 day / DIFJD .125 day
    # discrepancies, far larger than WRF's FMA spread. Preserve those two
    # exact REAL boundaries; nextafter(x,x) is an identity on finite inputs.
    elapsed = jnp.nextafter(elapsed,elapsed)
    dayi = jnp.trunc(elapsed) + 1.
    hour = (elapsed - dayi + 1.) * 24.
    yfctr = real(2000-year)
    difference = year-2000
    # Fortran integer division truncates toward zero, including negative years.
    addday = int(difference / 4) if difference < 0 else (difference+3)//4
    start = real(2451545.) + real(difference)*365. + real(addday) - .5
    datjul = start + elapsed
    datjul = jnp.nextafter(datjul,datjul)
    difjd = datjul - 2451545.
    slonm = (280.460 + .9856474*difjd)*degrad + yfctr*pi2
    anomaly = (357.528 + .9856003*difjd)*degrad
    slon = slonm + (1.915*jnp.sin(anomaly) + .020*jnp.sin(2.*anomaly))*degrad
    slon = jnp.where(slon > pi2, slon-pi2, slon)
    dec = jnp.arcsin(jnp.sin(slon)*jnp.sin(obliq))
    ra = jnp.arccos(jnp.cos(slon)/jnp.cos(dec))
    ra = jnp.where(slon > pi, pi2-ra, ra)
    datj0 = start + dayi - 1.
    tu = (datj0-2451545.)/36525.
    stim0 = 24110.54841 + tu*(8640184.812866 + .093104*tu - .0000062*tu*tu)
    sidtim = stim0/3600. + yfctr*24. + 1.00273791*hour
    sidtim = sidtim*15.*degrad
    sidtim = jnp.where(sidtim < 0., sidtim+pi2, sidtim)
    sidtim = jnp.where(sidtim > pi2, sidtim-pi2, sidtim)
    hrlcl = (sidtim-ra) + longitude + pi2
    sinalt = jnp.sin(dec)*jnp.sin(latitude) + jnp.cos(dec)*jnp.cos(hrlcl)*jnp.cos(latitude)
    czen = jnp.maximum(sinalt, 0.)
    days_in_year = 366. if year % 4 == 0 else 365.
    dayi = jnp.where(dayi > days_in_year, dayi-days_in_year, dayi)
    return czen, dayi, hour


def saturation_pressure(t):
    """FPVS_new alternative in the original module, pressure in Pa.

    The active FERRIER_GFDL path uses Ferrier's tabulated FPVS instead; callers
    must not silently substitute this alternative for that lookup.
    """
    x = jnp.maximum(-80., t-273.16)
    liquid = (.611583699e3, .444606896e2, .143177157e1, .264224321e-1,
              .299291081e-3, .203154182e-5, .702620698e-8,
              .379534310e-11, -.321582393e-13)
    ice = (.609868993e3, .499320233e2, .184672631e1, .402737184e-1,
           .565392987e-3, .521693933e-5, .307839583e-7,
           .105785160e-9, .161444444e-12)

    def polynomial(coefficients):
        value = jnp.asarray(coefficients[-1], t.dtype)
        for coefficient in reversed(coefficients[:-1]):
            value = coefficient + x*value
        return value

    return jnp.where(t >= 273.16, polynomial(liquid), polynomial(ice))


def zenith_dynamic(times,longitude,latitude,idat,start_hour):
    """ZENITH with explicit dynamic native (month,day,year) integer operands.

    Month-prefix additions are exact integers before conversion to REAL.
    The calendar rule and the two measured REAL Julian boundaries are the
    same as the qualified static-date function above; no host date lookup.
    """
    dtype=times.dtype
    real=lambda x:jnp.asarray(x,dtype)
    f=np.float32
    pi=real(f(3.1415926));pi2=real(f(f(2.)*f(3.1415926)))
    degrad=real(f(f(3.1415926)/f(180.)))
    obliq=real(f(f(23.440)*f(f(3.1415926)/f(180.))))
    month,day,year=idat
    leap=(year%jnp.int32(4))==0
    months=jnp.asarray([31,28,31,30,31,30,31,31,30,31,30,31],jnp.int32)
    months=months.at[1].set(jnp.where(leap,jnp.int32(29),jnp.int32(28)))
    prefix=jnp.sum(jnp.where(jnp.arange(12)<month-1,months,jnp.zeros_like(months)))
    elapsed=real(prefix)+real(day-1)+(real(start_hour)+times/3600.)/24.
    elapsed=jnp.nextafter(elapsed,elapsed)
    dayi=jnp.trunc(elapsed)+1.;hour=(elapsed-dayi+1.)*24.
    yfctr=real(2000.)-real(year)
    difference=year-jnp.int32(2000)
    addday=jnp.where(difference<0,lax.div(difference,jnp.int32(4)),
                    lax.div(difference+jnp.int32(3),jnp.int32(4)))
    start=real(2451545.)+real(difference)*365.+real(addday)-.5
    datjul=start+elapsed;datjul=jnp.nextafter(datjul,datjul)
    difjd=datjul-2451545.
    slonm=(280.460+.9856474*difjd)*degrad+yfctr*pi2
    anomaly=(357.528+.9856003*difjd)*degrad
    slon=slonm+(1.915*jnp.sin(anomaly)+.020*jnp.sin(2.*anomaly))*degrad
    slon=jnp.where(slon>pi2,slon-pi2,slon)
    dec=jnp.arcsin(jnp.sin(slon)*jnp.sin(obliq))
    ra=jnp.arccos(jnp.cos(slon)/jnp.cos(dec));ra=jnp.where(slon>pi,pi2-ra,ra)
    datj0=start+dayi-1.;tu=(datj0-2451545.)/36525.
    stim0=24110.54841+tu*(8640184.812866+.093104*tu-.0000062*tu*tu)
    sidtim=stim0/3600.+yfctr*24.+1.00273791*hour
    sidtim=sidtim*15.*degrad
    sidtim=jnp.where(sidtim<0.,sidtim+pi2,sidtim)
    sidtim=jnp.where(sidtim>pi2,sidtim-pi2,sidtim)
    hrlcl=(sidtim-ra)+longitude+pi2
    sine=jnp.sin(dec)*jnp.sin(latitude)+jnp.cos(dec)*jnp.cos(hrlcl)*jnp.cos(latitude)
    czen=jnp.maximum(sine,0.)
    year_days=jnp.where(leap,real(366.),real(365.))
    dayi=jnp.where(dayi>year_days,dayi-year_days,dayi)
    return czen,dayi,hour


def solar_distance(hour,day,month,year):
    """Native SOLARD R1, with its REAL Julian-day rounding and Newton exit.

    Date/hour operands are INTEGER. R1 is computed once per radiation call,
    outside the column solver. No Python calendar or host callback is used.
    """
    real=lambda x:jnp.asarray(x,jnp.float32)
    integer=lambda x:jnp.asarray(x,jnp.int32)
    hour,day,month,year=map(integer,(hour,day,month,year))
    div=lambda a,b:lax.div(a,integer(b))
    shift=div(month-14,12)
    jd=day-32075+div(1461*(year+4800+shift),4)
    jd=jd+div(367*(month-2-shift*12),12)-div(3*div(year+4900+shift,100),4)
    jd=jnp.where(hour<12,jd-1,jd)
    fjd=jnp.where(hour<12,.5+.041666667*real(hour),.041666667*real(hour-12))
    fjd1=real(jd)+fjd+real(hour)/24.
    # A REAL assignment before converting to INTEGER is essential at ~2.4e6.
    fjd1=jnp.nextafter(fjd1,fjd1)
    jd=fjd1.astype(jnp.int32)
    fjd=fjd1-real(jd)
    dat=real(jd-2415020)-1.55+fjd
    t=real(jd-2415020)/36525.
    year_length=.25964134+.304e-5*t
    eccentricity=.01675104-(.418e-4+.126e-6*t)*t
    year_length=year_length+365.
    date=jnp.fmod(dat,year_length)
    pi2=real(np.float32(np.float32(2.)*np.float32(3.1415926)))
    anomaly=pi2*date/year_length

    def advance(carry):
        iteration,e,_=carry
        ep=e-(e-eccentricity*jnp.sin(e)-anomaly)/(1.-eccentricity*jnp.cos(e))
        return iteration+1,ep,jnp.abs(e-ep)

    _,e,_=lax.while_loop(lambda s:(s[0]<=10)&(s[2]>1.3e-6),advance,
                         (integer(0),real(1.),real(1.)))
    return 1.-eccentricity*jnp.cos(e)
