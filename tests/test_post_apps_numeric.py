"""Independent published-formula / pvlib / CFFDRS application references."""
from pathlib import Path
import csv
import json
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from gpuwrf.post.apps.wind import (
    interpolate_profile, rotor_weights, rotor_wind, wind_power, read_power_curve,
    sampled_energy_wh,
)
from gpuwrf.post.apps.solar import erbs, plane_of_array, pvwatts_dc, solar_position
from gpuwrf.post.apps.hazards import (
    supercooled_liquid_water, frost_flags, visibility, hot_dry_windy,
)
from gpuwrf.post.apps.fire import daily_fwi


def test_rews_independent_disk_quadrature():
    # Independent integration of disk chords, not implementation's antiderivative.
    heights = np.array([50., 100., 150.])
    edges = [-50, -25, 25, 50]
    nodes, coefficients = np.polynomial.legendre.leggauss(32)
    areas = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a, b = math.asin(lo/50), math.asin(hi/50)
        theta = a + (b-a)*(nodes+1)/2
        areas.append((b-a)/2 * np.sum(coefficients * 5000 * np.cos(theta)**2))
    weights = np.array(areas) / (math.pi * 2500)
    assert_allclose(rotor_weights(heights, 100, 100), weights, rtol=1e-12)
    z = np.array([[10, 50, 100, 150, 200.]])
    u = z * .1
    hub, rews = rotor_wind(z, u, u*0, samples=3)
    assert hub[0] == 10
    assert_allclose(rews, [(weights @ np.array([5., 10., 15.])**3)**(1/3)], rtol=1e-12)


def test_veer_projection_and_rotation_invariance():
    z = np.array([[50., 100., 150.]])
    u = np.array([[10., 10., 0.]])
    v = np.array([[0., 0., 10.]])
    h0, r0 = rotor_wind(z, u, v, samples=3)
    h1, r1 = rotor_wind(z, -v, u, samples=3)
    weights = rotor_weights(z[0], 100, 100)
    assert_allclose(r0, [(weights[0]+weights[1])**(1/3)*10], rtol=1e-12)
    assert_allclose((h0, r0), (h1, r1), rtol=1e-12)
    assert_allclose(rotor_wind(z, u, v, samples=3, veer=False)[1], 10)


def test_veer_reversing_flow_retains_signed_cubic_flux():
    z = np.array([[50.,100.,150.]])
    u = np.array([[-10.,1.,-10.]])
    # Middle half-radius strip area from circle geometry, independent formula.
    middle = (math.sqrt(3)/2 + math.pi/3) / math.pi
    expected = np.cbrt(middle - 1000*(1-middle))
    assert_allclose(rotor_wind(z,u,u*0,samples=3)[1],expected,rtol=1e-14)
    assert expected < 0  # Positive-part clamping would be a different product.


def test_wind_curve_density_cutout_and_csv(tmp_path):
    path = tmp_path / "power.csv"
    path.write_text("speed_ms,power_w\n0,0\n10,1000\n20,1000\n25,0\n")
    speed, power = read_power_curve(path)
    assert_allclose(wind_power([5, 5, 30], [1.225, 1.225*8, 1.225], speed, power), [500, 1000, 0])
    path.write_text("speed_ms,power_w\n10,1000\n0,0\n")
    with pytest.raises(ValueError, match="increase"):
        read_power_curve(path)


def test_energy_irregular_sampling():
    assert_allclose(sampled_energy_wh([0., 1000., 1000.], [0, 3600, 10800]), [0, 500, 2500])
    with pytest.raises(ValueError, match="timestamps"):
        sampled_energy_wh([1, 2], [0, 0])


@pytest.mark.parametrize("ghi,zenith,day", [
    ([0, 50, 400, 800], [0, 30, 45, 60], 180),
    ([0, 5, 30, 150], [180, 90, 88, 87], 1),
    ([300, 900, 1200, 1600], [10, 30, 20, 5], 366),
])
def test_erbs_against_pvlib(ghi, zenith, day):
    pvlib = pytest.importorskip("pvlib")
    ref = pvlib.irradiance.erbs(np.array(ghi), np.array(zenith), day)
    dni, dhi = erbs(ghi, zenith, day)
    assert_allclose(dni, ref["dni"], rtol=2e-13, atol=1e-10)
    assert_allclose(dhi, ref["dhi"], rtol=2e-13, atol=1e-10)


def test_poa_pvwatts_against_pvlib():
    pvlib = pytest.importorskip("pvlib")
    g, n, d, z, az = [np.array(a) for a in ([800, 500, 0], [700, 400, 0], [100, 200, 0], [30, 70, 100], [180, 90, 300])]
    poa = plane_of_array(g, n, d, z, az, tilt=35, azimuth=170, albedo=.25)
    ref = pvlib.irradiance.get_total_irradiance(35, 170, z, az, n, g, d, model="isotropic", albedo=.25)
    assert_allclose(poa, ref["poa_global"], rtol=2e-14)
    temp = np.array([25, 50, 0])
    assert_allclose(pvwatts_dc(poa, temp, dc_rating_w=5000), pvlib.pvsystem.pvwatts_dc(poa, temp, 5000, -.004), rtol=1e-14)
    assert pvwatts_dc(1000, 25) == 1000  # NREL worked STC point.
    assert pvwatts_dc(1000, 50) == 900


def test_spa_nrel_published_point():
    pytest.importorskip("pvlib")
    # NREL SPA report example: Oct17 2003 12:30:30 MST, 39.742476N, -105.1786E.
    # Pressure/temperature defaults differ from the report, so use true zenith
    # (50.127954 deg) and azimuth (194.340241 deg), independent of refraction.
    z, apparent, az = solar_position(["2003-10-17T19:30:30"], [[39.742476]], [[-105.1786]], altitude=1830.14)
    assert_allclose(z, 50.127954, atol=0.0001)
    assert_allclose(az, 194.340241, atol=0.0001)
    assert apparent[0,0,0] < z[0,0,0]


def test_icing_and_frost_worked_screen():
    # Ideal-gas dry density exactly 1 kg/m3 when qv=0, p=Rd*T.
    t = np.array([263.15, 273.15, 233.15, 283.15])
    slw = supercooled_liquid_water(t, 287.05*t, 0, .0001, .0002)
    assert_allclose(slw, [.3, 0, 0, 0], rtol=1e-14)
    air, skin = frost_flags([273.15, 273.16, 270], [274, 271, 273.15])
    assert_allclose(air, [1,0,1]); assert_allclose(skin, [0,1,1])


def test_calvis_independent_scalar_formula_all_hydrometeors():
    # Literal UPP CALVIS R scalar calculation with independently evaluated masses.
    inputs = (270., 95000., .004, .0002, .0003, .0001, .0004)
    t,p,qv,qc,qr,qi,qs = inputs
    rho = p/(287.04*t*(1+.608*qv))
    volume = (1+qv)/rho + (qc+qr)/1000 + (qi+qs)/917
    extinction = sum(coef*(1000*q/volume)**exponent for q,coef,exponent in
                     [(qc,144.7,.88),(qr,2.24,.75),(qi,327.8,1),(qs,10.36,.7776)]) + 1e-10
    expected = 1000*min(24.135, -math.log(.02)/extinction)
    assert_allclose(visibility(*inputs), expected, rtol=1e-14)
    assert visibility(290, 100000, .001, 0,0,0,0) == 24135


def test_calvis_raw_negative_condensate_matches_upp_concentration_max():
    # Original CALVIS R uses raw hydrometeors in volume, then MAX(0,C).
    # A negative cloud value must not remove positive rain extinction.
    t,p,qv,qc,qr,qi,qs = 270.,95000.,.004,-.0002,.0003,.0001,.0004
    rho = p/(287.04*t*(1+.608*qv))
    volume = (1+qv)/rho + (qc+qr)/1000 + (qi+qs)/917
    extinction = sum(coef*max(0,1000*q/volume)**exponent for q,coef,exponent in
                     [(qc,144.7,.88),(qr,2.24,.75),(qi,327.8,1),(qs,10.36,.7776)]) + 1e-10
    assert_allclose(visibility(t,p,qv,qc,qr,qi,qs),1000*min(24.135,-math.log(.02)/extinction),rtol=1e-14)


def test_hdw_independent_adiabatic_example():
    # Level2 has higher wind; adjusted level3 has greater VPD. Separate maxima.
    t = np.array([[300., 299., 298., 270.]])
    p = np.array([[100000., 98000., 95000., 80000.]])
    q = np.array([[.01, .008, .006, .001]])
    u = np.array([[3., 10., 5., 50.]])
    z = np.array([[25., 200., 450., 1500.]])
    expected_vpd = []
    for ti,pi,qi in zip(t[0,:3],p[0,:3],q[0,:3]):
        tc = ti*(100000/pi)**(287.05/1004)-273.15
        expected_vpd.append((611.2*math.exp(17.67*tc/(tc+243.5))-100000*qi/(.622+qi))/100)
    assert_allclose(hot_dry_windy(t,p,q,u,u*0,z), 10*max(expected_vpd), rtol=1e-14)


def test_hdw_upper_stratosphere_is_not_adjusted_into_surface_layer():
    # A true200K/100hPa stratosphere reaches~386K at1000hPa; it is outside500m.
    # It must not trip the surface thermodynamic bounds or affect HDW.
    expected = 5 * 6.112 * math.exp(17.67*(300-273.15)/(300-29.65))
    assert_allclose(hot_dry_windy([[300.,200.]],[[100000.,10000.]],[[0.,0.]],
                                 [[5.,100.]],[[0.,0.]],[[25.,16000.]]),expected,rtol=1e-14)


def test_fwi_published_1985_sequential_table():
    path = Path(__file__).parent / "fixtures/post_apps/fwi_1985.csv"
    state = dict(ffmc_previous=85, dmc_previous=6, dc_previous=15)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            result = daily_fwi(*[float(row[k]) for k in ("TEMP", "RH", "WS", "PREC")],
                               latitude=float(row["LAT"]), month=int(row["MON"]), **state)
            for name, actual in result.items():
                tolerance = .0051 if name == "dsr" else .051
                assert float(actual) == pytest.approx(float(row[name.upper()]), abs=tolerance), (row, name, actual)
            state = {k + "_previous": result[k] for k in ("ffmc", "dmc", "dc")}


def test_fwi_full_precision_independent_cffdrs_reference():
    fixture = json.loads((Path(__file__).parent / "fixtures/post_apps/fwi_reference.json").read_text())
    for row in fixture["cases"]:
        actual = daily_fwi(**row["inputs"])
        for name, expected in row["expected"].items():
            assert_allclose(actual[name], expected, rtol=2e-12, atol=2e-12,
                            err_msg=f"{row['inputs']} / {name}")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("call", [
    lambda bad: wind_power(bad,1.225,[0,10],[0,1000]),
    lambda bad: wind_power(5,bad,[0,10],[0,1000]),
    lambda bad: pvwatts_dc(bad,25),
    lambda bad: pvwatts_dc(1000,bad),
    lambda bad: erbs(500,bad,1),
    lambda bad: plane_of_array(500,300,100,30,180,albedo=bad),
    lambda bad: frost_flags(bad,273),
    lambda bad: supercooled_liquid_water(263,90000,.01,bad,0),
    lambda bad: visibility(280,100000,.01,0,0,bad,0),
    lambda bad: daily_fwi(20,40,10,0,latitude=50,month=7,ffmc_previous=bad,dmc_previous=6,dc_previous=15),
])
def test_nonfinite_rejected_before_thresholds(bad, call):
    with pytest.raises(ValueError, match="finite"):
        call(bad)


def test_outside_profiles_and_bad_month_refused():
    with pytest.raises(ValueError, match="outside"):
        interpolate_profile([[20,100]], [[1,2]], [10])
    with pytest.raises(ValueError, match="integer"):
        daily_fwi(20,40,10,0,latitude=50,month=7.5,ffmc_previous=85,dmc_previous=6,dc_previous=15)
