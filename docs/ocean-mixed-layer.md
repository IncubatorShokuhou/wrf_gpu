# Ocean mixed layer and hurricane surface fluxes

**Status in v0.4.0: Validated 24 h.** Passed a 24-hour comparison with the original CPU WRF in the v0.4.0 release check, in one tested single-domain configuration. Other combinations and nested domains are not validated. In the tested configuration with `isftcflx=2` (YSU PBL, revised MM5 surface layer), the compiled GPU code exceeds the project's code-quality limits for speed and robustness; it runs correctly, but its speed has not been measured. The ocean model with `isftcflx=1` passed those limits.

`sf_ocean_physics=1` selects WRF 4.7.1 OML1D. It advances seven REAL surface
fields: TML/T0ML (K), HML/H0ML (m), HUML/HVML (depth-integrated currents,
m²/s), and TMOML (K). Only XLAND > 1.5 points evolve. Current surface heat,
latent and absorbed shortwave/longwave fluxes drive the model after the land
surface solve, with step-entry mass-grid winds and the surface timestep.

Both forecast pipelines bind `oml_hml0` (default 50 m), `oml_gamma` (0.14 K/m),
and `oml_relaxation_time` (seconds; <=0 disables relaxation). Positive HML0
initializes a constant depth and zero currents; zero uses wrfinput H0ML and
zero currents. Negative HML0 requires H0ML, HUML, HVML and TMOML in wrfinput.
The pristine coastal TMOML sentinel (200 < TMOML <= 201 K) becomes initial SST.
Restarts preserve every anchor and current, and reject missing or changed
active schemas/configuration. History exports the seven held Registry fields.

The current operational OML path requires Noah-MP (`sf_surface_physics=4`) for
surface emissivity and held radiation. `sst_update=1` with prognostic OML is
refused because the combination is not validated. Running the lake model at the
same time is refused until WRF's ocean-before-lake ordering is validated. `sf_ocean_physics=2` (PWP 3-D) is refused.
Gamma must be positive and all controls/initial fields finite.

`isftcflx=1/2` selects the original WRF hurricane branches for revised MM5
(`sf_sfclay_physics=1`), MYNN (`5`), and old MM5 (`91`). All use Davis momentum
roughness. MM5 option 1 fixes scalar roughness; option 2 uses its Brutsaert
resistances. MYNN option 1 retains Fairall scalar roughness; option 2 uses
MYNN's distinct Garratt relation. MYNN also carries USTM separately for
dissipative heating. Options 3/4 and other surface-layer combinations are refused.
Hurricane fluxes with RUC, SLUCM, BEP/BEM or epsilon-PBL are refused at both
namelist preflight and the operational API. These paths respectively repeat
an unvalidated surface solve, lack required CHS, or replace the supplied
hurricane diagnostics. Their combined surface interfaces are not validated.
NetCDF restart validates stored ocean, USTM and required LH precision and
horizontal grid shape before JAX conversion, including when x64 is disabled.

The code is also checked on the CPU against the original WRF routines.
Reference sources and the active-column census are in `proofs/s2_oml/fixtures`;
generation copies unmodified original WRF sources into a private build directory.
The default zero options retain the released program. New paths carry REAL32,
add seven OML and one USTM carry arrays (32 bytes/cell), up to seven traced initialization arrays retained by the namelist, and six held history arrays if history was previously disabled. Current surface_driver LH is held exactly; it is never reconstructed through QFX/rho conversions. They use elementwise
operations without host transfers or vertical iterations. GPU launch count
and compile cost have not been measured.
