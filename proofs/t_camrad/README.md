The c28 CAM radiation run has a nighttime cold T2 bias, accompanied by missing
downwelling longwave radiation. Original WRF `check_a_mundo.F:1802–1820` disables
`icloud_bl` outside MYNN. The CAM coupler omitted the PBL cloud selection already
implemented for RRTMG, so its default MYNN merge replaced resolved clouds with
unused `CLDFRA_BL` after the first step.

The fix passes the existing static WRF policy through CAM's land-flux and heating
consumers. The spectral solvers, absorptivity cadence, carry schema and release
defaults remain unchanged. `source_gate.py` checks the three CAM dispatch hunks
and exact physics-tree/original-cloud-oracle identity. The source deletion mutant
must fail with a numerical assertion, and the healthy caller tests must pass.

`retained_attribution.py` diagnoses the original 25 c28 frames. Interior statistics
exclude five boundary cells and are attribution only; the final D6 gate remains
the frozen manifest. History GLW/SWDOWN are held CAM land fluxes; other history
fluxes may be fallback RRTMG diagnostics and cannot prove CAM parity.

`operand_probe.py` records CPU timing and allocation for the changed operand
builder on real c28 initial inputs (clear at initialization). GPU performance,
E41 and the complete 24-hour D6 rerun belong to the manager's final-candidate arm.
No forecast qualification follows from these component tests alone.
