# P3 and Jensen ice history diagnostics

P3 (`mp_physics=50/51/53`) retains `D_ICE`, `RHO_ICE` and `V_ICE` from its existing
microphysics call. P3 two-category (`52`) also retains the `2` fields. Jensen
(`55`) retains those fields plus `PHI_ICE` for all three habits (suffixes `2` and
`3`). These are the original scheme outputs at the end of the microphysics
call, held until the next call and copied into history. They are not advected
and are not reconstructed from the later model state.

| Field | Units | Original producer |
| --- | --- | --- |
| D_ICE* | m | P3 `diag_di=f1pr15`; Jensen maximum of the two habit-axis gamma-ratio mean sizes |
| RHO_ICE* | kg m-3 | P3 `diag_rhoi=f1pr16`; Jensen `rhopo1d=rhobar` with the original size/ice conditions |
| V_ICE* | m s-1 | P3 `diag_vmi=f1pr02*rhofaci`; Jensen `vmi1d=vtrmi1` under the original conditions |
| PHI_ICE* | dimensionless | Jensen `phii1d=cni/ani*gamma_tab(gi)*i_gammnu` under the original conditions |

The references are pristine WRF `module_mp_p3.F:4856–4859`,
`module_mp_jensen_ishmael.F:2611–2632`, and `Registry.EM_COMMON:1599–1610`.
Only the categories supplied by the selected scheme have State arrays. Initial
values follow WRF's zero initialization; every subsequent value comes from the
scheme. P3 has no aspect-ratio producer and does not emit `PHI_ICE*`.

These mass-grid SoA arrays retain the existing State layout and column adapter.
They have no scalar transport or boundary exchange of their own; the schemes
retain their current single-domain restriction. All arrays use WRF REAL32. Checkpoint and NetCDF restart readers refuse active
P3/Jensen payloads missing these held arrays; restoring them as zeros would
lose the original diagnostic time level. Default Thompson configurations have
no added arrays. Native kernels retain their one-program-per-column design;
the additional output/carry is 12 bytes per cell for one-category P3, 24 for
two-category P3, and 48 for Jensen. GPU runtime, spills and forecast fidelity
remain separate qualification gates. This retention does not resolve the
disclosed P3 native-column residuals, the MP52 compiler blocker, or Jensen's
disclosed compiled deviations.

The expanded Jensen CPU native output check preserves all prior operational
outputs byte-for-byte. Its 17 original columns contain 165 diagnostic values
outside the unchanged original input-ULP/FMA enclosure (maximum relative excess
1.43e-5). These remain disclosed strict-enclosure failures. Independent review
accepted the small diagnostic-only deviation under the standing FMA policy;
no bounds or diagnostic values are adjusted to turn them into passes. The
[review acceptance](../../proofs/s2_integrate/batch16/rv-t-icediag-DELTA_VERDICT_583.md)
retains the separate GPU, E41 and full-forecast qualification gates.
