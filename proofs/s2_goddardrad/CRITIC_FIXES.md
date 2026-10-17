# rv-s2-goddardrad P1: selected land radiation surfaces

The accepted RUC3 caller was omitted at cadence refresh: initialization passed
`ruc_radiation_surface(ruc_land)`, while Goddard refresh passed only the
Noah-MP/lake view and silently selected MODIS fallback values. The fix forwards
the existing `RucRadiationSurface` exactly as the RRTMG refresh does. WRF's
`module_radiation_driver.F:1900–1902,2499–2501` passes selected ALBEDO/EMISS.

The regression builds actual RUC bundles using the critic R06 path, admits the
pair through CLI preflight and the runtime resolver, and captures the production
refresh operands with matching atmospheric dimensions. It checks array identity,
the common surface-property consumer, and the held-cadence behavior. These are
caller-wiring checks; they are not a new physical forecast qualification.

| Surface | Audit result for Goddard |
|---|---|
| Noah-MP4 | Existing held land ALBEDO/EMISS forwarded unchanged. |
| RUC3 | Existing two-array RUC view now forwarded at refresh, matching initialization and RRTMG. |
| SLUCM1 + Noah-MP4 | `urban_slucm_hook` merges its albedo into the Noah-MP land state; the same resident land view is forwarded. |
| Lake + Noah-MP4 | Overlay reaches the view, but `_surface_radiation_properties` ignores it on water-marked lake cells. Goddard+lake is explicitly refused pending qualification of that shared seam. |
| Slab1, Noah-classic2, Pleim-Xiu7 | Current land carries do not expose the required held ALBEDO/EMISS view here; Goddard pairings are refused before execution. |
| BEP2/BEM3 | Existing operational contract requires RRTMG4/4 and extra urban radiation operands; Goddard also refuses this pairing in backend-dark admission. |
| No-LSM0 | Existing common surface-table control path retained. |

Both CLI preflight and the direct operational resolver use the same backend-dark
Goddard admission. The restrictions apply only to option 5. The audit does not
qualify the shared RRTMG fallback behavior for other land schemes.

The RUC view contains references to two existing resident arrays: no additional
carry fields, array copies, host transfers, synchronization or device launches
are introduced by constructing it. The static non-RUC branch preserves existing
Noah-MP arithmetic; paired Goddard column code and runtime assets are unchanged.
Original numerical/component evidence is retained with source identity; changed
caller gates are rerun. GPU performance/E41/full D6 remain pending.

The original qualification receipt and critic verdict remain historical evidence.
The separate correction receipt is `critic_fix_receipt.json`; it records the
new experiments, genuine deletion mutant and source/trace reuse evidence.
