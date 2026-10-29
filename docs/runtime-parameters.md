# Runtime parameter families: release status

The diffusion/damping parameter-override family is **not shipped in v0.4.4**.
MYNN parameter overrides ship in v0.4.4 as **experimental**. They passed
component and short two-domain GPU root checks; three-nest D6, station
comparisons and 24–72-hour drift are not qualified. The existing forecast without these overrides is unaffected.

Other families have their own scope and evidence: see
[Thompson parameters](thompson-parameters.md) and
[Noah-MP soil, canopy and snow parameters](surface_params.md#noah-mp-soil-canopy-and-snow-parameters-research-candidate).
The shared parameter-file option does not establish support for every family.

## Surface controls

Continue using `gpuwrf surface-params expand` and `gpuwrf run --surface-controls
CONTROLS.npz`. The existing Noah-MP controls are `z0mvt`, `eg`, `vcmx25`, `mp`,
`albsat`, `albdry`, `rhol`, `rhos`, `laim`, `saim` and `shdmax`.

`ctopo` changes the already-traced topographic drag field for YSU/Shin-Hong.
Checkpoint authentication now includes the actual bound CTOPO array, together
with the other surface controls. A normal resume requires the same values.
Older topographic checkpoints without that authentication are refused. The
restrictions on two-way feedback and batched host controls remain in place.

## Known limitations / planned

| Family | v0.4.4 status |
|---|---|
| Diffusion and damping overrides (`diffusion_parameters`, including `c_s`) | Not shipped; refused after the GPU qualification failed; may come in a later release after GPU qualification; no date or commitment |
| MYNN coefficient overrides (`mynn_parameters`) | Experimental in v0.4.4; component/short two-domain GPU root PASS only |
| Noah-MP W2 soil, canopy and snow overrides (`smcmax`, `dksat`, `bexp`, `psisat`, `hvt`, `hvb`, `z0sno`, `snow_emis`, `rsmin`) | Not shipped; all nine remain refused after the GPU qualification failed. CPU follow-up is in progress; GPU qualification is still required. |

The deferred diffusion/damping family includes `c_s`, `diff_6th_factor`,
`dampcoef`, `zdamp`, `w_alpha`, `w_beta` and `w_crit_cfl` as parameter-file
operands. These overrides are refused in v0.4.4. They may come in a later release after
GPU qualification; no date or commitment. The optional path failed both the compiled-kernel stack limit and the comparison of
an override file containing WRF defaults with the program without that file.
These results concern the optional override path; they do not show a change
to the shipped default forecast. Existing static namelist support is unchanged.

## MYNN qualification scope and cost

The CLI help's `GPU qualification pending` means full coupled forecast
qualification, including three-nest runs; the component GPU check has passed.

The five MYNN coefficients are `sqfac`, `alp1`, `cphm_unst`, `am_st` and
`entrainment`. The actual two-domain Canary Islands GPU root replays on the
frozen source passed the default-value byte check, the four selected programs'
compiled-resource checks, and exact paired profiling labels. The original-WRF
component/oracle evidence carries at its original scope. Changed values loaded
the same ON keys/HLO and created no new native blobs; enabling a family or
changing its selected names can still require a first cold program.

Twelve unprofiled samples measured 42.742 ms OFF and 43.087 ms ON (+0.81%).
No A/A noise floor was measured, so this is a short-root cost observation,
not an end-to-end forecast benchmark or a precise subpercent slowdown claim.
It does not qualify three-nest D6, station skill, 24–72-hour drift, release
real-world gates or full-process peak VRAM.

See the [recorded qualification scope](../proofs/w44_docs/receipts/DOC44_MYNN_SCOPE.json)
for the pinned GPU evidence. Normal resume still requires the same parameter
values; [checkpoint branches](checkpoint-branching.md) have their own admission
and forecast qualification limits.

The development evidence is recorded in
[PARAMS.md](../proofs/w44_params/PARAMS.md). Earlier CPU-only results there
must not be read as release qualification for these overrides.
