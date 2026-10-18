# U2.5 stochastic physics — CPU-wave qualification

The feature is implemented, wired and admitted as **CPU-oracle-qualified;
GPU/coupled-forecast qualification pending**. Defaults remain OFF. Source authority
is pristine WRF 4.7.1 `dyn_em/module_stoch.F`, its actual RK callers, Registry,
MYNN and surface-layer bodies. No original source bodies or reference cases were edited.

Evidence lives in `<USER_HOME>/wrf_gpu2_lanes/u2-stoch/`; all experiment receipts
declare CPU affinity, bounded execution, inputs and disabled disposable JAX caches.
The coupled measurement source subtree was `26c25cccb43d267831e35287e461ef180f9fd79c`.
Final integration includes main `ad14fd5ae` (GPU qualification metadata and BEP/BEM).
ST74 records the final source tree and exact stochastic arithmetic byte identity;
ST70 preserves the complete active coupled CPU program, and ST71/ST72 preserve
both complete OFF release programs. The affected composed suite passes 108 tests.

| Gate | Measured result | Receipts |
|---|---|---|
| Complete original CPU-WRF patterns, vertical 0/1 | 64 history-field comparisons; 288 exact seed words; fixed `8e-6 + 64 eps maxref` field bound | ST17, ST18 |
| Long original trajectories | Seven 1,000-step trajectories, 700 samples; 5,600 exact seed words; fixed field and spectral-power bounds | ST39, ST40 |
| RK tendencies and MYNN SPP | Native C-grid adapter versus original routines; signed land/water/snow, capped entrainment, real drizzle-cloud and height-dependent diffusion gates pass | ST26, ST29, ST42 |
| Parent forcing | Independent source-literal WRF SINT; all eight fields, Registry staggering, vertical 0/1, batched/unbatched; no child Fourier/RNG state | ST42 |
| Restart and admission | Both formats × both vertical modes, all leaves/next step exact; missing/corrupt live plans refused; all eight switches admitted with honest notes | ST58: 98 passed; final ST73: 108 passed |
| Shared consumers | All required consumer files plus NetCDF process-lock gate | ST47: 254 passed |
| Actual coupled Swiss CPU | Three steps, zero nonfinite values, eight original final fields and 48 exact seed words, seven stochastic cap counters zero | ST59 |
| Coupled guard control | One `dynamics.qc` out-of-range visit + its repair in BOTH arms; same counts and all negative-location classes. Pre-guard minima ON `-9.893e-37`, OFF `-3.742e-37` kg/kg | ST60, ST63 |
| OFF device-program identity | Full Swiss `2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a`; PROD `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b` | ST55, ST61; final ST71, ST72 |
| Archive and mutation | Clean archive imports plus 106-array original fixture; three genuine deletion/replacement assertion kills plus active observation-order mutant kill | ST52, ST62, ST54 |
| Real IC/BC tool → ensemble workflow | PROD two nests: 40 scaled and 1,554 unchanged variables; four source SHA hashes unchanged; two prepared members feed existing batch plan | ST38 |

The paired coupled arms use the same Swiss inputs, physics, 42×42×44 grid and
three 18-second steps; the control disables only stochastic switches. ST63 applies
the manager's matched-control guard policy. ST56 and final ST70 prove the active CPU graph
matches ST59 exactly (`8061672f…`). Restart/graph comparisons are mechanics and
regression evidence; original WRF outputs and source oracles supply fidelity.

Changed-artifact CPU accounting (ST48, shared machine): constant/varying vertical
carry 214,764 / 3,483,976 bytes; warm median 4.808 / 15.932 ms; compile
3.004 / 2.895 s; compiled temporaries 179,760 / 6,673,136 bytes. These are
pattern-driver CPU measurements, not GPU timings or VRAM. Actual coupled ON
CPU wall 446.8 s includes cold compile and three steps; peak process RSS 7,716,144 KiB.
ST36 lowers RNG and both changed MYNN plume variants to Triton SM120 IR with
zero floating-point64 tokens. No GPU execution or E41 qualification is claimed.

GPU timing, launch/traffic accounting, peak VRAM, E41 and full forecast D6/skill
qualification remain pending in manager-authorized windows. GNU14 RNG profile,
SPPT raw-source limits, upstream convection/LSM SPP physical no-ops, and input
perturbation assumptions are disclosed in `docs/stochastic.md` and the catalog.


## Critic admission fixes

The two required `rv-u2-stoch` fixes reject Gaussian custom `*forct` bounds
exactly as `check_a_mundo:1155-1200` does, and validate all 32 Registry INTEGER
controls before conversion. Valid custom SKEBS-only bounds remain admitted.
ST75 passes 507 affected tests; ST78 kills both guard-removal mutants through
real public-admission assertion failures. ST77 proves 11 numerical files remain
byte-identical, both configuration classes retain their AST, and all 58 valid
configuration/domain cases (including the original vertical-0/1 reference
profiles) retain exactly the same resolved values. ST79 seals the final main
rebase to the tested source. Prior numerical/default-program receipts are reused;
these host-only changes add no timestep launches, device allocations, traffic,
synchronization or compiled-body work. No forecast was repeated.

## GPU qualification (lane w41-stoch, 2026-10-17)

Swiss d01 24 h GPU arms (SKEBS, SPPT, SPP, all switches, OFF control) against original serial CPU-WRF
runs of the identical namelists and seeds, frozen D6 + every-frame inventory + all stochastic history
fields and seed arrays + E41: `proofs/w41_stoch/GPU_QUALIFICATION.md`. The same lane replaced the serial
RNG kernel by a bitwise-identical parallel stream and fixed the SPP FIELD_* history inventory for cu0.
