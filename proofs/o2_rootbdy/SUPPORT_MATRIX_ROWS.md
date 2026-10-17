# o2-rootbdy — proposed support-matrix rows for v0.4.0 (replace the v0.3.4 "Refused: mp_physics=24/26/28/40" row)

## Experimental (CPU-qualified, GPU not run)

| Key = code | Scheme | Status | Scheme-specific gaps |
|---|---|---|---|
| `mp_physics=24` | WSM7 | Experimental, CPU-qualified (GPU not run), single domains | Column kernel runs as an fp64 island inside the fp32 model (slower than native fp32); hail `qh` uses WRF's flow-dependent root boundary (inflow 0 / outflow copy); nested runs refused |
| `mp_physics=28` | Aerosol-aware Thompson | Experimental, CPU-qualified (GPU not run), single domains | `use_aero_icbc=.false.` only (climatological self-init); aerosol numbers and droplet number use WRF's flow-dependent root boundary; droplet number now transported like in WRF; nested runs refused |
| `mp_physics=40` | Morrison-aerosol (`aercu_opt=0`) | Experimental, CPU-qualified (GPU not run), single domains | `aercu_opt>0` refused (needs MSKF); RRTMG does not receive the Morrison effective radii; Ns/Ng/Nc use WRF's flow-dependent root boundary; nested runs refused |

## Refused

| Selection | Reason |
|---|---|
| `mp_physics=26` (WDM7) | No fp32-carry island for the WDM7 kernel, and the WDM number concentrations Nc/Nn/Nr are not transported (WRF advects them; CCN also has a fixed-inflow boundary) |
| `mp_physics=18/24/28/40` with `--max-dom > 1` | Their extra species have no parent→child lateral forcing in this port (WRF forces them on nests) |

Evidence: `tests/v040/rootbdy/test_root_flow_extras.py` (pristine-WRF boundary chain on real PROD d01 operands; Swiss
production-step dispatch; pre-JAX refusals), `proofs/o2_rootbdy/smoke_release.json` (1 release-carry step per code on the
Swiss case, CPU, Pallas interpret, finite), release jaxpr identity Swiss `2bf12cb8` / PROD d01 `6fc715a2`.
Known shipped gap surfaced by this lane: `mp_physics=16` (WDM6) has the same untransported Nc/Nn/Nr as WDM7.
