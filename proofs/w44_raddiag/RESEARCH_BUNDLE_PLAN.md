# ALISIOS rad_diag research bundle — CPU preparation plan

Authority: manager 2026-10-23 18:40:40Z / 20:59:52Z. Requested before 2026-11-01 for Nord333 step3, A/R experiments with heating rates. This is a research build. The nightly stays on its existing sealed bundle until ALISIOS explicitly changes that pin. No GPU job is armed by this plan.

## Pin and delivery layout

After the **same critic returns MERGE**, the manager selects the integrated source commit and its final version string. Freeze that source before priming; no source/version edit after the bundle pin. The current measured 3-nest cache is an evidence/performance dependency, not a substitute for compiling the final research source and Nord333 shapes.

Proposed final absolute path (manager-owned namespace):

```
<DATA_ROOT>/wrf_gpu2/release_bundles/research-raddiag-<source12>/
  src/                 immutable detached source, including data/wrf_pristine link
  bundle.env           exact environment for prime AND consumer
  jax/                 persistent JAX + AOT + effective per-fusion tunes
  autotune/ triton/ cuda/
  tools/               exact compile_bundle + bundle_consumer tools
  sealed/              BUNDLE_MANIFEST.json, BUNDLE.json, deterministic tar.gz
  receipts/            input/key/native/E41/readback/replay receipts
  RESEARCH_BUILD.json   scope, qualification, source/tool/input SHAs, limitations
```

The absolute cache/autotune/source paths must be fixed **before** the first GPU compile. `compile_bundle.py` documents that moving a primed cache changes/refuses its target keys. The research label lives in metadata/path; do not add an ad-hoc `GPUWRF_*` model flag or edit `__version__` after priming. Freeze the source `commit`, `src_tree`, `gpuwrf_tree`, assets manifest, JAX/jaxlib, ptxas and GPU/driver target tuple with the existing seal.

## Required ALISIOS inputs — unresolved prerequisites

Manager was asked at21:18:02Z for exact staged paths/configuration authority. Do not infer what A/R means.

| Required fact | Why it binds the bundle |
|---|---|
| Exact A/R namelists and intended comparisons | Physics selections/cadences/options are static program keys; diagnostics must not silently change either arm |
| Nord333 input directory/date set, every file SHA | Prime the actual grid/nests; ALISIOS-root reads need sealed-facade admission or a manager-authorized staged copy |
| Actual d01–d04 dimensions, nesting ratios, dt, e_vert, p_top and geography | The completed WN3 0227 gate is 3 nests/44 mass levels; it is not evidence of 4-nest333m or e_vert≈85 qualification |
| Requested horizon/history/auxinput/restart settings | Check whole root-step divisibility and input coverage before expensive initialization |
| At least one second issue on identical geometry/options, if available | Cross-date warm-key and replay check; do not promise date coverage not tested |

The only presently authorized input available to this lane is the read-only staged WN3 0227 3-nest copy. It can remain a regression canary; it cannot stand in for missing Nord333 inputs.

## CPU stage (no GPU)

1. After critic MERGE and manager source pin, create the final source snapshot and preserve its pristine-WRF dependency. Record source/tool hashes, disk floor ≥200GiB and exact target paths. Keep every temporary cache/output in this lane or its allowed scratch namespace.
2. Stage admitted inputs read-only. Record all namelist/input/boundary/auxiliary file hashes and resolved settings. Make research variants as separate namelist copies; never edit ALISIOS or the shared staged source.
3. Run the product CLI `--dry-run` for every member (`JAX_PLATFORMS=cpu`). Check active RRTMG4/4, exact domain count/geometry, whole-step3h canary, history cadence, input horizon, LUT/geography admission and unsupported-combination refusals.
4. Use the existing CPU `gpuwrf.cli aot-key --input-dir … --compare-input-dir … --hours … --json …` workflow, private temporary caches and ≤12G RAM slot. Compare components for the actual final pin and every member/date; group only genuinely identical program keys. Never compile a full fused four-nest physics executable on CPU. If loading the requested large grid exceeds the CPU cap, log/refuse that probe and use shape/header preparation plus the admitted GPU producer inventory instead of an invented key.
5. Freeze a member manifest and a CPU-dry-tested night launcher. The manager sequencer supplies the real UTC window; the lane does not start it itself. Source the same bundle.env in prime and replay, keep the CLI backend-dark until allocator admission, and capture asserted GPU backend/source/work receipts later.

For each **authoritatively supplied A/R physics member**, include `rad_diag=1` in the research runs and its `rad_diag=0` control for the compiled observer gate. If A/R is only an input/date difference, the CPU key census should prove sharing rather than cold-priming the same program twice. Include `icloud_bl=0` or `icloud=3` only when the A/R specification actually requests it. Do not multiply unrequested scheme combinations.

## Night producer and acceptance sequence

| Stage | Required result |
|---|---|
| Prime each unique program family | Real Nord333 geometry/options, all required init/root/fused-child programs, declared private cache, timeout and official lease/lock |
| Observer gate | Same source/work per member, diagOFF vsON compiled prognostic/history bytes exact; STOP/report any difference. All expected new fields finite and metadata exact on every nest |
| Actual executable gate | E41 on rich JIT and AOT, with producer proof inventory requiring all four nests/fused descendants; no root-only scan |
| Seal | Existing `scripts/release/compile_bundle.py seal`; every AOT blob/meta digest and target field attested; input and assets hashes included |
| Overlay replay | Existing bundle_consumer.sh only, same absolute bundle path, distinct private layer per ALISIOS consumer. `compiled_new=0`, unchanged sealed files, allowed aliases/runtime state only |
| Same-input replay | Field/file identity to producer, all expected hourly fields present, source/native/program tuple pinned |
| Cross-date replay | Same geometry/options, second admitted issue, 0 new compiles and complete outputs; otherwise state NOT TESTED |
| Research delivery | `BUNDLE_OK`, manifest SHA + deterministic bundle SHA, source tuple, exact scope/limits, operator command and independent critic status |

The producer may write its new research tree until sealing. Consumers always use the existing copy-on-write overlay; they never write into the sealed bundle or modify the nightly tuple. No release/72h/85-level/day-case claim is inferred from a3h research canary. The manager owns the composed v0.4.4 RW1–RW4 release gates.

## Reuse and slot budget

Carry forward the accepted unchanged-source WRF column/oracle and 3-nest numerical evidence. Recheck the new compiled Nord333 artifact and its actual option/key coverage; avoid obsolete intermediate numerical runs. Seed only compatible per-fusion autotune data, never relabel old AOT blobs as the new pin.

Measured3-nest44-level cold setup was943/720/703s for OFF/ON/bl0, with shared initialization cache after the first arm. Four nests/other levels are **unknown**. Request a bounded night cold-setup phase per unique program family after the CPU key census, not a guessed all-in5min slot. Split prime and warm acceptance into separate night phases if necessary; every GPU arm stays within its lease/window and the project cold-setup limit. CPU E41/sealing/scoring can follow after GPU release.

Use the established owned process-group/supervisor pattern: spawn the target in its own process group; on timeout TERM then KILL that group, wait for descendants, and verify no owned child retains FD9 before reporting GPU END. The failed nsys prototype showed why a parent timeout alone is insufficient. Research priming needs no nsys child; optional cost profiling is a separate manager/perf phase.

## Operator handoff template

```
status: RESEARCH / not nightly / not a release
path: <final absolute research bundle>
source: <commit, src_tree, gpuwrf_tree, version>
bundle_sha256: <BUNDLE.json>
manifest_sha256: <BUNDLE.json>
cases/options: <exact admitted A/R + geometry + dates>
gates: <observer, E41, replay0compiles, finite/readback, cross-date>
limits: <untested shapes/levels/dates, counter-cost accounting status>
consumer: scripts/with_gpu_lock.sh --label <authorized-lane> --timeout 1 -- \
  <P>/tools/bundle_consumer.sh --bundle <P> --layer <private-layer> -- \
  <command sourcing P/bundle.env then gpuwrf run with the frozen case/options>
```

No bundle path/SHA is reported as delivered until actual night priming, sealing and overlay acceptance complete.
