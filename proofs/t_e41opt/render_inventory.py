"""Render the frozen E41 inventory as support-matrix supplement rows."""
import json
from pathlib import Path
import sys

inventory, output = map(Path, sys.argv[1:])
report = json.loads(inventory.read_text())
lines = ['# v0.4.0 experimental opt-in E41 supplement', '',
         f'Inventory cutoff: **{report["cutoff_utc"]}**. These rows describe exact retained GPU',
         'executables. They do not waive the release-default hard gate or qualify',
         'a different scheme combination. All values below are measured static',
         'artifact properties [M]. Per-fusion runtime impact is **unknown [U]** for',
         'every row: no matched per-kernel profile was retained for these programs.', '',
         'STACK is bytes. kInput size is **optimized HLO instructions**, not SASS.',
         'Limits remain STACK ≤256 B, kInput transpose ≤500 HLO and non-leaf callee',
         '≤1,000 SASS. The complete producer scans report no non-leaf violation.',
         'Large caller roots with small leaf callees are not non-leaf failures.', '',
         '## Configuration scope and independent CPU-WRF status', '',
         'Options are MP/CU/PBL/SFCLAY/LSM/LW/SW. D6 status is copied from the',
         'producer, without upgrading missing coverage to PASS. Full D6 PASS and',
         'raw finiteness do not justify F2 negative events.', '',
         '| Case | Options | Source | D6 | Frames | Program max STACK B | Flags |',
         '|---|---|---|---|---:|---:|---:|']
keys = ('mp_physics','cu_physics','bl_pbl_physics','sf_sfclay_physics',
        'sf_surface_physics','ra_lw_physics','ra_sw_physics')
for row in report['records']:
    opts = '/'.join(str(row['options'][k]) for k in keys)
    source = row['source_commit'] or 'see receipt'
    lines.append(f'| {row["case"]} | {opts} | `{source[:10]}` | {row["D6_verdict"]} | '
                 f'{row["D6_paired_frames"] or "unknown"} | {row["program_max_stack_B"]} | {len(row["flags"])} |')
lines += ['', '## Support-matrix disclosure rows', '',
          'Every listed configuration remains **experimental: E41 STACK** and/or',
          '**experimental: E41 kInput** as indicated by its measured flag. Source',
          'sites are recovered from rich HLO stack frames, not inferred from names.',
          'Full source distributions and additional frames are in the sealed maps.', '',
          '| Case | Fusion | STACK B | kInput HLO >500 | Source sites |',
          '|---|---|---:|---:|---|']
for row in report['records']:
    for name, flag in row['flags'].items():
        sources = flag.get('sources', [])
        assert sources, (row['case'],name,'source attribution incomplete')
        sites = '; '.join(f'`{Path(s["file"]).name}:{s["line"]}` ({s["function"]})' for s in sources[:2])
        lines.append(f'| {row["case"]} | `{name}` | {flag["stack_B"]} | {flag.get("kinput_HLO", "—")} | {sites} |')
lines += ['', '## Remedy disposition and resource consequences', '',
          'No production code change is included. The production `src/gpuwrf` tree',
          'is byte-identical to the lane base; default arithmetic, launches, traffic,',
          'temporary allocations, peak VRAM and executable construction are unchanged.',
          'No speedup, roofline closure or isolated memory-impact claim is made.', '',
          'The cheapest public-output firewall was screened on the real Swiss inputs',
          'through `MP_SCAN_ADAPTERS`, including its production fp64 island, State',
          'replacements, precipitation accumulation and theta conversion: MP24',
          '**164/164** and MP26 **158/158** State leaf results byte-identical over two',
          'successive MP calls, all finite. These are CPU regression/neutrality checks;',
          'dynamics and other physics were not stepped in this local screen. They',
          'are not WRF fidelity or native-resource proof. The public firewall is not',
          'shipped: its effect on the internal reduction/transpose mega-fusions has',
          'not been established on a whole-root native executable. Internal cuts',
          'would add materialization/traffic and launches and require fresh native',
          'measurements; fixed-loop rewrites also require the same byte gate.', '',
          'C03 Eta/mixed-surface remedies belong to `t-c03rain`; SF1/PBL work belongs',
          'to `t-c02nan`. Their files were not edited. Their retained failures are',
          'listed for complete disclosure, without claiming a fix or transferring',
          'runtime between similarly named fusions. MP40 bounded D6 passed, but its',
          'full comparison is INCOMPLETE: 12 CPU fields are missing. `t-writer` owns',
          'that independent coverage issue.', '',
          'EO03, the initial whole-root CPU harness, reused a donated buffer and',
          'failed before candidate execution. That setup failure is preserved and',
          'logged; it is neither a candidate numerical failure nor a passing gate.', '',
          '## Evidence', '',
          'The compact evidence files are in `proofs/t_e41opt/`.',
          '`INVENTORY.json` binds every producer gate and input manifest by SHA256.',
          'Producer `e41/pins` retain actual native executables and complete scans;',
          'the attribution maps bind those PINs, decompressed blobs and rich HLO.',
          '`SOURCE_MAPS.json` and `CPU_NEUTRALITY.json` preserve the compact evidence.',
          'Full maps remain under `<USER_HOME>/wrf_gpu2_lanes/t-e41opt/EO*`.',
          'Producer references and artifacts were not changed. Disposable lane',
          'snapshots were removed; no lane JAX/native cache is retained.', '']
output.write_text('\n'.join(lines))
