#!/usr/bin/env python3
"""Static ALISIOS path inventory. Never dereference a discovered data path.

Stdlib only. Repository inputs come from immutable Git objects. Receipt inputs
are restricted to our lane directory and opened with O_NOFOLLOW at EVERY level.
The output is a JSON list, starting with a scope/summary record. A literal path
is not proof that a payload was read. UNKNOWN is deliberately not false.
"""
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from context_dates import extract as extract_context

ROOT_RE = r'/mnt/(?:data/alisios|server/work/src/alisios)(?:/|(?=$|["\s]))'
PATH_RE = re.compile(ROOT_RE + r'''[^\s"'`<>\\,;\)\]\}]*''')
DAY_RE = re.compile(r'(?<!\d)(20\d{2})-?(\d{2})-?(\d{2})(?!\d)')
ISSUE_RE = re.compile(r'(?:wg_|/)(20\d{6})_(\d{2})z')
STAMP_RE = re.compile(r'20\d{2}-\d{2}-\d{2}[T_ ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?')
HOURS_RE = re.compile(r'''(?:run_hours|forecast_hours|--hours|hours)\s*["']?\s*(?:[:=]|\s)\s*["']?(\d+(?:\.\d+)?)''')
INIT_RE = re.compile(r'''(?:init_time_utc|start_time|start_datetime)["']?\s*[:=]\s*["'](20\d\d-\d\d-\d\d[T_ ]\d\d:\d\d:\d\d(?:Z|[+-]\d\d:\d\d)?)''')
TEXT_SUFFIX = {'.json', '.jsonl', '.md', '.txt', '.log', '.sh', '.py', '.yaml', '.yml', '.tsv'}
PRUNE = {'.git', '__pycache__', 'node_modules', '.venv', 'pytest', 'jit', 'cache', 'caches', 'aot', 'xla_gpu_per_fusion_autotune_cache_dir'}
LANES = '<USER_HOME>/wrf_gpu2_lanes'
MAX_BYTES = 8 * 1024 * 1024


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], timeout=240)


def paths_in(text):
    # Python adjacent literals are common in these harnesses. AST folds them
    # without importing or executing the harness. Dynamic expressions stay raw.
    strings = [text]
    try:
        if text.lstrip().startswith(('{', '[')) or len(text) > 262144:
            return sorted(found_paths_raw(text))
        tree = ast.parse(text)
        strings += [n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    except (SyntaxError, ValueError):
        pass
    found = set()
    for value in strings:
        found.update(m.group().rstrip('.:') for m in PATH_RE.finditer(value))
    # Drop incomplete fragments if adjacent-literal folding recovered the full path.
    return sorted(p for p in found if not any(q != p and q.startswith(p) for q in found))


def found_paths_raw(text):
    return {m.group().rstrip('.:') for m in PATH_RE.finditer(text)}


def dates_in(value):
    result = []
    for m in DAY_RE.finditer(value):
        try:
            result.append(datetime(*map(int, m.groups())).date().isoformat())
        except ValueError:
            pass
    return sorted(set(result))


def checks(days, unresolved):
    ranges = {'in_E32': ('2026-08-10', '2026-08-22'),
              'in_B1': ('2026-11-16', '2026-12-13'),
              'on_or_after_2026-09-15': ('2026-09-15', '9999-12-31')}
    out = {}
    for name, (lo, hi) in ranges.items():
        hits = [d for d in days if lo <= d <= hi]
        value = True if hits else ('UNKNOWN' if unresolved or not days else False)
        out[name] = {'value': value, 'evidence':
                     f'explicit/derived days={days}; inclusive range={lo}..{hi}; '
                     f'matching={hits}; unbounded/undetermined={unresolved}'}
    return out


def path_evidence(path, text):
    date_tokens = dates_in(path)
    issues = []
    for m in ISSUE_RE.finditer(path):
        issues.append(datetime.strptime(''.join(m.groups()), '%Y%m%d%H'))
    hour_values = sorted(set(float(m.group(1)) for m in HOURS_RE.finditer(text)))
    # Do not assume issue time is forecast start (WN3 often starts six hours
    # later). Only derive an illustrative horizon from an explicit init clock.
    starts = sorted({m.group(1) for m in INIT_RE.finditer(text)})
    derived = []
    for value in starts:
        init = datetime.fromisoformat(value.replace('_', 'T').replace('Z', '+00:00'))
        for hours in hour_values:
            if hours <= 240:
                end = init + timedelta(hours=hours)
                span = [(init + timedelta(days=i)).date().isoformat()
                        for i in range((end.date() - init.date()).days + 1)]
                derived.append({'start': init.isoformat(), 'run_hours': hours,
                                'end': end.isoformat(), 'days': span,
                                'status': 'POSSIBLE_FILE_GLOBAL_HORIZON_NOT_PROVEN_PER_PATH'})
    explicit_valid = bool(re.search(r'wrfout_[^/]*_20\d{2}-\d{2}-\d{2}', path))
    valid_days = dates_in(path.split('wrfout_', 1)[1]) if explicit_valid else []
    issue_days = sorted({x.date().isoformat() for x in issues})
    product = re.search(r'/days/(20\d\d-\d\d-\d\d)(?:/|$)', path)
    case_date = re.search(r'/registry/wrf_cases/payload/[^/]+/(20\d\d-\d\d-\d\d)(?:/|$)', path)
    if product:
        valid_days = dates_in(product.group(1))
    case_days = dates_in(case_date.group(1)) if case_date else []
    days = sorted(set(issue_days + valid_days + case_days))
    root_only_case = bool(issues) and not explicit_valid
    dynamic = any(c in path for c in '{}*$') or 'wg_2026' in path and not issues
    synthetic = any(s in path for s in ('__trackp_never_opened__', '__cr_trackp_no_payload__'))
    return {'path': path, 'purpose': 'SYNTHETIC_GUARD_PATH' if synthetic else 'UNKNOWN',
            'issue_days': issue_days, 'explicit_path_date_tokens': date_tokens,
            'unclassified_path_date_tokens': sorted(set(date_tokens) - set(days)),
            'forecast_or_valid_days': days, 'possible_horizons': derived,
            'file_run_hours_values': hour_values or 'UNKNOWN',
            'valid_days': valid_days if explicit_valid or product else 'UNKNOWN',
            'extent': 'EXPLICIT_PRODUCT_DAY' if product else 'EXPLICIT_WRFOUT_TIMESTAMP' if explicit_valid else 'UNKNOWN',
            'checks': checks(days, dynamic or root_only_case or not days or bool(case_days))}


def safe_open(path, max_bytes=MAX_BYTES):
    """No symlink resolution, including parent directory components."""
    path = os.path.abspath(path)  # lexical only; never realpath/stat a target
    if not path.startswith(LANES + '/'):
        raise ValueError('receipt outside allowed lane root')
    parts = path.split('/')[1:]
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = new
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(file_fd, 'rb') as f:
            import stat
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode):
                raise ValueError('not a regular receipt')
            if st.st_size > max_bytes:
                raise ValueError(f'receipt exceeds {max_bytes} B text limit')
            return f.read().decode('utf-8', errors='replace')
    finally:
        os.close(fd)


def receipt_files(root, skipped, active=False):
    # Never follow links: is_dir(follow_symlinks=False) is essential here.
    try:
        with os.scandir(root) as it:
            entries = sorted(it, key=lambda e: e.name)
    except OSError as e:
        skipped.append({'path': str(root), 'reason': type(e).__name__})
        return
    for e in entries:
        if e.is_symlink():
            skipped.append({'path': e.path, 'reason': 'SYMLINK_NOT_FOLLOWED'})
            continue
        if e.is_dir(follow_symlinks=False):
            if e.name in PRUNE or e.name.startswith(('snap_', 'snapshot', 'snap-')):
                skipped.append({'path': e.path, 'reason': 'CACHE_OR_SNAPSHOT_EXCLUDED'})
                continue
            on = active or e.name in {'bench', 'v-prep', 'wn3', 'release-docs'}
            yield from receipt_files(e.path, skipped, on)
        elif active and Path(e.name).suffix.lower() in TEXT_SUFFIX:
            yield e.path


def history_records(patch):
    changes = defaultdict(list)
    commit = filename = None
    for line in patch.splitlines():
        if line.startswith('@@COMMIT '):
            _, commit, date = line.split(maxsplit=2)
        elif line.startswith('diff --git '):
            filename = line.split(' b/', 1)[-1]
        elif line[:1] in {'+', '-'} and not line.startswith(('+++', '---')):
            for path in paths_in(line[1:]):
                changes[(filename, path)].append({'commit': commit, 'timestamp': date,
                                                  'change': 'ADD' if line[0] == '+' else 'REMOVE'})
    return changes


def stamp_evidence(text):
    found = []
    # Explicit metadata only: never use filesystem mtime or a forecast date as
    # a run date. Epoch values require timestamp-like JSON keys.
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        data = None
    def walk(v, key='$'):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, key + '.' + k)
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f'{key}[{i}]')
        elif (re.search(r'(?:\.utc$|timestamp|created|generated|audited|t_start|t_end|start_utc|end_utc|started_at|finished_at|run_date)', key, re.I)
              and not re.search(r'valid|init_time|forecast|clock', key, re.I)):
            if isinstance(v, str) and STAMP_RE.search(v):
                found.append({'key': key, 'value': v[:200]})
            elif isinstance(v, (int, float)) and 1700000000 < v < 1900000000:
                found.append({'key': key, 'value': datetime.fromtimestamp(v, timezone.utc).isoformat(), 'epoch': v})
    walk(data)
    if data is None:
        for i, line in enumerate(text.splitlines(), 1):
            if re.search(r'created|generated|timestamp|started|finished|UTC|^20\d\d-\d\d-\d\dT', line, re.I):
                for m in STAMP_RE.finditer(line):
                    found.append({'line': i, 'value': m.group()})
    return found or 'UNKNOWN'


def entry(kind, source, text, paths):
    evidence = [path_evidence(p, text) for p in paths]
    days = sorted({d for e in evidence for d in e['forecast_or_valid_days']})
    unresolved = any(any(v['value'] == 'UNKNOWN' for v in e['checks'].values()) for e in evidence)
    lines = [{'line': i, 'text': line[:1500]} for i, line in enumerate(text.splitlines(), 1)
             if re.search(ROOT_RE, line)]
    read_status = 'UNKNOWN_STATIC_REFERENCE_ONLY'
    receipt_evidence = []
    try:
        obj = json.loads(text)
        def inspect(v, key='$'):
            if isinstance(v, dict):
                if (v.get('schema') == 'wrfout-grid-comparison-v1'
                    and isinstance(v.get('summaries'), dict)
                    and v['summaries'].get('comparable_field_count', 0) > 0):
                    receipt_evidence.append(key + ': schema=wrfout-grid-comparison-v1; comparable_field_count>0')
                if 'pairing' in v and isinstance(v['pairing'], dict) and v['pairing'].get('common_file_count_before_lead_filter', 0) > 0:
                    receipt_evidence.append(key + '.pairing: common_file_count_before_lead_filter>0')
                for k, x in v.items():
                    inspect(x, key + '.' + k)
            elif isinstance(v, list):
                for i, x in enumerate(v):
                    inspect(x, f'{key}[{i}]')
        inspect(obj)
        if receipt_evidence:
            read_status = 'RECEIPT_REPORTS_COMPARISON_NOT_INDEPENDENT_READ_SPY'
    except (ValueError, TypeError):
        pass
    return {'record_type': kind, 'source': source,
            'sha256': hashlib.sha256(text.encode()).hexdigest(),
            'path_evidence': evidence, 'days': days, 'checks': checks(days, unresolved),
            'evidence_lines': lines,
            'read_execution': read_status, 'read_execution_evidence': receipt_evidence,
            'run_timestamp_evidence': stamp_evidence(text),
            'release_versions_explicit': sorted(set(re.findall(r'v0\.(?:3\.\d+|4\.0)\b', text))),
            'unknowns': ['Payload reads were not replayed; path presence alone does not establish execution.']}


def enrich(item, text, historical=False):
    facts = extract_context(text)
    bases = [p['path'] for p in item['path_evidence']]
    if historical:
        facts = [f for f in facts if any(q.startswith(b) or b.startswith(q)
                                        for q in f['paths'] for b in bases)]
        for f in facts:
            f['paths'] = [q for q in f['paths'] if any(q.startswith(b) or b.startswith(q) for b in bases)]
    for f in facts:
        f['forecast_or_valid_days'] = sorted(set(f['valid_days']) |
            {d for q in f['paths'] for d in path_evidence(q, '')['forecast_or_valid_days']})
    item['context_date_evidence'] = [f for f in facts if f['forecast_or_valid_days']]
    item['derived_paths'] = sorted({q for f in facts if f['forecast_or_valid_days'] for q in f['paths']})
    for p in item['path_evidence']:
        fresh = path_evidence(p['path'], text)
        for k, v in fresh.items():
            p[k] = v
        related = [f for f in facts if any(q.startswith(p['path']) or p['path'].startswith(q) for q in f['paths'])]
        p['contextual_forecast_or_valid_days'] = sorted({d for f in related for d in f['forecast_or_valid_days']})
        p['bounded_context_windows'] = [f for f in related if f['bounded_valid_extent']]
        ds = sorted(set(p['forecast_or_valid_days'] + p['contextual_forecast_or_valid_days']))
        uncertain = p['valid_days'] == 'UNKNOWN' and not p['bounded_context_windows']
        p['checks'] = checks(ds, uncertain)
    item['days'] = sorted({d for p in item['path_evidence'] for d in p['forecast_or_valid_days']} |
                          {d for f in facts for d in f['forecast_or_valid_days']})
    unknown = any(v['value'] == 'UNKNOWN' for p in item['path_evidence'] for v in p['checks'].values())
    item['checks'] = checks(item['days'], unknown)
    item['run_timestamp_evidence'] = stamp_evidence(text)
    item['context_source_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    return item


def enrich_existing(args):
    raw = safe_open(args.base_inventory, max_bytes=64 * 1024 * 1024)
    data = json.loads(raw)
    summary, entries = data[0], data[1:]
    start = time.monotonic()
    changes = []
    for item in entries:
        if os.path.exists('/tmp/wrf_gpu2_quiet'):
            raise RuntimeError('QUIET during contextual enrichment')
        old_days = item['days'][:]
        old_stamp = item['run_timestamp_evidence']
        if item['record_type'] == 'LANE_RECEIPT_OR_HARNESS_PATH_REFERENCE':
            text = safe_open(item['source'])
            assert hashlib.sha256(text.encode()).hexdigest() == item['sha256'], item['source']
        elif item['record_type'] == 'REPOSITORY_PATH_REFERENCE':
            text = git(args.repo, 'show', f"{item['source_commit']}:{item['source']}").decode()
            assert hashlib.sha256(text.encode()).hexdigest() == item['sha256'], item['source']
        else:
            cs = item['first_path_commit_on_main']
            assert isinstance(cs, dict), item['source']
            text = git(args.repo, 'show', f"{cs['commit']}:{item['source']}").decode()
            item['historical_context_commit'] = cs['commit']
        enrich(item, text, historical=item['record_type'].startswith('HISTORICAL'))
        added = sorted(set(item['days']) - set(old_days))
        if added or old_stamp != item['run_timestamp_evidence']:
            changes.append({'source': item['source'], 'historical_literal': item['path_evidence'][0]['path']
                            if item['record_type'].startswith('HISTORICAL') else None,
                            'added_days': added, 'run_timestamp_recovered': old_stamp == 'UNKNOWN' and item['run_timestamp_evidence'] != 'UNKNOWN'})
    summary['contextual_enrichment'] = {'base_inventory': args.base_inventory,
        'base_sha256': hashlib.sha256(raw.encode()).hexdigest(), 'elapsed_seconds': time.monotonic() - start,
        'changed_entries': changes, 'method': 'syntax-only bindings, guard mappings and same-object forecast windows; no reader execution'}
    summary['generated_utc'] = datetime.now(timezone.utc).isoformat()
    summary['checks_counts'] = {k: dict(Counter(str(e['checks'][k]['value']) for e in entries)) for k in ('in_E32', 'in_B1', 'on_or_after_2026-09-15')}
    summary['run_timestamp_unknown_count'] = sum(e['run_timestamp_evidence'] == 'UNKNOWN' for e in entries)
    summary['unique_forecast_or_valid_days'] = sorted({d for e in entries for d in e['days']})
    summary['unique_derived_paths'] = len({p for e in entries for p in e['derived_paths']})
    summary['unknown_valid_extent_sources'] = sorted({e['source'] for e in entries if any(v['value'] == 'UNKNOWN' for v in e['checks'].values())})
    summary['comparison_receipt_days'] = sorted({d for e in entries if e['read_execution'].startswith('RECEIPT_REPORTS') for d in e['days']})
    summary['unclassified_date_tokens'] = sorted({d for e in entries for p in e['path_evidence'] for d in p['unclassified_path_date_tokens']})
    Path(args.output).write_text(json.dumps([summary, *entries], indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('excluded', 'unknown_valid_extent_sources', 'known_synthetic_paths')}, indent=2))


def build(args):
    start = time.monotonic()
    repo = Path(args.repo)
    ref = git(repo, 'rev-parse', args.ref).decode().strip()
    patch = Path(args.history).read_text()
    changes = history_records(patch)
    current = git(repo, '-c', 'grep.threads=1', 'grep', '-l', '-I', '-E',
                  '/mnt/(data/alisios|server/work/src/alisios)/', ref, '--',
                  'tests', 'scripts', 'proofs').decode().splitlines()
    entries = []
    seen = set()
    for spec in current:
        filename = spec.split(':', 1)[1]
        text = git(repo, 'show', f'{ref}:{filename}').decode(errors='replace')
        item = entry('REPOSITORY_PATH_REFERENCE', filename, text, paths_in(text))
        item['source_commit'] = ref
        item['last_file_commit'] = git(repo, 'log', '-1', '--format=%H %cI', ref, '--', filename).decode().strip() or 'UNKNOWN'
        item['file_added_commit'] = git(repo, 'log', '--follow', '--diff-filter=A', '--format=%H %cI', ref, '--', filename).decode().splitlines() or 'UNKNOWN'
        file_changes = [c for (f, _), cs in changes.items() if f == filename for c in cs]
        item['first_path_commit_on_main'] = min(file_changes, key=lambda c: c['timestamp']) if file_changes else 'UNKNOWN'
        item['last_path_commit_on_main'] = max(file_changes, key=lambda c: c['timestamp']) if file_changes else 'UNKNOWN'
        for p in item['path_evidence']:
            followed = False
            cs = [c for (f, s), rows in changes.items() if f == filename and
                  (s == p['path'] or p['path'].startswith(s)) for c in rows]
            if not cs:
                # A pure rename has no -G match. Follow it explicitly for an
                # exact literal before classifying provenance as UNKNOWN.
                rows = git(repo, 'log', '--follow', '--first-parent', '--format=%H %cI',
                           '-S', p['path'], ref, '--', filename).decode().splitlines()
                cs = [{'commit': row.split()[0], 'timestamp': row.split()[1],
                       'change': 'PATH_COUNT_CHANGED_RENAME_FOLLOWED'} for row in reversed(rows)]
                if cs:
                    cs[0]['change'] = 'ADD'
                    followed = True
            cs.sort(key=lambda c: datetime.fromisoformat(c['timestamp']))
            p['main_path_changes'] = cs or 'UNKNOWN'
            p['history_match'] = ('RENAME_FOLLOWED_EXACT_LITERAL' if followed else
                                  'EXACT_LITERAL' if (filename, p['path']) in changes else 'PREFIX_FRAGMENT_ONLY')
            p['first_commit_on_main'] = next((c for c in cs if c['change'] == 'ADD'), 'UNKNOWN')
            p['last_commit_on_main'] = cs[-1] if cs else 'UNKNOWN'
            seen.add((filename, p['path']))
        all_changes = [c for p in item['path_evidence'] if isinstance(p['main_path_changes'], list) for c in p['main_path_changes']]
        if all_changes:
            item['first_path_commit_on_main'] = min(all_changes, key=lambda c: datetime.fromisoformat(c['timestamp']))
            item['last_path_commit_on_main'] = max(all_changes, key=lambda c: datetime.fromisoformat(c['timestamp']))
        entries.append(enrich(item, text))
    for (filename, path), cs in sorted(changes.items()):
        if (filename, path) in seen or any(f == filename and p.startswith(path) for f, p in seen):
            continue
        item = entry('HISTORICAL_REPOSITORY_PATH_REFERENCE', filename, path, [path])
        item['first_path_commit_on_main'] = next((c for c in cs if c['change'] == 'ADD'), 'UNKNOWN')
        item['last_path_commit_on_main'] = cs[-1]
        item['path_evidence'][0]['main_path_changes'] = cs
        historical_text = git(repo, 'show', f"{item['first_path_commit_on_main']['commit']}:{filename}").decode()
        item['historical_context_commit'] = item['first_path_commit_on_main']['commit']
        entries.append(enrich(item, historical_text, historical=True))
    skipped = []
    examined = 0
    for filename in receipt_files(LANES, skipped):
        if '/w41-readaudit/' in filename:
            continue
        # Abort between files as well as at shell entry if a timed arm begins.
        if os.path.exists('/tmp/wrf_gpu2_quiet'):
            raise RuntimeError('QUIET appeared during static scan; rerun later')
        try:
            text = safe_open(filename)
        except (OSError, ValueError) as e:
            skipped.append({'path': filename, 'reason': str(e)})
            continue
        examined += 1
        paths = paths_in(text) if re.search(ROOT_RE, text) else []
        if paths:
            entries.append(enrich(entry('LANE_RECEIPT_OR_HARNESS_PATH_REFERENCE', filename, text, paths), text))
    summary = {'record_type': 'SUMMARY', 'source_commit': ref,
               'generated_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'tests/, scripts/, proofs/ at source_commit plus historical first-parent main path changes; '
                        'lane text files below any bench/, v-prep/, wn3/, release-docs/ directory',
               'limitations': ['No ALISIOS payload, directory, symlink target or metadata was opened/listed/statted.',
                              'First commit means first-parent main integration; feature author commit may precede it.',
                              'Only explicit date strings are definitive. Issue day does not bound the valid-day extent.',
                              'run_hours elsewhere in a file is a possible horizon; it is not assigned to every arm.',
                              'Release versions and read execution remain UNKNOWN unless evidenced in the retained text.',
                              'Symlinks, caches, snapshots, binary files and text files larger than 8 MiB excluded; exclusions below.',
                              'Receipt tree is a point-in-time live scan; exact file hashes retained.'],
               'entries': len(entries), 'counts_by_record_type': dict(Counter(e['record_type'] for e in entries)),
               'receipt_text_files_examined': examined, 'excluded': skipped,
               'checks_counts': {k: dict(Counter(str(e['checks'][k]['value']) for e in entries))
                                 for k in ('in_E32', 'in_B1', 'on_or_after_2026-09-15')},
               'run_timestamp_unknown_count': sum(e['run_timestamp_evidence'] == 'UNKNOWN' for e in entries),
               'read_execution_counts': dict(Counter(e['read_execution'] for e in entries)),
               'unique_forecast_or_valid_days': sorted({d for e in entries for d in e['days']}),
               'unclassified_date_tokens': sorted({d for e in entries for p in e['path_evidence']
                                                   for d in p['unclassified_path_date_tokens']}),
               'comparison_receipt_days': sorted({d for e in entries if e['read_execution'].startswith('RECEIPT_REPORTS') for d in e['days']}),
               'known_synthetic_paths': sorted({p['path'] for e in entries for p in e['path_evidence'] if p['purpose'] == 'SYNTHETIC_GUARD_PATH'}),
               'unknown_valid_extent_sources': sorted({e['source'] for e in entries if any(p['valid_days'] == 'UNKNOWN' for p in e['path_evidence'])}),
               'unique_paths': len({p['path'] for e in entries for p in e['path_evidence']}),
               'cost': {'elapsed_seconds': time.monotonic() - start, 'GPU_commands': 0,
                        'JAX_imports': 0, 'compile_cache': 'NONE'}}
    Path(args.output).write_text(json.dumps([summary, *entries], indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in {'excluded', 'limitations'}}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='.')
    p.add_argument('--ref', required=True)
    p.add_argument('--history')
    p.add_argument('--base-inventory')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.base_inventory:
        enrich_existing(args)
    else:
        if not args.history:
            p.error('--history or --base-inventory is required')
        build(args)
