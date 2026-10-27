"""Publish the reconciled ordinary-root census; no new measurement/compile."""
from collections import defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

L=Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf')
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,default=L/'SOURCE_CENSUS_FINAL');args=ap.parse_args()
DATA=args.data
OUT=Path(__file__).resolve().parent
r=json.loads((DATA/'DELTA.json').read_text())
a,b=r['arms']['N32'],r['arms']['N43']
for arm,version in [('N32','v032'),('N43','v043')]:
    for domain in ('d01','d02'):
        proof=json.loads((L/'ORDINARY_THUNKS'/f'{version}_{domain}.json').read_text())['actual_execution_binding']
        assert proof['all_rest_fields_except_retained_hlo_equal'] and all(proof['records_0_1_2_equal'].values())
        r['arms'][arm]['source_execution_bindings'][domain]=proof
assert b['count']-a['count']==1935
assert b['device_activity_rows']['total']-a['device_activity_rows']['total']==1479

scope_rows=[]
for scope in ['d01','d02','nesting/force']:
    values=[]
    for arm in [a,b]:
        n=ns=0
        for key,v in arm['by_scope_family_class'].items():
            if key.split(':',1)[0]==scope:n+=v[0];ns+=v[1]
        values.append([n,ns])
    scope_rows.append(dict(scope=scope,old=values[0],new=values[1],delta=values[1][0]-values[0][0]))
ops=[]
for arm in [a,b]:
    total=defaultdict(lambda:[0,0])
    for key,v in arm['by_scope_family_class'].items():
        name=key.rsplit(':',1)[1];total[name][0]+=v[0];total[name][1]+=v[1]
    ops.append(total)
op_rows=[]
for name in ops[0].keys()|ops[1].keys():
    old,new=ops[0].get(name,[0,0]),ops[1].get(name,[0,0])
    op_rows.append(dict(operation_class=name,old=old,new=new,delta_count=new[0]-old[0]))
op_rows.sort(key=lambda row:-row['delta_count'])
assert sum(x['delta_count'] for x in op_rows)==1935

native=[]
top={}
for arm in ['N32','N43']:
    cohorts=defaultdict(lambda:[0,0]);choices=[]
    for w in json.loads((DATA/(arm+'_witnesses.json')).read_text()):
        sites=w['possible_source_sites']
        if sites and all(s['source']['operation_class']=='native Pallas' for s in sites):
            files=tuple(sorted(set(k.split(':',1)[0] for s in sites for k in s['source']['sources'])))
            cohorts[files][0]+=w['activities'];cohorts[files][1]+=w['kernel_ns']
        if w['label']=='dycore/advection' and sites and all(s['source']['operation_class']!='native Pallas' for s in sites):
            choices.append(w)
    native.append(cohorts)
    top[arm]=sorted(choices,key=lambda w:-w['kernel_ns'])[:4]
native_rows=[]
for files in native[0].keys()|native[1].keys():
    old,new=native[0].get(files,[0,0]),native[1].get(files,[0,0])
    native_rows.append(dict(files=list(files),old=old,new=new,delta_count=new[0]-old[0]))
native_rows.sort(key=lambda row:-row['delta_count'])

commits={'layout_shield':'6e38cbf173','rc_defaults':'ac278549e6','physics_rk':'150628ddf6',
         'root_scalar_boundary':'0dcd90e126','nest_scalar_final':'659482b8d0','qni_mixing':'a054d46773',
         'psiq_flux':'05a7bac11f','real_psi':'36fe022a0b','mynn_caller':'f771035c95',
         'noah_qsfc':'3814b53fc7','land_q2':'b4b36bcc66','w_damp':'f3c5967654',
         'thompson_mixed':'aaf2e85576','thompson_exits':'b7864e0553'}
commit_info={}
for key,ref in commits.items():
    full,subject=subprocess.check_output(['git','show','-s','--format=%H%n%s',ref],text=True).splitlines()[:2]
    commit_info[key]={'commit':full,'subject':subject}

copy_detail={}
for arm,version in [('N32','v032'),('N43','v043')]:
    x=r['arms'][arm];th=json.loads((L/'ORDINARY_THUNKS'/f'{version}_d01.json').read_text())
    sizes={t['bytes'] for t in th['thunks'] if t['kind']=='copy'};rows=[]
    for group in x['copy_helper_grid_counts']:
        gx,gy,gz,bx,by,bz=group['dimensions']
        candidates=sorted(z for z in sizes if math.ceil(z/(4*bx))==gx and gy==gz==by==bz==1)
        assert candidates
        rows.append({**group,'own_serialized_copy_byte_candidates':candidates})
    explicit=[c for c in x['copy_scope_kind_size_counts'] if c['scope']=='d01' and c['kind']==8]
    copy_detail[arm]={'explicit_d01_DtoD_count':sum(c['count'] for c in explicit),
                      'explicit_d01_DtoD_bytes':sum(c['count']*c['bytes'] for c in explicit),
                      'outside_native_copy_helper_count':sum(c['count'] for c in rows),
                      'copy_thunk_count':sum(t['kind']=='copy' for t in th['thunks']),
                      'static_copy_thunk_byte_sum':sum(t['bytes'] for t in th['thunks'] if t['kind']=='copy'),
                      'helper_own_size_grid_witnesses':rows}
    copy_detail[arm]['copy_activity_count_including_helpers']=copy_detail[arm]['explicit_d01_DtoD_count']+copy_detail[arm]['outside_native_copy_helper_count']
    copy_detail[arm]['inferred_helper_bytes_range']=[sum(v['count']*min(v['own_serialized_copy_byte_candidates']) for v in rows),sum(v['count']*max(v['own_serialized_copy_byte_candidates']) for v in rows)]

result={'schema':'w44-perf.ordinary-delta.v1',
 'scope':'Three complete ordinary PROD roots68-70,dt54/18,44lev. Existing N32/N43 node profiles only; no GPU access/compile. Counts are profile snapshots, not causal wall speed or CPU-WRF fidelity validation.',
 'raw_kernel_counts':[a['count'],b['count']],'kernel_count_delta':1935,'kernel_count_change_pct':1935/a['count']*100,
 'raw_device_activity_counts':[a['device_activity_rows']['total'],b['device_activity_rows']['total']],
 'raw_device_activity_delta':1479,'raw_device_activity_change_pct':1479/a['device_activity_rows']['total']*100,
 'kernel_duration_ms':[a['kernel_ns']/1e6,b['kernel_ns']/1e6],
 'by_callback':scope_rows,'by_source_family':r['delta'],'by_operation_class':op_rows,
 'native_source_cohorts':native_rows,'copy_lowering_witnesses':copy_detail,'source_commits':commit_info,
 'largest_bound_dycore_fusion_sites':top,
 'binding':{arm:{k:r['arms'][arm][k] for k in ['binding_methods','source_execution_bindings','helper_count_ns']} for arm in ['N32','N43']},
 'limits':['294 old/21 new kernel rows outside proven RK-loop module; -273 is not a proven optimization.',
           '138 old/144 new rows have no source frame; shared fusion time stays whole.',
           'Commit labels prove intent/activation, not per-commit counterfactual costs; no ablation.',
           'Earlier FAMILY_TIMES labels missed symbol reuse/small eager executables and are superseded; raw counts unchanged.',
           'Old wall A/A failed; this is not a23% wall regression or an accepted version speed verdict.'],
 'artifact_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'DELTA.json',DATA/'N32_witnesses.json',DATA/'N43_witnesses.json',L/'ACTUAL_HLO/INDEX.json',L/'ORDINARY_THUNKS/INDEX.json']}}
flag_names=['GPUWRF_LAYOUT_PIN','GPUWRF_CARRY_DONATE','GPUWRF_MYNN_QNI_MIXING','GPUWRF_MYNN_PSIQ_FLUX_WRF',
            'GPUWRF_MYNN_SFC_PSI_REAL','GPUWRF_NOAH_QSFC_WRF','GPUWRF_NEST_SCALAR_SPEC_FINAL',
            'GPUWRF_ROOT_SCALAR_BDY_RK1','GPUWRF_PHYS_TEND_RK_WRF','GPUWRF_W_DAMP_STAGE',
            'GPUWRF_THOMPSON_MIXED_PHASE_WRF','GPUWRF_PHY_PREP_REAL_CONST']
result['measured_flag_allowlist']={arm:{key:json.loads((L/'gpu/runs'/arm/'run/receipt.json').read_text())['env'].get(key) for key in flag_names} for arm in ['N32','N43']}
result['actual_hlo_e41']=json.loads((L/'ORDINARY_E41_HLO.json').read_text())
result['unresolved_source']={}
for arm in ['N32','N43']:
    missing=[r['arms'][arm]['family_count_ns'].get(label,[0,0]) for label in ['outside proven RK-loop module','unattributed source']]
    n,ns=sum(v[0] for v in missing),sum(v[1] for v in missing)
    result['unresolved_source'][arm]={'count':n,'kernel_ms':ns/1e6,'kernel_time_fraction_pct':ns/r['arms'][arm]['kernel_ns']*100}
(OUT/'receipts/ORDINARY_ROOT_DELTA.json').write_text(json.dumps(result,indent=2)+'\n')

labels={
 'dycore/advection':('Dycore/Advektion','150628ddf6, f3c5967654, 6e38cbf173','Physik→RK, W-Dämpfung sowie Layout/Fission; Q1 ohne qualifizierten Prototyp geschlossen'),
 'graph helper outside model CUBIN':('Graph-Helfer außerhalb Modell-CUBIN','kein einzelner kausaler Commit','405 davon Kopierpfad; keine zusätzliche Physik, nicht auf Anzahl optimieren'),
 'Noah-MP':('Noah-MP','3814b53fc7, b4b36bcc66, 6e38cbf173','QSFC/Q2 korrekt halten; Q2 No-Go ohne konkreten Export-Redundanzbeleg'),
 'boundary/nesting':('Randbedingungen im Step','0dcd90e126, 659482b8d0; Defaults 6e38cbf173','WRF-RK-/spec_bdy_final-Takt behalten'),
 'runtime/glue source':('Runtime-/Staging-Quellen','mehrere, nicht isoliert','kein eindeutig neues Physikverfahren'),
 'MYNN':('MYNN','a054d46773/ac278549e6, f771035c95','qni-Mischung, WRF-Caller und Konstanten behalten'),
 'surface-layer':('Oberflächenschicht','05a7bac11f/ac278549e6, 36fe022a0b, f771035c95','PSIQ, REAL-PSI und Callergrenzen behalten'),
 'scalar control':('Skalare HLO-Steuerung','kein einzelner kausaler Commit','Zähler/Steuerung, nicht neue Flächengleichungen'),
 'mixed/shared fusion':('Gemischte/geteilte Fusionen','mehrere Quellen pro Fusion','Zeit bleibt ungeteilt; kein Name→Familie-Transfer'),
 'nesting/force call scope':('Force-down-Aufrufbereich','6e38cbf173','kompakte Streifen-Fences; nur 0.057 ms Mehrzeit, geringe ROI'),
 'layout-only source':('Nur Layout-Quellen','6e38cbf173','Root-Pins sind Compiler-Schutz; kleines Gitter-Firewall hier inaktiv'),
 'hydrostatic-prep':('Hydrostatische Vorbereitung','6e38cbf173','nativer REAL-Pfad ersetzt Vorbereitung'),
 'unattributed source':('Quellframe nicht zuordenbar','offen','sichtbarer Rest, nicht einer Familie zugeschlagen'),
 'Thompson':('Thompson','aaf2e85576, b7864e0553, 6e38cbf173','weniger Familien-Kernel, mehr Zeit; Q3 ohne materiellen Gewinnbeleg geschlossen'),
 'gravity-wave-drag':('Schwerewellenwiderstand','mehrere','kleine Fusions-/Zuordnungsverschiebung'),
 'physics preparation/coupling':('Physik-Vorbereitung/Kopplung','150628ddf6, 6e38cbf173','coupling/ ist nicht pauschal Randbedingung'),
 'outside proven RK-loop module':('Präfix/Epilog/kleine Executables nicht gebunden','offen','-273 keine isoliert belegte Rückholung')}
lines=['# Ordinary-Root-Kernel-Delta: v0.3.2 → v0.4.3','',
 '**CPU-only, vorhandene Artefakte; keine neue GPU-Messung.** Drei vollständige PROD-Roots 68–70,',
 'd01/d02, 44 Level, dt 54/18. Quellkohorten sind beobachtete Ausführungen; Commit-Zuordnungen',
 'belegen den Änderungszweck, keine ohne Ablation isolierten Kosten.','',
 '| Metrik [M Trace] | v0.3.2 | v0.4.3 | Delta |','|---|---:|---:|---:|',
 f'| Kernelzeilen | 8,292 | 10,227 | +1,935 / +{result["kernel_count_change_pct"]:.2f}% |',
 f'| Kernel+Memcpy+Memset-Zeilen | 10,554 | 12,033 | +1,479 / +{result["raw_device_activity_change_pct"]:.2f}% |',
 '| Summe Kernelzeit ms | 93.476 | 103.146 | +9.669 / +10.34% |','',
 '**+23% ist keine Wallclock-Regression.** Die alte A/A-Regel ist verletzt; repeat_v4 klärt den',
 'Zeitvergleich. JAX-vs-JAX dient hier der Programmbindung, nicht der WRF-Fidelity.','',
 '| Aufrufbereich [M] | alt | neu | Mehrkernel | Anteil Netto-Delta |','|---|---:|---:|---:|---:|']
for s in scope_rows:lines.append(f'| {s["scope"]} | {s["old"][0]:,} | {s["new"][0]:,} | {s["delta"]:+,} | {s["delta"]/1935*100:.1f}% |')
lines+=['','## Quellengebundene Familien','',
 '| Quellkohorte | alt | neu | Δ Kernel | Δ ms | Commit-/Zweckbeleg; Entscheidung |','|---|---:|---:|---:|---:|---|']
for row in r['delta']:
    name,comm,intent=labels[row['family']]
    lines.append(f'| {name} | {row["old_count"]:,} | {row["new_count"]:,} | {row["delta_count"]:+,} | {row["delta_ms"]:+.3f} | {comm} — {intent} |')
lines+=['| **Summe** | **8,292** | **10,227** | **+1,935** | **+9.669** | jede Aktivität genau einmal |','',
 '138/144 Aktivitäten ohne Quellframe sowie 294/21 nicht modulgebundene Präfix-/Epilog-Aktivitäten bleiben',
 'unaufgelöst. Ein Teil der Familienverschiebung kann aus Fusion/Graph-Gruppierung kommen; die Tabelle',
 'erklärt nicht jeden Mehrkernel als zusätzliche Gleichung. Der unaufgelöste Zeitanteil beträgt alt',
 '0,854 ms (0,91%), neu 0,303 ms (0,29%); die Count-Unsicherheit ist größer als die Zeit-Unsicherheit.','',
 '## Was die 1.935 Mehrkernel technisch sind','',
 '| Operatorklasse innerhalb gebundener Programme [M] | alt | neu | Δ |','|---|---:|---:|---:|']
for row in op_rows:lines.append(f'| {row["operation_class"]} | {row["old"][0]:,} | {row["new"][0]:,} | {row["delta_count"]:+,} |')
lines+=['',
 'Nur **81 der 1.935** Mehrkernel sind zusätzliche native Pallas-Aufrufe (+4.19% des Netto-Deltas).',
 'Deren Kernelzeit steigt nur 47.966→48.394 ms. XLA-Array-/Staging-Fusionen steigen 5,367→6,693:',
 'Fission/Staging plus bewusste Caller-/Physikänderung, keine 1.326 neuen unabhängigen WRF-Prozesse.','',
 '## Kopierpfad: 405 Mehrkernel sind nicht 405 neue Kopien','',
 '- Graph-Helfer außerhalb Modell-CUBIN: 18→516; davon memcpy32_post 12→417 (+405),',
 '  set_while_condition 0→90 und set_case_condition 6→9.',
 '- Alle drei Helfersymbole fehlen im **exakt geladenen Modell-CUBIN**. Graph-Erzeugung/Klonkette',
 '  bindet sie an d01. Sie werden keiner Physikfamilie nach Symbolnamen zugeschrieben.',
 '- d01: explizite DtoD 417→6, Kopierkernel 12→417, zusammen **429→423**. Kopier-Thunk-Templates 137→139.',
 '  Grid-/Blockgrößen passen zu eigenen serialisierten Kopiergrößen; große Größen eindeutig.',
 '  Das stützt eine andere Kopierabbildung [I], keine zusätzlichen Gleichungen.',
 '- Der explizite d01-DtoD-Zähler fällt 238.932 MB→24 B; große Kopien erscheinen nun als Kernel.',
 '  Diese Speicherbewegung verschwindet nicht. Der Kopierpfad-Zeitunterschied beträgt rund 0.222 ms',
 '  je drei Roots ≈0.005 s/fc-h [I]. WHILE-Graphen nur für weniger Zählerzeilen abzuschalten ist keine schnelle Rückholung.','',
 '## Konkrete native Pfadänderungen','',
 '| Quelle, tatsächliche Runtime-Thunk-Bindung [M] | Aufrufe alt→neu | Bedeutung |','|---|---:|---|',
 '| dyn_rhsph_fp32.py | 0→36 | 6e38cbf173 erweitert den nativen RHS_PH-Pfad auch ohne top_lid; Ersatz eines vorhandenen Stencils. |',
 '| dyn_ring_relax_fp32.py | 63→75 | Root-/Nest-Ränder: 0dcd90e126,659482b8d0; Aktivierung6e38cbf173. Takt behalten. |',
 '| dyn_boundary_fp32.py | 153→150 | kleine Pfad-/Fusionsänderung, kein isolierter kausaler Revert. |',
 '| phys_hydrostatic_real.py | 0→12 | nativer REAL-Physikvorbereitungspfad in 6e38cbf173. |',
 '| phys_noahmp_radiation.py | 0→12 | nativer Noah-Strahlungspfad in 6e38cbf173 ersetzt Ausdruckspfad. |',
 '| phys_mynn_tridiag.py | 96→108 | zusätzliche Mischung konsistent mit qni: a054d46773/ac278549e6; keine per-Flag-Ablation behauptet. |',
 '| phys_thompson_full.py | 12→12 | Anzahl gleich, native Zeit 2.914→4.934 ms; Mixed/REAL/Driver-Copyback bleiben. |',
 '| dyn_acoustic_fp32.py | 528→528 | keine zusätzlichen nativen akustischen Aufrufe. |','',
 '## E82/E195-Bindung','',
 '1. Cheap-Keys aus dem gemessenen nested_pipeline_run.json wählen den tatsächlichen AOT-Blob.',
 '   ELF, Interface und **alle Ausführungsfelder außerhalb des behaltenen HLO** stimmen mit dem HLO-Spender überein,',
 '   für beide Versionen/beide Domains. Nicht nur ein gleiches Kernelsymbol wird vorausgesetzt.',
 '2. Runtime-Thunks liefern HLO-Instruktion, Symbol und sechs Launch-Dimensionen. NVTX-Aufrufstellen',
 '   und Graph-Erzeugungs-/Klonbeziehungen liefern Kontext. Geteilte Symbole behalten alle Quellstellen.',
 '   Eine Aktivität wird genau einmal gezählt; Zeit wird nicht nach Quell-Instruktionszahlen aufgeteilt.',
 '3. while.793 kollidiert alt zwischen d01-Einstieg und innerer d02-Schleife: äußerster Einstieg entscheidet.',
 '   loop_add_fusion ist kein McICA-Beleg: geteilter skalarer Zähler. Frühere FAMILY_TIMES-Familienlabels',
 '   werden ersetzt; rohe Counts/Zeiten bleiben gleich.',
 '4. SQLite: nur Aktivitäts-/NVTX-/Graph-Spalten und referenzierte kurze Strings mit Längengrenzen.',
 '   Kein TARGET_INFO,SELECT-Stern oder IR-/Environment-Dump.',
 '5. Kleine-Gitter-Firewall <=4,096 Spalten ist bei PROD 8,400/31,239 inaktiv und erklärt hier nichts.',
 '   Carry-Donation ist bereits in beiden Messungen an. Am exakt gebundenen reichen GPU-HLO sind',
 '   alte kInput-Hero-Fusionen >500: d01=8/d02=2; aktuell beide0. Der Root-Pin-/Firewall-Schutz',
 '   darf nicht für einen kleineren Kernelzähler entfernt werden.','',
 '[Zahlen/Bindungen/Commits](receipts/ORDINARY_ROOT_DELTA.json) · [Vier-Augen-Plan](ORDINARY_RECOVERY_PLAN.md).',
 'Rohdaten bleiben privat unter den referenzierten Pfaden. Kein neuer Compile und kein GPU-Zugriff.']
(OUT/'ORDINARY_ROOT_DELTA.md').write_text('\n'.join(lines)+'\n')
print('published',len(r['delta']),'source rows; total1935; total-device delta1479')

