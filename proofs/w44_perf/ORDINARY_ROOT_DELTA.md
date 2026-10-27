# Ordinary-Root-Kernel-Delta: v0.3.2 → v0.4.3

**CPU-only, vorhandene Artefakte; keine neue GPU-Messung.** Drei vollständige PROD-Roots 68–70,
d01/d02, 44 Level, dt 54/18. Quellkohorten sind beobachtete Ausführungen; Commit-Zuordnungen
belegen den Änderungszweck, keine ohne Ablation isolierten Kosten.

| Metrik [M Trace] | v0.3.2 | v0.4.3 | Delta |
|---|---:|---:|---:|
| Kernelzeilen | 8,292 | 10,227 | +1,935 / +23.34% |
| Kernel+Memcpy+Memset-Zeilen | 10,554 | 12,033 | +1,479 / +14.01% |
| Summe Kernelzeit ms | 93.476 | 103.146 | +9.669 / +10.34% |

**+23% ist keine Wallclock-Regression.** Die alte A/A-Regel ist verletzt; repeat_v4 klärt den
Zeitvergleich. JAX-vs-JAX dient hier der Programmbindung, nicht der WRF-Fidelity.

| Aufrufbereich [M] | alt | neu | Mehrkernel | Anteil Netto-Delta |
|---|---:|---:|---:|---:|
| d01 | 2,586 | 4,005 | +1,419 | 73.3% |
| d02 | 5,409 | 5,865 | +456 | 23.6% |
| nesting/force | 297 | 357 | +60 | 3.1% |

## Quellengebundene Familien

| Quellkohorte | alt | neu | Δ Kernel | Δ ms | Commit-/Zweckbeleg; Entscheidung |
|---|---:|---:|---:|---:|---|
| Dycore/Advektion | 3,132 | 3,681 | +549 | +4.683 | 150628ddf6, f3c5967654, 6e38cbf173 — Physik→RK, W-Dämpfung sowie Layout/Fission; Q1 ohne qualifizierten Prototyp geschlossen |
| Graph-Helfer außerhalb Modell-CUBIN | 18 | 516 | +498 | +0.824 | kein einzelner kausaler Commit — 405 davon Kopierpfad; keine zusätzliche Physik, nicht auf Anzahl optimieren |
| Noah-MP | 345 | 723 | +378 | +0.707 | 3814b53fc7, b4b36bcc66, 6e38cbf173 — QSFC/Q2 korrekt halten; Q2 No-Go ohne konkreten Export-Redundanzbeleg |
| Randbedingungen im Step | 678 | 894 | +216 | +0.975 | 0dcd90e126, 659482b8d0; Defaults 6e38cbf173 — WRF-RK-/spec_bdy_final-Takt behalten |
| Runtime-/Staging-Quellen | 288 | 414 | +126 | -0.502 | mehrere, nicht isoliert — kein eindeutig neues Physikverfahren |
| MYNN | 1,068 | 1,188 | +120 | +1.876 | a054d46773/ac278549e6, f771035c95 — qni-Mischung, WRF-Caller und Konstanten behalten |
| Oberflächenschicht | 264 | 351 | +87 | +0.300 | 05a7bac11f/ac278549e6, 36fe022a0b, f771035c95 — PSIQ, REAL-PSI und Callergrenzen behalten |
| Skalare HLO-Steuerung | 342 | 426 | +84 | +0.069 | kein einzelner kausaler Commit — Zähler/Steuerung, nicht neue Flächengleichungen |
| Gemischte/geteilte Fusionen | 1,179 | 1,245 | +66 | -0.567 | mehrere Quellen pro Fusion — Zeit bleibt ungeteilt; kein Name→Familie-Transfer |
| Force-down-Aufrufbereich | 297 | 357 | +60 | +0.057 | 6e38cbf173 — kompakte Streifen-Fences; nur 0.057 ms Mehrzeit, geringe ROI |
| Nur Layout-Quellen | 90 | 117 | +27 | +0.261 | 6e38cbf173 — Root-Pins sind Compiler-Schutz; kleines Gitter-Firewall hier inaktiv |
| Hydrostatische Vorbereitung | 0 | 12 | +12 | +0.368 | 6e38cbf173 — nativer REAL-Pfad ersetzt Vorbereitung |
| Quellframe nicht zuordenbar | 138 | 144 | +6 | -0.029 | offen — sichtbarer Rest, nicht einer Familie zugeschlagen |
| Physik-Vorbereitung/Kopplung | 42 | 39 | -3 | -0.081 | 150628ddf6, 6e38cbf173 — coupling/ ist nicht pauschal Randbedingung |
| Schwerewellenwiderstand | 54 | 48 | -6 | +0.017 | mehrere — kleine Fusions-/Zuordnungsverschiebung |
| Thompson | 63 | 51 | -12 | +1.235 | aaf2e85576, b7864e0553, 6e38cbf173 — weniger Familien-Kernel, mehr Zeit; Q3 ohne materiellen Gewinnbeleg geschlossen |
| Präfix/Epilog/kleine Executables nicht gebunden | 294 | 21 | -273 | -0.522 | offen — -273 keine isoliert belegte Rückholung |
| **Summe** | **8,292** | **10,227** | **+1,935** | **+9.669** | jede Aktivität genau einmal |

138/144 Aktivitäten ohne Quellframe sowie 294/21 nicht modulgebundene Präfix-/Epilog-Aktivitäten bleiben
unaufgelöst. Ein Teil der Familienverschiebung kann aus Fusion/Graph-Gruppierung kommen; die Tabelle
erklärt nicht jeden Mehrkernel als zusätzliche Gleichung. Der unaufgelöste Zeitanteil beträgt alt
0,854 ms (0,91%), neu 0,303 ms (0,29%); die Count-Unsicherheit ist größer als die Zeit-Unsicherheit.

## Was die 1.935 Mehrkernel technisch sind

| Operatorklasse innerhalb gebundener Programme [M] | alt | neu | Δ |
|---|---:|---:|---:|
| fusion arithmetic | 5,367 | 6,693 | +1,326 |
| unknown | 609 | 894 | +285 |
| layout/view | 225 | 378 | +153 |
| scalar control | 342 | 426 | +84 |
| native Pallas | 1,707 | 1,788 | +81 |
| layout/cast | 42 | 48 | +6 |

Nur **81 der 1.935** Mehrkernel sind zusätzliche native Pallas-Aufrufe (+4.19% des Netto-Deltas).
Deren Kernelzeit steigt nur 47.966→48.394 ms. XLA-Array-/Staging-Fusionen steigen 5,367→6,693:
Fission/Staging plus bewusste Caller-/Physikänderung, keine 1.326 neuen unabhängigen WRF-Prozesse.

## Kopierpfad: 405 Mehrkernel sind nicht 405 neue Kopien

- Graph-Helfer außerhalb Modell-CUBIN: 18→516; davon memcpy32_post 12→417 (+405),
  set_while_condition 0→90 und set_case_condition 6→9.
- Alle drei Helfersymbole fehlen im **exakt geladenen Modell-CUBIN**. Graph-Erzeugung/Klonkette
  bindet sie an d01. Sie werden keiner Physikfamilie nach Symbolnamen zugeschrieben.
- d01: explizite DtoD 417→6, Kopierkernel 12→417, zusammen **429→423**. Kopier-Thunk-Templates 137→139.
  Grid-/Blockgrößen passen zu eigenen serialisierten Kopiergrößen; große Größen eindeutig.
  Das stützt eine andere Kopierabbildung [I], keine zusätzlichen Gleichungen.
- Der explizite d01-DtoD-Zähler fällt 238.932 MB→24 B; große Kopien erscheinen nun als Kernel.
  Diese Speicherbewegung verschwindet nicht. Der Kopierpfad-Zeitunterschied beträgt rund 0.222 ms
  je drei Roots ≈0.005 s/fc-h [I]. WHILE-Graphen nur für weniger Zählerzeilen abzuschalten ist keine schnelle Rückholung.

## Konkrete native Pfadänderungen

| Quelle, tatsächliche Runtime-Thunk-Bindung [M] | Aufrufe alt→neu | Bedeutung |
|---|---:|---|
| dyn_rhsph_fp32.py | 0→36 | 6e38cbf173 erweitert den nativen RHS_PH-Pfad auch ohne top_lid; Ersatz eines vorhandenen Stencils. |
| dyn_ring_relax_fp32.py | 63→75 | Root-/Nest-Ränder: 0dcd90e126,659482b8d0; Aktivierung6e38cbf173. Takt behalten. |
| dyn_boundary_fp32.py | 153→150 | kleine Pfad-/Fusionsänderung, kein isolierter kausaler Revert. |
| phys_hydrostatic_real.py | 0→12 | nativer REAL-Physikvorbereitungspfad in 6e38cbf173. |
| phys_noahmp_radiation.py | 0→12 | nativer Noah-Strahlungspfad in 6e38cbf173 ersetzt Ausdruckspfad. |
| phys_mynn_tridiag.py | 96→108 | zusätzliche Mischung konsistent mit qni: a054d46773/ac278549e6; keine per-Flag-Ablation behauptet. |
| phys_thompson_full.py | 12→12 | Anzahl gleich, native Zeit 2.914→4.934 ms; Mixed/REAL/Driver-Copyback bleiben. |
| dyn_acoustic_fp32.py | 528→528 | keine zusätzlichen nativen akustischen Aufrufe. |

## E82/E195-Bindung

1. Cheap-Keys aus dem gemessenen nested_pipeline_run.json wählen den tatsächlichen AOT-Blob.
   ELF, Interface und **alle Ausführungsfelder außerhalb des behaltenen HLO** stimmen mit dem HLO-Spender überein,
   für beide Versionen/beide Domains. Nicht nur ein gleiches Kernelsymbol wird vorausgesetzt.
2. Runtime-Thunks liefern HLO-Instruktion, Symbol und sechs Launch-Dimensionen. NVTX-Aufrufstellen
   und Graph-Erzeugungs-/Klonbeziehungen liefern Kontext. Geteilte Symbole behalten alle Quellstellen.
   Eine Aktivität wird genau einmal gezählt; Zeit wird nicht nach Quell-Instruktionszahlen aufgeteilt.
3. while.793 kollidiert alt zwischen d01-Einstieg und innerer d02-Schleife: äußerster Einstieg entscheidet.
   loop_add_fusion ist kein McICA-Beleg: geteilter skalarer Zähler. Frühere FAMILY_TIMES-Familienlabels
   werden ersetzt; rohe Counts/Zeiten bleiben gleich.
4. SQLite: nur Aktivitäts-/NVTX-/Graph-Spalten und referenzierte kurze Strings mit Längengrenzen.
   Kein TARGET_INFO,SELECT-Stern oder IR-/Environment-Dump.
5. Kleine-Gitter-Firewall <=4,096 Spalten ist bei PROD 8,400/31,239 inaktiv und erklärt hier nichts.
   Carry-Donation ist bereits in beiden Messungen an. Am exakt gebundenen reichen GPU-HLO sind
   alte kInput-Hero-Fusionen >500: d01=8/d02=2; aktuell beide0. Der Root-Pin-/Firewall-Schutz
   darf nicht für einen kleineren Kernelzähler entfernt werden.

[Zahlen/Bindungen/Commits](receipts/ORDINARY_ROOT_DELTA.json) · [Vier-Augen-Plan](ORDINARY_RECOVERY_PLAN.md).
Rohdaten bleiben privat unter den referenzierten Pfaden. Kein neuer Compile und kein GPU-Zugriff.
