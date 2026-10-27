# CUDA-Mini-Paar für Repeat und Hostprofil

**Status 2026-10-24T21:17Z: CUDA MINI NOT TONIGHT** (MAILBOX). Der 16×16-
Ausschnitt ändert alle Programmformen; jede der drei Versionen bräuchte auf der
GPU einen Kaltcompile (gemessen v0.4.3 kalt 390 s) → ≤5 min ist ehrlich nicht
erreichbar. Vorschlag 25.10: PROD-Gitter-Warm-Mini über denselben v5-Builder
(ein v0.4.3-Arm auf dem bezahlten Cache, erwartet ≈1,5 min, Cap 5). Der
CPU-Hostprofil-Probelauf 17:08Z war PASS (E41 38 s), aber `cuda_cleanup.py`
scheiterte an overlayfs `work/work` mit Modus 000; korrigiert (Kindverzeichnisse
vor dem Abstieg öffnen) und am Restlauf geprüft.

**Vorbereitung, noch kein CUDA-PASS.** Ein CPU-Probelauf ersetzt keinen echten
CUDA-Mini-Lauf. Die endgültigen Einträge werden erst nach allen erfolgreichen
CPU-Phasen als CUDA MINI READY an den Manager gemeldet; der Manager beantragt
und startet das gemeinsame Smoke-Fenster.

## Gemeinsamer Builder und dokumentierter Diff

Die neuen Produktions- und Mini-Einträge verwenden denselben eingefrorenen
launch_v5.sh → repeat_session.sh/host_session.sh → arm.sh → capture.py-Pfad.
Die bisherigen v4-Einträge bleiben unverändert. Beide neuen Varianten haben
dieselben zusätzlichen Nachphasen; es gibt keine Mini-exklusive Ersatzpipeline.

| Parameter | Produktion | CUDA-Mini |
|---|---|---|
| Horizontales Gitter | d01 120×70, d02 267×117 | d01 16×16, d02 15×15 |
| Eingaben | ursprünglicher PROD-Fall | Inhalts-Hash-gebundener Ausschnitt derselben echten Dateien |
| Vertikal/Physik/dt | 44 Level; Thompson/RRTMG/MYNN/Noah-MP/KF(d01); 54/18 s | identisch |
| Dauer/Arme | 3 h; gleiche zehn Repeat-Arme bzw. ein H43-Arm | identisch |
| Host-Profilfenster | Segment 2 mit demselben Observer/NSYS-Builder | identisch |
| Child-Modellumgebung | explizit cuda, cuda_async, preallocate=true, fraction=0.25; gleiche Quelle/Caches/Flags | identisch; nur W44-Grid-/Output-Namensraum unterscheidet sich |
| Zeitbox | Repeat 600 s, Host 180 s | 300 s Gesamtreserve, Supervisor-Cap 270 s plus kurze Teardown-Reserve |

Ein öffentlicher Wrapper unterscheidet sich vom zugehörigen Produktionswrapper
nur im Größenargument prod/cuda_mini. Die Input-Pfadänderung repräsentiert das
kleinere Gitter; Ausgaben und private Cache-Overlay-Sichten bekommen getrennte
Run-Verzeichnisse. Absolute Modell-Cache-Pfade im Child bleiben unverändert.
Die Mini-Grenzstreifen sind entlang ihrer Tangente beschnittene originale
WRF-Randdaten. Das ist Launch-Abdeckung, kein geografisch vollständiger oder
gegen CPU-WRF qualifizierter Mini-Forecast.

## Wirklich durchlaufene Phasen

1. Inhalts-Siegel und Eingaben prüfen; Ga-Zeitfenster/Platten-Floors prüfen.
2. Ganzen Baum mit dem unverändert übernommenen Subreaper aus w44-params
   überwachen. **Eigene sequenzielle MemoryMax=12G/MemorySwapMax=0-cgroup**,
   keine Shared-RAM-Tickets für GPU-Eintrag, E41 oder Checker (Manager 15:09Z).
3. Im CUDA-Modus offizielle GPU-Lock-Prüfung; dieselben Source-Guards,
   Child-Umgebung und Modell-Commands für beide Größen.
4. Reale private Overlay-Sicht; bei v0.4.0/v0.4.3 tatsächlicher Bundle-Consumer
   mit beiden echten Bundle-Checkern. v0.3.2 bekommt eine private Overlay-Sicht
   seines eigenen bestehenden Caches.
5. Alle Repeat-Arme bzw. H43 einschließlich tatsächlichem Modellaufruf;
   derselbe Summary-/Receipt-Parser. CUDA-Ausgaben werden auf Endlichkeit geprüft.
6. E41 mit explizitem CPU/PYTHONPATH: Aus dem realen Run-Proof ausgewählte AOT-
   Programme werden an reichhaltige JIT-Spender gebunden (Records 0/1/2 und
   alle Nicht-HLO-Restfelder bytegleich). Danach echter wave_cubin_gate-Scanner,
   unveränderte Stack/Callee/kInput-Regeln und vollständige Domain-Abdeckung.
7. Nach vollständigem Reaping nur eigene Overlay-/Scratch-/NetCDF-Dateien
   entfernen; Output-SHA256 bleiben erhalten. Quellen, Fallreferenzen und
   bezahlte Cache-Unterbäume bleiben unberührt.

## CPU-Probelauf und Grenzen

Der vollständige CPU-Probelauf startet denselben öffentlichen Mini-Wrapper
mit --cpu-probe. Die Produkt-Child-Umgebung wird vor der Geräte-Substitution
erfasst, einschließlich explizitem cuda. Nur CUDA-Gerät/NVML/NSYS werden
ersetzt: echtes unverändertes Thompson-Modell auf einer echten PROD-1×44-
Regensäule, tatsächlicher Host-Observer und unveränderte CPU-Verweigerung
des GPU-Harness. Die Receipt-Prüfung behandelt diese Ergebnisse ausdrücklich
als CPU-Abdeckung und erfindet kein GPU-Feld.

Die CUDA-Artefakt-Substitution für die CPU-E41-Phase ist ebenso ausdrücklich:
ein vorhandener exakt gebundener v0.4.3-GPU-Spender mit seinem echten Run-Proof
wird vom wirklichen Scanner geprüft. Der spätere echte CUDA-Mini-Lauf prüft
seine **eigenen** Programme. CPU-Erfolg ist keine Mini-Codegen-/GPU-/Fidelity-
Qualifikation. Die echte Mini-Loader-/Force-down-Initialisierung wurde separat
auf CPU geprüft: 16×16/15×15×44, endlich, 98,868 s.

## Siegel, Ressourcen und offene Freigaben

Alle eigenen Siegel-/Receipt-Identitäten sind Inhalts-SHA256. Die vorhandenen
st_dev/st_ino-Stellen sind aktuelle Hardlink-/FD-Sicherheitsprüfungen oder
prozesslokale Alias-Memos; sie vergleichen keine gespeicherten Identitäten
über einen Neustart. Output-mtime bleibt eine frisch gemessene Zeitmetrik,
kein Identitätssiegel.

Neue schwere CPU-Proben nur bei freiem QUIET-/GPU-window-Flag und <DATA_ROOT>
≥365 GiB; CPU-Proben können separat einen Shared-RAM-Slot reservieren. Die
GPU-Einträge selbst haben diese Warteschlange nicht. Mini-Transienten sind
auf unter 3 GB geschätzt; ein voller Repeat hält bis zu etwa 8 GB Outputs bis
zum Checker und wurde separat als DISK-PLAN zur Managerfreigabe angemeldet.

Die historische E41-Policy des v0.3.2-/v0.4.0-Vergleichs wurde beim Manager
nachgefragt: frühere Baselines haben dokumentiert FAIL/INCOMPLETE, sind aber
keine Releasekandidaten. Der Entwurf prüft aktuell alle Varianten streng;
keine historische Ausnahme wird stillschweigend eingeführt. v0.4.3 bleibt
streng. README-A/A und der eigentliche Perf-Slot am 25.10 bleiben offen.
