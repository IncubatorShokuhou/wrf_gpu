# Q3: vorhandener Warm-/Cold-Selektor

**Entscheidung: kein qualifizierter Prototyp.** Nach dem Neustart am
2026-10-24 um 14:46Z wieder aufgenommen; bestehendes PTX um 14:48Z erneut
vollständig gelesen und Entscheidung um 14:49:49Z versiegelt, innerhalb
20 Minuten. Kein Lower/Compile und kein GPU-Zugriff. Der eine erlaubte
Prototyp wurde weder für Q1 noch für Q3 verbraucht.

Im tatsächlich geladenen v0.4.3-d02-Programm sind zwei getrennte Integer-MAX-
Reduktionen für `warm_any` und `cold_any` vorhanden. Das ist eine konkrete
Codegen-Beobachtung, über die bisherige statische BAR-Gesamtzahl hinaus:

| PTX-Body-Zeile, nullbasiert | Vorgang |
|---|---|
| 574–575 | Warm-Prädikat mit gültigen Levels (`k<44`) und Cloud/Rain > R1 |
| 576, 578–582 | eigene Barrier, `redux.sync.max.s32`, Shared-Store, Barrier, Load und Shuffle |
| 583–584 | Cold-Prädikat mit gültigen Levels, Ice/Snow/Graupel sowie Cold-/Nukleationspfad |
| 585–590 | zweiter entsprechender MAX-/Shared-/Barrier-/Shuffle-Pfad |

Die Prädikat-Def-use-Kette enthält dieselben Schwellen wie die gebundene
Quelle: R1=1e-12, T0=273,15 K, 253,15 K und Nukleation 0,2. Die Integer-MAX-
Instruktionen stehen bei 578/586; lokal sind vier `bar.sync` sichtbar.
Das PTX hat keine Source-Line-Debug-Tabelle. Die Zuordnung verwendet deshalb
die tatsächlichen Prädikate/Operanden und deren Schwellen, keine Übertragung
von Kernelnamen. Record 1 (PTX) und Record 2 (native ELF) sind durch die
vollständige Ausführungsbindung an den geladenen AOT-Blob gebunden.

Der dreistufige Integer-MAX-Selektor ist logisch äquivalent; Astras 512
Maskenkombinationen sind bereits geprüft und werden nicht wiederholt.
Ein geänderter Compiler könnte damit einen der beiden Reduktions-/Shared-
Pfade entfernen. **Es ist noch kein Sync aus einem geänderten Artefakt entfernt
oder als Zeitgewinn gemessen.** Die aktuelle ganze Thompson-Funktion hat
179 PTX-`bar.sync`, das fertige SASS 341 BAR-Instruktionen. Beide statischen
Zahlen sind keine dynamische Sync-Zeit; ein Verhältnis daraus ist kein Speedup.

Für ≥0,5 ms/root Nettogewinn gibt es keinen Beleg. Insbesondere gehört die
gesamte native Thompson-Versionsdifferenz von 2,020324 ms/drei Roots nicht
nachweislich zu diesem Selektor. Auch Register, Spills, endgültige Syncs und
kompilierte BYTE-/E41-Ergebnisse einer geänderten Variante sind unbekannt.
Unter der bindenden materiellen Aufwandsschwelle wird daher kein Prototyp,
CPU-Lower oder zusätzlicher GPU-Launcher begonnen. Das ist ein begrenztes
negatives Ergebnis, keine Behauptung, dass die Zusammenlegung niemals hilft.

Q2 bleibt No-Go: Der gespeicherte kurze Def-use-Screen zeigt für die
Noah-/Surface-Kohorten 176 gebundene Fusionsaufrufstellen mit 375 Ausgaben,
ohne tote SSA-Ausgabe oder exakt doppelt ausgegebene Berechnung. Er belegt
keinen konkreten toten/doppelten EF-/ET-Export und rechtfertigt keinen
Barrierensweep. Die EF/ET-, QSFC- und Root-Schutzgrenzen bleiben bestehen.
Der Screen ist eine Quellenanalyse, kein physischer Puffer-/Traffic-Beweis.

[Q3-Entscheidungsbeleg](receipts/Q3_SELECTOR_DECISION.json) ·
[PTX-Screen](q3_existing_ptx_audit.py) ·
[gespeicherter Q2-Kurzscreen](receipts/Q2_EXISTING_LIVENESS.json).
Keine neuen Caches oder Snapshots. Vorhandene PTX-/HLO-Def-use-Belege bleiben
bis zur Astra-Prüfung privat unter den im Receipt hashgebundenen Pfaden.
