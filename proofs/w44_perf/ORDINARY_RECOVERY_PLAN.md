# Vier-Augen-Plan: Ordinary-Root-Rückholung

Verbindlicher Stand: Manager 2026-10-24 07:02Z und nach Neustart 14:50Z;
Astra-Planreview 8f7ef66e45 zu 0cce3bb6a8.
Review.
Die Quellenbindung und geschlossene Zählung sind akzeptiert. Die frühere
Gewinnsprache und Reihenfolge dieses Plans sind durch den folgenden Stand ersetzt.

**Ergebnis: kein qualifizierter materieller Prototyp.** Q1 ist ohne Prototyp
geschlossen, Q3 ebenfalls; Q2 bleibt No-Go. Es gab keinen Modell-Lower/Compile
und keinen eigenen GPU-Zugriff. Der eine erlaubte Prototyp-/Lower-Arm blieb ungenutzt.

| Reihenfolge | Bindende Grenze | Ergebnis und Beleg |
|---|---|---|
| Q1 | Cross-Fusion-/Liveness-Audit bestehender Artefakte ≤45 min. Nur konkrete identische Operanden oder redundante materialisierte Producer derselben RK-Stufe; keine neuen Vollfeld-Temporaries/längere schädliche Liveness | 07:01–07:08Z; 366 gebundene Aufrufstellen/566 Ausgaben, alle SSA-live, keine exakt doppelt ausgegebene Fusion. d02-Arithmetikteilung benötigt neue Vollfelder bis 32,99 MB. Bereits ausgegebene d01-Zielkohorte insgesamt nur 0,045536 ms/root. [Audit](Q1_CROSS_FUSION_AUDIT.md). |
| Q3, vor Q2 | Kurze Reserve ≤20 min; zwei reale Reduktions-/Sync-Pfade belegen. Statische BAR-Zahlen und gesamte Thompson-Versionsdifferenz sind kein Gewinnbeleg | Nach Neustart innerhalb vier Minuten abgeschlossen. Zwei konkrete MAX-/Shared-/Barrier-Pfade am gebundenen PTX bestätigt. Keine ≥0,5-ms/root-Nettowirkung belegt; kein Prototyp. [Audit](Q3_SELECTOR_AUDIT.md). |
| Q2 | Default-No-Go. Nur ≤15-min-Bestandsaudit auf einen konkreten toten/doppelten EF-/ET-Export; kein weiterer Lower-/GPU-Arm | Gespeicherter kleiner Screen: 176 Noah-/Surface-Fusionsaufrufstellen, 375 Ausgaben, keine tote SSA-Ausgabe oder exakt doppelte Ausgabe. Kein EF-/ET-Revert gerechtfertigt. [Receipt](receipts/Q2_EXISTING_LIVENESS.json). |

Die +7,099678 ms nichtnative Dycore-Zeit, +1,007225 ms Noah+Surface und
+2,020324 ms native Thompson-Zeit je drei Roots sind **beobachtete
Zwischenversions-Kohortendifferenzen**, keine kausalen Rückholgrenzen und keine
versprochenen Gewinne. Die akzeptierte Tabelle zählt jede Aktivität einmal;
Pallas ist eine zweite Einteilung derselben Aktivitäten, kein additiver Posten.

## Harte Grenzen für einen künftigen, separat ausgewählten Versuch

- Maximal EIN materieller Prototyp, maximal ein halber Lane-Tag und EIN
  CPU-Lower ≤600 s/≤12 GiB. Wartezeit auf einen Slot eröffnet keine weiteren Prototypen.
- Keep ausschließlich bei ≥0,5 ms/root **netto über alle betroffenen Familien
  und Domains** und jenseits eigener A/A-Streuung; gleicher unveränderlicher
  Ausgangsbaum, Arbeit, Inputs, Allocator und Output, gepaartes A/B/A/B.
- Keine zusätzlichen Vollfeld-Temporaries, Peak-VRAM-Zunahme, Step-Transfers
  oder Syncs. Traffic, Compile und Readiness ausweisen.
- OFF/default exakt; kompilierte ON/OFF-BYTE-Gleichheit aller Carry- und
  relevanten Exportfelder auf echten PROD/WN3 Tag-/Nachtzuständen, endlich.
  Originale WRF-Orakel an geänderten Seams, unveränderte Operatorreihenfolge
  und Kind-/f64-Inseln. GPU-vs-GPU ist Regression, keine WRF-Wahrheit.
- E41 am tatsächlichen geänderten Artefakt: Stack ≤256 B, nonleaf ≤1.000 SASS,
  kInput-Transpose-Hero ≤500 HLO. Nicht-BYTE/E41-Regression: Stop und Managerentscheidung.
- WRF-Physik→RK, W-Dämpfung, Root-/Nest-Randtakt, qni, PSIQ/REAL-PSI,
  Noah-QSFC/Q2 und Thompson-Mixed-/Driver-Regeln bleiben. Root-Pins,
  QSFC-Firewall und notwendige Compiler-Schutzgrenzen bleiben ebenfalls.
- Keine Massencaches über RK-Stufen/Domains/Zeitpunkte, kein lru_cache auf
  getraceten jnp-Arrays (E206). Kleine Schutzgeometrien bei Fence-Eingriffen mitprüfen.
- GPU nur im expliziten Manager-Slotplan mit Launch-Smoke. Der Ersatzslot
  am 24.10 bis 18:00Z gehört warmstart→raddiag→budget. Perf-Repeat und
  Hostprofil sind für den 25.10 vorgesehen; keine eigene Queue/Go-Freigabe.

Die beiden registrierten v4-Launcher sind nach dem Neustart SHA256-identisch;
siehe [Integritätsbeleg](receipts/POST_POWER_LOSS_CHECK.json).
Die README-Vergleichsaussage bleibt bis zu einer neuen gültigen A/A-Session HOLD.

Das negative Audit-Ergebnis erfüllt diesen begrenzten CPU-Auftrag. Der
PERF.md-Abschnitt [Verlust verstanden](PERF.md#verlust-verstanden) dokumentiert
die bewusste Arbeit je Kohorte und die verbleibenden kausalen Grenzen.
Das Gesamtgate bleibt für Repeat/Hostprofil und offene vollständige
Strahlungs-/Idle-Zuordnung offen. READY-FOR-CRITIC bezieht sich auf diesen
CPU-Attributions-/Audit-Teil.
