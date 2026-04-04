# DNA NN Simulator

Dieses Repository enthält einen Python-Simulator für ein verteiltes DNA-/Molekularnetz-Szenario mit drei lokalen Entscheidungsstrategien (`RR`, `TR`, `EIR`), Gateway-Aggregation, Kalibrierung von Schwellenwerten sowie Export von CSV-Dateien und Plots.

Die zentrale Datei ist:

- `dna-nn-simulator.py`

Zusätzlich enthält das Repository einen Beispiel-Ordner `results/` mit bereits erzeugten Ergebnisdateien.

## 1. Voraussetzungen

Benötigt wird eine aktuelle Python-Installation.

Empfehlung:

- Python 3.10 oder neuer
- `pip` verfügbar

Der Simulator verwendet genau diese externen Bibliotheken:

- `numpy`
- `pandas`
- `matplotlib`

## 2. Virtuelle Umgebung anlegen und Bibliotheken installieren

### Windows PowerShell

Im Repository-Verzeichnis:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Falls PowerShell das Aktivieren von Skripten blockiert, kann temporär z. B. Folgendes nötig sein:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Linux / macOS / Git Bash

Im Repository-Verzeichnis:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Repository-Struktur

Die ZIP-Datei enthält in der vorliegenden Form im Wesentlichen:

```text
.
├── dna-nn-simulator.py
└── results/
    ├── calibration.json
    ├── summary.csv
    ├── trial_results.csv
    ├── sweep_anomaly.csv
    ├── sweep_noise.csv
    ├── sweep_nodes.csv
    ├── sweep_inference_delay.csv
    ├── plot_detection_vs_anomaly.png
    ├── plot_false_alarm_vs_noise.png
    ├── plot_comm_load_h0_vs_nodes.png
    ├── plot_comm_load_h1_vs_nodes.png
    ├── plot_delay_vs_inference_delay.png
    ├── plot_pareto.png
    └── diagnostics/
        ├── calibration_preview.json
        ├── local_diagnostics_summary.csv
        ├── local_samples_H0.csv
        ├── local_samples_H1.csv
        ├── state_dynamics_summary.csv
        ├── state_dynamics_trials.csv
        ├── local_marker_histograms.png
        ├── local_marker_scatter.png
        └── local_distance_send_profiles.png
```

## 4. Simulator aufrufen

Der Simulator wird direkt über das Python-Skript gestartet:

```bash
python dna-nn-simulator.py [OPTIONEN]
```

Verfügbare relevante Optionen:

- `--diagnostics`  
  Führt lokale Diagnostik und Zustandsdynamik-Diagnostik aus.
- `--demo`  
  Führt die vollständigen Sweeps aus und erzeugt die Demo-Plots.
- `--output-dir <PFAD>`  
  Zielverzeichnis für die Ausgaben. Standard ist `sim_output`.
- `--num-samples <N>`  
  Anzahl lokaler Samples pro Zustand für die Diagnostik. Standard: `5000`.
- `--num-trials-state <N>`  
  Anzahl State-Dynamics-Trials in der Diagnostik. Standard: `100`.

### 4.1 Diagnostik-Modus

Beispiel:

```bash
python dna-nn-simulator.py --diagnostics --output-dir results
```

Dabei werden Diagnostikdateien im Unterordner

```text
results/diagnostics/
```

erzeugt.

Optional können die Stichprobengröße und die Zahl der State-Trials angepasst werden:

```bash
python dna-nn-simulator.py --diagnostics --output-dir results --num-samples 5000 --num-trials-state 100
```

### 4.2 Demo-Modus

Beispiel:

```bash
python dna-nn-simulator.py --demo --output-dir results
```

Dabei erzeugt der Simulator die Hauptausgaben direkt in:

```text
results/
```

Der Demo-Modus umfasst:

- eine Baseline-Simulation,
- Kalibrierung lokaler und Gateway-Schwellen,
- Trial-Export,
- Zusammenfassungsdatei,
- mehrere Parametersweeps,
- Plot-Erzeugung.

### 4.3 Kombination beider Modi

Beide Flags können auch gemeinsam verwendet werden:

```bash
python dna-nn-simulator.py --diagnostics --demo --output-dir results
```

Dann entstehen:

- Demo-Ergebnisse in `results/`
- Diagnostik-Ergebnisse in `results/diagnostics/`

### 4.4 Standardlauf ohne Flags

Ohne `--diagnostics` und ohne `--demo` führt das Skript einen kleineren Standardlauf aus:

```bash
python dna-nn-simulator.py --output-dir sim_output
```

Dabei werden vor allem `summary.csv` und `calibration.json` geschrieben.

## 5. Was der Simulator inhaltlich macht

Der Simulator vergleicht drei lokale Entscheidungsstrategien:

- **RR**: Alarm, wenn mindestens einer von zwei Markern über seinem Schwellenwert liegt.
- **TR**: Alarm basierend auf dem ersten Marker (`x1`) und einem einzelnen Schwellenwert.
- **EIR**: Alarm auf Basis einer gewichteten linearen Kombination der Marker (`w1*x1 + w2*x2`) mit optionalem Gate auf `x2`.

Auf lokaler Ebene werden Markerwerte erzeugt, daraus lokale Entscheidungen abgeleitet und eventuelle Alarmemissionen an ein Gateway weitergegeben. Auf Gateway-Ebene wird dann geprüft, ob eine globale Detektion vorliegt.

## 6. Erklärung der Ergebnisdateien

## 6.1 Hauptausgaben im Demo-/Standardlauf

### `calibration.json`
JSON-Datei mit:

- der vollständigen verwendeten Simulationskonfiguration,
- den kalibrierten lokalen Schwellenwerten,
- den kalibrierten Gateway-Schwellenwerten.

Diese Datei ist die wichtigste Referenz, um eine Simulation später reproduzierbar nachzuvollziehen.

### `summary.csv`
Aggregierte Ergebnisübersicht mit genau einer Zeile pro Strategie (`RR`, `TR`, `EIR`).

Wichtige Spalten:

- `P_D`: Detektionswahrscheinlichkeit unter H1
- `P_pre_onset_alarm`: Wahrscheinlichkeit eines Alarms vor dem eigentlichen Onset unter H1
- `P_FA`: Falschalarmwahrscheinlichkeit unter H0
- `C_total_molecules_avg`: mittlere gesamte Kommunikationslast
- `C_H0_molecules_avg`: mittlere Kommunikationslast unter H0
- `C_H1_molecules_avg`: mittlere Kommunikationslast unter H1
- `R_H1_molecules_per_s`: mittlere Kommunikationsrate bis zur Detektion unter H1
- `D_avg_after_onset`: mittlere Detektionsverzögerung nach Onset

### `trial_results.csv`
Detaillierte Trial-Tabelle mit einer Zeile pro Simulationsdurchlauf und Strategie.

Wichtige Spalten:

- `strategy`: `RR`, `TR` oder `EIR`
- `state_h1`: `0` für H0, `1` für H1
- `detected`: ob unter H1 eine Detektion erfolgte
- `pre_onset_alarm`: ob schon vor `anomaly_start` ein Alarm auftrat
- `detection_time`: Zeitpunkt der Detektion
- `first_alarm_time`: Zeitpunkt des ersten Gateway-Alarms
- `transmissions`: Anzahl lokaler Sendungen
- `max_gateway_evidence`: maximale am Gateway akkumulierte Evidenz

## 6.2 Sweep-Dateien

### `sweep_anomaly.csv`
Ergebnisse eines Sweeps über die Anomalie-Stärke `a1`.

Zusätzliche Spalte:

- `a1`: jeweils getesteter Wert

### `sweep_noise.csv`
Ergebnisse eines Sweeps über das Rauschen des ersten Markers.

Zusätzliche Spalte:

- `sigma1`: jeweils getesteter Wert

### `sweep_nodes.csv`
Ergebnisse eines Sweeps über die Zahl der Knoten.

Zusätzliche Spalte:

- `num_nodes`: jeweils getesteter Wert

### `sweep_inference_delay.csv`
Ergebnisse eines Sweeps über die lokale Inferenzverzögerung.

Zusätzliche Spalte:

- `inference_delay`: jeweils getesteter Wert

Alle Sweep-Dateien enthalten neben dem Sweep-Parameter dieselben aggregierten Metriken wie `summary.csv`.

## 6.3 Plot-Dateien im Demo-Modus

### `plot_detection_vs_anomaly.png`
Detektionswahrscheinlichkeit `P_D` als Funktion der Anomalie-Stärke `a1`.

### `plot_false_alarm_vs_noise.png`
Falschalarmwahrscheinlichkeit `P_FA` als Funktion des Markerrauschens `sigma1`.

### `plot_comm_load_h0_vs_nodes.png`
Kommunikationslast unter H0 in Abhängigkeit von der Knotenzahl.

### `plot_comm_load_h1_vs_nodes.png`
Kommunikationslast unter H1 in Abhängigkeit von der Knotenzahl.

### `plot_delay_vs_inference_delay.png`
Detektionsverzögerung in Abhängigkeit von der lokalen Inferenzverzögerung.

### `plot_pareto.png`
Pareto-artige Darstellung von Kommunikationslast unter H1 versus Detektionswahrscheinlichkeit `P_D`.

## 6.4 Diagnostik-Dateien in `diagnostics/`

### `calibration_preview.json`
Vorschau der Konfiguration und der lokal kalibrierten Schwellenwerte im Diagnostiklauf.

Hinweis: In diesem Diagnostiklauf werden die Gateway-Schwellenwerte nicht vollständig kalibriert; deshalb stehen sie hier als `NaN`.

### `local_samples_H0.csv` und `local_samples_H1.csv`
Rohdaten lokaler Stichproben unter H0 bzw. H1.

Spalten:

- `state_h1`: Kennzeichnung des Zustands
- `x`: Position des Knotens
- `dist_to_anomaly`: Abstand zur Anomaliequelle
- `x1`, `x2`: gezogene Markerwerte
- `z_eir`: EIR-Score relativ zur Entscheidungsgrenze
- `send_rr`, `send_tr`, `send_eir`: lokale Sendeentscheidung je Strategie

Diese Dateien eignen sich besonders für eigene Nachanalysen der lokalen Entscheidungsräume.

### `local_diagnostics_summary.csv`
Verdichtete Übersicht der lokalen Stichproben unter H0 und H1.

Spalten u. a.:

- Mittelwerte von `x1` und `x2`
- mittlere lokale Sendewahrscheinlichkeiten je Strategie
- mittlerer EIR-Score

### `state_dynamics_trials.csv`
Detaillierte Ergebnisse der zeitlichen Zustandsdynamik über viele Trials.

Wichtige Spalten:

- `strategy`
- `state_h1`
- `trial`
- `mean_on_fraction`: Anteil aktiver Zeit
- `mean_rising_edges_per_node`: mittlere Zahl von Aktivierungsflanken pro Knoten
- `total_rising_edges`: gesamte Zahl von Aktivierungsflanken
- `mean_on_duration_s`: mittlere Dauer aktiver Phasen

### `state_dynamics_summary.csv`
Gemittelte Zustandsdynamik pro Strategie und Zustand (`H0`/`H1`).

### `local_marker_histograms.png`
Histogramme der Markerverteilungen `x1` und `x2` unter H0 und H1.

### `local_marker_scatter.png`
Streudiagramm von `x1` gegen `x2` mit eingezeichneter EIR-Entscheidungsgrenze sowie optionalem `x2`-Gate.

### `local_distance_send_profiles.png`
Darstellung der lokalen Sendewahrscheinlichkeit unter H1 in Abhängigkeit vom Abstand zur Anomaliequelle.

## 7. Typische Arbeitsabfolge

Für einen sauberen Reproduktionslauf bietet sich diese Reihenfolge an:

1. virtuelle Umgebung anlegen,
2. Abhängigkeiten installieren,
3. Diagnostiklauf starten,
4. Demo-Lauf starten,
5. `summary.csv`, Sweep-Dateien und Plots auswerten,
6. bei Bedarf `calibration.json` zur Dokumentation der exakten Konfiguration archivieren.

## 8. Beispielkommandos auf einen Blick

### Diagnostik

```bash
python dna-nn-simulator.py --diagnostics --output-dir results
```

### Demo

```bash
python dna-nn-simulator.py --demo --output-dir results
```

### Beides zusammen

```bash
python dna-nn-simulator.py --diagnostics --demo --output-dir results
```

