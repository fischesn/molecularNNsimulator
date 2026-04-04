# DNA NN Simulator

This repository contains a Python simulator for a distributed DNA/molecular network scenario with three local decision strategies (`RR`, `TR`, `EIR`), gateway aggregation, threshold calibration, and export of CSV files and plots.

The main file is:

- `dna-nn-simulator.py`

In addition, the repository contains an example folder `results/` with precomputed output files.

## 1. Requirements

A recent Python installation is required.

Recommended:

- Python 3.10 or newer
- `pip` available

The simulator uses exactly these external libraries:

- `numpy`
- `pandas`
- `matplotlib`

## 2. Create a Virtual Environment and Install the Libraries

### Windows PowerShell

In the repository directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks script execution, you may need to temporarily run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Linux / macOS / Git Bash

In the repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Repository Structure

In its current form, the ZIP file essentially contains:

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

## 4. Running the Simulator

The simulator is started directly via the Python script:

```bash
python dna-nn-simulator.py [OPTIONS]
```

Relevant available options:

- `--diagnostics`  
  Runs local diagnostics and state-dynamics diagnostics.
- `--demo`  
  Runs the full parameter sweeps and generates the demo plots.
- `--output-dir <PATH>`  
  Target directory for the outputs. Default is `sim_output`.
- `--num-samples <N>`  
  Number of local samples per state for diagnostics. Default: `5000`.
- `--num-trials-state <N>`  
  Number of state-dynamics trials in diagnostics. Default: `100`.

### 4.1 Diagnostics Mode

Example:

```bash
python dna-nn-simulator.py --diagnostics --output-dir results
```

This creates diagnostic files in the subdirectory:

```text
results/diagnostics/
```

Optionally, the sample size and the number of state trials can be adjusted:

```bash
python dna-nn-simulator.py --diagnostics --output-dir results --num-samples 5000 --num-trials-state 100
```

### 4.2 Demo Mode

Example:

```bash
python dna-nn-simulator.py --demo --output-dir results
```

This writes the main outputs directly to:

```text
results/
```

The demo mode includes:

- a baseline simulation,
- calibration of local and gateway thresholds,
- trial export,
- a summary file,
- multiple parameter sweeps,
- plot generation.

### 4.3 Combination of Both Modes

Both flags can also be used together:

```bash
python dna-nn-simulator.py --diagnostics --demo --output-dir results
```

This produces:

- demo results in `results/`
- diagnostics results in `results/diagnostics/`

### 4.4 Default Run Without Flags

Without `--diagnostics` and without `--demo`, the script performs a smaller default run:

```bash
python dna-nn-simulator.py --output-dir sim_output
```

This primarily writes `summary.csv` and `calibration.json`.

## 5. What the Simulator Does

The simulator compares three local decision strategies:

- **RR**: alarm if at least one of the two markers exceeds its threshold.
- **TR**: alarm based on the first marker (`x1`) and a single threshold.
- **EIR**: alarm based on a weighted linear combination of the markers (`w1*x1 + w2*x2`) with an optional gate on `x2`.

At the local level, marker values are generated, local decisions are derived, and potential alarm emissions are transmitted to a gateway. At the gateway level, the simulator then decides whether a global detection is present.

## 6. Explanation of the Output Files

## 6.1 Main Outputs in Demo/Default Mode

### `calibration.json`
A JSON file containing:

- the full simulation configuration used,
- the calibrated local thresholds,
- the calibrated gateway thresholds.

This is the most important reference file for reproducing a simulation later.

### `summary.csv`
An aggregated result overview with exactly one row per strategy (`RR`, `TR`, `EIR`).

Important columns:

- `P_D`: detection probability under H1
- `P_pre_onset_alarm`: probability of an alarm before the actual onset under H1
- `P_FA`: false alarm probability under H0
- `C_total_molecules_avg`: average total communication load
- `C_H0_molecules_avg`: average communication load under H0
- `C_H1_molecules_avg`: average communication load under H1
- `R_H1_molecules_per_s`: average communication rate until detection under H1
- `D_avg_after_onset`: average detection delay after onset

### `trial_results.csv`
A detailed trial table with one row per simulation run and strategy.

Important columns:

- `strategy`: `RR`, `TR`, or `EIR`
- `state_h1`: `0` for H0, `1` for H1
- `detected`: whether a detection occurred under H1
- `pre_onset_alarm`: whether an alarm occurred before `anomaly_start`
- `detection_time`: time of detection
- `first_alarm_time`: time of the first gateway alarm
- `transmissions`: number of local transmissions
- `max_gateway_evidence`: maximum evidence accumulated at the gateway

## 6.2 Sweep Files

### `sweep_anomaly.csv`
Results of a sweep over anomaly strength `a1`.

Additional column:

- `a1`: tested value

### `sweep_noise.csv`
Results of a sweep over the noise of the first marker.

Additional column:

- `sigma1`: tested value

### `sweep_nodes.csv`
Results of a sweep over the number of nodes.

Additional column:

- `num_nodes`: tested value

### `sweep_inference_delay.csv`
Results of a sweep over the local inference delay.

Additional column:

- `inference_delay`: tested value

All sweep files contain the same aggregated metrics as `summary.csv`, in addition to the sweep parameter.

## 6.3 Plot Files in Demo Mode

### `plot_detection_vs_anomaly.png`
Detection probability `P_D` as a function of anomaly strength `a1`.

### `plot_false_alarm_vs_noise.png`
False alarm probability `P_FA` as a function of marker noise `sigma1`.

### `plot_comm_load_h0_vs_nodes.png`
Communication load under H0 as a function of the number of nodes.

### `plot_comm_load_h1_vs_nodes.png`
Communication load under H1 as a function of the number of nodes.

### `plot_delay_vs_inference_delay.png`
Detection delay as a function of the local inference delay.

### `plot_pareto.png`
Pareto-style visualization of communication load under H1 versus detection probability `P_D`.

## 6.4 Diagnostics Files in `diagnostics/`

### `calibration_preview.json`
Preview of the configuration and the locally calibrated thresholds in the diagnostics run.

Note: in this diagnostics run, the gateway thresholds are not fully calibrated; therefore they appear as `NaN` here.

### `local_samples_H0.csv` and `local_samples_H1.csv`
Raw local sample data under H0 and H1, respectively.

Columns:

- `state_h1`: state indicator
- `x`: node position
- `dist_to_anomaly`: distance to the anomaly source
- `x1`, `x2`: sampled marker values
- `z_eir`: EIR score relative to the decision boundary
- `send_rr`, `send_tr`, `send_eir`: local send decision for each strategy

These files are particularly useful for custom post-analysis of the local decision spaces.

### `local_diagnostics_summary.csv`
Compressed overview of the local samples under H0 and H1.

Columns include:

- means of `x1` and `x2`
- average local send probabilities per strategy
- average EIR score

### `state_dynamics_trials.csv`
Detailed results of the temporal state dynamics across many trials.

Important columns:

- `strategy`
- `state_h1`
- `trial`
- `mean_on_fraction`: fraction of active time
- `mean_rising_edges_per_node`: average number of activation edges per node
- `total_rising_edges`: total number of activation edges
- `mean_on_duration_s`: average duration of active phases

### `state_dynamics_summary.csv`
Averaged state dynamics per strategy and state (`H0`/`H1`).

### `local_marker_histograms.png`
Histograms of the marker distributions `x1` and `x2` under H0 and H1.

### `local_marker_scatter.png`
Scatter plot of `x1` versus `x2` including the EIR decision boundary and, where applicable, the `x2` gate.

### `local_distance_send_profiles.png`
Visualization of the local send probability under H1 as a function of distance to the anomaly source.

## 7. Typical Workflow

A sensible workflow for a reproducible run is:

1. create the virtual environment,
2. install the dependencies,
3. run diagnostics mode,
4. run demo mode,
5. inspect `summary.csv`, the sweep files, and the plots,
6. archive `calibration.json` if the exact configuration should be preserved.

## 8. Example Commands at a Glance

### Diagnostics

```bash
python dna-nn-simulator.py --diagnostics --output-dir results
```

### Demo

```bash
python dna-nn-simulator.py --demo --output-dir results
```

### Both Together

```bash
python dna-nn-simulator.py --diagnostics --demo --output-dir results
```
