# DNA NN Simulator

A Python simulator for studying distributed anomaly detection and communication-efficiency trade-offs in DNA-inspired / molecular nanonetworks.

## GLOBECOM 2026 camera-ready artifact (v2.1)

Release v2.1 adds the corrected simulator, regression tests, and the complete
trial-level data used for the camera-ready version of *6G Network Empowering AI
Agents*. The archived artifact contains the high-confidence primary sweeps,
the 54-configuration EIR robustness grid, physical-parameter sensitivities,
and two scaling studies. See [`CAMERA_READY_DATA.md`](CAMERA_READY_DATA.md) for
the data dictionary, validation commands, and exact reproduction workflow.

The stable archival identifier is the Zenodo concept DOI
[`10.5281/zenodo.19416612`](https://doi.org/10.5281/zenodo.19416612); it always
resolves to the newest archived release. Release-specific identifiers are
listed in the corresponding GitHub release and Zenodo record.

The simulator compares three local decision strategies:

- **RR** — raw reporting based on two local marker thresholds
- **TR** — single-marker threshold reporting
- **EIR** — embedded inference reporting based on weighted multi-marker evidence, optional gating, and hysteresis

Version **2.0** introduces a configuration-driven workflow via `.config_sim`, a substantially improved physical and statistical model, deterministic per-trial reproducibility, and a built-in grid-search mechanism configurable directly from the TOML file.

---

## 1. Main features

- Config-driven execution via **`.config_sim`**
- Baseline, diagnostics, demo, and EIR grid-search run modes
- Diffusive 1D channel model with optional legacy channel fallback
- Leaky gateway evidence accumulation with threshold calibration
- Deterministic per-trial seeding for reproducible sweeps and grid searches
- Temporally correlated marker processes with variance-preserving AR(1)-style dynamics
- Robust threshold calibration for clipped-at-zero marker distributions
- Automatic CSV / JSON / PNG export for all major experiments
- Config-based parameter sweeps and config-based Cartesian-product grid search

---

## 2. Repository structure

A typical repository layout for v2.0 is:

```text
.
├── dna-nn-simulator-v2.0.py
├── .config_sim
├── requirements.txt
└── results/
```

The simulator writes outputs to the directory configured in `[run_modes].output_dir`.

---

## 3. Requirements

Recommended:

- Python **3.11+**
- `pip`

Required Python packages:

- `numpy`
- `pandas`
- `matplotlib`

If you use Python < 3.11, install `tomli` as well because the simulator uses TOML-based configuration files.

---

## 4. Installation

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks script execution, temporarily run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Linux / macOS / Git Bash

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Configuration file

Version 2.0 uses **`.config_sim`** as the default simulator configuration file.

### Default start

```bash
python dna-nn-simulator-v2.0.py
```

### Explicit config path

```bash
python dna-nn-simulator-v2.0.py --config my_experiment.toml
```

### Optional output override

```bash
python dna-nn-simulator-v2.0.py --output-dir my_results
```

### Print the resolved configuration

```bash
python dna-nn-simulator-v2.0.py --print-effective-config
```

At startup, the simulator writes the fully resolved configuration to:

```text
effective_config.json
```

inside the selected output directory.

---

## 6. Configuration structure

The TOML file is organized into the following sections:

- `[simulation]` — physical model, local decision model, channel model, gateway model, seeds
- `[run_modes]` — which run modes should execute and where outputs go
- `[baseline]` — trial counts for the default baseline run
- `[diagnostics]` — local sample export and state-dynamics diagnostics
- `[demo]` — baseline + sweep settings for the full demo mode
- `[sweeps]` — values for anomaly, noise, node-count, and inference-delay sweeps
- `[eir_grid]` — controls for EIR parameter search
- `[eir_grid.parameter_values]` — lists of parameter values interpreted as a Cartesian-product grid

### Example

```toml
[run_modes]
output_dir = "run_v20"
run_baseline = true
run_diagnostics = false
run_demo = false
run_eir_grid_search = false

[simulation]
channel_model = "diffusive_1d"
gateway_evidence_mode = "leaky_integrator"
eir_gate_quantile = 0.85
hysteresis_margin_eir = 0.02
w2 = 0.75
```

---

## 7. Run modes

### 7.1 Baseline run

If `run_baseline = true`, the simulator executes one baseline experiment and writes:

- `summary.csv`
- `calibration.json`
- `effective_config.json`

Minimal command:

```bash
python dna-nn-simulator-v2.0.py
```

### 7.2 Diagnostics mode

Enable in `.config_sim`:

```toml
[run_modes]
run_diagnostics = true
```

This creates a `diagnostics/` subdirectory with local marker and state-dynamics diagnostics.

Typical outputs:

- `local_diagnostics_summary.csv`
- `local_samples_H0.csv`
- `local_samples_H1.csv`
- `state_dynamics_summary.csv`
- `state_dynamics_trials.csv`
- `local_marker_histograms.png`
- `local_marker_scatter.png`
- `local_distance_send_profiles.png`

### 7.3 Demo mode

Enable:

```toml
[run_modes]
run_demo = true
```

Demo mode performs:

- one baseline run
- parameter sweeps over anomaly strength, noise, node count, and inference delay
- export of trial-level and summary data
- generation of result plots

Typical outputs:

- `trial_results.csv`
- `summary.csv`
- `sweep_anomaly.csv`
- `sweep_noise.csv`
- `sweep_nodes.csv`
- `sweep_inference_delay.csv`
- `plot_detection_vs_anomaly.png`
- `plot_false_alarm_vs_noise.png`
- `plot_comm_load_h0_vs_nodes.png`
- `plot_comm_load_h1_vs_nodes.png`
- `plot_delay_vs_inference_delay.png`
- `plot_pareto.png`

### 7.4 EIR grid search

Enable:

```toml
[run_modes]
run_eir_grid_search = true
```

The grid search evaluates the parameter combinations defined in `[eir_grid.parameter_values]`.

Outputs include:

- `eir_grid_anomaly.csv`
- `eir_grid_summary.csv`
- `eir_grid_best.json`
- heatmaps for detection, communication-cost difference, delay difference, and PFA deviation

---

## 8. How the grid search works

The table

```toml
[eir_grid.parameter_values]
```

contains lists of values. Each list defines one search dimension. The simulator automatically evaluates the full Cartesian product.

### Example

```toml
[eir_grid.parameter_values]
eir_gate_quantile = [0.84, 0.85, 0.86]
hysteresis_margin_eir = [0.02, 0.03, 0.04]
gate_hysteresis_margin = [0.01, 0.02, 0.03]
w2 = [0.75, 0.90]
```

This produces a 3 × 3 × 3 × 2 = **54-combination** grid search.

You may also grid-search other keys from `[simulation]` if they are valid simulator parameters, for example `inference_delay`, `w1`, `temporal_alpha`, or `receiver_gain`.

---

## 9. Simulation model

### 9.1 Scenario

The simulator models a distributed in-body nanonetwork in a one-dimensional domain. Nanonodes observe two biochemical markers and may emit alarm molecules toward a gateway. The simulator compares different local reporting strategies and studies the trade-offs between:

- detection probability
- false alarms
- event-driven communication cost
- nuisance communication under H0
- end-to-end delay

### 9.2 Local strategies

- **RR**: positive if either marker crosses its local threshold
- **TR**: positive if marker 1 crosses its threshold
- **EIR**: positive if a weighted evidence score exceeds threshold, optionally with an additional gate on marker 2

The EIR score is of the form:

```text
z = w1 * x1 + w2 * x2
```

with a calibrated threshold and optional gate/hysteresis logic.

### 9.3 Marker process

Under H0, both markers fluctuate around background means.
Under H1, spatially decaying anomaly amplitudes are added to the local means.

Version 2.0 uses a **variance-preserving temporally correlated latent process** rather than the older non-stationary smoothing approximation. This keeps the requested stationary variance of the unclipped observation model while preserving temporal persistence.

### 9.4 Local decision dynamics

The simulator supports:

- hysteresis for RR, TR, and EIR
- an additional hysteresis margin for the EIR gate
- edge-triggered alarm emission
- refractory suppression of repeated transmissions
- optional refresh transmissions for sustained positive states

### 9.5 Channel model

Version 2.0 supports two channel models:

- **`diffusive_1d`** — the preferred model in v2.0
- **`legacy_linear`** — retained for comparison and backward-style experiments

The preferred model uses a discretized diffusive first-passage style response with optional drift and decay. This replaces the earlier purely linear delay plus fixed Gaussian spread approximation as the recommended default.

### 9.6 Gateway model

Version 2.0 supports:

- **`instantaneous`** gateway evidence
- **`leaky_integrator`** gateway evidence (recommended default)

The leaky integrator provides a more stable and physically plausible aggregation of arriving alarm evidence over time.

### 9.7 Calibration

The simulator calibrates:

- local thresholds to a target local positive-state probability under H0
- gateway thresholds to a target false-alarm level under H0

Version 2.0 uses more robust threshold-search logic for clipped-at-zero marker distributions.

---

## 10. Reproducibility

A major v2.0 change is deterministic per-trial seeding.

Instead of consuming one shared RNG stream across entire sweeps, the simulator derives deterministic seeds from:

- the master seed
- phase
- strategy
- H0 / H1 state
- trial index
- sweep / namespace context

This makes sweeps and grid searches reproducible and avoids artifacts caused by variable random-number consumption across early-stopping trials.

---

## 11. Output files

### Core files

- **`effective_config.json`** — fully resolved configuration actually used
- **`calibration.json`** — calibrated thresholds and key simulation settings
- **`summary.csv`** — one summary row per strategy
- **`trial_results.csv`** — trial-level outcomes

### Sweep files

- **`sweep_anomaly.csv`**
- **`sweep_noise.csv`**
- **`sweep_nodes.csv`**
- **`sweep_inference_delay.csv`**

### Grid-search files

- **`eir_grid_anomaly.csv`**
- **`eir_grid_summary.csv`**
- **`eir_grid_best.json`**
- **`plot_eir_grid_mean_pd.png`**
- **`plot_eir_grid_delta_ch1_vs_tr.png`**
- **`plot_eir_grid_delta_delay_vs_tr.png`**
- **`plot_eir_grid_pfa_deviation.png`**

---

## 12. Recommended workflows

### Quick baseline

```bash
python dna-nn-simulator-v2.0.py
```

### Full demo

Set in `.config_sim`:

```toml
[run_modes]
run_demo = true
```

then run:

```bash
python dna-nn-simulator-v2.0.py
```

### Grid search only

Set:

```toml
[run_modes]
run_eir_grid_search = true
run_demo = false
run_diagnostics = false
run_baseline = false
```

then run:

```bash
python dna-nn-simulator-v2.0.py
```

### Separate experiment configs

Keep one default `.config_sim` for standard runs and copy it for variants, e.g.:

```bash
cp .config_sim experiment_low_noise.toml
python dna-nn-simulator-v2.0.py --config experiment_low_noise.toml
```

---

## 13. Migration notes from v1.0

Compared with the original release:

- configuration is no longer hard-coded as the primary workflow
- `.config_sim` is now the default experiment definition
- the recommended channel model is now diffusive rather than linear-delay-based
- the recommended gateway model is now a leaky evidence integrator
- the temporal marker model is variance-preserving
- calibration is more robust for clipped-at-zero observations
- sweeps are reproducible via deterministic per-trial seeds
- EIR parameter search can be configured directly from the TOML file

If you want to approximate older behavior, set:

```toml
[simulation]
channel_model = "legacy_linear"
gateway_evidence_mode = "instantaneous"
```

---

## 14. License

This project is released under the repository license.
