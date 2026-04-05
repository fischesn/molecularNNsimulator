# DNA-NN Simulator v2.0

This release introduces a major revision of the DNA-NN simulator for distributed anomaly detection in molecular nanonetwork scenarios.

Version **2.0** is not just a convenience update. It changes both the **user workflow** and the **recommended simulator model** in substantial ways.

---

## Highlights

### New configuration-driven workflow

The simulator is now driven by a TOML configuration file:

- default config file: **`.config_sim`**
- optional override via `--config <path>`
- full resolved configuration exported automatically as `effective_config.json`

This makes experiments easier to reproduce, modify, and document than in the original single-script workflow.

### Strongly improved simulator model

Version 2.0 replaces the earlier recommended default model by a more robust and physically plausible setup:

- **diffusive 1D channel model** as the default recommendation
- **leaky gateway evidence integration** as the default recommendation
- **variance-preserving temporally correlated marker model**
- **robust threshold calibration** for clipped-at-zero observations
- **deterministic per-trial seeding** for reproducible sweeps and parameter searches

The legacy linear-delay channel remains available for comparison, debugging, and backward-style experiments.

### Built-in config-based grid search

Version 2.0 adds a built-in EIR parameter search that can be specified directly in the config file through parameter lists. The simulator automatically evaluates the Cartesian product.

---

## What changed compared with v1.0

## 1. Usage model

### v1.0

The original release centered on a single Python entry point with command-line flags for diagnostics and demo mode.

### v2.0

The new release keeps a simple command-line interface, but the primary workflow is now:

```bash
python dna-nn-simulator-v2.0.py
```

with all experiment settings loaded from `.config_sim`.

You can still select another config explicitly:

```bash
python dna-nn-simulator-v2.0.py --config my_experiment.toml
```

This is the largest usability change in the release.

---

## 2. Model changes

### 2.1 Channel model

### v1.0

The earlier simulator used a simple distance-dependent communication approximation with linear delay and fixed spreading.

### v2.0

The recommended default is now a **diffusive 1D response model** with optional drift and decay:

- `channel_model = "diffusive_1d"`
- legacy fallback retained as `legacy_linear`

This gives a more meaningful distance/time coupling than the earlier linear-delay approximation.

### 2.2 Gateway aggregation

### v1.0

Gateway evidence was accumulated in a simpler way around raw arrival evidence and thresholding.

### v2.0

The preferred default is now a **leaky evidence integrator**:

- `gateway_evidence_mode = "leaky_integrator"`

This stabilizes gateway-side decision dynamics and avoids a purely instantaneous view of received alarm evidence.

### 2.3 Temporal marker dynamics

### v1.0

Temporal correlation was modeled in a simpler form that could distort the effective stationary variance.

### v2.0

Marker dynamics now use a **variance-preserving temporally correlated latent process**, so temporal persistence no longer unintentionally suppresses fluctuation strength in the same way.

### 2.4 Threshold calibration

### v1.0

Threshold calibration relied more directly on quantile-style logic, which can become brittle for clipped-at-zero marginals.

### v2.0

Calibration uses more robust search procedures designed for the actual clipped-at-zero observation model.

### 2.5 Reproducibility

### v1.0

Large sweeps could be affected by shared random-number-stream coupling across trials.

### v2.0

The simulator now derives **deterministic per-trial seeds** from phase, strategy, state, trial index, and experiment namespace. This makes sweeps and searches reproducible and avoids artifacts caused by different random-number consumption paths.

---

## 3. New functionality

### Diagnostics, demo, and baseline remain available

The familiar simulator outputs remain conceptually intact:

- baseline run
- diagnostics mode
- demo mode with parameter sweeps

### New: config-based EIR grid search

The new config section

```toml
[eir_grid.parameter_values]
```

lets you define search dimensions directly in the config file, for example:

```toml
[eir_grid.parameter_values]
eir_gate_quantile = [0.84, 0.85, 0.86]
hysteresis_margin_eir = [0.02, 0.03, 0.04]
gate_hysteresis_margin = [0.01, 0.02, 0.03]
w2 = [0.75, 0.90]
```

The simulator then evaluates the full Cartesian product automatically.

This is a major practical improvement over manually editing one parameter set after another.

---

## 4. Output changes

Version 2.0 still produces CSV, JSON, and PNG outputs, but now also writes:

- **`effective_config.json`** — the exact resolved configuration used for the run

For EIR parameter search, v2.0 additionally produces:

- `eir_grid_anomaly.csv`
- `eir_grid_summary.csv`
- `eir_grid_best.json`
- heatmaps for parameter-search interpretation

---

## 5. Recommended v2.0 default model

The intended default setup of v2.0 is:

- TOML config-driven execution via `.config_sim`
- `channel_model = "diffusive_1d"`
- `gateway_evidence_mode = "leaky_integrator"`
- deterministic per-trial seeding
- config-based sweeps and grid search

This should be regarded as the recommended public reference model for the release.

---

## 6. Backward-style experiments

If you want behavior closer to the older simulator lineage, v2.0 still supports a legacy-style configuration, for example:

```toml
[simulation]
channel_model = "legacy_linear"
gateway_evidence_mode = "instantaneous"
```

This is useful for method comparison and migration studies.

---

## 7. Typical usage

### Standard run

```bash
python dna-nn-simulator-v2.0.py
```

### Alternative config file

```bash
python dna-nn-simulator-v2.0.py --config experiment.toml
```

### Override output directory

```bash
python dna-nn-simulator-v2.0.py --output-dir results_v20
```

---

## 8. Release summary

Version 2.0 is the first release that turns the simulator into a more systematic experimental framework rather than only a fixed-script implementation.

The central differences to v1.0 are therefore:

- **better usability** through externalized configuration
- **better reproducibility** through deterministic seeding and saved effective configs
- **better model quality** through improved channel, gateway, temporal, and calibration models
- **better extensibility** through config-defined sweeps and grid search

