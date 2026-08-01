# GLOBECOM 2026 camera-ready data

This repository release is the executable and data artifact for the
camera-ready version of *6G Network Empowering AI Agents*. It preserves the
corrected simulator, the resolved configuration, the scripts that generated
the publication-facing studies, all trial-level observations, summaries,
calibration samples, and validation reports.

## Artifact contents

- `dna-nn-simulator-v2.0.py`: corrected simulation model used for all released
  runs. The historical filename is retained so existing commands remain valid.
- `.config_sim`: publication configuration.
- `tests/`: 27 regression and invariant tests for channel physics, local state
  logic, statistics, and deterministic seeding.
- `phase4_refine_primary_sweeps.py`: high-confidence anomaly and noise sweeps.
- `phase5_sensitivity_scaling.py`: EIR grid, physical sensitivities, and
  scaling experiments.
- `phase5_analyze_results.py`: compact summaries of Phase-5 results.
- `phase5_validate_results.py`: row-count, seed-separation, and paired-scenario
  validation.
- `results/phase4_camera_ready_2026_08_01/`: Phase-4 summaries, trial-level
  data, calibration samples, local-state diagnostics, plots, effective
  configuration, and metadata.
- `results/phase5_sensitivity_2026_08_01/`: Phase-5 robustness, sensitivity,
  and scaling data with a machine-readable validation report.
- `CAMERA_READY_SHA256.csv`: SHA-256 checksums and byte sizes for all released
  camera-ready scripts, tests, configurations, and result artifacts.

## Reproduction

Create an environment and install the dependencies described in `README.md`.
From the repository root, run:

```powershell
python -m unittest discover -s tests -v
python phase4_refine_primary_sweeps.py --output-dir reproduced_phase4
python phase5_sensitivity_scaling.py --output-dir reproduced_phase5
python phase5_validate_results.py --output-dir reproduced_phase5
python phase5_analyze_results.py --output-dir reproduced_phase5
```

The default trial counts are the publication counts. Reproduction is
deterministic for the released configuration, but a full run is computationally
substantial. The committed trial-level CSV files allow the reported statistics
to be audited without rerunning the Monte Carlo experiments.

## Experimental design and validation

The Phase-4 accuracy sweeps use 1,000 H0 and 1,000 H1 evaluation trials per
point and independent calibration samples. The Phase-5 artifact includes 54
unique EIR configurations, 18 physical/activation variations, and fixed-length
and constant-density scaling studies. Strategies are evaluated on paired
scenarios, while calibration and evaluation seed sets are disjoint.

The committed Phase-5 validation report has status `passed` and checks expected
row counts, configuration uniqueness, calibration/evaluation seed separation,
and RR/TR/EIR scenario pairing. The regression test suite must additionally
report 27 passing tests.

## Key files behind the manuscript

- Detection and false-alarm curves: Phase-4 `sweep_anomaly.csv` and
  `sweep_noise.csv` plus their `_trials.csv` files.
- Communication-load curves: Phase-4 `sweep_nodes.csv` plus trial-level data.
- Delay/trade-off results: Phase-4 `sweep_inference_delay.csv` plus trial-level
  data.
- Local ON-state and transition statistics: Phase-4
  `diagnostics/state_dynamics_summary.csv` plus trial-level diagnostics.
- Robustness claim: Phase-5 `eir_grid_search/eir_grid_summary.csv` and
  `phase5_robust_summary.json`.
- Transport and scaling limits: Phase-5 `scaling_summary.csv` and
  `scaling_trials.csv`.

## Citation and permanence

Use the Zenodo concept DOI
[`10.5281/zenodo.19416612`](https://doi.org/10.5281/zenodo.19416612) when a
stable link to the evolving software record is desired. For strict
reproducibility, cite the version-specific DOI shown on the Zenodo page for
release v2.1 together with the Git commit and tag.

The software is distributed under the repository's MIT license. The released
data are generated simulation results and contain no personal or confidential
information.
