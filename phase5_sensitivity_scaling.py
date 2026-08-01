from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parent / "dna-nn-simulator-v2.0.py"
SPEC = importlib.util.spec_from_file_location("dna_nn_simulator_v2_phase5", SCRIPT_PATH)
SIM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SIM
SPEC.loader.exec_module(SIM)


SENSITIVITY_VALUES: Dict[str, List[float]] = {
    "gateway_integration_tau": [10.0, 20.0, 40.0],
    "diffusion_coeff": [1500.0, 3000.0, 6000.0],
    "local_send_prob_target": [0.01, 0.02, 0.04],
}
SENSITIVITY_ANOMALIES = [0.03, 0.06]
FIXED_SEGMENT_NODES = [20, 50, 100, 200]
CONSTANT_DENSITY_LENGTHS = [500.0, 1000.0, 2000.0, 4000.0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase-5 EIR robustness, physical sensitivity, and scaling studies."
    )
    parser.add_argument("--config", default=".config_sim")
    parser.add_argument("--output-dir", default="phase5_sensitivity_2026_08_01")
    parser.add_argument("--sensitivity-trials", type=int, default=300)
    parser.add_argument("--sensitivity-calibration-trials", type=int, default=500)
    parser.add_argument("--scaling-trials", type=int, default=300)
    parser.add_argument("--scaling-calibration-trials", type=int, default=500)
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-scaling", action="store_true")
    return parser


def run_parameter_sensitivity(
    output_dir: Path,
    base_cfg,
    *,
    parameter_name: str,
    values: List[float],
    anomaly_a2_ratio: float,
    num_trials: int,
    calib_trials: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    trial_frames = []
    calibration_frames = []
    started = time.time()
    for index, value in enumerate(values, start=1):
        print(
            f"[phase5:{parameter_name}] value {index}/{len(values)}: {value}",
            flush=True,
        )
        cfg = SIM.SimulationConfig(**asdict(base_cfg))
        setattr(cfg, parameter_name, value)
        summary, trials, calibration = SIM.sweep_anomaly_pair(
            cfg,
            SENSITIVITY_ANOMALIES,
            anomaly_a2_ratio,
            num_h0=num_trials,
            num_h1=num_trials,
            calib_trials=calib_trials,
            return_trials=True,
        )
        for frame in [summary, trials, calibration]:
            frame["sensitivity_parameter"] = parameter_name
            frame["sensitivity_value"] = value
        summary_frames.append(summary)
        trial_frames.append(trials)
        calibration_frames.append(calibration)
        print(
            f"[phase5:{parameter_name}] completed {value} in "
            f"{time.time() - started:.1f}s cumulative",
            flush=True,
        )

    summary_df = pd.concat(summary_frames, ignore_index=True)
    trials_df = pd.concat(trial_frames, ignore_index=True)
    calibration_df = pd.concat(calibration_frames, ignore_index=True)
    stem = f"sensitivity_{parameter_name}"
    summary_df.to_csv(output_dir / f"{stem}.csv", index=False)
    trials_df.to_csv(output_dir / f"{stem}_trials.csv", index=False)
    calibration_df.to_csv(output_dir / f"{stem}_calibration_trials.csv", index=False)
    return summary_df, trials_df, calibration_df


def run_scaling_point(
    cfg,
    *,
    mode: str,
    point_index: int,
    num_trials: int,
    calib_trials: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    namespace = f"phase5:scaling:{mode}:shared_scenarios"
    trials, summary, _, _, calibration = SIM.run_experiment(
        cfg,
        num_h0=num_trials,
        num_h1=num_trials,
        calib_trials=calib_trials,
        seed_namespace=namespace,
        return_calibration_trials=True,
    )
    metadata = {
        "scaling_mode": mode,
        "scaling_point_index": point_index,
        "L": cfg.L,
        "num_nodes": cfg.num_nodes,
        "node_density_per_um": cfg.num_nodes / cfg.L,
        "anomaly_x": cfg.anomaly_x,
    }
    for frame in [summary, trials, calibration]:
        for key, value in metadata.items():
            frame[key] = value
    return summary, trials, calibration


def run_scaling_studies(
    output_dir: Path,
    base_cfg,
    *,
    num_trials: int,
    calib_trials: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    trial_frames = []
    calibration_frames = []
    started = time.time()

    for index, num_nodes in enumerate(FIXED_SEGMENT_NODES):
        cfg = SIM.SimulationConfig(**asdict(base_cfg))
        cfg.num_nodes = int(num_nodes)
        print(
            f"[phase5:fixed_segment_density] point {index + 1}/{len(FIXED_SEGMENT_NODES)}: "
            f"L={cfg.L}, N={cfg.num_nodes}",
            flush=True,
        )
        summary, trials, calibration = run_scaling_point(
            cfg,
            mode="fixed_segment_increasing_density",
            point_index=index,
            num_trials=num_trials,
            calib_trials=calib_trials,
        )
        summary_frames.append(summary)
        trial_frames.append(trials)
        calibration_frames.append(calibration)

    base_density = base_cfg.num_nodes / base_cfg.L
    for index, length in enumerate(CONSTANT_DENSITY_LENGTHS):
        cfg = SIM.SimulationConfig(**asdict(base_cfg))
        cfg.L = float(length)
        cfg.num_nodes = int(round(base_density * cfg.L))
        cfg.anomaly_x = 0.75 * cfg.L
        print(
            f"[phase5:constant_density_segment] point {index + 1}/{len(CONSTANT_DENSITY_LENGTHS)}: "
            f"L={cfg.L}, N={cfg.num_nodes}, xA={cfg.anomaly_x}",
            flush=True,
        )
        summary, trials, calibration = run_scaling_point(
            cfg,
            mode="constant_density_increasing_segment",
            point_index=index,
            num_trials=num_trials,
            calib_trials=calib_trials,
        )
        summary_frames.append(summary)
        trial_frames.append(trials)
        calibration_frames.append(calibration)

    summary_df = pd.concat(summary_frames, ignore_index=True)
    trials_df = pd.concat(trial_frames, ignore_index=True)
    calibration_df = pd.concat(calibration_frames, ignore_index=True)
    summary_df.to_csv(output_dir / "scaling_summary.csv", index=False)
    trials_df.to_csv(output_dir / "scaling_trials.csv", index=False)
    calibration_df.to_csv(output_dir / "scaling_calibration_trials.csv", index=False)
    print(f"[phase5:scaling] completed in {time.time() - started:.1f}s", flush=True)
    return summary_df, trials_df, calibration_df


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    cfg, _, _, _, _, sweeps, grid = SIM.load_bundle_from_config(config_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    block_seconds: Dict[str, float] = {}

    if not args.skip_grid:
        block_started = time.time()
        SIM.run_eir_grid_search(
            output_dir / "eir_grid_search",
            cfg,
            grid_parameters=grid.parameter_values,
            anomaly_a1_values=sweeps.anomaly_a1_values,
            anomaly_a2_ratio=sweeps.anomaly_a2_ratio,
            focus_max_a1=grid.focus_max_a1,
            target_pfa=grid.target_pfa,
            num_h0=grid.num_h0,
            num_h1=grid.num_h1,
            calib_trials=grid.calib_trials,
        )
        block_seconds["eir_grid_search"] = time.time() - block_started

    if not args.skip_sensitivity:
        block_started = time.time()
        for parameter_name, values in SENSITIVITY_VALUES.items():
            run_parameter_sensitivity(
                output_dir,
                cfg,
                parameter_name=parameter_name,
                values=values,
                anomaly_a2_ratio=sweeps.anomaly_a2_ratio,
                num_trials=args.sensitivity_trials,
                calib_trials=args.sensitivity_calibration_trials,
            )
        block_seconds["physical_and_operating_sensitivity"] = time.time() - block_started

    if not args.skip_scaling:
        block_started = time.time()
        run_scaling_studies(
            output_dir,
            cfg,
            num_trials=args.scaling_trials,
            calib_trials=args.scaling_calibration_trials,
        )
        block_seconds["scaling"] = time.time() - block_started

    metadata = {
        "purpose": "Phase-5 sensitivity and scaling study",
        "simulator": SCRIPT_PATH.name,
        "config_path": str(config_path.resolve()),
        "base_simulation": asdict(cfg),
        "eir_grid": asdict(grid),
        "sensitivity": {
            "values": SENSITIVITY_VALUES,
            "anomaly_a1_values": SENSITIVITY_ANOMALIES,
            "anomaly_a2_ratio": sweeps.anomaly_a2_ratio,
            "num_h0_per_point": args.sensitivity_trials,
            "num_h1_per_point": args.sensitivity_trials,
            "calibration_trials_per_strategy": args.sensitivity_calibration_trials,
        },
        "scaling": {
            "fixed_segment_nodes": FIXED_SEGMENT_NODES,
            "constant_density_lengths": CONSTANT_DENSITY_LENGTHS,
            "base_density_per_um": cfg.num_nodes / cfg.L,
            "num_h0_per_point": args.scaling_trials,
            "num_h1_per_point": args.scaling_trials,
            "calibration_trials_per_strategy": args.scaling_calibration_trials,
        },
        "block_elapsed_seconds": block_seconds,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "phase5_effective_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[phase5] completed in {metadata['elapsed_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
