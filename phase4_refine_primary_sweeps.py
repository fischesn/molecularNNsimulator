from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent / "dna-nn-simulator-v2.0.py"
SPEC = importlib.util.spec_from_file_location("dna_nn_simulator_v2_phase4", SCRIPT_PATH)
SIM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SIM
SPEC.loader.exec_module(SIM)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute the two publication-facing accuracy sweeps with higher trial counts."
    )
    parser.add_argument("--config", default=".config_sim")
    parser.add_argument("--output-dir", default="phase4_camera_ready_2026_08_01")
    parser.add_argument("--anomaly-trials", type=int, default=1000)
    parser.add_argument("--anomaly-calibration-trials", type=int, default=1000)
    parser.add_argument("--noise-trials", type=int, default=1000)
    parser.add_argument("--noise-calibration-trials", type=int, default=2000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    cfg, _, _, _, _, sweeps, _ = SIM.load_bundle_from_config(config_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("[phase4-refinement] recomputing anomaly-strength sweep", flush=True)
    anomaly_summary, anomaly_trials, anomaly_calibration = SIM.sweep_anomaly_pair(
        cfg,
        sweeps.anomaly_a1_values,
        sweeps.anomaly_a2_ratio,
        num_h0=args.anomaly_trials,
        num_h1=args.anomaly_trials,
        calib_trials=args.anomaly_calibration_trials,
        return_trials=True,
    )
    anomaly_summary.to_csv(output_dir / "sweep_anomaly.csv", index=False)
    anomaly_trials.to_csv(output_dir / "sweep_anomaly_trials.csv", index=False)
    anomaly_calibration.to_csv(
        output_dir / "sweep_anomaly_calibration_trials.csv", index=False
    )
    SIM.plot_sweep(
        anomaly_summary,
        "a1",
        "P_D",
        output_dir / "plot_detection_vs_anomaly.png",
        "Detection probability vs anomaly strength",
    )
    SIM.plot_pareto(
        anomaly_summary,
        output_dir / "plot_pareto.png",
        "Pareto view (communication load until detection vs detection)",
    )

    print("[phase4-refinement] recomputing noise sweep", flush=True)
    noise_summary, noise_trials, noise_calibration = SIM.sweep_parameter(
        cfg,
        "sigma1",
        sweeps.noise_values,
        num_h0=args.noise_trials,
        num_h1=args.noise_trials,
        calib_trials=args.noise_calibration_trials,
        return_trials=True,
    )
    noise_summary.to_csv(output_dir / "sweep_noise.csv", index=False)
    noise_trials.to_csv(output_dir / "sweep_noise_trials.csv", index=False)
    noise_calibration.to_csv(
        output_dir / "sweep_noise_calibration_trials.csv", index=False
    )
    SIM.plot_sweep(
        noise_summary,
        "sigma1",
        "P_FA",
        output_dir / "plot_false_alarm_vs_noise.png",
        "False alarm probability vs noise",
    )

    metadata = {
        "purpose": "Higher-confidence publication-facing Phase-4 accuracy sweeps",
        "simulator": SCRIPT_PATH.name,
        "config_path": str(config_path.resolve()),
        "simulation": asdict(cfg),
        "anomaly_sweep": {
            "num_h0_per_point": args.anomaly_trials,
            "num_h1_per_point": args.anomaly_trials,
            "calibration_trials_per_strategy": args.anomaly_calibration_trials,
            "shared_calibration_across_anomaly_strengths": True,
            "values": sweeps.anomaly_a1_values,
            "a2_ratio": sweeps.anomaly_a2_ratio,
        },
        "noise_sweep": {
            "num_h0_per_point": args.noise_trials,
            "num_h1_per_point": args.noise_trials,
            "calibration_trials_per_strategy_and_point": args.noise_calibration_trials,
            "values": sweeps.noise_values,
        },
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "phase4_primary_refinement.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"[phase4-refinement] completed in {metadata['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
