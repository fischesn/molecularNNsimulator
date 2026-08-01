from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_with_count(path: Path, expected: int, counts: dict[str, int]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    counts[str(path.name)] = len(frame)
    require(len(frame) == expected, f"{path.name}: expected {expected}, got {len(frame)}")
    return frame


def require_disjoint_seeds(evaluation: pd.DataFrame, calibration: pd.DataFrame, label: str) -> None:
    overlap = set(evaluation["scenario_seed"]) & set(calibration["scenario_seed"])
    require(not overlap, f"{label}: calibration/evaluation seed overlap: {len(overlap)}")


def require_paired(frame: pd.DataFrame, group_columns: list[str], expected_strategies: int, label: str) -> None:
    paired = frame.groupby(group_columns, dropna=False).agg(
        num_strategies=("strategy", "nunique"),
        num_seeds=("scenario_seed", "nunique"),
    )
    require(
        bool((paired["num_strategies"] == expected_strategies).all()),
        f"{label}: incomplete strategy pairing",
    )
    require(bool((paired["num_seeds"] == 1).all()), f"{label}: paired strategies use different seeds")


def validate(output_dir: Path) -> dict:
    counts: dict[str, int] = {}
    grid_dir = output_dir / "eir_grid_search"

    grid_summary = load_with_count(grid_dir / "eir_grid_summary.csv", 54, counts)
    require(
        len(grid_summary[["eir_gate_quantile", "hysteresis_margin_eir", "gate_hysteresis_margin", "w2"]].drop_duplicates()) == 54,
        "grid: parameter combinations are not unique",
    )
    grid_eir_eval = load_with_count(grid_dir / "eir_grid_eir_trials.csv", 81_000, counts)
    grid_eir_cal = load_with_count(grid_dir / "eir_grid_eir_calibration_trials.csv", 10_800, counts)
    grid_ref_eval = load_with_count(grid_dir / "eir_grid_reference_trials.csv", 3_000, counts)
    grid_ref_cal = load_with_count(grid_dir / "eir_grid_reference_calibration_trials.csv", 400, counts)
    require_disjoint_seeds(grid_eir_eval, grid_eir_cal, "grid EIR")
    require_disjoint_seeds(grid_ref_eval, grid_ref_cal, "grid reference")
    require_paired(
        grid_ref_eval,
        ["a1", "a2", "state_h1", "trial_index"],
        2,
        "grid RR/TR evaluation",
    )

    for parameter in ["gateway_integration_tau", "diffusion_coeff", "local_send_prob_target"]:
        summary = load_with_count(output_dir / f"sensitivity_{parameter}.csv", 18, counts)
        evaluation = load_with_count(output_dir / f"sensitivity_{parameter}_trials.csv", 10_800, counts)
        calibration = load_with_count(output_dir / f"sensitivity_{parameter}_calibration_trials.csv", 4_500, counts)
        require(summary["sensitivity_value"].nunique() == 3, f"{parameter}: expected three values")
        require_disjoint_seeds(evaluation, calibration, parameter)
        require_paired(
            evaluation,
            ["sensitivity_value", "a1", "a2", "state_h1", "trial_index"],
            3,
            f"{parameter} evaluation",
        )
        require_paired(
            calibration,
            ["sensitivity_value", "trial_index"],
            3,
            f"{parameter} calibration",
        )

    scaling_summary = load_with_count(output_dir / "scaling_summary.csv", 24, counts)
    scaling_eval = load_with_count(output_dir / "scaling_trials.csv", 14_400, counts)
    scaling_cal = load_with_count(output_dir / "scaling_calibration_trials.csv", 12_000, counts)
    require(scaling_summary["scaling_mode"].nunique() == 2, "scaling: expected two scaling modes")
    require_disjoint_seeds(scaling_eval, scaling_cal, "scaling")
    require_paired(
        scaling_eval,
        ["scaling_mode", "scaling_point_index", "state_h1", "trial_index"],
        3,
        "scaling evaluation",
    )
    require_paired(
        scaling_cal,
        ["scaling_mode", "scaling_point_index", "trial_index"],
        3,
        "scaling calibration",
    )

    return {
        "status": "passed",
        "row_counts": counts,
        "checks": [
            "expected row counts",
            "54 unique grid configurations",
            "disjoint calibration/evaluation seeds",
            "paired RR/TR grid reference scenarios",
            "paired RR/TR/EIR sensitivity scenarios",
            "paired RR/TR/EIR scaling scenarios",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase-5 result artifacts.")
    parser.add_argument("--output-dir", default="phase5_sensitivity_2026_08_01")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    report = validate(output_dir)
    (output_dir / "phase5_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
