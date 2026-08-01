from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


GRID_PARAMETERS = [
    "eir_gate_quantile",
    "hysteresis_margin_eir",
    "gate_hysteresis_margin",
    "w2",
]

PFA_TARGET = 0.05
PFA_TOLERANCE = 0.03
MAX_RELATIVE_TRAFFIC_OVERHEAD_VS_RR = 0.05
NUMERICAL_EPSILON = 1e-12


def json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def records_json(frame: pd.DataFrame):
    return [
        {key: json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def analyze_grid(output_dir: Path) -> dict:
    grid_dir = output_dir / "eir_grid_search"
    grid = pd.read_csv(grid_dir / "eir_grid_summary.csv")
    grid["pfa_tolerance"] = PFA_TOLERANCE
    grid["pfa_feasible"] = (
        grid["pfa_abs_deviation"] <= grid["pfa_tolerance"] + NUMERICAL_EPSILON
    )
    grid["detection_gain_vs_both"] = (
        (grid["delta_P_D_vs_RR"] > 0.0) & (grid["delta_P_D_vs_TR"] > 0.0)
    )
    grid["relative_C_H1_vs_RR"] = (
        grid["eir_mean_C_H1"] / grid["rr_mean_C_H1"] - 1.0
    )
    grid["max_relative_traffic_overhead_vs_RR"] = (
        MAX_RELATIVE_TRAFFIC_OVERHEAD_VS_RR
    )
    grid["traffic_competitive_vs_RR"] = (
        grid["relative_C_H1_vs_RR"]
        <= MAX_RELATIVE_TRAFFIC_OVERHEAD_VS_RR + NUMERICAL_EPSILON
    )
    grid["robust_success"] = (
        grid["pfa_feasible"]
        & grid["detection_gain_vs_both"]
        & grid["traffic_competitive_vs_RR"]
    )
    grid = grid.sort_values(
        [
            "robust_success",
            "pfa_feasible",
            "eir_mean_P_D",
            "pfa_abs_deviation",
            "relative_C_H1_vs_RR",
            "eir_mean_delay",
        ],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)
    grid["robust_rank"] = np.arange(1, len(grid) + 1)
    grid.to_csv(grid_dir / "eir_grid_summary_robust.csv", index=False)

    best = grid.iloc[0]
    (grid_dir / "eir_grid_best_robust.json").write_text(
        json.dumps({key: json_value(value) for key, value in best.items()}, indent=2),
        encoding="utf-8",
    )

    marginal_rows = []
    for parameter in GRID_PARAMETERS:
        for value, group in grid.groupby(parameter):
            marginal_rows.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "num_configurations": len(group),
                    "pfa_feasible_fraction": group["pfa_feasible"].mean(),
                    "robust_success_fraction": group["robust_success"].mean(),
                    "mean_eir_P_D": group["eir_mean_P_D"].mean(),
                    "min_eir_P_D": group["eir_mean_P_D"].min(),
                    "max_eir_P_D": group["eir_mean_P_D"].max(),
                    "mean_delta_P_D_vs_RR": group["delta_P_D_vs_RR"].mean(),
                    "mean_delta_P_D_vs_TR": group["delta_P_D_vs_TR"].mean(),
                    "mean_delta_C_H1_vs_RR": group["delta_C_H1_vs_RR"].mean(),
                    "mean_delta_C_H1_vs_TR": group["delta_C_H1_vs_TR"].mean(),
                }
            )
    marginals = pd.DataFrame(marginal_rows)
    marginals.to_csv(grid_dir / "eir_grid_parameter_marginals.csv", index=False)

    successes = grid[grid["robust_success"]].copy()
    if successes.empty:
        ranges = {}
    else:
        ranges = {
            parameter: sorted(float(v) for v in successes[parameter].unique())
            for parameter in GRID_PARAMETERS
        }
    near_optimal = successes[
        successes["eir_mean_P_D"] >= successes["eir_mean_P_D"].max() - 0.02
    ] if not successes.empty else successes
    near_optimal_ranges = {
        parameter: sorted(float(v) for v in near_optimal[parameter].unique())
        for parameter in GRID_PARAMETERS
    } if not near_optimal.empty else {}

    return {
        "num_configurations": len(grid),
        "num_pfa_feasible": int(grid["pfa_feasible"].sum()),
        "num_robust_success": int(grid["robust_success"].sum()),
        "criteria": {
            "P_FA_target": PFA_TARGET,
            "P_FA_tolerance": PFA_TOLERANCE,
            "positive_P_D_gain_vs_RR_and_TR": True,
            "max_relative_C_H1_overhead_vs_RR": MAX_RELATIVE_TRAFFIC_OVERHEAD_VS_RR,
        },
        "robust_success_parameter_values": ranges,
        "near_optimal_within_0.02_PD_parameter_values": near_optimal_ranges,
        "best_robust": {key: json_value(value) for key, value in best.items()},
        "top_five_robust": records_json(grid.head(5)),
    }


def comparative_sensitivity(frame: pd.DataFrame, parameter: str) -> pd.DataFrame:
    index_cols = ["sensitivity_value", "a1", "a2"]
    metrics = ["P_D", "P_FA", "C_H1_molecules_avg", "D_avg_after_onset"]
    rows = []
    for keys, group in frame.groupby(index_cols):
        by_strategy = group.set_index("strategy")
        if not {"RR", "TR", "EIR"}.issubset(by_strategy.index):
            continue
        row = {
            "sensitivity_parameter": parameter,
            "sensitivity_value": keys[0],
            "a1": keys[1],
            "a2": keys[2],
        }
        for strategy in ["RR", "TR", "EIR"]:
            for metric in metrics:
                row[f"{strategy}_{metric}"] = by_strategy.loc[strategy, metric]
        row["EIR_delta_P_D_vs_RR"] = row["EIR_P_D"] - row["RR_P_D"]
        row["EIR_delta_P_D_vs_TR"] = row["EIR_P_D"] - row["TR_P_D"]
        row["EIR_delta_C_H1_vs_RR"] = row["EIR_C_H1_molecules_avg"] - row["RR_C_H1_molecules_avg"]
        row["EIR_delta_C_H1_vs_TR"] = row["EIR_C_H1_molecules_avg"] - row["TR_C_H1_molecules_avg"]
        row["EIR_relative_C_H1_vs_RR"] = row["EIR_C_H1_molecules_avg"] / row["RR_C_H1_molecules_avg"] - 1.0
        row["EIR_relative_C_H1_vs_TR"] = row["EIR_C_H1_molecules_avg"] / row["TR_C_H1_molecules_avg"] - 1.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["sensitivity_value", "a1"])


def analyze_sensitivity(output_dir: Path) -> dict:
    compact_frames = []
    summary = {}
    for parameter in [
        "gateway_integration_tau",
        "diffusion_coeff",
        "local_send_prob_target",
    ]:
        frame = pd.read_csv(output_dir / f"sensitivity_{parameter}.csv")
        compact = comparative_sensitivity(frame, parameter)
        compact_frames.append(compact)
        summary[parameter] = {
            "values": sorted(float(v) for v in compact["sensitivity_value"].unique()),
            "EIR_delta_P_D_vs_RR_range": [
                float(compact["EIR_delta_P_D_vs_RR"].min()),
                float(compact["EIR_delta_P_D_vs_RR"].max()),
            ],
            "EIR_delta_P_D_vs_TR_range": [
                float(compact["EIR_delta_P_D_vs_TR"].min()),
                float(compact["EIR_delta_P_D_vs_TR"].max()),
            ],
            "EIR_relative_C_H1_vs_RR_range": [
                float(compact["EIR_relative_C_H1_vs_RR"].min()),
                float(compact["EIR_relative_C_H1_vs_RR"].max()),
            ],
            "EIR_relative_C_H1_vs_TR_range": [
                float(compact["EIR_relative_C_H1_vs_TR"].min()),
                float(compact["EIR_relative_C_H1_vs_TR"].max()),
            ],
            "all_points_positive_detection_gain_vs_RR": bool(
                (compact["EIR_delta_P_D_vs_RR"] > 0.0).all()
            ),
            "all_points_positive_detection_gain_vs_TR": bool(
                (compact["EIR_delta_P_D_vs_TR"] > 0.0).all()
            ),
        }
    all_compact = pd.concat(compact_frames, ignore_index=True)
    all_compact.to_csv(output_dir / "sensitivity_compact_comparison.csv", index=False)
    return summary


def analyze_scaling(output_dir: Path) -> dict:
    scaling = pd.read_csv(output_dir / "scaling_summary.csv")
    fixed = scaling[scaling["scaling_mode"] == "fixed_segment_increasing_density"].copy()
    constant = scaling[scaling["scaling_mode"] == "constant_density_increasing_segment"].copy()

    def selected_records(frame: pd.DataFrame):
        columns = [
            "L",
            "num_nodes",
            "node_density_per_um",
            "strategy",
            "P_D",
            "P_FA",
            "C_H0_molecules_avg",
            "C_H1_molecules_avg",
            "D_avg_after_onset",
        ]
        return records_json(frame[columns].sort_values(["L", "num_nodes", "strategy"]))

    return {
        "fixed_segment_increasing_density": selected_records(fixed),
        "constant_density_increasing_segment": selected_records(constant),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-process Phase-5 sensitivity results.")
    parser.add_argument("--output-dir", default="phase5_sensitivity_2026_08_01")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    payload = {
        "grid": analyze_grid(output_dir),
        "sensitivity": analyze_sensitivity(output_dir),
        "scaling": analyze_scaling(output_dir),
    }
    (output_dir / "phase5_robust_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["grid"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
