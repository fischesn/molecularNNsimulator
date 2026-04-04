from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    L: float = 1000.0
    gateway_x: float = 0.0
    anomaly_x: float = 750.0
    dt: float = 1.0
    t_max: float = 300.0
    anomaly_start: float = 30.0
    num_nodes: int = 50

    mu1_h0: float = 0.20
    mu2_h0: float = 0.10
    sigma1: float = 0.16
    sigma2: float = 0.05

    a1: float = 0.30
    a2: float = 0.12
    lambda1: float = 250.0
    lambda2: float = 200.0

    w1: float = 1.0
    w2: float = 0.75
    inference_delay: float = 20.0

    use_eir_gate: bool = True
    eir_gate_quantile: float = 0.85
    eir_gate_manual: Optional[float] = None

    use_hysteresis: bool = True
    hysteresis_margin_rr: float = 0.000
    hysteresis_margin_tr: float = 0.005
    hysteresis_margin_eir: float = 0.120
    gate_hysteresis_margin: float = 0.020

    tau0: float = 5.0
    kappa_delay: float = 0.05
    lambda_c: float = 400.0
    channel_spread_steps: int = 3
    alarm_molecules: float = 100.0

    local_send_prob_target: float = 0.02
    gateway_false_alarm_target: float = 0.05

    edge_triggered: bool = True
    refractory_period: float = 10.0
    allow_refresh_if_still_positive: bool = False
    refresh_period: float = 60.0

    use_temporal_correlation: bool = True
    temporal_alpha: float = 0.85

    random_seed: int = 13


@dataclass
class LocalThresholds:
    rr_tau1: float
    rr_tau2: float
    tr_tau: float
    eir_theta: float
    eir_gate_tau: float


@dataclass
class GatewayThresholds:
    rr: float
    tr: float
    eir: float


@dataclass
class TrialOutcome:
    state_h1: bool
    detected: bool
    pre_onset_alarm: bool
    detection_time: float
    first_alarm_time: float
    transmissions: int
    max_gateway_evidence: float


def truncated_normal(mean: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    x = mean + sigma * rng.standard_normal(size=mean.shape)
    return np.maximum(0.0, x)


def positions_uniform(n: int, L: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, L, size=n)


def anomaly_profile(positions: np.ndarray, anomaly_x: float, amplitude: float, lamb: float) -> np.ndarray:
    dist = np.abs(positions - anomaly_x)
    return amplitude * np.exp(-dist / lamb)


def compute_channel_params(positions: np.ndarray, cfg: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
    d = np.abs(positions - cfg.gateway_x)
    tau = cfg.tau0 + cfg.kappa_delay * d
    eta = np.exp(-d / cfg.lambda_c)
    return tau, eta


def make_memory_kernel(spread_steps: int) -> np.ndarray:
    if spread_steps <= 0:
        return np.array([1.0], dtype=float)
    grid = np.arange(0, 2 * spread_steps + 1)
    center = spread_steps
    sigma = max(1.0, spread_steps / 2.0)
    kernel = np.exp(-0.5 * ((grid - center) / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def add_emission(A: np.ndarray, emit_time: float, tau_i: float, eta_i: float, cfg: SimulationConfig, kernel: np.ndarray) -> None:
    arrival_time = emit_time + tau_i
    arrival_idx = int(round(arrival_time / cfg.dt))
    if arrival_idx >= len(A):
        return
    start_idx = arrival_idx - len(kernel) // 2
    for offset, weight in enumerate(kernel):
        idx = start_idx + offset
        if 0 <= idx < len(A):
            A[idx] += cfg.alarm_molecules * eta_i * weight


def sample_markers_for_time(positions: np.ndarray, t: float, cfg: SimulationConfig, state_h1: bool, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    base1 = np.full_like(positions, cfg.mu1_h0, dtype=float)
    base2 = np.full_like(positions, cfg.mu2_h0, dtype=float)
    if state_h1 and t >= cfg.anomaly_start:
        base1 = base1 + anomaly_profile(positions, cfg.anomaly_x, cfg.a1, cfg.lambda1)
        base2 = base2 + anomaly_profile(positions, cfg.anomaly_x, cfg.a2, cfg.lambda2)
    x1 = truncated_normal(base1, cfg.sigma1, rng)
    x2 = truncated_normal(base2, cfg.sigma2, rng)
    return x1, x2


def _rr_tau_for_target(cfg: SimulationConfig, samples: int = 100_000) -> Tuple[float, float]:
    rng = np.random.default_rng(cfg.random_seed + 101)
    x1 = truncated_normal(np.full(samples, cfg.mu1_h0), cfg.sigma1, rng)
    x2 = truncated_normal(np.full(samples, cfg.mu2_h0), cfg.sigma2, rng)
    q_target = cfg.local_send_prob_target
    lo, hi = 1e-6, 0.5
    best_tau1, best_tau2 = None, None
    for _ in range(40):
        p_single = 0.5 * (lo + hi)
        tau1 = float(np.quantile(x1, 1.0 - p_single))
        tau2 = float(np.quantile(x2, 1.0 - p_single))
        send_prob = float(np.mean((x1 > tau1) | (x2 > tau2)))
        best_tau1, best_tau2 = tau1, tau2
        if send_prob > q_target:
            hi = p_single
        else:
            lo = p_single
    return best_tau1, best_tau2


def _calibrate_eir_theta_with_gate(z: np.ndarray, x2: np.ndarray, gate_tau: float, target_prob: float) -> float:
    mask = x2 > gate_tau
    if mask.mean() <= target_prob + 1e-9:
        return float(np.quantile(z, 0.01))
    z_masked = z[mask]
    cond_target = target_prob / mask.mean()
    cond_target = min(max(cond_target, 1e-6), 1.0 - 1e-6)
    return float(np.quantile(z_masked, 1.0 - cond_target))


def calibrate_local_thresholds(cfg: SimulationConfig, samples: int = 100_000) -> LocalThresholds:
    rng = np.random.default_rng(cfg.random_seed + 102)
    x1 = truncated_normal(np.full(samples, cfg.mu1_h0), cfg.sigma1, rng)
    x2 = truncated_normal(np.full(samples, cfg.mu2_h0), cfg.sigma2, rng)
    rr_tau1, rr_tau2 = _rr_tau_for_target(cfg, samples=samples)
    tr_tau = float(np.quantile(x1, 1.0 - cfg.local_send_prob_target))
    z = cfg.w1 * x1 + cfg.w2 * x2
    if cfg.eir_gate_manual is not None:
        eir_gate_tau = float(cfg.eir_gate_manual)
    else:
        eir_gate_tau = float(np.quantile(x2, cfg.eir_gate_quantile))
    if cfg.use_eir_gate:
        eir_theta = _calibrate_eir_theta_with_gate(z=z, x2=x2, gate_tau=eir_gate_tau, target_prob=cfg.local_send_prob_target)
    else:
        eir_theta = float(np.quantile(z, 1.0 - cfg.local_send_prob_target))
    return LocalThresholds(rr_tau1=rr_tau1, rr_tau2=rr_tau2, tr_tau=tr_tau, eir_theta=eir_theta, eir_gate_tau=eir_gate_tau)


def raw_positive_state(strategy: str, x1: np.ndarray, x2: np.ndarray, thresholds: LocalThresholds, cfg: SimulationConfig) -> np.ndarray:
    if strategy == "RR":
        return (x1 > thresholds.rr_tau1) | (x2 > thresholds.rr_tau2)
    if strategy == "TR":
        return x1 > thresholds.tr_tau
    if strategy == "EIR":
        z = cfg.w1 * x1 + cfg.w2 * x2
        state = z > thresholds.eir_theta
        if cfg.use_eir_gate:
            state = state & (x2 > thresholds.eir_gate_tau)
        return state
    raise ValueError(f"Unknown strategy: {strategy}")


def hysteretic_positive_state(strategy: str, x1: np.ndarray, x2: np.ndarray, prev_state: np.ndarray, thresholds: LocalThresholds, cfg: SimulationConfig) -> np.ndarray:
    if strategy == "RR":
        on = (x1 > thresholds.rr_tau1) | (x2 > thresholds.rr_tau2)
        off = (x1 < (thresholds.rr_tau1 - cfg.hysteresis_margin_rr)) & (x2 < (thresholds.rr_tau2 - cfg.hysteresis_margin_rr))
        return (prev_state & (~off)) | ((~prev_state) & on)
    if strategy == "TR":
        on = x1 > thresholds.tr_tau
        off = x1 < (thresholds.tr_tau - cfg.hysteresis_margin_tr)
        return (prev_state & (~off)) | ((~prev_state) & on)
    if strategy == "EIR":
        z = cfg.w1 * x1 + cfg.w2 * x2
        on = z > thresholds.eir_theta
        off = z < (thresholds.eir_theta - cfg.hysteresis_margin_eir)
        if cfg.use_eir_gate:
            gate_on = x2 > thresholds.eir_gate_tau
            gate_off = x2 < (thresholds.eir_gate_tau - cfg.gate_hysteresis_margin)
            on = on & gate_on
            off = off & gate_off
        return (prev_state & (~off)) | ((~prev_state) & on)
    raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_positive_state(strategy: str, x1: np.ndarray, x2: np.ndarray, prev_state: np.ndarray, thresholds: LocalThresholds, cfg: SimulationConfig) -> np.ndarray:
    if cfg.use_hysteresis:
        return hysteretic_positive_state(strategy, x1, x2, prev_state, thresholds, cfg)
    return raw_positive_state(strategy, x1, x2, thresholds, cfg)


def should_emit_alarm(current_positive: np.ndarray, prev_positive: np.ndarray, last_send_time: np.ndarray, t: float, cfg: SimulationConfig) -> np.ndarray:
    if cfg.edge_triggered:
        send_now = current_positive & (~prev_positive)
        if cfg.allow_refresh_if_still_positive:
            refresh_ok = current_positive & ((t - last_send_time) >= cfg.refresh_period)
            send_now = send_now | refresh_ok
        send_now = send_now & ((t - last_send_time) >= cfg.refractory_period)
        return send_now
    return current_positive & ((t - last_send_time) >= cfg.refractory_period)


def sample_local_decisions(cfg: SimulationConfig, thresholds: LocalThresholds, state_h1: bool, num_samples: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed + (500 if state_h1 else 400))
    positions = positions_uniform(num_samples, cfg.L, rng)
    t = cfg.anomaly_start + cfg.dt if state_h1 else 0.0
    x1, x2 = sample_markers_for_time(positions, t, cfg, state_h1, rng)
    prev = np.zeros(num_samples, dtype=bool)
    rr = evaluate_positive_state("RR", x1, x2, prev, thresholds, cfg)
    tr = evaluate_positive_state("TR", x1, x2, prev, thresholds, cfg)
    eir = evaluate_positive_state("EIR", x1, x2, prev, thresholds, cfg)
    z = cfg.w1 * x1 + cfg.w2 * x2 - thresholds.eir_theta
    dist = np.abs(positions - cfg.anomaly_x)
    return pd.DataFrame({"state_h1": int(state_h1), "x": positions, "dist_to_anomaly": dist, "x1": x1, "x2": x2, "z_eir": z, "send_rr": rr.astype(int), "send_tr": tr.astype(int), "send_eir": eir.astype(int)})


def summarize_local_decisions(df_h0: pd.DataFrame, df_h1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df in [("H0", df_h0), ("H1", df_h1)]:
        rows.append({"state": label, "x1_mean": float(df["x1"].mean()), "x2_mean": float(df["x2"].mean()), "send_rr_prob": float(df["send_rr"].mean()), "send_tr_prob": float(df["send_tr"].mean()), "send_eir_prob": float(df["send_eir"].mean()), "eir_score_mean": float(df["z_eir"].mean())})
    return pd.DataFrame(rows)


def distance_send_profile(df: pd.DataFrame, send_col: str, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, df["dist_to_anomaly"].max(), bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    out = []
    vals = df["dist_to_anomaly"].to_numpy()
    sends = df[send_col].to_numpy()
    for left, right, mid in zip(edges[:-1], edges[1:], mids):
        mask = (vals >= left) & (vals < right if right < edges[-1] else vals <= right)
        prob = np.nan if mask.sum() == 0 else float(sends[mask].mean())
        out.append({"distance_mid": mid, "send_probability": prob})
    return pd.DataFrame(out)


def plot_marker_histograms(df_h0: pd.DataFrame, df_h1: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.hist(df_h0["x1"], bins=40, alpha=0.5, density=True, label="x1, H0")
    plt.hist(df_h1["x1"], bins=40, alpha=0.5, density=True, label="x1, H1")
    plt.hist(df_h0["x2"], bins=40, alpha=0.5, density=True, label="x2, H0")
    plt.hist(df_h1["x2"], bins=40, alpha=0.5, density=True, label="x2, H1")
    plt.xlabel("marker concentration")
    plt.ylabel("density")
    plt.title("Local marker distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_marker_scatter(df_h0: pd.DataFrame, df_h1: pd.DataFrame, thresholds: LocalThresholds, cfg: SimulationConfig, outpath: Path, max_points: int = 1200) -> None:
    rng = np.random.default_rng(cfg.random_seed + 901)
    h0 = df_h0.sample(n=min(max_points, len(df_h0)), random_state=int(rng.integers(0, 1_000_000)))
    h1 = df_h1.sample(n=min(max_points, len(df_h1)), random_state=int(rng.integers(0, 1_000_000)))
    plt.figure(figsize=(5.5, 5))
    plt.scatter(h0["x1"], h0["x2"], s=10, alpha=0.35, label="H0")
    plt.scatter(h1["x1"], h1["x2"], s=10, alpha=0.35, label="H1")
    x_vals = np.linspace(0, max(df_h0["x1"].max(), df_h1["x1"].max()) * 1.05, 200)
    y_vals = (thresholds.eir_theta - cfg.w1 * x_vals) / cfg.w2
    plt.plot(x_vals, y_vals, linestyle="--", linewidth=1.5, label="EIR linear boundary")
    if cfg.use_eir_gate:
        plt.axhline(y=thresholds.eir_gate_tau, linestyle=":", linewidth=1.5, label="EIR gate on x2")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Two-marker local decision space")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_distance_send_profiles(df_h1: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(6, 4))
    for send_col, label in [("send_rr", "RR"), ("send_tr", "TR"), ("send_eir", "EIR")]:
        prof = distance_send_profile(df_h1, send_col, bins=10)
        plt.plot(prof["distance_mid"], prof["send_probability"], marker="o", label=label)
    plt.xlabel("distance to anomaly source")
    plt.ylabel("local send probability under H1")
    plt.title("Distance-dependent local send probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def collect_state_dynamics_metrics(cfg: SimulationConfig, thresholds: LocalThresholds, strategy: str, state_h1: bool, num_trials: int = 100) -> pd.DataFrame:
    rows = []
    for trial in range(num_trials):
        rng = np.random.default_rng(cfg.random_seed + 10_000 + trial + (0 if not state_h1 else 100_000) + sum(ord(c) for c in strategy))
        positions = positions_uniform(cfg.num_nodes, cfg.L, rng)
        prev_state = np.zeros(cfg.num_nodes, dtype=bool)
        K = int(round(cfg.t_max / cfg.dt)) + 1
        on_time_steps = np.zeros(cfg.num_nodes, dtype=int)
        rising_edges = np.zeros(cfg.num_nodes, dtype=int)
        current_on_start = np.full(cfg.num_nodes, -1, dtype=int)
        on_durations = []
        x1_prev, x2_prev = sample_markers_for_time(positions, 0.0, cfg, state_h1, rng)
        alpha = cfg.temporal_alpha
        for k in range(K):
            t = k * cfg.dt
            x1_raw, x2_raw = sample_markers_for_time(positions, t, cfg, state_h1, rng)
            if cfg.use_temporal_correlation:
                x1 = alpha * x1_prev + (1.0 - alpha) * x1_raw
                x2 = alpha * x2_prev + (1.0 - alpha) * x2_raw
                x1_prev, x2_prev = x1, x2
            else:
                x1, x2 = x1_raw, x2_raw
            current_state = evaluate_positive_state(strategy, x1, x2, prev_state, thresholds, cfg)
            rising = current_state & (~prev_state)
            falling = (~current_state) & prev_state
            rising_edges += rising.astype(int)
            on_time_steps += current_state.astype(int)
            for i in np.where(rising)[0]:
                current_on_start[i] = k
            for i in np.where(falling)[0]:
                if current_on_start[i] >= 0:
                    on_durations.append((k - current_on_start[i]) * cfg.dt)
                    current_on_start[i] = -1
            prev_state = current_state.copy()
        for i in range(cfg.num_nodes):
            if current_on_start[i] >= 0:
                on_durations.append((K - current_on_start[i]) * cfg.dt)
        rows.append({"strategy": strategy, "state_h1": int(state_h1), "trial": trial, "mean_on_fraction": float(on_time_steps.sum()) / float(cfg.num_nodes * K), "mean_rising_edges_per_node": float(rising_edges.mean()), "total_rising_edges": int(rising_edges.sum()), "mean_on_duration_s": float(np.mean(on_durations)) if len(on_durations) > 0 else 0.0})
    return pd.DataFrame(rows)


def run_state_dynamics_diagnostics(cfg: SimulationConfig, output_dir: Path, num_trials: int = 100) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = calibrate_local_thresholds(cfg)
    dfs = []
    for state_h1 in [False, True]:
        for strategy in ["RR", "TR", "EIR"]:
            dfs.append(collect_state_dynamics_metrics(cfg=cfg, thresholds=thresholds, strategy=strategy, state_h1=state_h1, num_trials=num_trials))
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(output_dir / "state_dynamics_trials.csv", index=False)
    summary = (all_df.groupby(["strategy", "state_h1"], as_index=False)
               .agg(mean_on_fraction=("mean_on_fraction", "mean"),
                    mean_rising_edges_per_node=("mean_rising_edges_per_node", "mean"),
                    mean_on_duration_s=("mean_on_duration_s", "mean"))
               .sort_values(["state_h1", "strategy"])
               .reset_index(drop=True))
    summary.to_csv(output_dir / "state_dynamics_summary.csv", index=False)
    return summary


def run_local_diagnostics(cfg: SimulationConfig, output_dir: Path, num_samples: int = 5000, num_trials_state: int = 100):
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = calibrate_local_thresholds(cfg)
    df_h0 = sample_local_decisions(cfg, thresholds, state_h1=False, num_samples=num_samples)
    df_h1 = sample_local_decisions(cfg, thresholds, state_h1=True, num_samples=num_samples)
    local_summary = summarize_local_decisions(df_h0, df_h1)
    dynamics_summary = run_state_dynamics_diagnostics(cfg, output_dir, num_trials=num_trials_state)
    df_h0.to_csv(output_dir / "local_samples_H0.csv", index=False)
    df_h1.to_csv(output_dir / "local_samples_H1.csv", index=False)
    local_summary.to_csv(output_dir / "local_diagnostics_summary.csv", index=False)
    plot_marker_histograms(df_h0, df_h1, output_dir / "local_marker_histograms.png")
    plot_marker_scatter(df_h0, df_h1, thresholds, cfg, output_dir / "local_marker_scatter.png")
    plot_distance_send_profiles(df_h1, output_dir / "local_distance_send_profiles.png")
    return df_h0, df_h1, local_summary, dynamics_summary, thresholds


def simulate_one_trial(cfg: SimulationConfig, thresholds: LocalThresholds, gateway_threshold: float, strategy: str, state_h1: bool, rng: np.random.Generator) -> TrialOutcome:
    K = int(round(cfg.t_max / cfg.dt)) + 1
    A = np.zeros(K, dtype=float)
    positions = positions_uniform(cfg.num_nodes, cfg.L, rng)
    tau, eta = compute_channel_params(positions, cfg)
    kernel = make_memory_kernel(cfg.channel_spread_steps)
    prev_positive = np.zeros(cfg.num_nodes, dtype=bool)
    last_send_time = np.full(cfg.num_nodes, -np.inf, dtype=float)
    transmissions = 0
    detected = False
    pre_onset_alarm = False
    detection_time = np.inf
    first_alarm_time = np.inf
    gateway_above_prev = False
    x1_prev, x2_prev = sample_markers_for_time(positions, 0.0, cfg, state_h1, rng)
    alpha = cfg.temporal_alpha

    for k in range(K):
        t = k * cfg.dt
        x1_raw, x2_raw = sample_markers_for_time(positions, t, cfg, state_h1, rng)
        if cfg.use_temporal_correlation:
            x1 = alpha * x1_prev + (1.0 - alpha) * x1_raw
            x2 = alpha * x2_prev + (1.0 - alpha) * x2_raw
            x1_prev, x2_prev = x1, x2
        else:
            x1, x2 = x1_raw, x2_raw

        current_positive = evaluate_positive_state(strategy, x1, x2, prev_positive, thresholds, cfg)
        send_now = should_emit_alarm(current_positive=current_positive, prev_positive=prev_positive, last_send_time=last_send_time, t=t, cfg=cfg)

        idxs = np.where(send_now)[0]
        if len(idxs) > 0:
            for i in idxs:
                emit_time = t + (cfg.inference_delay if strategy == "EIR" else 0.0)
                add_emission(A, emit_time, tau[i], eta[i], cfg, kernel)
            last_send_time[idxs] = t
            transmissions += int(len(idxs))

        prev_positive = current_positive.copy()
        gateway_above = A[k] > gateway_threshold
        upcross = gateway_above and (not gateway_above_prev)

        if upcross:
            if not np.isfinite(first_alarm_time):
                first_alarm_time = t
            if state_h1:
                if t < cfg.anomaly_start:
                    pre_onset_alarm = True
                elif not detected:
                    detected = True
                    detection_time = t
                    break

        gateway_above_prev = gateway_above

    return TrialOutcome(state_h1=bool(state_h1), detected=bool(detected), pre_onset_alarm=bool(pre_onset_alarm), detection_time=float(detection_time), first_alarm_time=float(first_alarm_time), transmissions=int(transmissions), max_gateway_evidence=float(np.max(A)))


def calibrate_gateway_threshold(cfg: SimulationConfig, thresholds: LocalThresholds, strategy: str, num_trials: int = 250) -> float:
    rng = np.random.default_rng(cfg.random_seed + sum(ord(c) for c in strategy) + 200)
    maxima = []
    dummy_threshold = np.inf
    for _ in range(num_trials):
        outcome = simulate_one_trial(cfg, thresholds, dummy_threshold, strategy, False, rng)
        maxima.append(outcome.max_gateway_evidence)
    q = 1.0 - cfg.gateway_false_alarm_target
    return float(np.quantile(np.asarray(maxima), q))


def calibrate_all_gateway_thresholds(cfg: SimulationConfig, thresholds: LocalThresholds, num_trials: int = 250) -> GatewayThresholds:
    return GatewayThresholds(rr=calibrate_gateway_threshold(cfg, thresholds, "RR", num_trials=num_trials),
                             tr=calibrate_gateway_threshold(cfg, thresholds, "TR", num_trials=num_trials),
                             eir=calibrate_gateway_threshold(cfg, thresholds, "EIR", num_trials=num_trials))


def run_trials(cfg: SimulationConfig, thresholds: LocalThresholds, gateway_thresholds: GatewayThresholds, strategy: str, num_h0: int = 200, num_h1: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed + sum(ord(c) for c in strategy) + 300)
    rows = []
    gt = {"RR": gateway_thresholds.rr, "TR": gateway_thresholds.tr, "EIR": gateway_thresholds.eir}[strategy]
    for state_h1, count in [(False, num_h0), (True, num_h1)]:
        for _ in range(count):
            outcome = simulate_one_trial(cfg=cfg, thresholds=thresholds, gateway_threshold=gt, strategy=strategy, state_h1=state_h1, rng=rng)
            rows.append({"strategy": strategy, "state_h1": int(outcome.state_h1), "detected": int(outcome.detected), "pre_onset_alarm": int(outcome.pre_onset_alarm), "detection_time": outcome.detection_time, "first_alarm_time": outcome.first_alarm_time, "transmissions": outcome.transmissions, "max_gateway_evidence": outcome.max_gateway_evidence})
    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame, cfg: SimulationConfig) -> pd.DataFrame:
    rows = []
    for strategy, group in df.groupby("strategy"):
        h1 = group[group["state_h1"] == 1]
        h0 = group[group["state_h1"] == 0]
        p_fa = float((h0["first_alarm_time"] < np.inf).mean()) if len(h0) else float("nan")
        p_d = float(h1["detected"].mean()) if len(h1) else float("nan")
        p_pre = float(h1["pre_onset_alarm"].mean()) if len(h1) else float("nan")
        comm_load_all = float(group["transmissions"].mean()) * cfg.alarm_molecules
        comm_load_h0 = float(h0["transmissions"].mean()) * cfg.alarm_molecules if len(h0) else float("nan")
        comm_load_h1 = float(h1["transmissions"].mean()) * cfg.alarm_molecules if len(h1) else float("nan")
        detected_h1 = h1[h1["detected"] == 1]["detection_time"]
        delay = float("inf") if len(detected_h1) == 0 else float(detected_h1.mean() - cfg.anomaly_start)
        detected_h1_full = h1[h1["detected"] == 1].copy()
        if len(detected_h1_full) > 0:
            durations = np.maximum(detected_h1_full["detection_time"].to_numpy() - cfg.anomaly_start, cfg.dt)
            molecules = detected_h1_full["transmissions"].to_numpy() * cfg.alarm_molecules
            rate_h1 = float(np.mean(molecules / durations))
        else:
            rate_h1 = float("nan")
        rows.append({"strategy": strategy, "P_D": p_d, "P_pre_onset_alarm": p_pre, "P_FA": p_fa, "C_total_molecules_avg": comm_load_all, "C_H0_molecules_avg": comm_load_h0, "C_H1_molecules_avg": comm_load_h1, "R_H1_molecules_per_s": rate_h1, "D_avg_after_onset": delay})
    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)


def run_experiment(cfg: SimulationConfig, num_h0: int = 200, num_h1: int = 200, calib_trials: int = 250):
    thresholds = calibrate_local_thresholds(cfg)
    gateway_thresholds = calibrate_all_gateway_thresholds(cfg, thresholds, num_trials=calib_trials)
    dfs = []
    for strategy in ["RR", "TR", "EIR"]:
        dfs.append(run_trials(cfg, thresholds, gateway_thresholds, strategy, num_h0=num_h0, num_h1=num_h1))
    all_trials = pd.concat(dfs, ignore_index=True)
    summary = summarize_results(all_trials, cfg)
    return all_trials, summary, thresholds, gateway_thresholds


def sweep_parameter(base_cfg: SimulationConfig, parameter_name: str, values: List[float], num_h0: int = 150, num_h1: int = 150, calib_trials: int = 200) -> pd.DataFrame:
    rows = []
    for value in values:
        cfg = SimulationConfig(**asdict(base_cfg))
        setattr(cfg, parameter_name, value)
        _, summary, _, _ = run_experiment(cfg, num_h0=num_h0, num_h1=num_h1, calib_trials=calib_trials)
        summary[parameter_name] = value
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def plot_sweep(df: pd.DataFrame, x_col: str, y_col: str, outpath: Path, title: Optional[str] = None) -> None:
    plt.figure(figsize=(6, 4))
    for strategy in ["RR", "TR", "EIR"]:
        g = df[df["strategy"] == strategy].sort_values(x_col)
        plt.plot(g[x_col], g[y_col], marker="o", label=strategy)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pareto(df: pd.DataFrame, outpath: Path, title: Optional[str] = None) -> None:
    plt.figure(figsize=(6, 4))
    for strategy in ["RR", "TR", "EIR"]:
        g = df[df["strategy"] == strategy]
        plt.scatter(g["C_H1_molecules_avg"], g["P_D"], label=strategy)
    plt.xlabel("C_H1_molecules_avg")
    plt.ylabel("P_D")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_metadata(cfg: SimulationConfig, thresholds: LocalThresholds, gateway_thresholds: GatewayThresholds, outpath: Path) -> None:
    data = {"config": asdict(cfg), "local_thresholds": asdict(thresholds), "gateway_thresholds": asdict(gateway_thresholds)}
    outpath.write_text(json.dumps(data, indent=2), encoding="utf-8")


def demo(output_dir: Path, cfg: Optional[SimulationConfig] = None) -> None:
    cfg = SimulationConfig() if cfg is None else cfg
    output_dir.mkdir(parents=True, exist_ok=True)
    all_trials, summary, thresholds, gateway_thresholds = run_experiment(cfg, num_h0=250, num_h1=250, calib_trials=250)
    all_trials.to_csv(output_dir / "trial_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    save_metadata(cfg, thresholds, gateway_thresholds, output_dir / "calibration.json")
    df_anomaly = sweep_parameter(cfg, "a1", [0.15, 0.25, 0.35, 0.50], num_h0=150, num_h1=150, calib_trials=200)
    df_noise = sweep_parameter(cfg, "sigma1", [0.08, 0.12, 0.16, 0.20], num_h0=150, num_h1=150, calib_trials=200)
    df_nodes = sweep_parameter(cfg, "num_nodes", [20, 40, 60, 100], num_h0=150, num_h1=150, calib_trials=200)
    df_ti = sweep_parameter(cfg, "inference_delay", [0.0, 10.0, 30.0, 60.0, 120.0], num_h0=150, num_h1=150, calib_trials=200)
    df_anomaly.to_csv(output_dir / "sweep_anomaly.csv", index=False)
    df_noise.to_csv(output_dir / "sweep_noise.csv", index=False)
    df_nodes.to_csv(output_dir / "sweep_nodes.csv", index=False)
    df_ti.to_csv(output_dir / "sweep_inference_delay.csv", index=False)
    plot_sweep(df_anomaly, "a1", "P_D", output_dir / "plot_detection_vs_anomaly.png", "Detection probability vs anomaly strength")
    plot_sweep(df_noise, "sigma1", "P_FA", output_dir / "plot_false_alarm_vs_noise.png", "False alarm probability vs noise")
    plot_sweep(df_nodes, "num_nodes", "C_H0_molecules_avg", output_dir / "plot_comm_load_h0_vs_nodes.png", "Communication load under H0 vs number of nodes")
    plot_sweep(df_nodes, "num_nodes", "C_H1_molecules_avg", output_dir / "plot_comm_load_h1_vs_nodes.png", "Communication load until detection under H1 vs number of nodes")
    plot_sweep(df_ti, "inference_delay", "D_avg_after_onset", output_dir / "plot_delay_vs_inference_delay.png", "Detection delay vs local inference delay")
    plot_pareto(df_anomaly, output_dir / "plot_pareto.png", "Pareto view (communication load until detection vs detection)")
    print("=== Baseline summary ===")
    print(summary.to_string(index=False))
    print("Output written to:", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test A simulator: temporally correlated markers + corrected post-onset detection.")
    parser.add_argument("--demo", action="store_true", help="Run full sweeps and export plots.")
    parser.add_argument("--diagnostics", action="store_true", help="Run local and state-dynamics diagnostics.")
    parser.add_argument("--output-dir", type=str, default="sim_output", help="Directory for exported files.")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of local diagnostic samples per state.")
    parser.add_argument("--num-trials-state", type=int, default=100, help="Number of state-dynamics trials.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    cfg = SimulationConfig(
        a1=0.30,
        a2=0.12,
        sigma1=0.16,
        sigma2=0.05,
        w1=1.0,
        w2=0.75,
        inference_delay=20.0,
        use_eir_gate=True,
        eir_gate_quantile=0.85,
        use_hysteresis=True,
        hysteresis_margin_rr=0.000,
        hysteresis_margin_tr=0.005,
        hysteresis_margin_eir=0.120,
        gate_hysteresis_margin=0.020,
        edge_triggered=True,
        refractory_period=10.0,
        allow_refresh_if_still_positive=False,
        use_temporal_correlation=True,
        temporal_alpha=0.85,
        random_seed=13,
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.diagnostics:
        _, _, local_summary, dynamics_summary, thresholds = run_local_diagnostics(cfg, outdir / "diagnostics", num_samples=args.num_samples, num_trials_state=args.num_trials_state)
        save_metadata(cfg, thresholds, GatewayThresholds(rr=float("nan"), tr=float("nan"), eir=float("nan")), outdir / "diagnostics" / "calibration_preview.json")
        print("=== Local diagnostics summary ===")
        print(local_summary.to_string(index=False))
        print("\n=== State dynamics summary ===")
        print(dynamics_summary.to_string(index=False))
        print("\nDiagnostics written to:", outdir / "diagnostics")

    if args.demo:
        demo(outdir, cfg)
    elif not args.diagnostics:
        _, summary, thresholds, gateway_thresholds = run_experiment(cfg, num_h0=120, num_h1=120, calib_trials=150)
        save_metadata(cfg, thresholds, gateway_thresholds, outdir / "calibration.json")
        summary.to_csv(outdir / "summary.csv", index=False)
        print(summary.to_string(index=False))
        print("Use --diagnostics or --demo for more detailed runs.")
