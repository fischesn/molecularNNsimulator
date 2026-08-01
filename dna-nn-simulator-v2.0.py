from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
import warnings
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    hysteresis_margin_eir: float = 0.020
    gate_hysteresis_margin: float = 0.020

    # Legacy channel parameters kept for comparison/debugging.
    tau0: float = 5.0
    kappa_delay: float = 0.05
    lambda_c: float = 400.0
    channel_spread_steps: int = 3

    # New channel model parameters.
    channel_model: str = "diffusive_1d"  # diffusive_1d | legacy_linear
    diffusion_coeff: float = 3000.0  # um^2 / s
    drift_velocity: float = 0.0      # um / s
    channel_decay_rate: float = 0.0  # 1 / s
    receiver_min_distance: float = 5.0  # um, effective near-gateway cutoff
    receiver_gain: float = 1.0

    gateway_evidence_mode: str = "leaky_integrator"  # leaky_integrator | instantaneous
    gateway_integration_tau: float = 20.0  # s

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


@dataclass
class MarkerProcessState:
    latent1: np.ndarray
    latent2: np.ndarray



def deterministic_seed(
    cfg: SimulationConfig,
    *,
    phase: str,
    state_h1: bool = False,
    trial_index: int = 0,
    namespace: str = "",
) -> int:
    """Derive a scenario seed independent of the reporting strategy.

    RR, TR, and EIR therefore see identical positions and marker-noise paths for
    a given phase, state, trial index, and namespace. Distinct phase names keep
    threshold calibration, gateway calibration, and evaluation independent.
    """
    payload = "|".join(
        [
            "dna-nn-simulator-v2.0",
            str(cfg.random_seed),
            phase,
            str(int(state_h1)),
            str(trial_index),
            namespace,
        ]
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, "little") % (2**63 - 1)
    return seed if seed > 0 else 1


def gateway_evidence_step(previous: float, raw_signal: float, cfg: SimulationConfig) -> float:
    if cfg.gateway_evidence_mode == "instantaneous":
        return float(raw_signal)
    if cfg.gateway_evidence_mode == "leaky_integrator":
        tau = max(float(cfg.gateway_integration_tau), float(cfg.dt))
        decay = float(np.exp(-cfg.dt / tau))
        return float(decay * previous + raw_signal)
    raise ValueError(f"Unknown gateway evidence mode: {cfg.gateway_evidence_mode}")


def clipped_normal(mean: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Normal sample clipped at zero.

    This is intentionally *not* a mathematically truncated normal. The name
    reflects the actual implementation so later analyses do not mistake the
    marginal distribution.
    """
    x = mean + sigma * rng.standard_normal(size=mean.shape)
    return np.maximum(0.0, x)


def positions_uniform(n: int, L: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, L, size=n)


def anomaly_profile(positions: np.ndarray, anomaly_x: float, amplitude: float, lamb: float) -> np.ndarray:
    dist = np.abs(positions - anomaly_x)
    return amplitude * np.exp(-dist / lamb)


def mean_markers_for_time(positions: np.ndarray, t: float, cfg: SimulationConfig, state_h1: bool) -> Tuple[np.ndarray, np.ndarray]:
    base1 = np.full_like(positions, cfg.mu1_h0, dtype=float)
    base2 = np.full_like(positions, cfg.mu2_h0, dtype=float)
    if state_h1 and t >= cfg.anomaly_start:
        base1 = base1 + anomaly_profile(positions, cfg.anomaly_x, cfg.a1, cfg.lambda1)
        base2 = base2 + anomaly_profile(positions, cfg.anomaly_x, cfg.a2, cfg.lambda2)
    return base1, base2


def sample_markers_for_time(
    positions: np.ndarray,
    t: float,
    cfg: SimulationConfig,
    state_h1: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    mean1, mean2 = mean_markers_for_time(positions, t, cfg, state_h1)
    x1 = clipped_normal(mean1, cfg.sigma1, rng)
    x2 = clipped_normal(mean2, cfg.sigma2, rng)
    return x1, x2


def initialize_marker_process(
    positions: np.ndarray,
    cfg: SimulationConfig,
    state_h1: bool,
    rng: np.random.Generator,
) -> MarkerProcessState:
    # Standard-normal latent state gives the requested stationary variance in
    # the *unclipped* observation model x = mu + sigma * latent.
    return MarkerProcessState(
        latent1=rng.standard_normal(size=positions.shape),
        latent2=rng.standard_normal(size=positions.shape),
    )


def step_marker_process(
    positions: np.ndarray,
    t: float,
    cfg: SimulationConfig,
    state_h1: bool,
    rng: np.random.Generator,
    process: MarkerProcessState,
) -> Tuple[np.ndarray, np.ndarray]:
    mean1, mean2 = mean_markers_for_time(positions, t, cfg, state_h1)
    if cfg.use_temporal_correlation:
        alpha = float(cfg.temporal_alpha)
        alpha = min(max(alpha, 0.0), 0.999999)
        noise_scale = np.sqrt(max(1.0 - alpha * alpha, 0.0))
        process.latent1 = alpha * process.latent1 + noise_scale * rng.standard_normal(size=positions.shape)
        process.latent2 = alpha * process.latent2 + noise_scale * rng.standard_normal(size=positions.shape)
    else:
        process.latent1 = rng.standard_normal(size=positions.shape)
        process.latent2 = rng.standard_normal(size=positions.shape)

    x1 = np.maximum(0.0, mean1 + cfg.sigma1 * process.latent1)
    x2 = np.maximum(0.0, mean2 + cfg.sigma2 * process.latent2)
    return x1, x2


def make_memory_kernel(spread_steps: int) -> np.ndarray:
    if spread_steps <= 0:
        return np.array([1.0], dtype=float)
    grid = np.arange(0, 2 * spread_steps + 1)
    center = spread_steps
    sigma = max(1.0, spread_steps / 2.0)
    kernel = np.exp(-0.5 * ((grid - center) / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def legacy_channel_response(distance: float, cfg: SimulationConfig, K: int) -> np.ndarray:
    tau = cfg.tau0 + cfg.kappa_delay * distance
    eta = np.exp(-distance / cfg.lambda_c)
    kernel = make_memory_kernel(cfg.channel_spread_steps)
    response = np.zeros(K, dtype=float)
    arrival_idx = int(round(tau / cfg.dt))
    start_idx = arrival_idx - len(kernel) // 2
    for offset, weight in enumerate(kernel):
        idx = start_idx + offset
        if 0 <= idx < K:
            response[idx] += cfg.alarm_molecules * eta * weight
    return response


def first_passage_density_1d(
    times: np.ndarray,
    distance: float,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Arrival-time density at an absorbing boundary in an effective 1D channel.

    The gateway is located at x=0 and the emitter at positive distance ``d``.
    Positive drift velocity therefore denotes drift toward the gateway. Molecular
    decay is represented as independent exponential killing before first arrival.
    """
    if cfg.diffusion_coeff <= 0.0:
        raise ValueError("diffusion_coeff must be positive.")
    if cfg.channel_decay_rate < 0.0:
        raise ValueError("channel_decay_rate must be non-negative.")
    if cfg.receiver_min_distance <= 0.0:
        raise ValueError("receiver_min_distance must be positive.")

    sample_times = np.asarray(times, dtype=float)
    if np.any(sample_times <= 0.0):
        raise ValueError("First-passage density is defined only for positive times.")

    D = cfg.diffusion_coeff
    v = cfg.drift_velocity
    d_eff = max(abs(float(distance)), cfg.receiver_min_distance)

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        denominator = np.sqrt(4.0 * np.pi * D * sample_times ** 3)
        exponent = -((d_eff - v * sample_times) ** 2) / (4.0 * D * sample_times)
        exponent -= cfg.channel_decay_rate * sample_times
        density = (d_eff / denominator) * np.exp(exponent)

    density[~np.isfinite(density)] = 0.0
    return np.maximum(density, 0.0)


def diffusive_channel_response(distance: float, cfg: SimulationConfig, K: int) -> np.ndarray:
    """Midpoint-discretized molecule arrivals caused by one alarm burst."""
    if cfg.dt <= 0.0:
        raise ValueError("dt must be positive.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if cfg.alarm_molecules < 0.0:
        raise ValueError("alarm_molecules must be non-negative.")
    if cfg.receiver_gain < 0.0:
        raise ValueError("receiver_gain must be non-negative.")

    times = (np.arange(K, dtype=float) + 0.5) * cfg.dt
    density = first_passage_density_1d(times, distance, cfg)
    return cfg.alarm_molecules * cfg.receiver_gain * density * cfg.dt


def build_channel_responses(positions: np.ndarray, cfg: SimulationConfig, K: int) -> List[np.ndarray]:
    distances = np.abs(positions - cfg.gateway_x)
    responses: List[np.ndarray] = []
    for d in distances:
        if cfg.channel_model == "legacy_linear":
            responses.append(legacy_channel_response(float(d), cfg, K))
        elif cfg.channel_model == "diffusive_1d":
            responses.append(diffusive_channel_response(float(d), cfg, K))
        else:
            raise ValueError(f"Unknown channel model: {cfg.channel_model}")
    return responses


def add_emission(A: np.ndarray, emit_time: float, response: np.ndarray, cfg: SimulationConfig) -> None:
    emit_idx = int(round(emit_time / cfg.dt))
    if emit_idx >= len(A):
        return
    remaining = len(A) - emit_idx
    if remaining <= 0:
        return
    A[emit_idx:] += response[:remaining]


def empirical_prob_above(x: np.ndarray, tau: float) -> float:
    return float(np.mean(x > tau))


def threshold_for_tail_prob(
    x: np.ndarray,
    target_prob: float,
    *,
    name: str,
    warn: bool = True,
    tol: float = 1e-5,
) -> float:
    x = np.asarray(x, dtype=float)
    if target_prob <= 0.0:
        return float(np.max(x) + 1.0)

    p0 = empirical_prob_above(x, 0.0)
    xmax = float(np.max(x))
    if xmax <= 0.0:
        if warn:
            warnings.warn(f"All samples are zero in {name}; returning threshold 0.0.")
        return 0.0
    if target_prob > p0 + tol:
        if warn:
            warnings.warn(
                f"Target tail probability {target_prob:.6f} exceeds attainable mass above zero {p0:.6f} in {name}; returning 0.0."
            )
        return 0.0

    lo = 0.0
    hi = xmax + 1e-12
    best_tau = 0.0
    best_err = abs(empirical_prob_above(x, best_tau) - target_prob)

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p = empirical_prob_above(x, mid)
        err = abs(p - target_prob)
        if err < best_err:
            best_err = err
            best_tau = mid
        if p > target_prob:
            lo = mid
        else:
            hi = mid

    return float(best_tau)


def search_theta_for_target_probability(
    z: np.ndarray,
    event_prob_fn,
    target_prob: float,
    *,
    name: str,
    warn: bool = True,
) -> float:
    z = np.asarray(z, dtype=float)
    lo = float(np.min(z) - 1e-9)
    hi = float(np.max(z) + 1e-9)
    p_lo = float(event_prob_fn(lo))
    if target_prob > p_lo + 1e-6:
        if warn:
            warnings.warn(
                f"Target probability {target_prob:.6f} exceeds attainable event probability {p_lo:.6f} in {name}; using lowest threshold."
            )
        return lo

    best_theta = lo
    best_err = abs(p_lo - target_prob)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p = float(event_prob_fn(mid))
        err = abs(p - target_prob)
        if err < best_err:
            best_err = err
            best_theta = mid
        if p > target_prob:
            lo = mid
        else:
            hi = mid
    return float(best_theta)


def _rr_tau_for_target(x1: np.ndarray, x2: np.ndarray, cfg: SimulationConfig) -> Tuple[float, float]:
    q_target = float(cfg.local_send_prob_target)
    positive_mass_1 = empirical_prob_above(x1, 0.0)
    positive_mass_2 = empirical_prob_above(x2, 0.0)
    hi = min(0.5, positive_mass_1, positive_mass_2)
    if hi <= 0.0:
        warnings.warn("RR calibration encountered non-positive support; returning zero thresholds.")
        return 0.0, 0.0

    lo = 0.0
    best_tau1 = 0.0
    best_tau2 = 0.0
    best_err = float("inf")

    for _ in range(60):
        p_single = 0.5 * (lo + hi)
        tau1 = threshold_for_tail_prob(x1, p_single, name="RR marker 1", warn=False)
        tau2 = threshold_for_tail_prob(x2, p_single, name="RR marker 2", warn=False)
        send_prob = float(np.mean((x1 > tau1) | (x2 > tau2)))
        err = abs(send_prob - q_target)
        if err < best_err:
            best_err = err
            best_tau1, best_tau2 = tau1, tau2
        if send_prob > q_target:
            hi = p_single
        else:
            lo = p_single

    return float(best_tau1), float(best_tau2)


def calibrate_local_thresholds(cfg: SimulationConfig, samples: int = 100_000) -> LocalThresholds:
    rng = np.random.default_rng(
        deterministic_seed(cfg, phase="local_threshold_calibration")
    )
    x1 = clipped_normal(np.full(samples, cfg.mu1_h0), cfg.sigma1, rng)
    x2 = clipped_normal(np.full(samples, cfg.mu2_h0), cfg.sigma2, rng)

    rr_tau1, rr_tau2 = _rr_tau_for_target(x1, x2, cfg)
    tr_tau = threshold_for_tail_prob(x1, cfg.local_send_prob_target, name="TR marker 1")

    score = eir_score(x1, x2, cfg)
    if cfg.eir_gate_manual is not None:
        eir_gate_tau = float(cfg.eir_gate_manual)
    else:
        eir_gate_tau = threshold_for_tail_prob(x2, 1.0 - cfg.eir_gate_quantile, name="EIR gate marker 2")

    if cfg.use_eir_gate:
        gate_mask = x2 > eir_gate_tau
        eir_theta = search_theta_for_target_probability(
            score,
            lambda theta: np.mean((score > theta) & gate_mask),
            cfg.local_send_prob_target,
            name="EIR gated score",
        )
    else:
        eir_theta = search_theta_for_target_probability(
            score,
            lambda theta: np.mean(score > theta),
            cfg.local_send_prob_target,
            name="EIR score",
        )

    return LocalThresholds(
        rr_tau1=float(rr_tau1),
        rr_tau2=float(rr_tau2),
        tr_tau=float(tr_tau),
        eir_theta=float(eir_theta),
        eir_gate_tau=float(eir_gate_tau),
    )


def eir_score(x1: np.ndarray, x2: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    """Return the unthresholded weighted EIR evidence score."""
    return cfg.w1 * x1 + cfg.w2 * x2


def eir_transition_conditions(
    x1: np.ndarray,
    x2: np.ndarray,
    thresholds: LocalThresholds,
    cfg: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the ON and OFF conditions of the gated EIR state machine.

    An inactive node turns ON only when both the weighted score and the
    specificity gate exceed their upper thresholds. An active node turns OFF
    when either quantity falls below its respective lower threshold.
    """
    score = eir_score(x1, x2, cfg)
    on = score > thresholds.eir_theta
    off = score < (thresholds.eir_theta - cfg.hysteresis_margin_eir)
    if cfg.use_eir_gate:
        gate_on = x2 > thresholds.eir_gate_tau
        gate_off = x2 < (thresholds.eir_gate_tau - cfg.gate_hysteresis_margin)
        on = on & gate_on
        off = off | gate_off
    return on, off


def raw_positive_state(strategy: str, x1: np.ndarray, x2: np.ndarray, thresholds: LocalThresholds, cfg: SimulationConfig) -> np.ndarray:
    if strategy == "RR":
        return (x1 > thresholds.rr_tau1) | (x2 > thresholds.rr_tau2)
    if strategy == "TR":
        return x1 > thresholds.tr_tau
    if strategy == "EIR":
        score = eir_score(x1, x2, cfg)
        state = score > thresholds.eir_theta
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
        on, off = eir_transition_conditions(x1, x2, thresholds, cfg)
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
    rng = np.random.default_rng(deterministic_seed(cfg, phase="local_decisions", state_h1=state_h1))
    positions = positions_uniform(num_samples, cfg.L, rng)
    t = cfg.anomaly_start + cfg.dt if state_h1 else 0.0
    x1, x2 = sample_markers_for_time(positions, t, cfg, state_h1, rng)
    prev = np.zeros(num_samples, dtype=bool)
    rr = evaluate_positive_state("RR", x1, x2, prev, thresholds, cfg)
    tr = evaluate_positive_state("TR", x1, x2, prev, thresholds, cfg)
    eir = evaluate_positive_state("EIR", x1, x2, prev, thresholds, cfg)
    score_margin = eir_score(x1, x2, cfg) - thresholds.eir_theta
    dist = np.abs(positions - cfg.anomaly_x)
    return pd.DataFrame(
        {
            "state_h1": int(state_h1),
            "x": positions,
            "dist_to_anomaly": dist,
            "x1": x1,
            "x2": x2,
            "z_eir": score_margin,
            "send_rr": rr.astype(int),
            "send_tr": tr.astype(int),
            "send_eir": eir.astype(int),
        }
    )


def summarize_local_decisions(df_h0: pd.DataFrame, df_h1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df in [("H0", df_h0), ("H1", df_h1)]:
        rows.append(
            {
                "state": label,
                "x1_mean": float(df["x1"].mean()),
                "x2_mean": float(df["x2"].mean()),
                "send_rr_prob": float(df["send_rr"].mean()),
                "send_tr_prob": float(df["send_tr"].mean()),
                "send_eir_prob": float(df["send_eir"].mean()),
                "eir_score_mean": float(df["z_eir"].mean()),
            }
        )
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
        rng = np.random.default_rng(
            deterministic_seed(
                cfg,
                phase="state_dynamics",
                state_h1=state_h1,
                trial_index=trial,
            )
        )
        positions = positions_uniform(cfg.num_nodes, cfg.L, rng)
        prev_state = np.zeros(cfg.num_nodes, dtype=bool)
        K = int(round(cfg.t_max / cfg.dt)) + 1
        on_time_steps = np.zeros(cfg.num_nodes, dtype=int)
        rising_edges = np.zeros(cfg.num_nodes, dtype=int)
        current_on_start = np.full(cfg.num_nodes, -1, dtype=int)
        on_durations = []
        process = initialize_marker_process(positions, cfg, state_h1, rng)

        for k in range(K):
            t = k * cfg.dt
            x1, x2 = step_marker_process(positions, t, cfg, state_h1, rng, process)
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
        rows.append(
            {
                "strategy": strategy,
                "state_h1": int(state_h1),
                "trial": trial,
                "mean_on_fraction": float(on_time_steps.sum()) / float(cfg.num_nodes * K),
                "mean_rising_edges_per_node": float(rising_edges.mean()),
                "total_rising_edges": int(rising_edges.sum()),
                "mean_on_duration_s": float(np.mean(on_durations)) if len(on_durations) > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def run_state_dynamics_diagnostics(cfg: SimulationConfig, output_dir: Path, num_trials: int = 100) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = calibrate_local_thresholds(cfg)
    dfs = []
    for state_h1 in [False, True]:
        for strategy in ["RR", "TR", "EIR"]:
            dfs.append(
                collect_state_dynamics_metrics(
                    cfg=cfg,
                    thresholds=thresholds,
                    strategy=strategy,
                    state_h1=state_h1,
                    num_trials=num_trials,
                )
            )
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(output_dir / "state_dynamics_trials.csv", index=False)
    summary_rows: List[Dict[str, object]] = []
    for (strategy, state_h1), group in all_df.groupby(["strategy", "state_h1"]):
        row: Dict[str, object] = {
            "strategy": strategy,
            "state_h1": state_h1,
            "n_trials": len(group),
        }
        for metric in [
            "mean_on_fraction",
            "mean_rising_edges_per_node",
            "mean_on_duration_s",
        ]:
            _add_mean_with_interval(
                row,
                metric,
                group[metric].to_numpy(dtype=float),
                lower_bound=0.0,
            )
        summary_rows.append(row)
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["state_h1", "strategy"])
        .reset_index(drop=True)
    )
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



def simulate_one_trial(
    cfg: SimulationConfig,
    thresholds: LocalThresholds,
    gateway_threshold: float,
    strategy: str,
    state_h1: bool,
    rng: np.random.Generator,
) -> TrialOutcome:
    K = int(round(cfg.t_max / cfg.dt)) + 1
    raw_gateway_signal = np.zeros(K, dtype=float)
    gateway_signal = np.zeros(K, dtype=float)
    positions = positions_uniform(cfg.num_nodes, cfg.L, rng)
    responses = build_channel_responses(positions, cfg, K)
    prev_positive = np.zeros(cfg.num_nodes, dtype=bool)
    last_send_time = np.full(cfg.num_nodes, -np.inf, dtype=float)
    transmissions = 0
    detected = False
    pre_onset_alarm = False
    detection_time = np.inf
    first_alarm_time = np.inf
    gateway_above_prev = False
    process = initialize_marker_process(positions, cfg, state_h1, rng)
    previous_gateway_evidence = 0.0

    for k in range(K):
        t = k * cfg.dt
        x1, x2 = step_marker_process(positions, t, cfg, state_h1, rng, process)
        current_positive = evaluate_positive_state(strategy, x1, x2, prev_positive, thresholds, cfg)
        send_now = should_emit_alarm(
            current_positive=current_positive,
            prev_positive=prev_positive,
            last_send_time=last_send_time,
            t=t,
            cfg=cfg,
        )

        idxs = np.where(send_now)[0]
        if len(idxs) > 0:
            for i in idxs:
                emit_time = t + (cfg.inference_delay if strategy == "EIR" else 0.0)
                add_emission(raw_gateway_signal, emit_time, responses[int(i)], cfg)
            last_send_time[idxs] = t
            transmissions += int(len(idxs))

        prev_positive = current_positive.copy()
        gateway_signal[k] = gateway_evidence_step(previous_gateway_evidence, raw_gateway_signal[k], cfg)
        previous_gateway_evidence = gateway_signal[k]
        gateway_above = gateway_signal[k] > gateway_threshold
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

    return TrialOutcome(
        state_h1=bool(state_h1),
        detected=bool(detected),
        pre_onset_alarm=bool(pre_onset_alarm),
        detection_time=float(detection_time),
        first_alarm_time=float(first_alarm_time),
        transmissions=int(transmissions),
        max_gateway_evidence=float(np.max(gateway_signal)),
    )




def calibrate_gateway_threshold(
    cfg: SimulationConfig,
    thresholds: LocalThresholds,
    strategy: str,
    num_trials: int = 250,
    *,
    seed_namespace: str = "",
    return_trials: bool = False,
):
    rows = []
    dummy_threshold = np.inf
    for trial in range(num_trials):
        scenario_seed = deterministic_seed(
            cfg,
            phase="gateway_calibration",
            state_h1=False,
            trial_index=trial,
            namespace=seed_namespace,
        )
        rng = np.random.default_rng(scenario_seed)
        outcome = simulate_one_trial(cfg, thresholds, dummy_threshold, strategy, False, rng)
        rows.append(
            {
                "strategy": strategy,
                "state_h1": 0,
                "trial_index": trial,
                "scenario_seed": scenario_seed,
                "seed_phase": "gateway_calibration",
                "seed_namespace": seed_namespace,
                "max_gateway_evidence": outcome.max_gateway_evidence,
                "transmissions": outcome.transmissions,
            }
        )
    records = pd.DataFrame(rows)
    q = 1.0 - cfg.gateway_false_alarm_target
    gateway_threshold = float(np.quantile(records["max_gateway_evidence"].to_numpy(), q))
    records["gateway_threshold"] = gateway_threshold
    if return_trials:
        return gateway_threshold, records
    return gateway_threshold


def calibrate_all_gateway_thresholds(
    cfg: SimulationConfig,
    thresholds: LocalThresholds,
    num_trials: int = 250,
    *,
    seed_namespace: str = "",
    return_trials: bool = False,
    strategies: Optional[List[str]] = None,
):
    selected_strategies = ["RR", "TR", "EIR"] if strategies is None else list(strategies)
    if not selected_strategies or any(s not in {"RR", "TR", "EIR"} for s in selected_strategies):
        raise ValueError(f"Invalid strategy subset: {selected_strategies}")
    threshold_values: Dict[str, float] = {"RR": float("nan"), "TR": float("nan"), "EIR": float("nan")}
    calibration_frames: List[pd.DataFrame] = []
    for strategy in selected_strategies:
        result = calibrate_gateway_threshold(
            cfg,
            thresholds,
            strategy,
            num_trials=num_trials,
            seed_namespace=seed_namespace,
            return_trials=return_trials,
        )
        if return_trials:
            threshold, records = result
            calibration_frames.append(records)
        else:
            threshold = result
        threshold_values[strategy] = float(threshold)
    gateway_thresholds = GatewayThresholds(
        rr=threshold_values["RR"],
        tr=threshold_values["TR"],
        eir=threshold_values["EIR"],
    )
    if return_trials:
        return gateway_thresholds, pd.concat(calibration_frames, ignore_index=True)
    return gateway_thresholds


def run_trials(
    cfg: SimulationConfig,
    thresholds: LocalThresholds,
    gateway_thresholds: GatewayThresholds,
    strategy: str,
    num_h0: int = 200,
    num_h1: int = 200,
    *,
    seed_namespace: str = "",
) -> pd.DataFrame:
    rows = []
    gt = {"RR": gateway_thresholds.rr, "TR": gateway_thresholds.tr, "EIR": gateway_thresholds.eir}[strategy]
    for state_h1, count in [(False, num_h0), (True, num_h1)]:
        for trial in range(count):
            scenario_seed = deterministic_seed(
                cfg,
                phase="evaluation",
                state_h1=state_h1,
                trial_index=trial,
                namespace=seed_namespace,
            )
            rng = np.random.default_rng(scenario_seed)
            outcome = simulate_one_trial(
                cfg=cfg,
                thresholds=thresholds,
                gateway_threshold=gt,
                strategy=strategy,
                state_h1=state_h1,
                rng=rng,
            )
            rows.append(
                {
                    "strategy": strategy,
                    "state_h1": int(outcome.state_h1),
                    "trial_index": trial,
                    "scenario_seed": scenario_seed,
                    "seed_phase": "evaluation",
                    "seed_namespace": seed_namespace,
                    "gateway_threshold": gt,
                    "detected": int(outcome.detected),
                    "pre_onset_alarm": int(outcome.pre_onset_alarm),
                    "detection_time": outcome.detection_time,
                    "first_alarm_time": outcome.first_alarm_time,
                    "transmissions": outcome.transmissions,
                    "max_gateway_evidence": outcome.max_gateway_evidence,
                }
            )
    return pd.DataFrame(rows)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        return float("nan"), float("nan")
    p = float(successes) / float(total)
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half_width = z * np.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denominator
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def mean_confidence_interval(values: np.ndarray, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Normal-approximation interval for a sample mean.

    Phase-4 operating points use at least hundreds of trials, so the bounded
    transmission counts and conditional delays are covered by the CLT. A
    degenerate one-value sample is reported without artificial width.
    """
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if len(sample) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(sample))
    if len(sample) == 1:
        return mean, mean
    standard_error = float(np.std(sample, ddof=1) / np.sqrt(len(sample)))
    return mean - z * standard_error, mean + z * standard_error


def _add_mean_with_interval(
    row: Dict[str, object],
    name: str,
    values: np.ndarray,
    *,
    lower_bound: Optional[float] = None,
) -> None:
    sample = np.asarray(values, dtype=float)
    finite = sample[np.isfinite(sample)]
    row[name] = float(np.mean(finite)) if len(finite) else float("nan")
    low, high = mean_confidence_interval(finite)
    if lower_bound is not None and np.isfinite(low):
        low = max(lower_bound, low)
    row[f"{name}_ci_low"] = low
    row[f"{name}_ci_high"] = high



def summarize_results(df: pd.DataFrame, cfg: SimulationConfig) -> pd.DataFrame:
    rows = []
    for strategy, group in df.groupby("strategy"):
        h1 = group[group["state_h1"] == 1]
        h0 = group[group["state_h1"] == 0]
        p_fa = float((h0["first_alarm_time"] < np.inf).mean()) if len(h0) else float("nan")
        p_d = float(h1["detected"].mean()) if len(h1) else float("nan")
        p_pre = float(h1["pre_onset_alarm"].mean()) if len(h1) else float("nan")
        detected_h1 = h1[h1["detected"] == 1]["detection_time"]
        delay = float("inf") if len(detected_h1) == 0 else float(detected_h1.mean() - cfg.anomaly_start)
        detected_h1_full = h1[h1["detected"] == 1].copy()
        if len(detected_h1_full) > 0:
            durations = np.maximum(detected_h1_full["detection_time"].to_numpy() - cfg.anomaly_start, cfg.dt)
            molecules = detected_h1_full["transmissions"].to_numpy() * cfg.alarm_molecules
            rate_h1 = float(np.mean(molecules / durations))
        else:
            rate_h1 = float("nan")
        row: Dict[str, object] = {
            "strategy": strategy,
            "n_H0": len(h0),
            "n_H1": len(h1),
            "n_detected_H1": len(detected_h1),
            "P_D": p_d,
            "P_pre_onset_alarm": p_pre,
            "P_FA": p_fa,
        }
        for name, successes, total in [
            ("P_D", int(h1["detected"].sum()), len(h1)),
            ("P_pre_onset_alarm", int(h1["pre_onset_alarm"].sum()), len(h1)),
            ("P_FA", int((h0["first_alarm_time"] < np.inf).sum()), len(h0)),
        ]:
            low, high = wilson_interval(successes, total)
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high

        _add_mean_with_interval(
            row,
            "C_total_molecules_avg",
            group["transmissions"].to_numpy(dtype=float) * cfg.alarm_molecules,
            lower_bound=0.0,
        )
        _add_mean_with_interval(
            row,
            "C_H0_molecules_avg",
            h0["transmissions"].to_numpy(dtype=float) * cfg.alarm_molecules,
            lower_bound=0.0,
        )
        _add_mean_with_interval(
            row,
            "C_H1_molecules_avg",
            h1["transmissions"].to_numpy(dtype=float) * cfg.alarm_molecules,
            lower_bound=0.0,
        )
        rate_values = (
            detected_h1_full["transmissions"].to_numpy(dtype=float)
            * cfg.alarm_molecules
            / np.maximum(
                detected_h1_full["detection_time"].to_numpy(dtype=float) - cfg.anomaly_start,
                cfg.dt,
            )
            if len(detected_h1_full)
            else np.asarray([], dtype=float)
        )
        _add_mean_with_interval(row, "R_H1_molecules_per_s", rate_values, lower_bound=0.0)
        delay_values = detected_h1.to_numpy(dtype=float) - cfg.anomaly_start
        _add_mean_with_interval(row, "D_avg_after_onset", delay_values, lower_bound=0.0)
        if not np.isfinite(delay):
            row["D_avg_after_onset"] = float("inf")
        if not np.isfinite(rate_h1):
            row["R_H1_molecules_per_s"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)


def run_experiment(
    cfg: SimulationConfig,
    num_h0: int = 200,
    num_h1: int = 200,
    calib_trials: int = 250,
    *,
    seed_namespace: str = "",
    return_calibration_trials: bool = False,
    precalibrated_thresholds: Optional[LocalThresholds] = None,
    precalibrated_gateway_thresholds: Optional[GatewayThresholds] = None,
    strategies: Optional[List[str]] = None,
):
    selected_strategies = ["RR", "TR", "EIR"] if strategies is None else list(strategies)
    if not selected_strategies or any(s not in {"RR", "TR", "EIR"} for s in selected_strategies):
        raise ValueError(f"Invalid strategy subset: {selected_strategies}")
    overrides = (precalibrated_thresholds, precalibrated_gateway_thresholds)
    if (overrides[0] is None) != (overrides[1] is None):
        raise ValueError("Both local and gateway precalibrated thresholds must be supplied together.")
    if overrides[0] is not None:
        thresholds = overrides[0]
        gateway_thresholds = overrides[1]
        calibration_trials = pd.DataFrame()
    else:
        thresholds = calibrate_local_thresholds(cfg)
        calibration_result = calibrate_all_gateway_thresholds(
            cfg,
            thresholds,
            num_trials=calib_trials,
            seed_namespace=seed_namespace,
            return_trials=return_calibration_trials,
            strategies=selected_strategies,
        )
        if return_calibration_trials:
            gateway_thresholds, calibration_trials = calibration_result
        else:
            gateway_thresholds = calibration_result
    dfs = []
    for strategy in selected_strategies:
        dfs.append(
            run_trials(
                cfg,
                thresholds,
                gateway_thresholds,
                strategy,
                num_h0=num_h0,
                num_h1=num_h1,
                seed_namespace=seed_namespace,
            )
        )
    all_trials = pd.concat(dfs, ignore_index=True)
    summary = summarize_results(all_trials, cfg)
    if return_calibration_trials:
        return all_trials, summary, thresholds, gateway_thresholds, calibration_trials
    return all_trials, summary, thresholds, gateway_thresholds


def sweep_parameter(
    base_cfg: SimulationConfig,
    parameter_name: str,
    values: List[float],
    num_h0: int = 150,
    num_h1: int = 150,
    calib_trials: int = 200,
    *,
    return_trials: bool = False,
):
    rows = []
    trial_frames: List[pd.DataFrame] = []
    calibration_frames: List[pd.DataFrame] = []
    sweep_started = time.time()
    for value_index, value in enumerate(values, start=1):
        print(
            f"[sweep:{parameter_name}] value {value_index}/{len(values)}: {value}",
            flush=True,
        )
        cfg = SimulationConfig(**asdict(base_cfg))
        setattr(cfg, parameter_name, value)
        namespace = f"sweep:{parameter_name}={value}"
        result = run_experiment(
            cfg,
            num_h0=num_h0,
            num_h1=num_h1,
            calib_trials=calib_trials,
            seed_namespace=namespace,
            return_calibration_trials=return_trials,
        )
        if return_trials:
            trials, summary, _, _, calibration_trials_df = result
            trials[parameter_name] = value
            calibration_trials_df[parameter_name] = value
            trial_frames.append(trials)
            calibration_frames.append(calibration_trials_df)
        else:
            _, summary, _, _ = result
        summary[parameter_name] = value
        rows.append(summary)
        print(
            f"[sweep:{parameter_name}] completed value {value} "
            f"in {time.time() - sweep_started:.1f}s cumulative",
            flush=True,
        )
    summary_df = pd.concat(rows, ignore_index=True)
    if return_trials:
        return (
            summary_df,
            pd.concat(trial_frames, ignore_index=True),
            pd.concat(calibration_frames, ignore_index=True),
        )
    return summary_df


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


def plot_heatmap(df: pd.DataFrame, x_col: str, y_col: str, value_col: str, outpath: Path, title: str) -> None:
    pivot = df.pivot(index=y_col, columns=x_col, values=value_col).sort_index().sort_index(axis=1)
    plt.figure(figsize=(6, 4.5))
    im = plt.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
    plt.xticks(range(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [f"{y:.2f}" for y in pivot.index])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel(value_col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()




def _preferred_grid_axes(parameter_columns: List[str]) -> Tuple[Optional[str], Optional[str], List[str]]:
    preferred = [
        "eir_gate_quantile",
        "hysteresis_margin_eir",
        "gate_hysteresis_margin",
        "w2",
    ]
    ordered = [c for c in preferred if c in parameter_columns] + [c for c in parameter_columns if c not in preferred]
    if not ordered:
        return None, None, []
    if len(ordered) == 1:
        return ordered[0], None, []
    return ordered[0], ordered[1], ordered[2:]


def plot_hypergrid_best_slice(df_summary: pd.DataFrame, output_dir: Path, parameter_columns: List[str], suffix: str = "") -> None:
    if df_summary.empty or not parameter_columns:
        return
    x_col, y_col, slice_cols = _preferred_grid_axes(parameter_columns)
    if x_col is None or y_col is None:
        return
    best = df_summary.iloc[0]
    slice_df = df_summary.copy()
    slice_parts = []
    for col in slice_cols:
        slice_df = slice_df[slice_df[col] == best[col]]
        try:
            slice_parts.append(f"{col}={float(best[col]):.3f}")
        except Exception:
            slice_parts.append(f"{col}={best[col]}")
    if slice_df.empty:
        return
    slice_label = "best slice" if not slice_parts else f"best slice: {', '.join(slice_parts)}"
    plot_heatmap(slice_df, x_col, y_col, "eir_mean_P_D", output_dir / f"plot_eir_hypergrid_mean_pd_best_slice{suffix}.png", f"EIR hypergrid: mean P_D ({slice_label})")
    plot_heatmap(slice_df, x_col, y_col, "pfa_abs_deviation", output_dir / f"plot_eir_hypergrid_pfa_deviation_best_slice{suffix}.png", f"EIR hypergrid: |mean P_FA - target| ({slice_label})")
    plot_heatmap(slice_df, x_col, y_col, "delta_C_H1_vs_TR", output_dir / f"plot_eir_hypergrid_delta_ch1_vs_tr_best_slice{suffix}.png", f"EIR hypergrid: Δ C_H1 vs TR ({slice_label})")
    plot_heatmap(slice_df, x_col, y_col, "delta_delay_vs_TR", output_dir / f"plot_eir_hypergrid_delta_delay_vs_tr_best_slice{suffix}.png", f"EIR hypergrid: Δ delay vs TR ({slice_label})")


def summarize_eir_grid(
    df_grid: pd.DataFrame,
    focus_max_a1: float,
    *,
    parameter_columns: List[str],
    target_pfa: float = 0.05,
) -> pd.DataFrame:
    rows = []
    group_cols = list(parameter_columns)
    if not group_cols:
        raise ValueError("parameter_columns must not be empty for EIR grid-search summarization.")
    for keys, group in df_grid.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        focus = group[group["a1"] <= focus_max_a1].copy()
        if focus.empty:
            focus = group.copy()
        eir = focus[focus["strategy"] == "EIR"].sort_values("a1")
        tr = focus[focus["strategy"] == "TR"].sort_values("a1")
        rr = focus[focus["strategy"] == "RR"].sort_values("a1")
        if len(eir) != len(tr) or len(eir) != len(rr):
            continue
        row = {col: float(val) if isinstance(val, (int, float)) else val for col, val in zip(group_cols, keys)}
        row.update({
            "focus_max_a1": float(focus_max_a1),
            "num_focus_points": int(len(eir)),
            "eir_mean_P_D": float(eir["P_D"].mean()),
            "eir_mean_P_FA": float(eir["P_FA"].mean()),
            "eir_mean_C_H1": float(eir["C_H1_molecules_avg"].mean()),
            "eir_mean_delay": float(eir["D_avg_after_onset"].mean()),
            "tr_mean_P_D": float(tr["P_D"].mean()),
            "tr_mean_P_FA": float(tr["P_FA"].mean()),
            "tr_mean_C_H1": float(tr["C_H1_molecules_avg"].mean()),
            "tr_mean_delay": float(tr["D_avg_after_onset"].mean()),
            "rr_mean_P_D": float(rr["P_D"].mean()),
            "rr_mean_P_FA": float(rr["P_FA"].mean()),
            "rr_mean_C_H1": float(rr["C_H1_molecules_avg"].mean()),
            "rr_mean_delay": float(rr["D_avg_after_onset"].mean()),
        })
        row["delta_P_D_vs_TR"] = row["eir_mean_P_D"] - row["tr_mean_P_D"]
        row["delta_C_H1_vs_TR"] = row["eir_mean_C_H1"] - row["tr_mean_C_H1"]
        row["delta_delay_vs_TR"] = row["eir_mean_delay"] - row["tr_mean_delay"]
        row["delta_P_D_vs_RR"] = row["eir_mean_P_D"] - row["rr_mean_P_D"]
        row["delta_C_H1_vs_RR"] = row["eir_mean_C_H1"] - row["rr_mean_C_H1"]
        row["delta_delay_vs_RR"] = row["eir_mean_delay"] - row["rr_mean_delay"]
        row["traffic_penalty_vs_TR"] = max(0.0, row["delta_C_H1_vs_TR"])
        row["delay_penalty_vs_TR"] = max(0.0, row["delta_delay_vs_TR"])
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["target_P_FA"] = float(target_pfa)
    out["pfa_abs_deviation"] = (out["eir_mean_P_FA"] - float(target_pfa)).abs()
    sort_cols = [
        "eir_mean_P_D",
        "pfa_abs_deviation",
        "traffic_penalty_vs_TR",
        "delay_penalty_vs_TR",
        "eir_mean_C_H1",
        "eir_mean_delay",
    ] + list(group_cols)
    sort_ascending = [False, True, True, True, True, True] + [True] * len(group_cols)
    out = out.sort_values(sort_cols, ascending=sort_ascending).reset_index(drop=True)
    out["soft_rank"] = np.arange(1, len(out) + 1)
    return out


def _write_grid_progress(progress_path: Path, payload: Dict[str, object]) -> None:
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_grid_artifacts(
    output_dir: Path,
    df_grid: pd.DataFrame,
    *,
    parameter_columns: List[str],
    focus_max_a1: float,
    target_pfa: float,
    partial: bool,
) -> pd.DataFrame:
    suffix = "_partial" if partial else ""
    df_grid.to_csv(output_dir / f"eir_grid_anomaly{suffix}.csv", index=False)
    df_summary = summarize_eir_grid(df_grid, focus_max_a1=focus_max_a1, parameter_columns=parameter_columns, target_pfa=target_pfa)
    df_summary.to_csv(output_dir / f"eir_grid_summary{suffix}.csv", index=False)
    if not df_summary.empty:
        best = df_summary.iloc[0].to_dict()
        (output_dir / f"eir_grid_best{suffix}.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        plot_hypergrid_best_slice(df_summary, output_dir, parameter_columns, suffix=suffix)
    return df_summary


def run_eir_grid_search(
    output_dir: Path,
    base_cfg: SimulationConfig,
    grid_parameters: Dict[str, List[float]],
    anomaly_a1_values: List[float],
    anomaly_a2_ratio: float,
    *,
    focus_max_a1: float = 0.10,
    target_pfa: float = 0.05,
    num_h0: int = 150,
    num_h1: int = 150,
    calib_trials: int = 200,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_columns = list(grid_parameters.keys())
    if not parameter_columns:
        raise ValueError("No grid parameters configured. Use [eir_grid.parameter_values] in .config_sim.")
    for name in parameter_columns:
        if name not in SimulationConfig.__dataclass_fields__:
            raise ValueError(f"Unknown simulation parameter in grid search: {name}")
    value_lists = [list(grid_parameters[name]) for name in parameter_columns]
    if any(len(v) == 0 for v in value_lists):
        raise ValueError("Every grid-search parameter must have at least one value.")
    combos = list(itertools.product(*value_lists))

    progress_path = output_dir / "eir_grid_progress.json"
    total_combos = len(combos)
    started = time.time()
    _write_grid_progress(progress_path, {
        "status": "running",
        "started_epoch": started,
        "completed_combinations": 0,
        "total_combinations": total_combos,
        "parameter_columns": parameter_columns,
        "current_parameters": None,
        "elapsed_seconds": 0.0,
    })
    print(f"[grid] starting EIR parameter grid search with {total_combos} combinations over {parameter_columns}...", flush=True)
    print("[grid] computing one shared RR/TR reference sweep...", flush=True)
    reference_summary, reference_trials, reference_calibration = sweep_anomaly_pair(
        base_cfg,
        anomaly_a1_values,
        anomaly_a2_ratio,
        num_h0=num_h0,
        num_h1=num_h1,
        calib_trials=calib_trials,
        return_trials=True,
        strategies=["RR", "TR"],
    )
    reference_summary.to_csv(output_dir / "eir_grid_reference_summary.csv", index=False)
    reference_trials.to_csv(output_dir / "eir_grid_reference_trials.csv", index=False)
    reference_calibration.to_csv(output_dir / "eir_grid_reference_calibration_trials.csv", index=False)
    grid_rows = []
    eir_trial_frames: List[pd.DataFrame] = []
    eir_calibration_frames: List[pd.DataFrame] = []
    completed = 0
    for combo in combos:
        combo_started = time.time()
        current_params = {name: float(value) for name, value in zip(parameter_columns, combo)}
        human = ", ".join(f"{k}={v:.3f}" for k, v in current_params.items())
        print(f"[grid] combination {completed + 1}/{total_combos}: {human}", flush=True)
        cfg = SimulationConfig(**asdict(base_cfg))
        for name, value in current_params.items():
            setattr(cfg, name, value)
        eir_summary, eir_trials, eir_calibration = sweep_anomaly_pair(
            cfg,
            anomaly_a1_values,
            anomaly_a2_ratio,
            num_h0=num_h0,
            num_h1=num_h1,
            calib_trials=calib_trials,
            return_trials=True,
            strategies=["EIR"],
        )
        df = pd.concat([reference_summary.copy(), eir_summary], ignore_index=True)
        for name, value in current_params.items():
            df[name] = value
            eir_trials[name] = value
            eir_calibration[name] = value
        grid_rows.append(df)
        eir_trial_frames.append(eir_trials)
        eir_calibration_frames.append(eir_calibration)
        completed += 1
        df_grid_partial = pd.concat(grid_rows, ignore_index=True)
        df_summary_partial = _write_grid_artifacts(
            output_dir,
            df_grid_partial,
            parameter_columns=parameter_columns,
            focus_max_a1=focus_max_a1,
            target_pfa=target_pfa,
            partial=True,
        )
        best_payload = None if df_summary_partial.empty else df_summary_partial.iloc[0].to_dict()
        _write_grid_progress(progress_path, {
            "status": "running",
            "started_epoch": started,
            "completed_combinations": completed,
            "total_combinations": total_combos,
            "parameter_columns": parameter_columns,
            "current_parameters": current_params,
            "elapsed_seconds": time.time() - started,
            "last_combination_seconds": time.time() - combo_started,
            "best_partial": best_payload,
        })
        print(f"[grid] finished combination {completed}/{total_combos} in {time.time() - combo_started:.1f}s", flush=True)
    if not grid_rows:
        raise ValueError("No EIR grid-search rows were generated.")
    df_grid = pd.concat(grid_rows, ignore_index=True)
    df_summary = _write_grid_artifacts(
        output_dir,
        df_grid,
        parameter_columns=parameter_columns,
        focus_max_a1=focus_max_a1,
        target_pfa=target_pfa,
        partial=False,
    )
    pd.concat(eir_trial_frames, ignore_index=True).to_csv(
        output_dir / "eir_grid_eir_trials.csv", index=False
    )
    pd.concat(eir_calibration_frames, ignore_index=True).to_csv(
        output_dir / "eir_grid_eir_calibration_trials.csv", index=False
    )
    _write_grid_progress(progress_path, {
        "status": "done",
        "started_epoch": started,
        "completed_combinations": completed,
        "total_combinations": total_combos,
        "parameter_columns": parameter_columns,
        "elapsed_seconds": time.time() - started,
        "best_final": None if df_summary.empty else df_summary.iloc[0].to_dict(),
    })
    plt.close("all")
    print(f"[grid] completed EIR parameter grid search in {time.time() - started:.1f}s", flush=True)


def save_metadata(cfg: SimulationConfig, thresholds: LocalThresholds, gateway_thresholds: GatewayThresholds, outpath: Path) -> None:
    data: Dict[str, object] = {
        "simulator_version": "dna-nn-simulator-v2.0",
        "config": asdict(cfg),
        "local_thresholds": asdict(thresholds),
        "gateway_thresholds": asdict(gateway_thresholds),
        "statistics": {
            "confidence_level": 0.95,
            "binomial_interval": "Wilson score interval",
            "mean_interval": "normal approximation: mean +/- 1.959963984540054 standard errors",
            "common_random_scenarios_across_strategies": True,
            "independent_seed_phases": [
                "local_threshold_calibration",
                "gateway_calibration",
                "evaluation",
            ],
        },
    }
    outpath.write_text(json.dumps(data, indent=2), encoding="utf-8")


def demo(
    output_dir: Path,
    cfg: Optional[SimulationConfig] = None,
    *,
    anomaly_a1_values: Optional[List[float]] = None,
    anomaly_a2_ratio: float = 0.4,
    noise_values: Optional[List[float]] = None,
    nodes_values: Optional[List[float]] = None,
    delay_values: Optional[List[float]] = None,
    baseline_num_h0: int = 250,
    baseline_num_h1: int = 250,
    baseline_calib_trials: int = 250,
    sweep_num_h0: int = 150,
    sweep_num_h1: int = 150,
    sweep_calib_trials: int = 200,
) -> None:
    cfg = SimulationConfig() if cfg is None else cfg
    output_dir.mkdir(parents=True, exist_ok=True)
    anomaly_a1_values = [0.03, 0.06, 0.10, 0.15, 0.20] if anomaly_a1_values is None else anomaly_a1_values
    noise_values = [0.08, 0.12, 0.16, 0.20] if noise_values is None else noise_values
    nodes_values = [20, 40, 60, 100] if nodes_values is None else nodes_values
    delay_values = [0.0, 10.0, 30.0, 60.0, 120.0] if delay_values is None else delay_values

    demo_started = time.time()
    print("[demo] starting high-confidence baseline", flush=True)
    all_trials, summary, thresholds, gateway_thresholds, baseline_calibration_trials = run_experiment(
        cfg,
        num_h0=baseline_num_h0,
        num_h1=baseline_num_h1,
        calib_trials=baseline_calib_trials,
        seed_namespace="demo_baseline",
        return_calibration_trials=True,
    )
    all_trials.to_csv(output_dir / "trial_results.csv", index=False)
    baseline_calibration_trials.to_csv(output_dir / "gateway_calibration_trials.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    save_metadata(cfg, thresholds, gateway_thresholds, output_dir / "calibration.json")
    print(f"[demo] baseline completed in {time.time() - demo_started:.1f}s", flush=True)

    df_anomaly, trials_anomaly, calibration_anomaly = sweep_anomaly_pair(
        cfg,
        anomaly_a1_values,
        anomaly_a2_ratio,
        num_h0=sweep_num_h0,
        num_h1=sweep_num_h1,
        calib_trials=sweep_calib_trials,
        return_trials=True,
    )
    df_noise, trials_noise, calibration_noise = sweep_parameter(
        cfg, "sigma1", noise_values, num_h0=sweep_num_h0, num_h1=sweep_num_h1,
        calib_trials=sweep_calib_trials, return_trials=True,
    )
    df_nodes, trials_nodes, calibration_nodes = sweep_parameter(
        cfg, "num_nodes", [int(v) for v in nodes_values], num_h0=sweep_num_h0,
        num_h1=sweep_num_h1, calib_trials=sweep_calib_trials, return_trials=True,
    )
    df_ti, trials_ti, calibration_ti = sweep_parameter(
        cfg, "inference_delay", delay_values, num_h0=sweep_num_h0,
        num_h1=sweep_num_h1, calib_trials=sweep_calib_trials, return_trials=True,
    )

    df_anomaly.to_csv(output_dir / "sweep_anomaly.csv", index=False)
    df_noise.to_csv(output_dir / "sweep_noise.csv", index=False)
    df_nodes.to_csv(output_dir / "sweep_nodes.csv", index=False)
    df_ti.to_csv(output_dir / "sweep_inference_delay.csv", index=False)
    trials_anomaly.to_csv(output_dir / "sweep_anomaly_trials.csv", index=False)
    trials_noise.to_csv(output_dir / "sweep_noise_trials.csv", index=False)
    trials_nodes.to_csv(output_dir / "sweep_nodes_trials.csv", index=False)
    trials_ti.to_csv(output_dir / "sweep_inference_delay_trials.csv", index=False)
    calibration_anomaly.to_csv(output_dir / "sweep_anomaly_calibration_trials.csv", index=False)
    calibration_noise.to_csv(output_dir / "sweep_noise_calibration_trials.csv", index=False)
    calibration_nodes.to_csv(output_dir / "sweep_nodes_calibration_trials.csv", index=False)
    calibration_ti.to_csv(output_dir / "sweep_inference_delay_calibration_trials.csv", index=False)

    plot_sweep(df_anomaly, "a1", "P_D", output_dir / "plot_detection_vs_anomaly.png", "Detection probability vs anomaly strength")
    plot_sweep(df_noise, "sigma1", "P_FA", output_dir / "plot_false_alarm_vs_noise.png", "False alarm probability vs noise")
    plot_sweep(df_nodes, "num_nodes", "C_H0_molecules_avg", output_dir / "plot_comm_load_h0_vs_nodes.png", "Communication load under H0 vs number of nodes")
    plot_sweep(df_nodes, "num_nodes", "C_H1_molecules_avg", output_dir / "plot_comm_load_h1_vs_nodes.png", "Communication load until detection under H1 vs number of nodes")
    plot_sweep(df_ti, "inference_delay", "D_avg_after_onset", output_dir / "plot_delay_vs_inference_delay.png", "Detection delay vs local inference delay")
    plot_pareto(df_anomaly, output_dir / "plot_pareto.png", "Pareto view (communication load until detection vs detection)")
    print("=== Baseline summary ===")
    print(summary.to_string(index=False))
    print("=== Demo anomaly sweep ===")
    print(f"a1 values: {anomaly_a1_values}")
    print(f"a2 ratio: {anomaly_a2_ratio}")
    print("Output written to:", output_dir)


def parse_float_list(spec: str, *, cast=float) -> List[float]:
    values: List[float] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(cast(float(part)))
    if not values:
        raise ValueError(f"No numeric values parsed from: {spec!r}")
    return values


def sweep_anomaly_pair(
    base_cfg: SimulationConfig,
    a1_values: List[float],
    a2_ratio: float,
    num_h0: int = 150,
    num_h1: int = 150,
    calib_trials: int = 200,
    *,
    return_trials: bool = False,
    strategies: Optional[List[str]] = None,
):
    rows = []
    trial_frames: List[pd.DataFrame] = []
    calibration_frames: List[pd.DataFrame] = []
    shared_thresholds: Optional[LocalThresholds] = None
    shared_gateway_thresholds: Optional[GatewayThresholds] = None
    sweep_started = time.time()
    for value_index, a1 in enumerate(a1_values):
        print(
            f"[sweep:anomaly_pair] value {value_index + 1}/{len(a1_values)}: "
            f"a1={float(a1)}, a2={float(a2_ratio) * float(a1)}",
            flush=True,
        )
        cfg = SimulationConfig(**asdict(base_cfg))
        cfg.a1 = float(a1)
        cfg.a2 = float(a2_ratio) * float(a1)
        namespace = "sweep:anomaly_pair:shared_scenarios"
        result = run_experiment(
            cfg,
            num_h0=num_h0,
            num_h1=num_h1,
            calib_trials=calib_trials,
            seed_namespace=namespace,
            return_calibration_trials=return_trials,
            precalibrated_thresholds=shared_thresholds,
            precalibrated_gateway_thresholds=shared_gateway_thresholds,
            strategies=strategies,
        )
        if return_trials:
            trials, summary, local_thresholds, gateway_thresholds, calibration_trials_df = result
            if value_index == 0:
                shared_thresholds = local_thresholds
                shared_gateway_thresholds = gateway_thresholds
            trials["a1"] = cfg.a1
            trials["a2"] = cfg.a2
            if not calibration_trials_df.empty:
                calibration_trials_df["a1"] = np.nan
                calibration_trials_df["a2"] = np.nan
                calibration_trials_df["sweep_scope"] = "shared_all_anomaly_strengths"
            trial_frames.append(trials)
            if not calibration_trials_df.empty:
                calibration_frames.append(calibration_trials_df)
        else:
            _, summary, local_thresholds, gateway_thresholds = result
            if value_index == 0:
                shared_thresholds = local_thresholds
                shared_gateway_thresholds = gateway_thresholds
        summary["a1"] = cfg.a1
        summary["a2"] = cfg.a2
        rows.append(summary)
        print(
            f"[sweep:anomaly_pair] completed a1={cfg.a1} "
            f"in {time.time() - sweep_started:.1f}s cumulative",
            flush=True,
        )
    summary_df = pd.concat(rows, ignore_index=True)
    if return_trials:
        return (
            summary_df,
            pd.concat(trial_frames, ignore_index=True),
            pd.concat(calibration_frames, ignore_index=True),
        )
    return summary_df


@dataclass
class RunModeConfig:
    output_dir: str = "sim_output"
    run_baseline: bool = True
    run_diagnostics: bool = False
    run_demo: bool = False
    run_eir_grid_search: bool = False


@dataclass
class BaselineConfig:
    num_h0: int = 120
    num_h1: int = 120
    calib_trials: int = 150


@dataclass
class DiagnosticsConfig:
    num_samples: int = 5000
    num_trials_state: int = 100


@dataclass
class DemoRunConfig:
    baseline_num_h0: int = 250
    baseline_num_h1: int = 250
    baseline_calib_trials: int = 250
    sweep_num_h0: int = 150
    sweep_num_h1: int = 150
    sweep_calib_trials: int = 200


@dataclass
class SweepConfig:
    anomaly_a1_values: List[float] = None
    anomaly_a2_ratio: float = 0.4
    noise_values: List[float] = None
    nodes_values: List[int] = None
    delay_values: List[float] = None

    def __post_init__(self) -> None:
        if self.anomaly_a1_values is None:
            self.anomaly_a1_values = [0.03, 0.06, 0.10, 0.15, 0.20]
        if self.noise_values is None:
            self.noise_values = [0.08, 0.12, 0.16, 0.20]
        if self.nodes_values is None:
            self.nodes_values = [20, 40, 60, 100]
        if self.delay_values is None:
            self.delay_values = [0.0, 10.0, 30.0, 60.0, 120.0]


@dataclass
class GridSearchConfig:
    gate_values: List[float] = None
    hysteresis_values: List[float] = None
    gate_hysteresis_values: List[float] = None
    w2_values: List[float] = None
    parameter_values: Dict[str, List[float]] = None
    focus_max_a1: float = 0.10
    target_pfa: float = 0.05
    num_h0: int = 150
    num_h1: int = 150
    calib_trials: int = 200

    def __post_init__(self) -> None:
        # Backwards-compatible legacy fields remain supported, but the preferred
        # v2.0 config style is [eir_grid.parameter_values].
        if self.parameter_values is None:
            if self.gate_values is None:
                self.gate_values = [0.84, 0.85, 0.86]
            if self.hysteresis_values is None:
                self.hysteresis_values = [0.02, 0.03, 0.04]
            if self.gate_hysteresis_values is None:
                self.gate_hysteresis_values = [0.01, 0.02, 0.03]
            if self.w2_values is None:
                self.w2_values = [0.75, 0.90]
            self.parameter_values = {
                "eir_gate_quantile": list(self.gate_values),
                "hysteresis_margin_eir": list(self.hysteresis_values),
                "gate_hysteresis_margin": list(self.gate_hysteresis_values),
                "w2": list(self.w2_values),
            }
        else:
            cleaned = {}
            for key, values in self.parameter_values.items():
                if not isinstance(values, list):
                    raise ValueError(f"Grid-search parameter '{key}' must be a TOML array/list.")
                cleaned[key] = [float(v) for v in values]
            self.parameter_values = cleaned


def _pick_known_fields(dataclass_type, values: Dict[str, object]) -> Dict[str, object]:
    names = set(dataclass_type.__dataclass_fields__.keys())
    return {k: v for k, v in values.items() if k in names}


def _coerce_path_like(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return value


def export_effective_config(
    cfg: SimulationConfig,
    run_modes: RunModeConfig,
    baseline: BaselineConfig,
    diagnostics: DiagnosticsConfig,
    demo_cfg: DemoRunConfig,
    sweeps: SweepConfig,
    grid: GridSearchConfig,
    outpath: Path,
) -> None:
    payload = {
        "simulation": {k: _coerce_path_like(v) for k, v in asdict(cfg).items()},
        "run_modes": asdict(run_modes),
        "baseline": asdict(baseline),
        "diagnostics": asdict(diagnostics),
        "demo": asdict(demo_cfg),
        "sweeps": asdict(sweeps),
        "eir_grid": asdict(grid),
    }
    outpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def locate_config_path(explicit: Optional[str]) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    else:
        candidates.extend([Path('.config_sim')])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ', '.join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"No simulator config file found. Expected one of: {searched}. "
        "Use --config to provide an explicit path."
    )


def load_bundle_from_config(config_path: Path) -> Tuple[SimulationConfig, RunModeConfig, BaselineConfig, DiagnosticsConfig, DemoRunConfig, SweepConfig, GridSearchConfig]:
    with config_path.open('rb') as fh:
        raw = tomllib.load(fh)

    raw_simulation = dict(raw.get('simulation', {}))
    if 'receiver_min_distance' not in raw_simulation and 'receiver_radius' in raw_simulation:
        warnings.warn(
            "Configuration key 'receiver_radius' is deprecated; use "
            "'receiver_min_distance'. Its value is interpreted as an effective "
            "near-gateway cutoff, not as a physical receiver radius.",
            FutureWarning,
            stacklevel=2,
        )
        raw_simulation['receiver_min_distance'] = raw_simulation['receiver_radius']
    sim_values = _pick_known_fields(SimulationConfig, raw_simulation)
    if 'eir_gate_manual' in sim_values and sim_values['eir_gate_manual'] in (0, 0.0, '', False):
        sim_values['eir_gate_manual'] = None
    cfg = SimulationConfig(**sim_values)
    run_modes = RunModeConfig(**_pick_known_fields(RunModeConfig, raw.get('run_modes', {})))
    baseline = BaselineConfig(**_pick_known_fields(BaselineConfig, raw.get('baseline', {})))
    diagnostics = DiagnosticsConfig(**_pick_known_fields(DiagnosticsConfig, raw.get('diagnostics', {})))
    demo_cfg = DemoRunConfig(**_pick_known_fields(DemoRunConfig, raw.get('demo', {})))
    sweeps = SweepConfig(**_pick_known_fields(SweepConfig, raw.get('sweeps', {})))
    grid = GridSearchConfig(**_pick_known_fields(GridSearchConfig, raw.get('eir_grid', {})))
    return cfg, run_modes, baseline, diagnostics, demo_cfg, sweeps, grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DNA NN simulator v2.0. All simulator parameters are loaded from a TOML-based .config_sim file (or a custom file via --config)."
    )
    parser.add_argument('--config', type=str, default=None, help='Path to the TOML simulator config file. Defaults to .config_sim.')
    parser.add_argument('--output-dir', type=str, default=None, help='Optional override for run_modes.output_dir from the config file.')
    parser.add_argument('--print-effective-config', action='store_true', help='Print the loaded effective configuration to stdout before execution.')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = locate_config_path(args.config)
    cfg, run_modes, baseline_cfg, diag_cfg, demo_cfg, sweeps_cfg, grid_cfg = load_bundle_from_config(config_path)

    if args.output_dir is not None:
        run_modes.output_dir = args.output_dir

    outdir = Path(run_modes.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    export_effective_config(cfg, run_modes, baseline_cfg, diag_cfg, demo_cfg, sweeps_cfg, grid_cfg, outdir / 'effective_config.json')

    if args.print_effective_config:
        print((outdir / 'effective_config.json').read_text(encoding='utf-8'))

    ran_any = False

    if run_modes.run_diagnostics:
        ran_any = True
        _, _, local_summary, dynamics_summary, thresholds = run_local_diagnostics(
            cfg,
            outdir / 'diagnostics',
            num_samples=diag_cfg.num_samples,
            num_trials_state=diag_cfg.num_trials_state,
        )
        save_metadata(
            cfg,
            thresholds,
            GatewayThresholds(rr=float('nan'), tr=float('nan'), eir=float('nan')),
            outdir / 'diagnostics' / 'calibration_preview.json',
        )
        print('=== Local diagnostics summary ===')
        print(local_summary.to_string(index=False))
        print('\n=== State dynamics summary ===')
        print(dynamics_summary.to_string(index=False))
        print('\nDiagnostics written to:', outdir / 'diagnostics')

    if run_modes.run_demo:
        ran_any = True
        demo(
            outdir,
            cfg,
            anomaly_a1_values=sweeps_cfg.anomaly_a1_values,
            anomaly_a2_ratio=sweeps_cfg.anomaly_a2_ratio,
            noise_values=sweeps_cfg.noise_values,
            nodes_values=sweeps_cfg.nodes_values,
            delay_values=sweeps_cfg.delay_values,
            baseline_num_h0=demo_cfg.baseline_num_h0,
            baseline_num_h1=demo_cfg.baseline_num_h1,
            baseline_calib_trials=demo_cfg.baseline_calib_trials,
            sweep_num_h0=demo_cfg.sweep_num_h0,
            sweep_num_h1=demo_cfg.sweep_num_h1,
            sweep_calib_trials=demo_cfg.sweep_calib_trials,
        )

    if run_modes.run_eir_grid_search:
        ran_any = True
        run_eir_grid_search(
            outdir / 'eir_grid_search',
            cfg,
            grid_parameters=grid_cfg.parameter_values,
            anomaly_a1_values=sweeps_cfg.anomaly_a1_values,
            anomaly_a2_ratio=sweeps_cfg.anomaly_a2_ratio,
            focus_max_a1=grid_cfg.focus_max_a1,
            target_pfa=grid_cfg.target_pfa,
            num_h0=grid_cfg.num_h0,
            num_h1=grid_cfg.num_h1,
            calib_trials=grid_cfg.calib_trials,
        )
        print('EIR grid-search written to:', outdir / 'eir_grid_search')

    if run_modes.run_baseline or not ran_any:
        baseline_outdir = outdir / 'baseline' if ran_any else outdir
        baseline_outdir.mkdir(parents=True, exist_ok=True)
        trials, summary, thresholds, gateway_thresholds, calibration_trials = run_experiment(
            cfg,
            num_h0=baseline_cfg.num_h0,
            num_h1=baseline_cfg.num_h1,
            calib_trials=baseline_cfg.calib_trials,
            seed_namespace='baseline',
            return_calibration_trials=True,
        )
        save_metadata(cfg, thresholds, gateway_thresholds, baseline_outdir / 'calibration.json')
        trials.to_csv(baseline_outdir / 'trial_results.csv', index=False)
        calibration_trials.to_csv(baseline_outdir / 'gateway_calibration_trials.csv', index=False)
        summary.to_csv(baseline_outdir / 'summary.csv', index=False)
        print(summary.to_string(index=False))
        if not ran_any and not run_modes.run_baseline:
            print('No explicit run mode enabled in config; baseline was executed by default.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
