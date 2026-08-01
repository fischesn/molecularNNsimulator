import importlib.util
import math
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "dna-nn-simulator-v2.0.py"
SPEC = importlib.util.spec_from_file_location("dna_nn_simulator_v2_channel", SCRIPT_PATH)
SIM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SIM
SPEC.loader.exec_module(SIM)


class FirstPassageChannelTests(unittest.TestCase):
    def test_density_is_finite_and_nonnegative(self):
        cfg = SIM.SimulationConfig(
            diffusion_coeff=2.0,
            drift_velocity=0.3,
            channel_decay_rate=0.02,
            receiver_min_distance=0.1,
        )
        times = np.geomspace(1e-6, 100.0, 10000)
        density = SIM.first_passage_density_1d(times, 3.0, cfg)
        self.assertTrue(np.all(np.isfinite(density)))
        self.assertTrue(np.all(density >= 0.0))

    def test_zero_drift_cumulative_probability_matches_analytic_cdf(self):
        cfg = SIM.SimulationConfig(
            dt=0.001,
            diffusion_coeff=1.0,
            drift_velocity=0.0,
            channel_decay_rate=0.0,
            receiver_min_distance=0.01,
            alarm_molecules=1.0,
            receiver_gain=1.0,
        )
        horizon = 100.0
        response = SIM.diffusive_channel_response(1.0, cfg, int(horizon / cfg.dt))
        expected = math.erfc(1.0 / math.sqrt(4.0 * cfg.diffusion_coeff * horizon))
        self.assertAlmostEqual(float(response.sum()), expected, delta=2e-5)

    def test_zero_drift_peak_time_matches_analytic_mode(self):
        cfg = SIM.SimulationConfig(
            diffusion_coeff=2.0,
            drift_velocity=0.0,
            channel_decay_rate=0.0,
            receiver_min_distance=0.01,
        )
        times = np.linspace(0.001, 3.0, 100000)
        density = SIM.first_passage_density_1d(times, 3.0, cfg)
        measured_mode = float(times[np.argmax(density)])
        expected_mode = 3.0 ** 2 / (6.0 * cfg.diffusion_coeff)
        self.assertAlmostEqual(measured_mode, expected_mode, delta=5e-4)

    def test_positive_drift_toward_gateway_increases_early_arrivals(self):
        base = dict(
            dt=0.01,
            diffusion_coeff=1.0,
            channel_decay_rate=0.0,
            receiver_min_distance=0.01,
            alarm_molecules=1.0,
            receiver_gain=1.0,
        )
        toward = SIM.diffusive_channel_response(
            3.0, SIM.SimulationConfig(**base, drift_velocity=0.5), 500
        ).sum()
        no_drift = SIM.diffusive_channel_response(
            3.0, SIM.SimulationConfig(**base, drift_velocity=0.0), 500
        ).sum()
        away = SIM.diffusive_channel_response(
            3.0, SIM.SimulationConfig(**base, drift_velocity=-0.5), 500
        ).sum()
        self.assertGreater(toward, no_drift)
        self.assertGreater(no_drift, away)

    def test_minimum_distance_is_an_effective_cutoff(self):
        cfg = SIM.SimulationConfig(
            dt=0.1,
            diffusion_coeff=1.0,
            receiver_min_distance=0.5,
        )
        at_gateway = SIM.diffusive_channel_response(0.0, cfg, 100)
        at_cutoff = SIM.diffusive_channel_response(0.5, cfg, 100)
        np.testing.assert_array_equal(at_gateway, at_cutoff)

    def test_discrete_response_includes_burst_gain_and_time_step(self):
        cfg = SIM.SimulationConfig(
            dt=0.2,
            diffusion_coeff=4.0,
            receiver_min_distance=0.1,
            alarm_molecules=80.0,
            receiver_gain=0.25,
        )
        times = (np.arange(20, dtype=float) + 0.5) * cfg.dt
        density = SIM.first_passage_density_1d(times, 2.0, cfg)
        response = SIM.diffusive_channel_response(2.0, cfg, len(times))
        np.testing.assert_allclose(
            response,
            cfg.alarm_molecules * cfg.receiver_gain * density * cfg.dt,
        )

    def test_invalid_physical_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            SIM.first_passage_density_1d(
                np.asarray([1.0]),
                1.0,
                SIM.SimulationConfig(diffusion_coeff=0.0),
            )
        with self.assertRaises(ValueError):
            SIM.first_passage_density_1d(
                np.asarray([1.0]),
                1.0,
                SIM.SimulationConfig(channel_decay_rate=-0.1),
            )
        with self.assertRaises(ValueError):
            SIM.diffusive_channel_response(
                1.0,
                SIM.SimulationConfig(dt=0.0),
                10,
            )

    def test_legacy_receiver_radius_key_is_migrated(self):
        config_text = """
[simulation]
receiver_radius = 7.5

[run_modes]
run_baseline = false
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".config_sim"
            path.write_text(config_text, encoding="utf-8")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cfg, *_ = SIM.load_bundle_from_config(path)
        self.assertEqual(cfg.receiver_min_distance, 7.5)
        self.assertTrue(any("receiver_radius" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
