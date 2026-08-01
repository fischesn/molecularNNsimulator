import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "dna-nn-simulator-v2.0.py"
SPEC = importlib.util.spec_from_file_location("dna_nn_simulator_v2_stats", SCRIPT_PATH)
SIM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SIM
SPEC.loader.exec_module(SIM)


class StatisticalIntervalTests(unittest.TestCase):
    def test_wilson_interval_matches_known_half_success_case(self):
        low, high = SIM.wilson_interval(5, 10)
        self.assertAlmostEqual(low, 0.2366, places=4)
        self.assertAlmostEqual(high, 0.7634, places=4)

    def test_wilson_interval_handles_empty_sample(self):
        low, high = SIM.wilson_interval(0, 0)
        self.assertTrue(np.isnan(low))
        self.assertTrue(np.isnan(high))

    def test_constant_mean_has_zero_width_interval(self):
        low, high = SIM.mean_confidence_interval(np.asarray([3.0, 3.0, 3.0]))
        self.assertEqual(low, 3.0)
        self.assertEqual(high, 3.0)

    def test_summary_contains_sample_sizes_and_intervals(self):
        rows = []
        for state_h1 in [0, 1]:
            for trial in range(10):
                detected = int(state_h1 == 1 and trial < 5)
                first_alarm = 10.0 if (state_h1 == 0 and trial == 0) else np.inf
                rows.append(
                    {
                        "strategy": "RR",
                        "state_h1": state_h1,
                        "detected": detected,
                        "pre_onset_alarm": 0,
                        "detection_time": 40.0 if detected else np.inf,
                        "first_alarm_time": first_alarm,
                        "transmissions": trial + 1,
                        "max_gateway_evidence": 1.0,
                    }
                )
        summary = SIM.summarize_results(pd.DataFrame(rows), SIM.SimulationConfig())
        row = summary.iloc[0]
        self.assertEqual(row["n_H0"], 10)
        self.assertEqual(row["n_H1"], 10)
        self.assertEqual(row["n_detected_H1"], 5)
        self.assertLess(row["P_D_ci_low"], row["P_D"])
        self.assertGreater(row["P_D_ci_high"], row["P_D"])
        self.assertIn("C_H1_molecules_avg_ci_low", summary.columns)
        self.assertIn("D_avg_after_onset_ci_high", summary.columns)


class CommonScenarioSeedTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SIM.SimulationConfig(
            num_nodes=3,
            dt=1.0,
            t_max=2.0,
            anomaly_start=1.0,
            diffusion_coeff=10.0,
            receiver_min_distance=0.1,
        )
        self.thresholds = SIM.LocalThresholds(
            rr_tau1=0.3,
            rr_tau2=0.2,
            tr_tau=0.3,
            eir_theta=0.4,
            eir_gate_tau=0.15,
        )
        self.gateway_thresholds = SIM.GatewayThresholds(
            rr=np.inf,
            tr=np.inf,
            eir=np.inf,
        )

    def test_calibration_and_evaluation_streams_are_distinct(self):
        calibration = SIM.deterministic_seed(
            self.cfg,
            phase="gateway_calibration",
            trial_index=7,
            namespace="test",
        )
        evaluation = SIM.deterministic_seed(
            self.cfg,
            phase="evaluation",
            trial_index=7,
            namespace="test",
        )
        self.assertNotEqual(calibration, evaluation)

    def test_evaluation_scenario_seeds_are_shared_across_strategies(self):
        frames = [
            SIM.run_trials(
                self.cfg,
                self.thresholds,
                self.gateway_thresholds,
                strategy,
                num_h0=2,
                num_h1=2,
                seed_namespace="paired-test",
            )
            for strategy in ["RR", "TR", "EIR"]
        ]
        combined = pd.concat(frames, ignore_index=True)
        unique_per_scenario = combined.groupby(
            ["state_h1", "trial_index"]
        )["scenario_seed"].nunique()
        self.assertTrue((unique_per_scenario == 1).all())

    def test_gateway_calibration_seeds_are_shared_across_strategies(self):
        _, records = SIM.calibrate_all_gateway_thresholds(
            self.cfg,
            self.thresholds,
            num_trials=2,
            seed_namespace="paired-calibration-test",
            return_trials=True,
        )
        unique_per_trial = records.groupby("trial_index")["scenario_seed"].nunique()
        self.assertTrue((unique_per_trial == 1).all())

    def test_experiment_can_evaluate_only_selected_strategy(self):
        trials, summary, _, gateway, calibration = SIM.run_experiment(
            self.cfg,
            num_h0=2,
            num_h1=2,
            calib_trials=2,
            seed_namespace="eir-only-test",
            return_calibration_trials=True,
            strategies=["EIR"],
        )
        self.assertEqual(set(trials["strategy"]), {"EIR"})
        self.assertEqual(set(summary["strategy"]), {"EIR"})
        self.assertEqual(set(calibration["strategy"]), {"EIR"})
        self.assertTrue(np.isnan(gateway.rr))
        self.assertTrue(np.isnan(gateway.tr))
        self.assertTrue(np.isfinite(gateway.eir))


class Phase4PipelineSmokeTests(unittest.TestCase):
    def test_demo_persists_summary_evaluation_and_calibration_trials(self):
        cfg = SIM.SimulationConfig(
            num_nodes=3,
            dt=1.0,
            t_max=3.0,
            anomaly_start=1.0,
            diffusion_coeff=10.0,
            receiver_min_distance=0.1,
        )
        expected = [
            "summary.csv",
            "trial_results.csv",
            "gateway_calibration_trials.csv",
            "sweep_anomaly.csv",
            "sweep_anomaly_trials.csv",
            "sweep_anomaly_calibration_trials.csv",
            "sweep_noise_trials.csv",
            "sweep_nodes_trials.csv",
            "sweep_inference_delay_trials.csv",
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            SIM.demo(
                output_dir,
                cfg,
                anomaly_a1_values=[0.03, 0.06],
                noise_values=[0.1],
                nodes_values=[3],
                delay_values=[0.0],
                baseline_num_h0=2,
                baseline_num_h1=2,
                baseline_calib_trials=2,
                sweep_num_h0=2,
                sweep_num_h1=2,
                sweep_calib_trials=2,
            )
            for filename in expected:
                self.assertTrue((output_dir / filename).is_file(), filename)
            trials = pd.read_csv(output_dir / "trial_results.csv")
            self.assertIn("scenario_seed", trials.columns)
            summary = pd.read_csv(output_dir / "summary.csv")
            self.assertIn("P_D_ci_low", summary.columns)
            anomaly_calibration = pd.read_csv(
                output_dir / "sweep_anomaly_calibration_trials.csv"
            )
            self.assertTrue(
                (anomaly_calibration["sweep_scope"] == "shared_all_anomaly_strengths").all()
            )


class Phase5GridSmokeTests(unittest.TestCase):
    def test_grid_reuses_reference_strategies_and_persists_eir_trials(self):
        cfg = SIM.SimulationConfig(
            num_nodes=3,
            dt=1.0,
            t_max=3.0,
            anomaly_start=1.0,
            diffusion_coeff=10.0,
            receiver_min_distance=0.1,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            SIM.run_eir_grid_search(
                output_dir,
                cfg,
                grid_parameters={"w2": [0.9]},
                anomaly_a1_values=[0.03, 0.06],
                anomaly_a2_ratio=0.4,
                focus_max_a1=0.06,
                num_h0=2,
                num_h1=2,
                calib_trials=2,
            )
            reference = pd.read_csv(output_dir / "eir_grid_reference_trials.csv")
            eir = pd.read_csv(output_dir / "eir_grid_eir_trials.csv")
            summary = pd.read_csv(output_dir / "eir_grid_summary.csv")
            self.assertEqual(set(reference["strategy"]), {"RR", "TR"})
            self.assertEqual(set(eir["strategy"]), {"EIR"})
            self.assertEqual(len(summary), 1)


if __name__ == "__main__":
    unittest.main()
