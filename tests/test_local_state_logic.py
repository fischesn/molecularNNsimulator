import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "dna-nn-simulator-v2.0.py"
SPEC = importlib.util.spec_from_file_location("dna_nn_simulator_v2", SCRIPT_PATH)
SIM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SIM
SPEC.loader.exec_module(SIM)


class EirStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SIM.SimulationConfig(
            w1=1.0,
            w2=1.0,
            use_eir_gate=True,
            use_hysteresis=True,
            hysteresis_margin_eir=0.2,
            gate_hysteresis_margin=0.1,
        )
        self.thresholds = SIM.LocalThresholds(
            rr_tau1=0.5,
            rr_tau2=0.5,
            tr_tau=0.5,
            eir_theta=1.0,
            eir_gate_tau=0.4,
        )

    def evaluate(self, x1, x2, previous):
        return SIM.hysteretic_positive_state(
            "EIR",
            np.asarray(x1, dtype=float),
            np.asarray(x2, dtype=float),
            np.asarray(previous, dtype=bool),
            self.thresholds,
            self.cfg,
        )

    def test_activation_requires_score_and_gate(self):
        result = self.evaluate(
            x1=[0.8, 0.5, 0.8],
            x2=[0.5, 0.5, 0.3],
            previous=[False, False, False],
        )
        np.testing.assert_array_equal(result, [True, False, False])

    def test_active_state_turns_off_when_score_alone_is_low(self):
        result = self.evaluate(x1=[0.1], x2=[0.5], previous=[True])
        np.testing.assert_array_equal(result, [False])

    def test_active_state_turns_off_when_gate_alone_is_low(self):
        result = self.evaluate(x1=[1.2], x2=[0.2], previous=[True])
        np.testing.assert_array_equal(result, [False])

    def test_active_state_persists_inside_both_hysteresis_bands(self):
        result = self.evaluate(x1=[0.55], x2=[0.35], previous=[True])
        np.testing.assert_array_equal(result, [True])

    def test_inactive_state_does_not_turn_on_inside_hysteresis_band(self):
        result = self.evaluate(x1=[0.55], x2=[0.35], previous=[False])
        np.testing.assert_array_equal(result, [False])

    def test_exact_lower_thresholds_do_not_force_transition(self):
        self.cfg.hysteresis_margin_eir = 0.25
        self.cfg.gate_hysteresis_margin = 0.125
        self.thresholds.eir_gate_tau = 0.5
        result = self.evaluate(x1=[0.375], x2=[0.375], previous=[True])
        np.testing.assert_array_equal(result, [True])

    def test_gate_can_be_disabled(self):
        self.cfg.use_eir_gate = False
        result = self.evaluate(x1=[0.9], x2=[0.2], previous=[False])
        np.testing.assert_array_equal(result, [True])


class AlarmEmissionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SIM.SimulationConfig(
            edge_triggered=True,
            refractory_period=10.0,
            allow_refresh_if_still_positive=False,
        )

    def test_only_a_rising_edge_emits(self):
        emitted = SIM.should_emit_alarm(
            current_positive=np.asarray([True, True, False]),
            prev_positive=np.asarray([False, True, True]),
            last_send_time=np.asarray([-np.inf, -np.inf, -np.inf]),
            t=20.0,
            cfg=self.cfg,
        )
        np.testing.assert_array_equal(emitted, [True, False, False])

    def test_refractory_period_suppresses_early_new_edge(self):
        emitted = SIM.should_emit_alarm(
            current_positive=np.asarray([True, True]),
            prev_positive=np.asarray([False, False]),
            last_send_time=np.asarray([15.0, 10.0]),
            t=20.0,
            cfg=self.cfg,
        )
        np.testing.assert_array_equal(emitted, [False, True])


if __name__ == "__main__":
    unittest.main()
