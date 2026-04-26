import unittest

import pandas as pd

from fund_quant_lab.risk import summarize_equity
from fund_quant_lab.sample_data import generate_sample_prices
from fund_quant_lab.strategy import StrategySettings, backtest_rotation, build_signal_snapshot, target_weights_for_date


class CoreBehaviorTest(unittest.TestCase):
    def setUp(self):
        self.prices = generate_sample_prices(periods=420, seed=9)
        self.settings = StrategySettings()

    def test_target_weights_are_bounded(self):
        weights = target_weights_for_date(self.prices, self.prices.index[-1], self.settings)

        self.assertLessEqual(float(weights.sum()), 1.000001)
        self.assertGreaterEqual(float(weights.min()), 0.0)
        self.assertLessEqual(float(weights.max()), self.settings.max_weight + 1e-9)

    def test_signal_snapshot_has_explanations(self):
        signals, weights = build_signal_snapshot(self.prices, self.settings)

        self.assertFalse(signals.empty)
        self.assertIn("解释", signals.columns)
        self.assertTrue(signals["解释"].str.len().gt(0).all())
        self.assertGreater(float(weights.sum()), 0.0)

    def test_backtest_returns_equity_curve(self):
        equity, weight_history = backtest_rotation(self.prices, 100_000, self.settings)
        metrics = summarize_equity(pd.Series(equity["equity"].values, index=equity["date"]))

        self.assertFalse(equity.empty)
        self.assertFalse(weight_history.empty)
        self.assertGreater(float(equity["equity"].iloc[-1]), 0.0)
        self.assertIn("max_drawdown", metrics)


if __name__ == "__main__":
    unittest.main()

