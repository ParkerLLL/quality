import unittest

import pandas as pd

from fund_quant_lab.data_sources import _extract_close_frame
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

    def test_extract_akshare_close_frame(self):
        raw = pd.DataFrame(
            {
                "日期": ["2024-01-02", "2024-01-03"],
                "收盘": ["1.23", "1.25"],
                "成交额": [1000, 1200],
            }
        )
        frame = _extract_close_frame(raw, "510300.SH")

        self.assertEqual(frame.columns.tolist(), ["510300.SH"])
        self.assertEqual(frame.index[0], pd.Timestamp("2024-01-02"))
        self.assertAlmostEqual(float(frame.iloc[-1, 0]), 1.25)


if __name__ == "__main__":
    unittest.main()
