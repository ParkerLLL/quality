from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ETF_UNIVERSE


def generate_sample_prices(
    start: str = "2022-01-04",
    periods: int = 820,
    seed: int = 42,
) -> pd.DataFrame:
    """Create plausible ETF-like daily prices for learning and demos."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    t = np.arange(periods)

    market_regime = (
        0.00025
        + 0.00055 * np.sin(t / 95)
        - 0.00045 * np.sin(t / 41)
        + rng.normal(0, 0.0065, periods)
    )
    growth_factor = 0.0001 + 0.0008 * np.sin(t / 70 + 1.2) + rng.normal(0, 0.008, periods)
    value_factor = 0.00008 + 0.00035 * np.sin(t / 120 + 2.0) + rng.normal(0, 0.0045, periods)
    bond_factor = 0.00008 + rng.normal(0, 0.0012, periods)
    gold_factor = 0.00012 + 0.00045 * np.cos(t / 80) + rng.normal(0, 0.005, periods)
    cash_factor = np.full(periods, 0.000055) + rng.normal(0, 0.00003, periods)

    returns = {
        "510300.SH": market_regime + rng.normal(0, 0.003, periods),
        "510500.SH": 1.08 * market_regime + 0.35 * growth_factor + rng.normal(0, 0.004, periods),
        "159915.SZ": 0.95 * market_regime + 0.85 * growth_factor + rng.normal(0, 0.006, periods),
        "512880.SH": 1.2 * market_regime + rng.normal(0, 0.010, periods),
        "512800.SH": 0.75 * market_regime + 0.7 * value_factor + rng.normal(0, 0.004, periods),
        "518880.SH": gold_factor + rng.normal(0, 0.002, periods),
        "511010.SH": bond_factor,
        "511880.SH": cash_factor,
    }

    data = {}
    for index, fund in enumerate(ETF_UNIVERSE):
        base_price = 1.0 + index * 0.08
        clipped_returns = np.clip(returns[fund.code], -0.095, 0.095)
        data[fund.code] = base_price * np.cumprod(1 + clipped_returns)

    prices = pd.DataFrame(data, index=dates)
    prices.index.name = "date"
    return prices.round(4)


def latest_price_table(prices: pd.DataFrame) -> pd.DataFrame:
    latest = prices.iloc[-1]
    previous = prices.iloc[-2]
    rows = []
    for fund in ETF_UNIVERSE:
        rows.append(
            {
                "代码": fund.code,
                "名称": fund.name,
                "类别": fund.category,
                "最新价格": latest[fund.code],
                "日涨跌": latest[fund.code] / previous[fund.code] - 1,
                "风险等级": fund.risk_level,
                "新手备注": fund.beginner_note,
            }
        )
    return pd.DataFrame(rows)

