from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ETF_UNIVERSE, FUND_BY_CODE
from .risk import classify_market_state, current_drawdown


@dataclass(frozen=True)
class StrategySettings:
    lookback_days: int = 60
    top_n: int = 3
    max_weight: float = 0.35
    cash_buffer: float = 0.05
    trading_fee: float = 0.0003
    rebalance_frequency: str = "M"


def momentum_score_table(prices: pd.DataFrame, as_of: pd.Timestamp, lookback_days: int = 60) -> pd.DataFrame:
    if as_of not in prices.index:
        as_of = prices.index[prices.index <= as_of][-1]

    window = prices.loc[:as_of]
    if len(window) <= lookback_days + 20:
        raise ValueError("Not enough price history to calculate scores.")

    latest = window.iloc[-1]
    ret20 = latest / window.iloc[-20] - 1
    ret_lookback = latest / window.iloc[-lookback_days] - 1
    returns = window.pct_change()
    vol20 = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
    dd60 = window.tail(60).apply(current_drawdown)

    rows = []
    for fund in ETF_UNIVERSE:
        code = fund.code
        raw_score = ret_lookback[code] - 0.45 * vol20[code] + 0.25 * dd60[code]
        rows.append(
            {
                "code": code,
                "name": fund.name,
                "category": fund.category,
                "risk_level": fund.risk_level,
                "defensive": fund.defensive,
                "price": latest[code],
                "return_20d": ret20[code],
                "return_lookback": ret_lookback[code],
                "volatility_20d": vol20[code],
                "drawdown_60d": dd60[code],
                "score": raw_score,
            }
        )

    table = pd.DataFrame(rows)
    table["rank"] = table["score"].rank(ascending=False, method="first").astype(int)
    return table.sort_values("score", ascending=False).reset_index(drop=True)


def target_weights_for_date(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    settings: StrategySettings,
) -> pd.Series:
    scores = momentum_score_table(prices, as_of, settings.lookback_days)
    market_state = classify_market_state(prices.loc[:as_of])
    weights = pd.Series(0.0, index=prices.columns)

    tradable_budget = max(0.0, 1 - settings.cash_buffer)

    if market_state["state"] == "偏弱":
        defensive_codes = ["511010.SH", "511880.SH", "518880.SH"]
        defensive_weights = [0.45, 0.30, 0.20]
        for code, weight in zip(defensive_codes, defensive_weights, strict=True):
            if code in weights.index:
                weights[code] = min(weight, settings.max_weight)
        leftover = tradable_budget - weights.sum()
        if leftover > 0 and "511880.SH" in weights.index:
            weights["511880.SH"] += leftover
        return weights.clip(upper=settings.max_weight)

    candidates = scores[~scores["defensive"]].head(settings.top_n)
    if market_state["state"] == "震荡":
        equity_budget = tradable_budget * 0.72
        defensive_budget = tradable_budget - equity_budget
    else:
        equity_budget = tradable_budget
        defensive_budget = 0.0

    if not candidates.empty:
        raw_scores = candidates.set_index("code")["score"].clip(lower=0)
        if raw_scores.sum() == 0:
            selected_weights = pd.Series(1 / len(candidates), index=candidates["code"])
        else:
            selected_weights = raw_scores / raw_scores.sum()
        selected_weights = selected_weights * equity_budget
        selected_weights = selected_weights.clip(upper=settings.max_weight)
        for code, weight in selected_weights.items():
            weights[code] = weight

    if defensive_budget > 0:
        weights["511010.SH"] += min(defensive_budget * 0.65, settings.max_weight - weights.get("511010.SH", 0.0))
        weights["511880.SH"] += max(0.0, defensive_budget - weights["511010.SH"])

    leftover = tradable_budget - weights.sum()
    if leftover > 0 and "511880.SH" in weights.index:
        weights["511880.SH"] += leftover

    return weights.clip(lower=0, upper=settings.max_weight)


def build_signal_snapshot(prices: pd.DataFrame, settings: StrategySettings) -> tuple[pd.DataFrame, pd.Series]:
    as_of = prices.index[-1]
    scores = momentum_score_table(prices, as_of, settings.lookback_days)
    weights = target_weights_for_date(prices, as_of, settings)

    rows = []
    for _, row in scores.iterrows():
        code = row["code"]
        weight = float(weights.get(code, 0.0))
        action = "买入/持有" if weight > 0.02 else "观察"
        if FUND_BY_CODE[code].defensive and weight > 0.15:
            action = "防守配置"
        rows.append(
            {
                "代码": code,
                "名称": row["name"],
                "类别": row["category"],
                "目标仓位": weight,
                "操作": action,
                "60日表现": row["return_lookback"],
                "20日波动": row["volatility_20d"],
                "60日回撤": row["drawdown_60d"],
                "评分": row["score"],
                "解释": _signal_reason(row, weight),
            }
        )
    return pd.DataFrame(rows), weights


def backtest_rotation(
    prices: pd.DataFrame,
    initial_capital: float,
    settings: StrategySettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_history = settings.lookback_days + 30
    returns = prices.pct_change().fillna(0.0)
    equity = pd.Series(index=prices.index[min_history:], dtype=float)
    weight_history = []
    current_weights = pd.Series(0.0, index=prices.columns)
    nav = float(initial_capital)

    rebalance_dates = _rebalance_dates(prices.index[min_history:], settings.rebalance_frequency)
    rebalance_set = set(rebalance_dates)

    for date in prices.index[min_history:]:
        if date in rebalance_set:
            new_weights = target_weights_for_date(prices.loc[:date], date, settings)
            turnover = float((new_weights - current_weights).abs().sum())
            nav *= 1 - turnover * settings.trading_fee
            current_weights = new_weights
            weight_history.append(
                {
                    "date": date,
                    "turnover": turnover,
                    **{code: float(weight) for code, weight in current_weights.items()},
                }
            )

        daily_return = float((current_weights * returns.loc[date]).sum())
        nav *= 1 + daily_return
        equity.loc[date] = nav

    result = pd.DataFrame(
        {
            "date": equity.index,
            "equity": equity.values,
            "daily_return": equity.pct_change().fillna(0.0).values,
        }
    )
    weights = pd.DataFrame(weight_history)
    return result, weights


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> list[pd.Timestamp]:
    series = pd.Series(index, index=index)
    if frequency == "W":
        groups = series.groupby(index.to_period("W-FRI")).last().dropna()
    else:
        groups = series.groupby(index.to_period("M")).last().dropna()
    return [pd.Timestamp(date) for date in groups.values]


def _signal_reason(row: pd.Series, weight: float) -> str:
    if weight <= 0.02:
        if row["return_lookback"] < 0:
            return "中期表现为负，先放在观察区。"
        if row["volatility_20d"] > 0.28:
            return "短期波动偏高，系统暂时不追。"
        return "评分不在前列，等待更清晰的机会。"
    if row["defensive"]:
        return "当前组合需要防守资产来降低波动。"
    if row["return_lookback"] > 0 and row["drawdown_60d"] > -0.08:
        return "中期趋势较好，且近期回撤可控。"
    return "入选组合，但仓位受风险控制限制。"
