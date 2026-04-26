from __future__ import annotations

import math

import pandas as pd


def max_drawdown(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return 0.0
    running_max = clean.cummax()
    drawdown = clean / running_max - 1
    return float(drawdown.min())


def current_drawdown(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return 0.0
    return float(clean.iloc[-1] / clean.cummax().iloc[-1] - 1)


def annualized_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    clean = equity.dropna()
    if len(clean) < 2:
        return 0.0
    total_return = clean.iloc[-1] / clean.iloc[0] - 1
    years = max((len(clean) - 1) / periods_per_year, 1 / periods_per_year)
    return float((1 + total_return) ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    return float(clean.std(ddof=0) * math.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.015) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    excess_daily = clean - risk_free_rate / 252
    vol = clean.std(ddof=0)
    if vol == 0:
        return 0.0
    return float(excess_daily.mean() / vol * math.sqrt(252))


def summarize_equity(equity: pd.Series) -> dict[str, float]:
    clean = equity.dropna()
    returns = clean.pct_change().dropna()
    total_return = float(clean.iloc[-1] / clean.iloc[0] - 1) if len(clean) > 1 else 0.0
    mdd = max_drawdown(clean)
    ann_return = annualized_return(clean)
    ann_vol = annualized_volatility(returns)
    sharpe = sharpe_ratio(returns)
    win_rate = float((returns > 0).mean()) if not returns.empty else 0.0
    return {
        "total_return": total_return,
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "max_drawdown": mdd,
        "current_drawdown": current_drawdown(clean),
        "sharpe": sharpe,
        "win_rate": win_rate,
    }


def classify_market_state(prices: pd.DataFrame, benchmark: str = "510300.SH") -> dict[str, str | float]:
    series = prices[benchmark].dropna()
    if len(series) < 130:
        return {
            "state": "数据不足",
            "tone": "neutral",
            "score": 0.0,
            "explanation": "示例数据还不够长，先观察，不急着交易。",
        }

    ma20 = series.rolling(20).mean().iloc[-1]
    ma120 = series.rolling(120).mean().iloc[-1]
    drawdown = current_drawdown(series)
    ret20 = series.iloc[-1] / series.iloc[-20] - 1

    score = 0
    score += 1 if ma20 > ma120 else -1
    score += 1 if ret20 > 0 else -1
    score += 1 if drawdown > -0.08 else -1

    if score >= 2:
        state = "偏强"
        tone = "good"
        explanation = "大盘趋势和短期表现都较健康，系统允许适度承担权益风险。"
    elif score <= -2:
        state = "偏弱"
        tone = "bad"
        explanation = "大盘趋势偏弱或回撤较深，系统会更偏向防守仓位。"
    else:
        state = "震荡"
        tone = "neutral"
        explanation = "市场信号不够一致，系统会控制仓位，避免频繁追涨杀跌。"

    return {
        "state": state,
        "tone": tone,
        "score": float(score),
        "drawdown": float(drawdown),
        "ret20": float(ret20),
        "explanation": explanation,
    }


def risk_sentence(metrics: dict[str, float]) -> str:
    mdd = metrics.get("max_drawdown", 0.0)
    vol = metrics.get("annual_volatility", 0.0)
    if mdd <= -0.25:
        return "历史最大回撤较深，新手不适合直接重仓跟随。"
    if mdd <= -0.15:
        return "历史回撤中等偏高，建议先用模拟账户观察。"
    if vol <= 0.08:
        return "历史波动较低，但仍然可能出现连续小亏。"
    return "历史风险处在可观察区间，实盘前仍要先理解亏损情景。"

