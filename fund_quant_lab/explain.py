from __future__ import annotations

import pandas as pd


GLOSSARY = {
    "目标仓位": "系统建议这只基金占总资金的比例。比如 20% 表示每 10 万元里配置 2 万元。",
    "动量": "最近一段时间表现更强的资产，未来短期可能继续强，但不是保证。",
    "回撤": "从阶段高点跌下来的幅度。回撤越大，心理压力通常越大。",
    "波动率": "价格上下波动的剧烈程度。波动高不等于一定赚钱，往往也意味着亏损更快。",
    "夏普比率": "衡量每承担一份波动，换来多少收益。它适合比较策略，但不能预测未来。",
    "换手率": "调仓时买卖变化的比例。换手越高，手续费和执行误差越值得注意。",
    "防守资产": "通常指债券、货币、黄金等用于降低组合波动的资产。",
}


def plain_signal_summary(market_state: dict[str, str | float], target_weights: pd.Series) -> str:
    risky_weight = float(
        target_weights.drop(labels=["511010.SH", "511880.SH", "518880.SH"], errors="ignore").sum()
    )
    defensive_weight = float(target_weights.sum() - risky_weight)

    if market_state["state"] == "偏弱":
        return f"今天系统偏谨慎，权益仓位约 {risky_weight:.0%}，防守仓位约 {defensive_weight:.0%}。重点是先保护本金。"
    if market_state["state"] == "震荡":
        return f"今天市场不够干脆，权益仓位约 {risky_weight:.0%}。系统会参与，但不把仓位打满。"
    return f"今天市场状态偏强，权益仓位约 {risky_weight:.0%}。仍保留少量现金，避免追得太满。"


def beginner_trade_checklist() -> list[str]:
    return [
        "先确认这只是模拟信号，不需要马上实盘照抄。",
        "如果连续亏损，先降低仓位，不要急着加码摊平。",
        "每次只看一个主要策略，避免多个策略互相打架。",
        "不要因为一天涨跌改变系统规则，至少用周或月来复盘。",
        "实盘前先记录你能接受的最大亏损金额。",
    ]


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def format_money(value: float) -> str:
    return f"¥{value:,.0f}"

