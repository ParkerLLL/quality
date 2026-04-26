from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from fund_quant_lab.config import (
    DEFAULT_CASH_BUFFER,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_REBALANCE,
    DEFAULT_TOP_N,
    DEFAULT_TRADING_FEE,
    ETF_UNIVERSE,
)
from fund_quant_lab.explain import (
    GLOSSARY,
    beginner_trade_checklist,
    format_money,
    format_pct,
    plain_signal_summary,
)
from fund_quant_lab.risk import classify_market_state, risk_sentence, summarize_equity
from fund_quant_lab.sample_data import generate_sample_prices, latest_price_table
from fund_quant_lab.strategy import StrategySettings, backtest_rotation, build_signal_snapshot


st.set_page_config(page_title="Fund Quant Lab", page_icon="FQL", layout="wide")


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        border: 1px solid #e6e8ec;
        border-radius: 8px;
        padding: 12px 14px;
        background: #ffffff;
    }
    div[data-testid="stMetric"] label {font-size: 0.88rem;}
    .small-note {color: #5f6975; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_prices() -> pd.DataFrame:
    return generate_sample_prices()


def percent_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    formatted = df.copy()
    for column in columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2%}")
    return formatted


prices = load_prices()

with st.sidebar:
    st.header("模拟设置")
    initial_capital = st.number_input("模拟本金", min_value=10_000, max_value=5_000_000, value=DEFAULT_INITIAL_CAPITAL, step=10_000)
    rebalance_label = st.segmented_control("调仓频率", options=["每月", "每周"], default="每月")
    rebalance_frequency = "M" if rebalance_label == "每月" else "W"
    lookback_days = st.slider("观察周期", min_value=40, max_value=120, value=DEFAULT_LOOKBACK_DAYS, step=5)
    top_n = st.slider("最多持有几只进攻型ETF", min_value=1, max_value=5, value=DEFAULT_TOP_N)
    max_weight = st.slider("单只最高仓位", min_value=0.15, max_value=0.50, value=DEFAULT_MAX_WEIGHT, step=0.05)
    cash_buffer = st.slider("保留现金", min_value=0.00, max_value=0.30, value=DEFAULT_CASH_BUFFER, step=0.05)
    trading_fee = st.number_input("单边交易成本", min_value=0.0, max_value=0.01, value=DEFAULT_TRADING_FEE, step=0.0001, format="%.4f")

settings = StrategySettings(
    lookback_days=lookback_days,
    top_n=top_n,
    max_weight=max_weight,
    cash_buffer=cash_buffer,
    trading_fee=trading_fee,
    rebalance_frequency=rebalance_frequency,
)

signals, target_weights = build_signal_snapshot(prices, settings)
equity, weight_history = backtest_rotation(prices, float(initial_capital), settings)
metrics = summarize_equity(equity.set_index("date")["equity"])
market_state = classify_market_state(prices)

st.title("Fund Quant Lab")
st.caption("个人基金量化驾驶舱 · 示例数据 · 模拟账户 · 不自动下单")

tab_dashboard, tab_backtest, tab_pool, tab_learn, tab_data = st.tabs(
    ["今日驾驶舱", "策略回测", "基金池", "学习卡片", "数据设置"]
)

with tab_dashboard:
    left, mid, right = st.columns([1.1, 1.1, 1.3])
    left.metric("市场状态", str(market_state["state"]), help=str(market_state["explanation"]))
    mid.metric("模拟账户", format_money(float(equity["equity"].iloc[-1])), delta=format_pct(metrics["total_return"]))
    right.metric("最大回撤", format_pct(metrics["max_drawdown"]), help="历史上从高点到低点的最大跌幅。")

    st.subheader("今日建议")
    st.write(plain_signal_summary(market_state, target_weights))
    st.markdown(f'<p class="small-note">{market_state["explanation"]}</p>', unsafe_allow_html=True)

    plan = signals[signals["目标仓位"] > 0.01].sort_values("目标仓位", ascending=False)
    display_plan = percent_columns(plan[["代码", "名称", "类别", "目标仓位", "操作", "解释"]], ["目标仓位"])
    st.dataframe(display_plan, width="stretch", hide_index=True)

    st.subheader("交易前检查")
    st.write(" · ".join(beginner_trade_checklist()))

with tab_backtest:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("累计收益", format_pct(metrics["total_return"]))
    c2.metric("年化收益", format_pct(metrics["annual_return"]))
    c3.metric("年化波动", format_pct(metrics["annual_volatility"]))
    c4.metric("夏普比率", f"{metrics['sharpe']:.2f}")

    fig = px.line(equity, x="date", y="equity", title="模拟账户净值曲线")
    fig.update_layout(yaxis_title="账户资产", xaxis_title="", height=420, margin=dict(l=10, r=10, t=48, b=10))
    st.plotly_chart(fig, width="stretch")

    st.info(risk_sentence(metrics))

    if not weight_history.empty:
        recent_weights = weight_history.tail(6).copy()
        recent_weights["date"] = recent_weights["date"].dt.strftime("%Y-%m-%d")
        weight_columns = [fund.code for fund in ETF_UNIVERSE if fund.code in recent_weights.columns]
        renamed = recent_weights[["date", "turnover", *weight_columns]].rename(
            columns={"date": "日期", "turnover": "换手率", **{fund.code: fund.name for fund in ETF_UNIVERSE}}
        )
        pct_cols = [column for column in renamed.columns if column != "日期"]
        st.subheader("最近调仓记录")
        st.dataframe(percent_columns(renamed, pct_cols), width="stretch", hide_index=True)

with tab_pool:
    st.subheader("基金池状态")
    price_table = latest_price_table(prices)
    price_table["日涨跌"] = price_table["日涨跌"].map(lambda value: f"{value:.2%}")
    st.dataframe(price_table, width="stretch", hide_index=True)

    st.subheader("评分明细")
    score_table = signals[["代码", "名称", "类别", "60日表现", "20日波动", "60日回撤", "评分", "解释"]]
    st.dataframe(
        percent_columns(score_table, ["60日表现", "20日波动", "60日回撤"]),
        width="stretch",
        hide_index=True,
    )

with tab_learn:
    st.subheader("指标卡片")
    for term, explanation in GLOSSARY.items():
        with st.expander(term):
            st.write(explanation)

    st.subheader("新手规则")
    for item in beginner_trade_checklist():
        st.checkbox(item, value=False)

with tab_data:
    st.subheader("当前数据源")
    st.write("第一版使用内置示例行情，方便先学习完整流程。")
    st.write("准备接真实数据时，可以从 AkShare 开始；真实数据入口已经留在 `fund_quant_lab/data_sources.py`。")
    st.dataframe(prices.tail().reset_index(), width="stretch", hide_index=True)
