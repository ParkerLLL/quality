from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FundInfo:
    code: str
    name: str
    category: str
    beginner_note: str
    risk_level: str
    defensive: bool = False


ETF_UNIVERSE: tuple[FundInfo, ...] = (
    FundInfo(
        code="510300.SH",
        name="沪深300ETF",
        category="宽基权益",
        beginner_note="代表A股大盘蓝筹，适合作为权益市场温度计。",
        risk_level="中高",
    ),
    FundInfo(
        code="510500.SH",
        name="中证500ETF",
        category="宽基权益",
        beginner_note="偏中盘成长，波动通常比沪深300更大。",
        risk_level="高",
    ),
    FundInfo(
        code="159915.SZ",
        name="创业板ETF",
        category="成长权益",
        beginner_note="成长风格更强，涨跌弹性都更明显。",
        risk_level="高",
    ),
    FundInfo(
        code="512880.SH",
        name="证券ETF",
        category="行业主题",
        beginner_note="券商板块波动较大，容易出现快速轮动。",
        risk_level="高",
    ),
    FundInfo(
        code="512800.SH",
        name="银行ETF",
        category="行业主题",
        beginner_note="偏价值和分红风格，走势常和成长板块不同步。",
        risk_level="中高",
    ),
    FundInfo(
        code="518880.SH",
        name="黄金ETF",
        category="商品避险",
        beginner_note="常被用作组合防守资产，但也会独立波动。",
        risk_level="中",
        defensive=True,
    ),
    FundInfo(
        code="511010.SH",
        name="国债ETF",
        category="债券防守",
        beginner_note="波动较低，常用于降低组合回撤。",
        risk_level="低",
        defensive=True,
    ),
    FundInfo(
        code="511880.SH",
        name="货币ETF",
        category="现金管理",
        beginner_note="接近现金仓位，用来等待机会和降低波动。",
        risk_level="低",
        defensive=True,
    ),
)


FUND_BY_CODE = {fund.code: fund for fund in ETF_UNIVERSE}


DEFAULT_INITIAL_CAPITAL = 100_000
DEFAULT_LOOKBACK_DAYS = 60
DEFAULT_REBALANCE = "M"
DEFAULT_TOP_N = 3
DEFAULT_MAX_WEIGHT = 0.35
DEFAULT_CASH_BUFFER = 0.05
DEFAULT_TRADING_FEE = 0.0003

