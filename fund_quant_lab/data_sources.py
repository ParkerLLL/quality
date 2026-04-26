from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd


class DataSourceError(RuntimeError):
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "akshare_etf_prices_hfq.csv"
DEFAULT_META_FILE = DEFAULT_CACHE_DIR / "akshare_etf_prices_hfq.meta.json"


@dataclass(frozen=True)
class PriceDataResult:
    prices: pd.DataFrame
    source: str
    price_adjustment: str
    is_cached: bool
    fetched_at: str | None
    latest_trading_day: str | None
    message: str


def load_akshare_cached_prices(
    codes: list[str],
    start_date: str | date,
    end_date: str | date,
    *,
    force_refresh: bool = False,
    adjust: str = "hfq",
    cache_file: Path = DEFAULT_CACHE_FILE,
    meta_file: Path = DEFAULT_META_FILE,
) -> PriceDataResult:
    """Load ETF prices from AkShare with a local CSV cache."""
    if cache_file.exists() and not force_refresh:
        prices = _read_cached_prices(cache_file, codes)
        requested_start = pd.to_datetime(_format_akshare_date(start_date), format="%Y%m%d")
        if prices.index.min() <= requested_start + pd.Timedelta(days=10):
            prices = prices.loc[prices.index >= requested_start]
            metadata = _read_metadata(meta_file)
            return PriceDataResult(
            prices=prices,
            source=metadata.get("source", "AkShare 缓存"),
            price_adjustment=metadata.get("price_adjustment", "未知"),
            is_cached=True,
            fetched_at=metadata.get("fetched_at"),
            latest_trading_day=_latest_trading_day(prices),
                message="已使用本地缓存。点击“刷新真实数据”可重新联网拉取。",
            )

    try:
        prices = load_akshare_fund_daily(codes, start_date, end_date, adjust=adjust)
    except DataSourceError:
        if cache_file.exists():
            prices = _read_cached_prices(cache_file, codes)
            metadata = _read_metadata(meta_file)
            return PriceDataResult(
            prices=prices,
            source=metadata.get("source", "AkShare 缓存"),
            price_adjustment=metadata.get("price_adjustment", "未知"),
            is_cached=True,
            fetched_at=metadata.get("fetched_at"),
            latest_trading_day=_latest_trading_day(prices),
                message="联网刷新失败，已自动使用上一次成功缓存。",
            )
        raise

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_file, index_label="date")
    fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
    provider = prices.attrs.get("provider", "AkShare / Eastmoney")
    price_adjustment = prices.attrs.get("price_adjustment", "后复权收盘价")
    metadata = {
        "source": provider,
        "adjust": adjust,
        "price_adjustment": price_adjustment,
        "fetched_at": fetched_at,
        "start_date": _format_akshare_date(start_date),
        "end_date": _format_akshare_date(end_date),
        "latest_trading_day": _latest_trading_day(prices),
        "rows": len(prices),
        "codes": codes,
    }
    meta_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return PriceDataResult(
        prices=prices,
        source=provider,
        price_adjustment=price_adjustment,
        is_cached=False,
        fetched_at=fetched_at,
        latest_trading_day=metadata["latest_trading_day"],
        message="已成功联网刷新真实行情。",
    )


def load_akshare_fund_daily(
    codes: list[str],
    start_date: str | date,
    end_date: str | date,
    *,
    adjust: str = "hfq",
) -> pd.DataFrame:
    """Fetch ETF daily close prices from AkShare.

    `adjust="hfq"` means back-adjusted prices, which are usually better for
    long-horizon signal research because distributions/splits are reflected.
    """
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise DataSourceError("AkShare is not installed. Run: pip install -r requirements-real-data.txt") from exc

    frames = []
    start = _format_akshare_date(start_date)
    end = _format_akshare_date(end_date)
    providers_used = set()
    for code in codes:
        symbol = code.split(".")[0]
        try:
            raw = ak.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
            providers_used.add("Eastmoney")
        except Exception as exc:  # pragma: no cover - depends on remote source
            try:
                raw = ak.fund_etf_hist_sina(symbol=_sina_symbol(code))
                providers_used.add("Sina")
            except Exception as fallback_exc:  # pragma: no cover - depends on remote source
                raise DataSourceError(
                    f"Failed to fetch {code} from AkShare. Eastmoney: {exc}; Sina: {fallback_exc}"
                ) from fallback_exc

        if raw.empty:
            raise DataSourceError(f"AkShare returned no data for {code}.")
        frame = _extract_close_frame(raw, code)
        start_ts = pd.to_datetime(start, format="%Y%m%d")
        end_ts = pd.to_datetime(end, format="%Y%m%d")
        frame = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)]
        frames.append(frame)

    prices = pd.concat(frames, axis=1).sort_index().ffill().dropna()
    missing_codes = [code for code in codes if code not in prices.columns]
    if missing_codes:
        raise DataSourceError(f"Missing price columns: {', '.join(missing_codes)}")
    if len(prices) < 160:
        raise DataSourceError("真实数据历史太短，暂时不足以计算当前策略。")
    if providers_used == {"Eastmoney"}:
        prices.attrs["provider"] = "AkShare / Eastmoney"
        prices.attrs["price_adjustment"] = "后复权收盘价"
    elif providers_used == {"Sina"}:
        prices.attrs["provider"] = "AkShare / Sina"
        prices.attrs["price_adjustment"] = "真实收盘价（未复权）"
    else:
        prices.attrs["provider"] = "AkShare / Eastmoney + Sina fallback"
        prices.attrs["price_adjustment"] = "混合口径：东方财富后复权 + 新浪未复权"
    return prices[codes]


def _extract_close_frame(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    date_col = _find_column(raw, ("日期", "date", "Date"))
    close_col = _find_column(raw, ("收盘", "close", "Close"))
    frame = raw[[date_col, close_col]].rename(columns={date_col: "date", close_col: code})
    frame["date"] = pd.to_datetime(frame["date"])
    frame[code] = pd.to_numeric(frame[code], errors="coerce")
    return frame.set_index("date").sort_index().dropna()


def _read_cached_prices(cache_file: Path, codes: list[str]) -> pd.DataFrame:
    prices = pd.read_csv(cache_file, parse_dates=["date"]).set_index("date").sort_index()
    missing_codes = [code for code in codes if code not in prices.columns]
    if missing_codes:
        raise DataSourceError(f"缓存缺少这些基金：{', '.join(missing_codes)}")
    return prices[codes].dropna()


def _read_metadata(meta_file: Path) -> dict[str, str]:
    if not meta_file.exists():
        return {}
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _find_column(raw: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in raw.columns:
            return candidate
    raise DataSourceError(f"AkShare 返回字段缺少 {candidates}，实际字段：{list(raw.columns)}")


def _format_akshare_date(value: str | date) -> str:
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    return value.replace("-", "")


def _latest_trading_day(prices: pd.DataFrame) -> str | None:
    if prices.empty:
        return None
    return prices.index[-1].strftime("%Y-%m-%d")


def _sina_symbol(code: str) -> str:
    symbol, market = code.split(".")
    prefix = "sh" if market.upper() == "SH" else "sz"
    return f"{prefix}{symbol}"
