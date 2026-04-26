from __future__ import annotations

import pandas as pd


class DataSourceError(RuntimeError):
    pass


def load_akshare_fund_daily(codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Reserved real-data adapter.

    The app uses sample data by default. Install requirements-real-data.txt before
    using this function, then add source-specific cleaning and caching.
    """
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise DataSourceError("AkShare is not installed. Run: pip install -r requirements-real-data.txt") from exc

    frames = []
    for code in codes:
        symbol = code.split(".")[0]
        try:
            raw = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start_date, end_date=end_date)
        except Exception as exc:  # pragma: no cover - depends on remote source
            raise DataSourceError(f"Failed to fetch {code} from AkShare: {exc}") from exc

        if raw.empty:
            continue
        date_col = "日期"
        close_col = "收盘"
        frame = raw[[date_col, close_col]].rename(columns={date_col: "date", close_col: code})
        frame["date"] = pd.to_datetime(frame["date"])
        frames.append(frame.set_index("date"))

    if not frames:
        raise DataSourceError("No price data returned from AkShare.")

    return pd.concat(frames, axis=1).sort_index().ffill().dropna(how="all")

