"""データ取得層 — backtesting.py が要求する OHLCV 形式に正規化して返す。

返却 DataFrame の規約:
  - index : tz-aware DatetimeIndex (UTC)
  - columns : Open, High, Low, Close, Volume (先頭大文字)

取得結果は pickle でローカルキャッシュし、再取得を避ける
(Dukascopy の intraday は取得が重いため)。
"""
from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import pandas as pd

_CACHE = Path(__file__).parent / "_cache"
_CACHE.mkdir(exist_ok=True)

# backtesting.py が要求する列
_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return _CACHE / f"{h}.pkl"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """列名を Capitalize し、UTC tz-aware にそろえる。"""
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    # 欠けている Volume を 0 で補完 (一部指数で volume なし)
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df[[c for c in _OHLCV if c in df.columns]].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df.sort_index()


# ---------------------------------------------------------------- 日足 (yfinance)

def load_daily(symbol: str, years: int = 10, use_cache: bool = True) -> pd.DataFrame:
    """yfinance から日足を取得 (^N225, AAPL, 7203.T など)。"""
    key = f"daily::{symbol}::{years}"
    cp = _cache_path(key)
    if use_cache and cp.exists():
        return pd.read_pickle(cp)

    import yfinance as yf

    raw = yf.download(
        symbol, period=f"{years}y", interval="1d",
        auto_adjust=True, progress=False, multi_level_index=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance: {symbol} のデータが空です")
    df = _normalize(raw)
    df.to_pickle(cp)
    return df


# ------------------------------------------------------------ intraday (Dukascopy)

_INTERVAL_MAP = {
    "1m": "INTERVAL_MIN_1", "5m": "INTERVAL_MIN_5", "10m": "INTERVAL_MIN_10",
    "15m": "INTERVAL_MIN_15", "30m": "INTERVAL_MIN_30",
    "1h": "INTERVAL_HOUR_1", "4h": "INTERVAL_HOUR_4", "1d": "INTERVAL_DAY_1",
}


def load_intraday(
    instrument: str,
    tf: str,
    start: dt.datetime,
    end: dt.datetime,
    offer: str = "bid",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Dukascopy から intraday OHLCV を取得。

    instrument : dukascopy_python.instruments の定数名
                 (例 "INSTRUMENT_IDX_AMERICA_E_NQ_100" = NAS100)
    tf         : "1m"/"5m"/"15m"/"1h" など (_INTERVAL_MAP のキー)
    """
    key = f"intra::{instrument}::{tf}::{start:%Y%m%d}::{end:%Y%m%d}::{offer}"
    cp = _cache_path(key)
    if use_cache and cp.exists():
        return pd.read_pickle(cp)

    import dukascopy_python as dk
    from dukascopy_python import instruments as inst

    if tf not in _INTERVAL_MAP:
        raise ValueError(f"未対応 tf: {tf} (対応: {list(_INTERVAL_MAP)})")
    interval = getattr(dk, _INTERVAL_MAP[tf])
    offer_side = dk.OFFER_SIDE_BID if offer == "bid" else dk.OFFER_SIDE_ASK
    instr = getattr(inst, instrument)

    raw = dk.fetch(instr, interval, offer_side, start, end)
    if raw is None or raw.empty:
        raise RuntimeError(f"dukascopy: {instrument} {tf} のデータが空です")
    df = _normalize(raw)
    df.to_pickle(cp)
    return df


# よく使う instrument のショートハンド (実名を覚えなくて済むように)
INSTRUMENTS = {
    "NAS100": "INSTRUMENT_IDX_AMERICA_E_NQ_100",
    "SP500": "INSTRUMENT_IDX_AMERICA_E_SANDP_500",
    "N225": "INSTRUMENT_IDX_ASIA_E_N225JAP",
    "EURUSD": "INSTRUMENT_FX_MAJORS_EUR_USD",
}


def resolve_instrument(name: str) -> str:
    """"NAS100" のような短縮名を Dukascopy 定数名に解決。未知ならそのまま返す。"""
    return INSTRUMENTS.get(name.upper(), name)
