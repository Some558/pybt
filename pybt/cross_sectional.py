"""クロスセクション(ロングショート)ファクター検証。

単一銘柄タイミング戦略が軒並みノーエッジだったため、戦略の"種類"を変える:
毎期、全銘柄をシグナルでランク付けし 上位 Q をロング・下位 Q をショート
(等ウェイト・市場中立)。市場全体の β を相殺して相対的エッジを抽出する。

ルックアヘッド厳禁:
  - ウェイトは t 時点で過去データのみから決定
  - リターンは t→t+1 の forward return に適用 (fwd = pct_change().shift(-1))

データは pybt.data の日足キャッシュを再利用 (MA25 全プライム検証で取得済み)。
コストはターンオーバー × 片道 bp で控除。

注意(Codex 監査): universe は現プライム構成 = 生存バイアスあり。
公開時は「2026/6 時点構成の過去検証」と明記すること。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D


def build_panel(tickers: list[dict], years: int = 10, min_obs: int = 250) -> pd.DataFrame:
    """銘柄リストから日足終値パネル (date × ticker) を構築 (キャッシュ利用)。"""
    cols = {}
    for t in tickers:
        sym = t["ticker"]
        try:
            df = D.load_daily(sym, years=years)
            if len(df) >= min_obs:
                s = df["Close"].astype(float)
                s = s.where(s > 0)  # 非正価格(データ異常)は NaN 化
                cols[sym] = s
        except Exception:  # noqa: BLE001
            continue
    panel = pd.DataFrame(cols).sort_index()
    return panel


# ----------------------------------------------------------------- シグナル
# いずれも「高いほどロング」になる向きで返す (月次パネル前提)

def sig_momentum(monthly: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    """モメンタム(lookback-skip ヶ月)。直近 skip ヶ月を除く過去 lookback ヶ月リターン。"""
    return monthly.shift(skip) / monthly.shift(lookback) - 1.0


def sig_reversal(monthly: pd.DataFrame, lookback: int = 1) -> pd.DataFrame:
    """短期リバーサル。直近 lookback ヶ月リターンの符号反転(下げた銘柄を買う)。"""
    return -(monthly / monthly.shift(lookback) - 1.0)


def sig_lowvol(monthly: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """低ボラ。過去 lookback ヶ月の月次リターン標準偏差の符号反転。"""
    rets = monthly.pct_change()
    return -rets.rolling(lookback).std()


SIGNALS = {
    "momentum": sig_momentum,
    "reversal": sig_reversal,
    "lowvol": sig_lowvol,
}


# ----------------------------------------------------------------- バックテスト

def _metrics(rets: pd.Series, periods_per_year: int = 12) -> dict:
    rets = rets.dropna()
    if len(rets) < 6:
        return {"ann_ret": float("nan"), "ann_vol": float("nan"),
                "sharpe": float("nan"), "maxdd": float("nan"), "n": len(rets)}
    ann_ret = rets.mean() * periods_per_year
    ann_vol = rets.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol else float("nan")
    eq = (1 + rets).cumprod()
    maxdd = (eq / eq.cummax() - 1).min()
    return {"ann_ret": round(ann_ret * 100, 2), "ann_vol": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 2), "maxdd": round(maxdd * 100, 2), "n": len(rets)}


def run_cross_sectional(
    panel: pd.DataFrame, signal: str, quantile: float = 0.2,
    cost_bps: float = 10.0, min_names: int = 20,
    sector_neutral: bool = False, sectors: dict | None = None,
    return_series: bool = False, **sig_kwargs,
) -> dict:
    """月次リバランスのロングショート検証。

    quantile       : 上位/下位の比率 (0.2 = 上位20%ロング・下位20%ショート)
    cost_bps       : 片道コスト(bp)。ターンオーバーに掛けて控除
    sector_neutral : True で各期セクター内デミーン (セクター傾斜を除去)
    sectors        : {ticker: sector} (sector_neutral 時に必須)
    """
    monthly = panel.resample("ME").last()
    sig = SIGNALS[signal](monthly, **sig_kwargs)
    if sector_neutral and sectors:
        sec = pd.Series(sectors).reindex(monthly.columns)
        sig = sig - sig.T.groupby(sec).transform("mean").T
    fwd = monthly.pct_change().shift(-1)  # t→t+1 の forward return (ルックアヘッド回避)
    fwd = fwd.replace([np.inf, -np.inf], np.nan)
    # 各期クロスセクションで 1/99% ウィンソライズ (未調整分割等の異常値が
    # ショート側で LS を破壊するのを防ぐ・ファクター研究の標準処理)
    lo = fwd.quantile(0.01, axis=1)
    hi = fwd.quantile(0.99, axis=1)
    fwd = fwd.clip(lower=lo, upper=hi, axis=0)

    dates = monthly.index
    ls_rets, long_rets, short_rets, mkt_rets = [], [], [], []
    prev_w = pd.Series(dtype=float)
    turnover_sum = 0.0
    idx_used = []

    for t in dates:
        s = sig.loc[t].dropna()
        f = fwd.loc[t]
        valid = s.index.intersection(f.dropna().index)
        s = s.loc[valid]
        if len(s) < min_names:
            continue
        n_side = max(1, int(len(s) * quantile))
        ranked = s.sort_values()
        shorts = ranked.index[:n_side]
        longs = ranked.index[-n_side:]

        w = pd.Series(0.0, index=s.index)
        w.loc[longs] = 1.0 / n_side
        w.loc[shorts] = -1.0 / n_side

        # ターンオーバー(前期ウェイトとの差・両側)
        all_names = w.index.union(prev_w.index)
        turn = (w.reindex(all_names, fill_value=0)
                - prev_w.reindex(all_names, fill_value=0)).abs().sum()
        turnover_sum += turn
        cost = turn * cost_bps / 10000.0
        prev_w = w

        fwd_t = f.reindex(s.index).fillna(0.0)
        ls = float((w * fwd_t).sum()) - cost
        long_r = float(fwd_t.loc[longs].mean())
        short_r = float(fwd_t.loc[shorts].mean())
        ls_rets.append(ls)
        long_rets.append(long_r)
        short_rets.append(short_r)
        mkt_rets.append(float(f.reindex(s.index).dropna().mean()))
        idx_used.append(t)

    ls = pd.Series(ls_rets, index=idx_used)
    half = len(ls) // 2
    out = {
        "signal": signal,
        "longshort": _metrics(ls),
        "long_only": _metrics(pd.Series(long_rets, index=idx_used)),
        "short_only": _metrics(pd.Series(short_rets, index=idx_used)),
        "market_ew": _metrics(pd.Series(mkt_rets, index=idx_used)),
        "oos_1st": _metrics(ls.iloc[:half]),
        "oos_2nd": _metrics(ls.iloc[half:]),
        "avg_turnover": round(turnover_sum / max(1, len(idx_used)), 2),
        "rebalances": len(idx_used),
    }
    if return_series:
        out["ls_series"] = ls
    return out
