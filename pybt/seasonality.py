"""カレンダー/季節性アノマリーの検証 (TOM / Sell-in-May / オーバーナイト効果)。

リサーチ(2026-06-23)で「証拠グレードA・当 harness で安く測れる」と判定した
3つの通説を検証する。いずれも本質は「いつ株を買い持ちするか」のタイミングで、
死因は β交絡(ただ長く持ってるだけ) / コスト / データスヌーピング / 近年の減衰。
従って各テストで必ず:
  - buy&hold(常時ロング) との比較 (β交絡の切り分け)
  - 効果日 vs 非効果日/季節 のHAC(Newey-West) t値 (有意性)
  - コスト感度 (特にオーバーナイトは round-trip/日でコストが死因)
  - 前後半OOS (近年の減衰チェック)
を出す。

ルックアヘッド: カレンダーは事前確定のため、ポジションは前日終値時点で確定でき
未来情報を使わない (pos[t] は t-1 までで決まる)。実装も pos[t]*ret[t] とする。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from .cross_sectional import _metrics

# 流動性の高い指数 + 一部 ETF (オーバーナイトは指数 Open が同期ズレしうるため
# 取引可能 ETF も併用して現実性を確認する)。
DEFAULT_SYMBOLS = ["^GSPC", "^N225", "^GDAXI", "^FTSE", "^STOXX50E",
                   "SPY", "EWJ", "1306.T"]  # ETF=取引可能 Open でオーバーナイト現実性確認


# ----------------------------------------------------------------- 統計基盤

def _ols_hac(y: np.ndarray, X: np.ndarray, lags: int | None = None):
    """OLS + Newey-West(HAC) 標準誤差。coef, se, lags を返す。"""
    n = len(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    XtX_inv = np.linalg.inv(X.T @ X)
    if lags is None:
        lags = max(1, int(4 * (n / 100.0) ** (2.0 / 9.0)))
    u = X * resid[:, None]
    S = u.T @ u
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = u[lag:].T @ u[:-lag]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return coef, se, lags


def _dummy_test(ret: pd.Series, mask: pd.Series) -> dict:
    """日次リターンを「効果日ダミー」に回帰し、超過分の年率と HAC t値を返す。

    ret = a + b*dummy + e。a=非効果日の平均・b=効果日の超過。t(b) で有意性判定。
    """
    df = pd.concat([ret.rename("y"), mask.rename("d").astype(float)], axis=1).dropna()
    y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(y)), df["d"].to_numpy()])
    coef, se, lags = _ols_hac(y, X)
    t_b = coef[1] / se[1] if se[1] else float("nan")
    in_mean = df.loc[df["d"] == 1, "y"].mean()
    out_mean = df.loc[df["d"] == 0, "y"].mean()
    return {
        "in_ann": round(in_mean * 252 * 100, 2),    # 効果日 平均×252 (年率換算の目安)
        "out_ann": round(out_mean * 252 * 100, 2),   # 非効果日
        "excess_ann": round(coef[1] * 252 * 100, 2),  # 超過(日次差×252)
        "t_hac": round(t_b, 2),
        "hac_lags": lags,
        "n_in": int((df["d"] == 1).sum()),
        "n_days": len(y),
    }


def _strategy(ret: pd.Series, pos: pd.Series, cost_bps: float) -> dict:
    """pos∈{0,1} の日次タイミング戦略のメトリクス。pos[t] は t-1 までで確定。

    日次戦略リターン = pos[t]*ret[t] - コスト*|pos[t]-pos[t-1]|。
    """
    pos = pos.reindex(ret.index).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    strat = pos * ret - turn * cost_bps / 10000.0
    m = _metrics(strat.dropna(), periods_per_year=252)
    m["exposure"] = round(float(pos.mean()) * 100, 1)  # 市場にいた時間割合%
    m["roundtrips_yr"] = round(float(turn.sum()) / 2 / (len(ret) / 252), 1)
    # β交絡の切り分け: 戦略を buy&hold に回帰し 対市場 alpha(年率)+HAC t を出す
    # (Sharpe 比較はエクスポージャー差を含むため β 切り分けにならない=Codex High)
    df = pd.concat([strat.rename("s"), ret.rename("m")], axis=1).dropna()
    coef, se, _ = _ols_hac(df["s"].to_numpy(),
                           np.column_stack([np.ones(len(df)), df["m"].to_numpy()]))
    m["alpha_ann"] = round(coef[0] * 252 * 100, 2)
    m["t_alpha"] = round(coef[0] / se[0], 2) if se[0] else float("nan")
    m["beta"] = round(coef[1], 2)
    return m


def _trim_partial_months(s: pd.Series) -> pd.Series:
    """先頭・末尾の不完全月を除外 (末尾は『最後に取得できた日=月末』の誤認を防ぐ)。"""
    per = s.index.to_period("M")
    return s[(per != per[0]) & (per != per[-1])]


def _paired_diff_hac(a: pd.Series, b: pd.Series) -> dict:
    """a-b のペア差の平均が 0 と有意に違うか (HAC)。年率(×252)と t値。"""
    d = (a - b).dropna()
    coef, se, lags = _ols_hac(d.to_numpy(), np.ones((len(d), 1)))
    return {"diff_ann": round(coef[0] * 252 * 100, 2),
            "t_diff": round(coef[0] / se[0], 2) if se[0] else float("nan"),
            "hac_lags": lags, "n": len(d)}


# ----------------------------------------------------------------- 効果別テスト

def test_tom(df: pd.DataFrame, before: int = 1, after: int = 3,
             cost_bps: float = 2.0) -> dict:
    """月末ターン(TOM): 月末 before 日 + 月初 after 日に超過リターンが集中するか。"""
    close = df["Close"].astype(float)
    ret = close.pct_change()
    idx = close.index
    # 各日の「月内 何営業日目か」「月末から何営業日前か」
    g = pd.Series(idx, index=idx).groupby([idx.year, idx.month])
    dom_from_start = g.cumcount()                    # 0=月初1日目
    dom_from_end = g.transform("size") - 1 - dom_from_start  # 0=月末最終日
    is_tom = (dom_from_start < after) | (dom_from_end < before)

    # 先頭・末尾の不完全月を除外 (末尾は size 逆算で月末を誤認する=Codex High)
    ret = _trim_partial_months(ret)
    is_tom = is_tom.reindex(ret.index)

    eff = _dummy_test(ret, is_tom)
    strat = _strategy(ret, is_tom.astype(float), cost_bps)
    bh = _metrics(ret.dropna(), periods_per_year=252)
    half = len(ret) // 2
    oos1 = _dummy_test(ret.iloc[:half], is_tom.iloc[:half])
    oos2 = _dummy_test(ret.iloc[half:], is_tom.iloc[half:])
    return {"effect": eff, "strategy": strat, "buyhold": bh,
            "oos1_t": oos1["t_hac"], "oos2_t": oos2["t_hac"],
            "oos1_excess": oos1["excess_ann"], "oos2_excess": oos2["excess_ann"]}


def test_sell_in_may(df: pd.DataFrame, cost_bps: float = 2.0) -> dict:
    """Sell-in-May: 冬(11-4月)が夏(5-10月)を上回るか。冬ロング戦略 vs buy&hold。"""
    close = df["Close"].astype(float)
    ret = close.pct_change()
    month = pd.Series(close.index.month, index=close.index)
    is_winter = month.isin([11, 12, 1, 2, 3, 4])

    eff = _dummy_test(ret, is_winter)
    strat = _strategy(ret, is_winter.astype(float), cost_bps)  # 冬ロング・夏フラット
    bh = _metrics(ret.dropna(), periods_per_year=252)
    half = len(ret) // 2
    oos1 = _dummy_test(ret.iloc[:half], is_winter.iloc[:half])
    oos2 = _dummy_test(ret.iloc[half:], is_winter.iloc[half:])
    return {"effect": eff, "strategy": strat, "buyhold": bh,
            "oos1_t": oos1["t_hac"], "oos2_t": oos2["t_hac"],
            "oos1_excess": oos1["excess_ann"], "oos2_excess": oos2["excess_ann"]}


def test_overnight(df: pd.DataFrame, cost_sweep=(0.0, 2.0, 5.0, 10.0, 20.0)) -> dict:
    """オーバーナイト効果: 引け→翌寄り のリターン vs 寄り→引け(日中)。

    コストが死因のため round-trip/日 のコスト感度を主軸に。1引け→翌寄りで1往復。
    """
    o = df["Open"].astype(float)
    c = df["Close"].astype(float)
    overnight = (o / c.shift(1) - 1.0).dropna()       # close[t-1] → open[t]
    intraday = (c / o - 1.0).dropna()                 # open[t] → close[t]
    full = c.pct_change().dropna()

    # データ品質: Open が壊れている指数 (^FTSE 等) は overnight がほぼ 0 に張り付く。
    # ゼロ近傍率が高いものは「Open 不良」として不採用フラグを立てる (Codex High)。
    zero_frac = float((overnight.abs() < 1e-6).mean())
    reliable = zero_frac < 0.10

    # 通説は「overnight > 0」でなく「overnight > intraday」。ペア差を HAC 検定する
    # (overnight≠0 だけだと株式の上昇ドリフトを夜間計上しているだけでも有意化する)。
    diff = _paired_diff_hac(overnight, intraday)
    # 参考: overnight 平均が 0 と有意に違うか
    coef, se, lags = _ols_hac(overnight.to_numpy(), np.ones((len(overnight), 1)))
    t_on = coef[0] / se[0] if se[0] else float("nan")

    on_m = _metrics(overnight, periods_per_year=252)
    id_m = _metrics(intraday, periods_per_year=252)
    bh_m = _metrics(full, periods_per_year=252)

    # コスト感度: 毎日 引けで買い→寄りで売り = 2レッグ/日。片道 bps の2倍を控除
    # (_strategy の turn=2 と整合させる=片道コスト基準)。
    cost_rows = []
    for bps in cost_sweep:
        net = overnight - 2.0 * bps / 10000.0
        m = _metrics(net, periods_per_year=252)
        cost_rows.append((bps, m))

    half = len(overnight) // 2
    return {
        "overnight": on_m, "intraday": id_m, "buyhold": bh_m,
        "t_overnight": round(t_on, 2), "hac_lags": lags,
        "diff_ann": diff["diff_ann"], "t_diff": diff["t_diff"],  # 主役: overnight-intraday
        "zero_frac": round(zero_frac * 100, 1), "reliable": reliable,
        "cost": cost_rows,
        "oos1_ann": _metrics(overnight.iloc[:half], 252)["ann_ret"],
        "oos2_ann": _metrics(overnight.iloc[half:], 252)["ann_ret"],
        "n_days": len(overnight),
    }


def load_universe(symbols: list[str], years: int = 25, min_obs: int = 1000) -> dict[str, pd.DataFrame]:
    """検証対象の日足を辞書で返す (取得失敗・短すぎは除外)。"""
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            df = D.load_daily(s, years=years)
            if len(df) >= min_obs:
                out[s] = df
        except Exception:  # noqa: BLE001
            continue
    return out
