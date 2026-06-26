"""ペアトレード(距離法/コインテグレーション近似) — 唯一の未検証『市場中立』系。

経済的に関連する2資産のスプレッドの平均回帰を取る。ペアは事後スキャンでなく
ex-ante に経済合理性で選ぶ(全組合せから best を選ぶ=データスヌーピングを回避)。

ルックアヘッド規律:
  - ローリング・ヘッジ比 β = cov/var は過去窓のみ (shift(1))
  - スプレッドの mean/std も過去窓のみ (shift(1))
  - z は当日終値で算出するが、約定は翌日 (held = pos.shift(2)=close[t]決定→翌日約定)

統計ライブラリ非依存 (numpy/pandas のみ)。ADF コインテグレーション検定は statsmodels
不在のため省き、ローリング・ヘッジ比 + z-score 平均回帰 (実務の標準実装) で代替する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from .cross_sectional import _metrics

# ex-ante に経済合理性で選んだペア (スキャンで best を拾わない=スヌーピング回避)
DEFAULT_PAIRS = [
    ("EWA", "EWC"),   # 豪 / 加: 資源国 ETF (教科書的ペア)
    ("GLD", "GDX"),   # 金 / 金鉱株
    ("XLE", "XOP"),   # エネルギーセクター / 石油探査
    ("KO", "PEP"),    # コカ・コーラ / ペプシ
    ("XLK", "XLY"),   # テック / 一般消費 (相関高セクター)
    ("EWA", "EWZ"),   # 豪 / ブラジル (資源国)
]


def run_pair(s1: str, s2: str, df1: pd.DataFrame, df2: pd.DataFrame,
             window: int = 60, z_entry: float = 2.0, z_exit: float = 0.5,
             z_stop: float = 4.0, cost_bps: float = 5.0,
             return_series: bool = False) -> dict:
    """1 ペアの z-score 平均回帰戦略を検証。"""
    px = pd.concat([df1["Close"].astype(float).rename("a"),
                    df2["Close"].astype(float).rename("b")], axis=1).dropna()
    if len(px) < window * 3:
        return {"pair": f"{s1}/{s2}", "sharpe": float("nan"), "n_trades": 0,
                "ann_ret": float("nan"), "maxdd": float("nan"), "n": len(px)}
    lp = np.log(px)
    r = px.pct_change()

    # ローリング・ヘッジ比 β=cov/var・スプレッド統計 (すべて過去窓のみ=shift(1))
    cov = lp["a"].rolling(window).cov(lp["b"])
    var = lp["b"].rolling(window).var()
    beta = (cov / var).shift(1)
    spread = lp["a"] - beta * lp["b"]
    m = spread.rolling(window).mean().shift(1)
    sd = spread.rolling(window).std().shift(1)
    z = (spread - m) / sd

    spread_ret = r["a"] - beta * r["b"]   # β は過去ベース
    gross = 1.0 + beta.abs()              # 2レッグの総建玉 (正規化用)

    # ステートフルにポジションを決める (z は当日終値=既知・約定は翌日)
    zv = z.to_numpy()
    pos = 0
    positions = np.zeros(len(px))
    for t in range(len(px)):
        zt = zv[t]
        if not np.isfinite(zt):
            positions[t] = 0
            pos = 0
            continue
        if pos == 0:
            if zt > z_entry:
                pos = -1                  # スプレッド高→ショート(縮小に賭け)
            elif zt < -z_entry:
                pos = 1                   # スプレッド低→ロング
        else:
            reverted = abs(zt) < z_exit
            blewout = abs(zt) > z_stop
            flipped = (pos == 1 and zt > 0) or (pos == -1 and zt < 0)
            if reverted or blewout or flipped:
                pos = 0
        positions[t] = pos
    pos_s = pd.Series(positions, index=px.index)

    # 約定ラグ: z は close[t] で決定 → 翌日 close[t+1] でエントリー → 以降を保有。
    # held[t] = 2日前(close[t-2])に決めたポジション = pos.shift(2)。これで「終値tを見て
    # 終値tで約定」の同一終値約定を排除 (Codex Critical)。
    held = pos_s.shift(2).fillna(0.0)
    pnl = held * (spread_ret / gross)
    # コストは実レッグ・ウェイトの日次変化から算出 (建玉の入替 + 日々の β/gross
    # ドリフトに伴うリバランス売買を両方計上=保守的。Codex High)。
    w_a = held / gross
    w_b = -held * beta / gross
    turn = (w_a.diff().abs() + w_b.diff().abs()).fillna(w_a.abs() + w_b.abs())
    strat = (pnl - turn * cost_bps / 10000.0).dropna()

    mm = _metrics(strat, periods_per_year=252)
    half = len(strat) // 2
    n_trades = int((held.diff().abs() > 0).sum())          # 実約定回数(執行ベース)
    out = {"pair": f"{s1}/{s2}", **mm, "n_trades": n_trades,
           "exposure": round(float((held != 0).mean()) * 100, 1),
           "oos1_sharpe": _metrics(strat.iloc[:half], 252)["sharpe"],
           "oos2_sharpe": _metrics(strat.iloc[half:], 252)["sharpe"]}
    if return_series:
        out["series"] = strat
    return out


def load_pairs(pairs: list[tuple[str, str]], years: int = 20) -> dict:
    """ペアに必要な銘柄の日足を取得して辞書で返す。"""
    syms = sorted({s for p in pairs for s in p})
    data: dict[str, pd.DataFrame] = {}
    for s in syms:
        try:
            data[s] = D.load_daily(s, years=years)
        except Exception:  # noqa: BLE001
            continue
    return data
