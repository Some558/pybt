"""タイムシリーズ・モメンタム (トレンドフォロー / CTA) の多資産ポートフォリオ検証。

クロスセクション(相対ランキング)が潰れたため、機構の違う「絶対トレンド」を検証する:
各資産について、自分自身の過去 lookback ヶ月リターンの符号で方向を決め
(正=ロング / 負=ショート)、ex-ante ボラの逆数でサイズを正規化し、全資産を
等リスクで分散平均する。Moskowitz/Ooi/Pedersen (2012) "Time Series Momentum"。

edge の源泉は「多数の独立トレンドの分散集約」。単一資産では過小評価されるため、
株価指数・債券先物・コモディティ先物・FX の 4 資産クラス横断で検証する。

ルックアヘッド厳禁:
  - シグナル(過去 lookback ヶ月リターンの符号)は t 時点で過去のみから決定
  - ex-ante ボラ(63 日)も t 時点で過去のみ
  - リターンは t→t+1 forward return に適用 (fwd = monthly.pct_change().shift(-1))

データ品質ゲート:
  - 非正価格は NaN 化 (build_panel)
  - 月次リターンを ±50% でクリップ (未調整分割・データ異常の万%バグ防止)

注意(公開時): yfinance の継続先物 (=F) / FX (=X) / 指数は連続系列の近似であり、
ロール調整やキャリーを厳密に再現しない。CFD/EA 実装時のスワップ・金利は別途要検証。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from .cross_sectional import _metrics

# 4 資産クラス横断のデフォルト・ユニバース (yfinance シンボル: クラス)
# 株価指数 / 債券先物 / コモディティ先物 / FX。EA は CFD で同等にトレード可能。
DEFAULT_UNIVERSE: dict[str, str] = {
    "^GSPC": "equity", "^NDX": "equity", "^N225": "equity",
    "^GDAXI": "equity", "^FTSE": "equity", "^HSI": "equity",
    "ZN=F": "bond", "ZB=F": "bond",
    "GC=F": "commodity", "SI=F": "commodity", "CL=F": "commodity",
    "HG=F": "commodity", "NG=F": "commodity",
    "EURUSD=X": "fx", "USDJPY=X": "fx", "GBPUSD=X": "fx", "AUDUSD=X": "fx",
}


def build_panel(symbols: list[str], years: int = 15, min_obs: int = 500) -> pd.DataFrame:
    """シンボルリストから日足終値パネル (date × symbol) を構築 (キャッシュ利用)。"""
    cols: dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            df = D.load_daily(sym, years=years)
            if len(df) >= min_obs:
                s = df["Close"].astype(float)
                cols[sym] = s.where(s > 0)  # 非正価格(データ異常)は NaN 化
        except Exception:  # noqa: BLE001 — 1 銘柄の失敗で全体を止めない
            continue
    return pd.DataFrame(cols).sort_index()


def run_tsmom(
    panel: pd.DataFrame,
    lookback: int = 12,        # シグナルの過去リターン窓 (ヶ月)
    vol_window: int = 63,      # ex-ante ボラ推定の日数
    vol_target: float = 0.10,  # 資産あたり年率目標ボラ
    max_lev: float = 5.0,      # 資産あたりレバレッジ上限 (逆ボラ爆発の抑制)
    cost_bps: float = 5.0,     # 片道コスト(bp)。ターンオーバーに掛けて控除
    min_assets: int = 3,       # 分散の意味が出る最小資産数 (クラス別検証では緩める)
    long_only: bool = False,   # True で下落資産をショートせずフラット (β寄与の切り分け)
    benchmark: bool = False,   # True で常時ロング等リスク(タイミングなし)の対照群
    return_series: bool = False,
) -> dict:
    """月次リバランスの多資産 TSMOM 検証。ポートフォリオ月次リターン系列を集計。

    時間整合 (Codex 監査 Critical 対応):
      - シグナル(mom)・ex-ante ボラはリバランス日 d (各月の最終取引日) の終値まで
      - 約定は資産別に「d より後の最初の実価格」(休場銘柄の ffill 約定=非因果を回避)
      - 保有リターンは entry(>d) → 翌月 entry(>d') の実価格間 (シグナル日と非重複)
      - 投資可否は d 時点の情報のみで判定 (翌月リターンの有無は使わない)
    """
    # --- ex-ante ボラ(日次→年率)。d 時点で過去のみ ---
    # ミックスカレンダー(各国指数/先物は取引日が違う)の NaN が rolling を全滅させる
    # ため、資産ごとに ffill (休場日=リターン0扱い) してからボラを推定する。
    # ffill は過去値の前送りのみ=ルックアヘッドなし。先頭の未上場期間は NaN のまま。
    prices = panel.ffill()
    daily_ret = prices.pct_change(fill_method=None)
    ex_vol = daily_ret.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)

    # 各月の最終取引日 = リバランス(シグナル)日。
    reb_dates = [grp.index[-1] for _, grp in prices.groupby(prices.index.to_period("M"))]

    # 資産別の「実観測価格」(ffill 前)。約定価格はこれの d より後の最初の値を使う
    # ことで、休場銘柄をシグナル日以前の ffill 値で約定する非因果を防ぐ。
    raw_valid = {c: (panel[c].dropna().index, panel[c].dropna().to_numpy())
                 for c in panel.columns}

    def _price_after(c: str, cutoff) -> float:
        idx, vals = raw_valid[c]
        p = idx.searchsorted(cutoff, side="right")  # cutoff より後の最初の実価格
        return float(vals[p]) if p < len(vals) else float("nan")

    port_rets: list[float] = []
    idx_used: list[pd.Timestamp] = []
    prev_w = pd.Series(dtype=float)
    turnover_sum = 0.0

    for i in range(lookback, len(reb_dates) - 1):
        d = reb_dates[i]                       # シグナル日(月末終値)
        d_past = reb_dates[i - lookback]
        d_next = reb_dates[i + 1]
        mom = prices.loc[d] / prices.loc[d_past] - 1.0
        vol = ex_vol.loc[d]
        # 投資可否は d 時点の情報のみ (翌月の存在は使わない=未来のデータ可用性で選別しない)
        valid = [c for c in prices.columns
                 if np.isfinite(mom[c]) and np.isfinite(vol[c]) and vol[c] > 0]
        if len(valid) < min_assets:
            continue

        # 約定 = 資産別に d より後の最初の実価格 → 翌月 d_next より後の最初の実価格。
        fwd_d = {}
        for c in valid:
            e = _price_after(c, d)
            x = _price_after(c, d_next)
            fwd_d[c] = (x / e - 1.0) if (np.isfinite(e) and np.isfinite(x) and e > 0) else 0.0
        fwd = pd.Series(fwd_d).clip(-0.5, 0.5)  # 異常値クリップ (約定不能は既に 0)

        if benchmark:
            sign = pd.Series(1.0, index=valid)              # 常時ロング(タイミングなし対照)
        else:
            sign = np.sign(mom[valid])
            if long_only:
                sign = sign.clip(lower=0.0)                 # ショートを切る

        # 逆ボラで等リスク化 → 資産あたりレバ上限 → 分散(等加重平均)
        scale = (vol_target / vol[valid]).clip(upper=max_lev)
        wp = (sign * scale) / len(valid)                     # 分散後の実ウェイト

        names = wp.index.union(prev_w.index)
        turn = (wp.reindex(names, fill_value=0.0)
                - prev_w.reindex(names, fill_value=0.0)).abs().sum()
        turnover_sum += turn
        cost = turn * cost_bps / 10000.0
        prev_w = wp

        port_rets.append(float((wp * fwd.reindex(wp.index)).sum()) - cost)
        idx_used.append(d_next)                 # リターンが実現する月でインデックス

    rets = pd.Series(port_rets, index=idx_used)
    half = len(rets) // 2
    out = {
        "lookback": lookback,
        "portfolio": _metrics(rets),
        "oos_1st": _metrics(rets.iloc[:half]),
        "oos_2nd": _metrics(rets.iloc[half:]),
        "avg_turnover": round(turnover_sum / max(1, len(idx_used)), 2),
        "rebalances": len(idx_used),
    }
    if return_series:
        out["series"] = rets
    return out


def alpha_beta(strat: pd.Series, bench: pd.Series, periods_per_year: int = 12) -> dict:
    """戦略リターンを対照群に OLS 回帰し、年率 alpha・beta・alpha の t 値を返す。

    「対照群より Sharpe が低い」だけでは timing alpha 不在は言えない (Codex High)。
    alpha の符号と t 値で「ベンチを超える寄与があるか」を統計的に判定する。
    t 値は Newey-West (HAC) 標準誤差で算出 (月次リターンの自己相関・不均一分散に頑健)。
    """
    df = pd.concat([strat.rename("y"), bench.rename("x")], axis=1).dropna()
    if len(df) < 24:
        return {"alpha_ann": float("nan"), "beta": float("nan"),
                "t_alpha": float("nan"), "n": len(df)}
    y = df["y"].to_numpy()
    x = df["x"].to_numpy()
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    XtX_inv = np.linalg.inv(X.T @ X)

    # Newey-West HAC: meat = S0 + Σ_l w_l (G_l + G_l'), w_l = 1 - l/(L+1)
    u = X * resid[:, None]                       # スコア (n×2)
    L = max(1, int(4 * (n / 100.0) ** (2.0 / 9.0)))  # 標準的なラグ選択
    S = u.T @ u
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        G = u[lag:].T @ u[:-lag]
        S += w * (G + G.T)
    cov_hac = XtX_inv @ S @ XtX_inv
    se_a = float(np.sqrt(cov_hac[0, 0]))
    a = float(coef[0])
    return {"alpha_ann": round(a * periods_per_year * 100, 2),
            "beta": round(float(coef[1]), 2),
            "t_alpha": round(a / se_a, 2) if se_a else float("nan"),
            "hac_lags": L,
            "n": n}


def run_by_class(panel: pd.DataFrame, classes: dict[str, str], **kw) -> dict[str, dict]:
    """資産クラス別の TSMOM (edge が全クラスに広いか・1 クラス集中かの炙り出し)。"""
    out: dict[str, dict] = {}
    for cls in sorted(set(classes.values())):
        syms = [s for s in panel.columns if classes.get(s) == cls]
        if len(syms) >= 2:
            out[cls] = run_tsmom(panel[syms], min_assets=2, **kw)
    return out


def run_tsmom_mf(panel: pd.DataFrame, lookbacks=(3, 6, 12), vol_window: int = 63,
                 asset_vol_target: float = 0.10, max_lev: float = 5.0,
                 port_vol_target: float = 0.10, port_vol_window: int = 12,
                 lev_cap: float = 3.0, cost_bps: float = 5.0,
                 min_assets: int = 3, benchmark: bool = False,
                 return_series: bool = False) -> dict:
    """managed-futures 流 TSMOM: 多ホライズン符号合成 + ポートフォリオ vol target。

    単一ホライズン簡易版(run_tsmom)への Codex 指摘(月次単一・資産別volのみ・portfolio
    vol target なし)に応える『ちゃんとした版』。これでもダメなら trend の扉を閉じる。
      - シグナル = 複数 lookback の sign 平均 (-1〜+1)
      - サイズ = 資産別 逆ボラ等リスク → /n 分散
      - 最後にポートフォリオ全体を目標 vol へ動的レバレッジ
        (レバは過去実現volのみから算出=因果)
    約定/選別の因果規律は run_tsmom と同一 (資産別 d より後の最初の実価格)。
    """
    prices = panel.ffill()
    daily_ret = prices.pct_change(fill_method=None)
    ex_vol = daily_ret.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
    reb_dates = [grp.index[-1] for _, grp in prices.groupby(prices.index.to_period("M"))]
    raw_valid = {c: (panel[c].dropna().index, panel[c].dropna().to_numpy())
                 for c in panel.columns}

    def _price_after(c, cutoff):
        idx, vals = raw_valid[c]
        p = idx.searchsorted(cutoff, side="right")
        return float(vals[p]) if p < len(vals) else float("nan")

    max_lb = max(lookbacks)
    # --- pass 1: 無レバの base ポートフォリオ ---
    base_wp: list[pd.Series] = []
    base_ret: list[float] = []
    idx_used: list[pd.Timestamp] = []
    for i in range(max_lb, len(reb_dates) - 1):
        d, d_next = reb_dates[i], reb_dates[i + 1]
        vol = ex_vol.loc[d]
        sig_sum = pd.Series(0.0, index=prices.columns)
        cnt = pd.Series(0.0, index=prices.columns)
        for lb in lookbacks:
            mom = prices.loc[d] / prices.loc[reb_dates[i - lb]] - 1.0
            sig_sum = sig_sum.add(np.sign(mom), fill_value=0.0)
            cnt = cnt.add(mom.notna().astype(float), fill_value=0.0)
        signal = sig_sum / cnt.replace(0, np.nan)        # 多ホライズン符号平均
        valid = [c for c in prices.columns
                 if np.isfinite(signal[c]) and np.isfinite(vol[c]) and vol[c] > 0]
        if len(valid) < min_assets:
            continue
        fwd = {}
        for c in valid:
            e = _price_after(c, d)
            x = _price_after(c, d_next)
            fwd[c] = (x / e - 1.0) if (np.isfinite(e) and np.isfinite(x) and e > 0) else 0.0
        fwd = pd.Series(fwd).clip(-0.5, 0.5)
        scale = (asset_vol_target / vol[valid]).clip(upper=max_lev)
        # benchmark=常時ロング (universe/vol target/レバ/コストは MF と同一=フェア対照)
        svec = pd.Series(1.0, index=valid) if benchmark else signal.reindex(valid)
        wp = (svec * scale) / len(valid)
        base_wp.append(wp)
        base_ret.append(float((wp * fwd.reindex(wp.index)).sum()))
        idx_used.append(d_next)

    base = pd.Series(base_ret, index=idx_used)
    # --- pass 2: ポートフォリオ vol target (過去実現 vol のみ=因果) ---
    # base[k-1] の exit は period k の signal 日 d の翌日価格を含む(d 時点で未確定)
    # ため、shift(2) で「d より前に完全実現した期」だけからレバを決める (Codex Critical)。
    est_vol = base.rolling(port_vol_window).std().shift(2) * np.sqrt(12)
    lev = (port_vol_target / est_vol).clip(upper=lev_cap).fillna(1.0)

    final_ret: list[float] = []
    prev_w = pd.Series(dtype=float)
    turnover_sum = 0.0
    for k in range(len(idx_used)):
        L = float(lev.iloc[k])
        wp = base_wp[k] * L
        names = wp.index.union(prev_w.index)
        turn = (wp.reindex(names, fill_value=0.0)
                - prev_w.reindex(names, fill_value=0.0)).abs().sum()
        turnover_sum += turn
        cost = turn * cost_bps / 10000.0
        final_ret.append(L * base.iloc[k] - cost)
        prev_w = wp

    rets = pd.Series(final_ret, index=idx_used)
    half = len(rets) // 2
    out = {
        "portfolio": _metrics(rets), "oos_1st": _metrics(rets.iloc[:half]),
        "oos_2nd": _metrics(rets.iloc[half:]),
        "avg_turnover": round(turnover_sum / max(1, len(idx_used)), 2),
        "avg_lev": round(float(lev.mean()), 2), "rebalances": len(idx_used),
    }
    if return_series:
        out["series"] = rets
    return out
