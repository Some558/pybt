"""検証スイート — tradingview-method-verify の往復を 1 関数に畳む。

実行する検定 (TV で手動で回していたもの):
  1. ベースライン        : 基準コストで PF/DD/勝率/トレード数
  2. コスト感度          : コストを 0〜3倍に振り、エッジが生き残るか
  3. 方向                : Long単独 / Short単独 / 反転 (逆でも勝つ=ノイズ兆候)
  4. RR近傍              : RR を振り、結果が「尖り(過剰最適化)」か「台地」か
  5. 期間分割OOS         : 前後半・3分割で局面横断の一貫性

最後に「エッジが durable か」のヒューリスティック判定を付す
(断定でなく、Codex ピストンに渡す前の一次フィルタ)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from backtesting import Backtest

from .costs import CostModel, sweep_grid


def _num(x) -> float:
    try:
        f = float(x)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def run_one(df: pd.DataFrame, strat, params: dict, cost: CostModel,
            cash: float = 1_000_000, trade_on_close: bool = False) -> dict:
    """1 構成を回し、主要メトリクスを dict で返す。

    trade_on_close=False (既定): 終値で判定したシグナルを翌バー寄りで約定。
    同一終値約定 (=True) は当日終値を使った条件を当日終値で約定する
    ルックアヘッドになるため、現実的な既定は False (Codex 監査 Critical-1)。
    """
    bt = Backtest(
        df, strat, cash=cash,
        spread=cost.spread, commission=cost.commission,
        trade_on_close=trade_on_close, margin=1.0, finalize_trades=True,
    )
    # 戦略が宣言したパラメータだけ渡す (戦略間で共通 CLI を使うため)
    valid = {k: v for k, v in params.items() if hasattr(strat, k)}
    stats = bt.run(**valid)
    return {
        "pf": _num(stats.get("Profit Factor")),
        "dd": _num(stats.get("Max. Drawdown [%]")),
        "winrate": _num(stats.get("Win Rate [%]")),
        "trades": int(_num(stats.get("# Trades")) or 0),
        "ret": _num(stats.get("Return [%]")),
        "sharpe": _num(stats.get("Sharpe Ratio")),
    }


@dataclass
class SuiteResult:
    name: str
    baseline: dict
    cost: list[tuple[str, dict]]
    direction: list[tuple[str, dict]]
    rr: list[tuple[str, dict]]
    oos: list[tuple[str, dict]]
    verdict: dict


def _split(df: pd.DataFrame, n: int) -> list[tuple[str, pd.DataFrame]]:
    size = len(df) // n
    out = []
    for k in range(n):
        lo = k * size
        hi = len(df) if k == n - 1 else (k + 1) * size
        seg = df.iloc[lo:hi]
        lab = f"{seg.index[0]:%Y-%m} 〜 {seg.index[-1]:%Y-%m}"
        out.append((lab, seg))
    return out


def run_suite(
    df: pd.DataFrame, strat, params: dict, base_cost: CostModel,
    name: str = "strategy",
) -> SuiteResult:
    base = run_one(df, strat, params, base_cost)

    # コスト感度
    cost = [(c.label, run_one(df, strat, params, c)) for c in sweep_grid(base_cost)]

    # 方向
    direction = [
        ("Long単独", run_one(df, strat, {**params, "direction": "Long"}, base_cost)),
        ("Short単独", run_one(df, strat, {**params, "direction": "Short"}, base_cost)),
        ("反転(Both)", run_one(df, strat, {**params, "invert": True}, base_cost)),
    ]

    # パラメータ近傍 (戦略が宣言した TUNE 軸を振る。尖り/台地の判定用)
    tune_name, tune_grid = getattr(strat, "TUNE", ("rr", (1.0, 1.5, 2.0, 2.5, 3.0)))
    rr = [(f"{tune_name}={v:g}", run_one(df, strat, {**params, tune_name: v}, base_cost))
          for v in tune_grid]

    # 期間分割OOS (前後半 + 3分割)
    oos = []
    for lab, seg in _split(df, 2):
        oos.append((f"前後半 {lab}", run_one(seg, strat, params, base_cost)))
    for lab, seg in _split(df, 3):
        oos.append((f"3分割 {lab}", run_one(seg, strat, params, base_cost)))

    verdict = _judge(base, cost, direction, rr, oos)
    return SuiteResult(name, base, cost, direction, rr, oos, verdict)


def _judge(base, cost, direction, rr, oos) -> dict:
    """durable エッジかの一次ヒューリスティック。断定でなく flag。"""
    reasons = []
    ok = True

    # 1x コストで PF>1.1 か
    one_x = next((m for lbl, m in cost if lbl.endswith("x1")), base)
    if not (one_x["pf"] > 1.1):
        ok = False
        reasons.append(f"1xコストで PF={one_x['pf']:.3f} ≤ 1.1 (コスト負け)")

    # OOS 全区間で PF>1
    oos_pf = [m["pf"] for _, m in oos]
    if any(not (p > 1.0) for p in oos_pf):
        ok = False
        reasons.append("OOS で PF≤1 の局面あり (窓依存の疑い)")

    # 反転でも強く勝っていないか (ノイズ兆候)
    inv = next((m for lbl, m in direction if "反転" in lbl), None)
    if inv and inv["pf"] > 1.2:
        ok = False
        reasons.append(f"反転でも PF={inv['pf']:.3f}>1.2 (方向ノイズの疑い)")

    # RR が単一の尖りでないか (最良 RR を外すと PF が急落)
    rr_pf = [m["pf"] for _, m in rr if math.isfinite(m["pf"])]
    if rr_pf:
        best = max(rr_pf)
        others = [p for p in rr_pf if p != best]
        if others and best > 1.2 and max(others) < 1.0:
            ok = False
            reasons.append("RR の好成績が単一点のみ (過剰最適化の尖り)")

    # サンプル数
    if base["trades"] < 100:
        reasons.append(f"トレード数 {base['trades']} < 100 (統計的に弱い・参考値)")

    return {"durable": ok, "reasons": reasons or ["主要検定をすべて通過"]}
