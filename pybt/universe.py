"""複数銘柄ランナー — 「検証ノート」ブランドの東証全銘柄横断検証。

単一銘柄では公開素材にならない (brand 規約 = 東証プライム全銘柄)。
各銘柄にヘッドライン構成を 1 本ずつ回し、PF>1 の銘柄割合・中央値 PF 等の
分布として集計する。これが検証記事の本体になる。

ティッカー源: nitekabu/loop/app/wwwroot/data/stocks.json (JPX 公式由来)。
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from . import data as D
from .costs import CostModel
from .suite import run_one

# nitekabu リポジトリ内の JPX 由来銘柄リスト
_STOCKS_JSON = Path(
    "/Users/some/Projects/nitekabu/loop/app/wwwroot/data/stocks.json"
)


def load_prime_tickers(limit: int | None = None) -> list[dict]:
    """プライム銘柄を [{ticker, name, sector}] で返す (yfinance 用 .T 付き)。"""
    rows = json.loads(_STOCKS_JSON.read_text(encoding="utf-8"))
    prime = [r for r in rows if r.get("market") == "プライム"]
    if limit:
        prime = prime[:limit]
    return [{"ticker": f"{r['ticker']}.T", "name": r.get("name", ""),
             "sector": r.get("sector", "")} for r in prime]


def run_universe(
    tickers: list[dict], strat, params: dict, cost: CostModel,
    years: int = 10, min_trades: int = 10, progress_every: int = 50,
) -> tuple[list[dict], list[str]]:
    """各銘柄に 1 構成を回し、メトリクス行と失敗銘柄を返す。"""
    rows: list[dict] = []
    failed: list[str] = []
    n = len(tickers)
    for i, t in enumerate(tickers, 1):
        sym = t["ticker"]
        try:
            df = D.load_daily(sym, years=years)
            if len(df) < 60:  # データ不足
                failed.append(sym)
                continue
            m = run_one(df, strat, params, cost)
            rows.append({"ticker": sym, "name": t["name"],
                         "sector": t["sector"], **m})
        except Exception as e:  # noqa: BLE001 — 1 銘柄の失敗で全体を止めない
            failed.append(f"{sym}:{type(e).__name__}")
        if progress_every and i % progress_every == 0:
            print(f"  ... {i}/{n} ({len(failed)} 失敗)")
    return rows, failed


def aggregate(rows: list[dict], min_trades: int = 10) -> dict:
    """横断分布を集計。検証記事に載せる主要統計。"""
    traded = [r for r in rows if r["trades"] >= min_trades]
    pf = [r["pf"] for r in traded if r["pf"] == r["pf"]]  # NaN除外
    ret = [r["ret"] for r in traded if r["ret"] == r["ret"]]
    wr = [r["winrate"] for r in traded if r["winrate"] == r["winrate"]]
    dd = [r["dd"] for r in traded if r["dd"] == r["dd"]]
    total_trades = sum(r["trades"] for r in traded)

    def pct(cond_list):
        return 100.0 * sum(cond_list) / len(cond_list) if cond_list else float("nan")

    return {
        "銘柄数(検証実行)": len(rows),
        f"銘柄数(取引{min_trades}+件)": len(traded),
        "総取引数": total_trades,
        "PF中央値": round(st.median(pf), 3) if pf else float("nan"),
        "PF平均": round(st.fmean(pf), 3) if pf else float("nan"),
        "PF>1の銘柄割合%": round(pct([p > 1 for p in pf]), 1) if pf else float("nan"),
        "プラス収益の銘柄割合%": round(pct([r > 0 for r in ret]), 1) if ret else float("nan"),
        "勝率中央値%": round(st.median(wr), 1) if wr else float("nan"),
        "最大DD中央値%": round(st.median(dd), 1) if dd else float("nan"),
    }
