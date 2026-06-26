#!/usr/bin/env python
"""pybt CLI — TV 検証ループを 1 コマンドで回す。

例:
  # NAS100 15分 ORB を Dukascopy intraday で (指数CFDコスト)
  python run.py orb --market intraday --symbol NAS100 --tf 15m \
      --start 2018-01-01 --end 2024-01-01 --cost cfd_index

  # 日経225 日足スイング戦略を yfinance で
  python run.py orb --market daily --symbol "^N225" --years 10 --cost zero

結果は標準出力 + reports/<name>_<timestamp>.md に保存。
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from pybt import data as D
from pybt.costs import PRESETS
from pybt.report import to_markdown
from pybt.strategies import REGISTRY
from pybt.suite import run_suite


def parse_args():
    p = argparse.ArgumentParser(description="pybt — TV検証ループの Python 版")
    p.add_argument("strategy", choices=list(REGISTRY), help="戦略名")
    p.add_argument("--market", choices=["daily", "intraday"], required=True)
    p.add_argument("--symbol", required=True, help="銘柄 (NAS100 / ^N225 / 7203.T 等)")
    p.add_argument("--tf", default="15m", help="intraday の時間足 (15m 等)")
    p.add_argument("--years", type=int, default=10, help="daily の取得年数")
    p.add_argument("--start", help="intraday 開始 YYYY-MM-DD")
    p.add_argument("--end", help="intraday 終了 YYYY-MM-DD")
    p.add_argument("--cost", default="zero", choices=list(PRESETS), help="コストプリセット")
    p.add_argument("--rr", type=float, default=2.0)
    p.add_argument("--risk", type=float, default=0.5, help="1トレードのリスク%%")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def load(args):
    if args.market == "daily":
        return D.load_daily(args.symbol, years=args.years, use_cache=not args.no_cache)
    if not (args.start and args.end):
        raise SystemExit("intraday は --start / --end が必須です")
    start = dt.datetime.fromisoformat(args.start)
    end = dt.datetime.fromisoformat(args.end)
    instrument = D.resolve_instrument(args.symbol)
    return D.load_intraday(instrument, args.tf, start, end, use_cache=not args.no_cache)


def main():
    args = parse_args()
    df = load(args)
    strat = REGISTRY[args.strategy]
    cost = PRESETS[args.cost]
    params = {"rr": args.rr, "risk_pct": args.risk, "direction": "Both"}

    name = f"{args.strategy}_{args.symbol}_{args.market}"
    result = run_suite(df, strat, params, cost, name=name)

    meta = {
        "戦略": args.strategy, "銘柄": args.symbol, "市場": args.market,
        "時間足": args.tf if args.market == "intraday" else "1d",
        "期間": f"{df.index[0]:%Y-%m-%d} 〜 {df.index[-1]:%Y-%m-%d} ({len(df)}本)",
        "基準コスト": cost.label, "RR": args.rr, "リスク%": args.risk,
    }
    md = to_markdown(result, meta)
    print(md)

    out_dir = Path(__file__).parent / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = f"{df.index[-1]:%Y%m%d}"
    out = out_dir / f"{name}_{stamp}.md"
    out.write_text(md, encoding="utf-8")
    print(f"\n→ 保存: {out}")


if __name__ == "__main__":
    main()
