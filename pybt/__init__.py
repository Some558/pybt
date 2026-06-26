"""pybt — TradingView 検証ループを Python で完結させる軽量 harness。

tradingview-method-verify の検証スイート
(ベースライン → コスト感度 → 方向 → RR近傍 → 期間分割OOS) を
backtesting.py 上で 1 コマンド実行に畳むためのパッケージ。

データ源:
  - 日足/スイング : yfinance (10年+)
  - 日中/intraday : Dukascopy 無料ヒストリカル (FX/CFD/指数, 2003年〜)
"""
