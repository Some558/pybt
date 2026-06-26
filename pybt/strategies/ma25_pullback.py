"""MA25 押し目買い — 「上昇トレンド中の25日線タッチで反発買い」の検証。

通説: 上昇トレンド(価格>MA25 かつ MA25 上向き)で 25日線まで押したら買い場。
客観ルール化:
  - 上昇押し目(Long): 終値>MA25 & MA25上向き & 安値が MA25 へ touch_tol 以内 & 終値は MA25 上で引け
  - 下降戻り(Short) : 終値<MA25 & MA25下向き & 高値が MA25 へ touch_tol 以内 & 終値は MA25 下で引け (対称)
  - 手仕舞い: RR ターゲット/損切り + トレンド割れ(終値が MA25 を逆方向へ抜け)
  - invert: 順張りを逆張りに反転 (方向ノイズ検定)

daily 専用 (yfinance)。コストは Backtest 側 (jp_stock 等) で外付け。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy


class MA25Pullback(Strategy):
    # --- パラメータ ---
    ma_period = 25
    slope_lookback = 10     # MA25 の上向き/下向きを測る期間
    touch_tol = 0.01        # MA25 へ何% 以内まで寄ったら「タッチ」
    sl_buf = 0.005          # 損切りバッファ
    rr = 2.0
    risk_pct = 0.5
    direction = "Both"      # "Long" / "Short" / "Both"
    invert = False
    exit_on_ma_break = True  # トレンド割れで手仕舞い
    TUNE = ("rr", (1.0, 1.5, 2.0, 2.5, 3.0))  # スイートのパラメータ近傍軸

    def init(self):
        close = pd.Series(np.asarray(self.data.Close, dtype=float))
        ma = close.rolling(self.ma_period).mean()
        self._ma = ma.to_numpy()
        self._slope = (ma - ma.shift(self.slope_lookback)).to_numpy()
        self._high = np.asarray(self.data.High, dtype=float)
        self._low = np.asarray(self.data.Low, dtype=float)

    def _size(self, price: float, stop_dist: float) -> float:
        if stop_dist <= 0:
            return 0.0
        frac = (self.risk_pct / 100.0) * price / stop_dist
        return max(0.0, min(frac, 0.99))

    def next(self):
        i = len(self.data) - 1
        ma, slope = self._ma[i], self._slope[i]
        if ma != ma or slope != slope:  # ウォームアップ中の NaN
            return
        price = self.data.Close[-1]

        # トレンド割れ手仕舞い (RR/SL より先に判定)
        if self.position and self.exit_on_ma_break:
            if self.position.is_long and price < ma:
                self.position.close()
                return
            if self.position.is_short and price > ma:
                self.position.close()
                return
        if self.position:
            return

        low, high = self._low[i], self._high[i]
        up_pullback = (price > ma) and (slope > 0) and (low <= ma * (1 + self.touch_tol))
        down_rally = (price < ma) and (slope < 0) and (high >= ma * (1 - self.touch_tol))

        side = "long" if up_pullback else ("short" if down_rally else None)
        if side is None:
            return
        if self.invert:
            side = "short" if side == "long" else "long"

        allow = {"long": self.direction in ("Long", "Both"),
                 "short": self.direction in ("Short", "Both")}
        if not allow[side]:
            return

        if side == "long":
            sl = min(low, ma) * (1 - self.sl_buf)
            stop_dist = price - sl
            tp = price + self.rr * stop_dist
            sz = self._size(price, stop_dist)
            if sz > 0 and sl < price < tp:
                self.buy(size=sz, sl=sl, tp=tp)
        else:
            sl = max(high, ma) * (1 + self.sl_buf)
            stop_dist = sl - price
            tp = price - self.rr * stop_dist
            sz = self._size(price, stop_dist)
            if sz > 0 and tp < price < sl:
                self.sell(size=sz, sl=sl, tp=tp)
