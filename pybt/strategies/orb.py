"""ORB (Opening Range Breakout) — ORB_kensho_v1.pine の忠実移植。

Pine 側ルール (改変禁止の基準):
  - セッション 09:30-16:00 America/New_York / OR = 寄り15分 / 15分足
  - close > orHigh で買い (sl=orLow, tp=entry + rr*stopDist)
  - close < orLow  で売り (sl=orHigh, tp=entry - rr*stopDist)
  - リスク riskPct% / 1セッション1トレード / EOD 手仕舞い
  - commission はここでは持たず、Backtest 側 (spread/commission) で外付け
"""
from __future__ import annotations

from backtesting import Strategy

from .base import compute_sessions


class ORB(Strategy):
    # --- パラメータ (run() / Backtest() で上書き可能) ---
    tz = "America/New_York"
    sess_start = "09:30"
    sess_end = "16:00"
    or_min = 15
    rr = 2.0
    risk_pct = 0.5
    direction = "Both"   # "Long" / "Short" / "Both"
    use_eod = True
    invert = False       # True で逆方向 (方向ノイズ検定用)
    TUNE = ("rr", (1.0, 1.5, 2.0, 2.5, 3.0))  # スイートのパラメータ近傍軸

    def init(self):
        s = compute_sessions(
            self.data.df, self.tz, self.sess_start, self.sess_end, self.or_min
        )
        # next() から位置参照する numpy 配列として保持
        self._in_sess = s["in_sess"].to_numpy()
        self._new_sess = s["new_sess"].to_numpy()
        self._sess_end = s["sess_end"].to_numpy()
        self._or_high = s["or_high"].to_numpy()
        self._or_low = s["or_low"].to_numpy()
        self._or_done = s["or_done"].to_numpy()
        self._traded = False

    def _size(self, price: float, stop_dist: float) -> float:
        """equity の risk_pct% を stop_dist で割った建玉を equity 比率で返す。"""
        if stop_dist <= 0:
            return 0.0
        frac = (self.risk_pct / 100.0) * price / stop_dist
        # backtesting.py の fraction sizing は (0,1)。レバレッジ相当は 0.99 で打ち止め
        return max(0.0, min(frac, 0.99))

    def next(self):
        i = len(self.data) - 1
        if self._new_sess[i]:
            self._traded = False

        # EOD 手仕舞い
        if self.use_eod and self._sess_end[i] and self.position:
            self.position.close()
            return

        if not self._in_sess[i] or not self._or_done[i]:
            return
        if self._traded or self.position:
            return

        price = self.data.Close[-1]
        orh, orl = self._or_high[i], self._or_low[i]
        if orh != orh or orl != orl:  # NaN ガード
            return

        # ブレイク方向と、その時の建玉サイド・損切りを決める。
        # 通常は順張り (上抜けで買い)。invert はフェード = 同じリスク幅のまま逆張り
        # (反対でも勝つなら方向シグナルがノイズという検定)。
        if price > orh:        # 上抜け
            if not self.invert:
                side, sl, stop_dist = "long", orl, price - orl
            else:
                stop_dist = price - orl
                side, sl = "short", price + stop_dist
        elif price < orl:      # 下抜け
            if not self.invert:
                side, sl, stop_dist = "short", orh, orh - price
            else:
                stop_dist = orh - price
                side, sl = "long", price - stop_dist
        else:
            return

        allow = {"long": self.direction in ("Long", "Both"),
                 "short": self.direction in ("Short", "Both")}
        if not allow[side]:
            return

        sz = self._size(price, stop_dist)
        if sz <= 0:
            return
        if side == "long":
            tp = price + self.rr * stop_dist
            if sl < price < tp:
                self.buy(size=sz, sl=sl, tp=tp)
                self._traded = True
        else:
            tp = price - self.rr * stop_dist
            if tp < price < sl:
                self.sell(size=sz, sl=sl, tp=tp)
                self._traded = True
