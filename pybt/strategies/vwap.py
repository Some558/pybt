"""VWAP 回帰 — 「寄り後 VWAP からの乖離は回帰する」の検証 (intraday)。

通説/仮説: セッション内 VWAP から大きく乖離した価格は VWAP へ戻りやすい。
客観ルール化:
  - エントリー: |close - VWAP| > k_entry × ATR
      close < VWAP - k×ATR → 買い (上への回帰を狙う)
      close > VWAP + k×ATR → 売り
  - 利確: VWAP へ回帰 (long なら close>=VWAP / short なら close<=VWAP) ※動的なので手動管理
  - 損切り: 乖離拡大側へ sl_atr × ATR (注文に付与)
  - EOD 手仕舞い / 1セッション複数可だが同時1ポジション
  - invert: 逆張りを順張りに反転 (方向ノイズ検定)

intraday 専用 (Dukascopy)。VWAP は出来高加重 (volume 不在時は典型価格の累積平均へフォールバック)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting import Strategy

from .base import compute_sessions


class VWAPReversion(Strategy):
    # --- パラメータ ---
    tz = "America/New_York"
    sess_start = "09:30"
    sess_end = "16:00"
    k_entry = 1.5      # 何 ATR 乖離でエントリー
    sl_atr = 1.5       # 乖離拡大側の損切り幅 (ATR 倍)
    atr_period = 14
    risk_pct = 0.5
    direction = "Both"  # "Long" / "Short" / "Both"
    invert = False
    use_eod = True
    TUNE = ("k_entry", (1.0, 1.5, 2.0, 2.5, 3.0))  # スイートのパラメータ近傍軸

    def init(self):
        df = self.data.df
        s = compute_sessions(df, self.tz, self.sess_start, self.sess_end, or_min=0)
        self._in_sess = s["in_sess"].to_numpy()
        self._sess_end = s["sess_end"].to_numpy()

        sid = s["new_sess"].cumsum()
        h, l, c = df["High"], df["Low"], df["Close"]
        typ = (h + l + c) / 3.0
        vol = df["Volume"].astype(float)
        # セッション内累積で VWAP。出来高ゼロなら典型価格の累積平均にフォールバック
        pv = (typ * vol).groupby(sid).cumsum()
        cv = vol.groupby(sid).cumsum()
        cnt = pd.Series(1.0, index=df.index).groupby(sid).cumsum()
        cum_typ = typ.groupby(sid).cumsum()
        vwap = np.where(cv > 0, pv / cv.replace(0, np.nan), cum_typ / cnt)
        self._vwap = np.asarray(vwap, dtype=float)

        # ATR
        prev_c = c.shift(1)
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        self._atr = tr.rolling(self.atr_period).mean().to_numpy()

    def _size(self, price: float, stop_dist: float) -> float:
        if stop_dist <= 0:
            return 0.0
        frac = (self.risk_pct / 100.0) * price / stop_dist
        return max(0.0, min(frac, 0.99))

    def next(self):
        i = len(self.data) - 1

        # EOD 手仕舞い
        if self.use_eod and self._sess_end[i] and self.position:
            self.position.close()
            return
        if not self._in_sess[i]:
            return

        vwap, atr = self._vwap[i], self._atr[i]
        if vwap != vwap or atr != atr or atr <= 0:  # NaN / ウォームアップ
            return
        price = self.data.Close[-1]

        # 利確: VWAP へ回帰したら手仕舞い (損切りは注文側 SL が処理)
        if self.position:
            if self.position.is_long and price >= vwap:
                self.position.close()
            elif self.position.is_short and price <= vwap:
                self.position.close()
            return

        dev = price - vwap
        long_sig = dev < -self.k_entry * atr
        short_sig = dev > self.k_entry * atr
        side = "long" if long_sig else ("short" if short_sig else None)
        if side is None:
            return
        if self.invert:
            side = "short" if side == "long" else "long"

        allow = {"long": self.direction in ("Long", "Both"),
                 "short": self.direction in ("Short", "Both")}
        if not allow[side]:
            return

        stop_dist = self.sl_atr * atr
        sz = self._size(price, stop_dist)
        if sz <= 0:
            return
        if side == "long":
            sl = price - stop_dist
            if sl < price:
                self.buy(size=sz, sl=sl)
        else:
            sl = price + stop_dist
            if sl > price:
                self.sell(size=sz, sl=sl)
