"""戦略共通の前計算ヘルパー。

intraday の取引所セッションとオープニングレンジ(OR)を
バーごとの配列にして返す。ORB 以外のセッション型戦略でも再利用する。
"""
from __future__ import annotations

import pandas as pd


def compute_sessions(
    df: pd.DataFrame,
    tz: str = "America/New_York",
    sess_start: str = "09:30",
    sess_end: str = "16:00",
    or_min: int = 15,
) -> pd.DataFrame:
    """UTC index の OHLCV から、セッション/OR 情報をバーごとに算出。

    返却列 (index は df と同一):
      in_sess  : 取引所セッション内か
      new_sess : セッション最初のバーか
      sess_end : セッション最後のバーか (EOD 手仕舞い判定用)
      or_high  : そのセッションの OR 高値 (セッション全バーへ broadcast)
      or_low   : そのセッションの OR 安値
      or_done  : OR 形成が完了しエントリー可能か
    """
    local = df.index.tz_convert(tz)
    tod = pd.Series(local.time, index=df.index)

    sh, sm = map(int, sess_start.split(":"))
    eh, em = map(int, sess_end.split(":"))
    import datetime as _dt
    t_start, t_end = _dt.time(sh, sm), _dt.time(eh, em)

    in_sess = (tod >= t_start) & (tod < t_end)
    prev = in_sess.shift(1, fill_value=False)
    nxt = in_sess.shift(-1, fill_value=False)
    new_sess = in_sess & ~prev
    sess_end_bar = in_sess & ~nxt

    # セッションID (新セッションごとに採番。セッション外は ffill された ID を持つが後段で in_sess で弾く)
    sid = new_sess.cumsum()

    # セッション開始時刻を各バーへ ffill → 経過分を計算
    sess_start_ts = pd.Series(df.index, index=df.index).where(new_sess)
    sess_start_ts = sess_start_ts.groupby(sid).ffill()
    mins_el = (pd.Series(df.index, index=df.index) - sess_start_ts).dt.total_seconds() / 60.0

    in_or = in_sess & (mins_el < or_min)
    # OR 窓の高安をセッション全体へ broadcast (使うのは or_done 後のみ)
    or_high = df["High"].where(in_or).groupby(sid).transform("max")
    or_low = df["Low"].where(in_or).groupby(sid).transform("min")
    or_done = in_sess & (mins_el >= or_min)

    return pd.DataFrame(
        {
            "in_sess": in_sess,
            "new_sess": new_sess,
            "sess_end": sess_end_bar,
            "or_high": or_high,
            "or_low": or_low,
            "or_done": or_done,
        },
        index=df.index,
    )
