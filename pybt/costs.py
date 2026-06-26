"""コストモデル — TV の「commission 0」楽観を Python 側で潰すための層。

backtesting.py の Backtest は spread / commission を
価格に対する比率 (fraction) で受け取る。ここでは
  - spread      : スプレッド (price に対する比率, 往復で対称適用)
  - commission  : 約定額に対する手数料率
をまとめた CostModel と、感度分析用のスイープ格子を提供する。

TV PF1.235 → 実ティック PF1.05 の二重楽観 (検証 665bbd5e) を
「コストを上げると PF が崩れるか」で再現するのが目的。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    spread: float = 0.0       # 価格に対する比率 (例 0.0001 = 1bp)
    commission: float = 0.0   # 約定額に対する比率
    label: str = "baseline"

    def scaled(self, k: float) -> "CostModel":
        """コストを k 倍した派生モデル (感度分析用)。"""
        return CostModel(self.spread * k, self.commission * k, f"cost x{k:g}")


# 代表的なプリセット (必要に応じて調整)
PRESETS = {
    # 無コスト = TV のデフォルト (楽観の上限)
    "zero": CostModel(0.0, 0.0, "zero (TV default)"),
    # 指数CFD 標準的な実コスト感 (スプレッド ~1bp + 手数料)
    "cfd_index": CostModel(spread=0.0001, commission=0.00005, label="cfd_index"),
    # FX メジャー
    "fx_major": CostModel(spread=0.00008, commission=0.00003, label="fx_major"),
    # 日本株 (片道手数料 ~0.05% + スプレッド/スリッページ ~3bp)
    "jp_stock": CostModel(spread=0.0003, commission=0.0005, label="jp_stock"),
}

# コスト感度スイープの倍率 (基準コストに対して)
SWEEP_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0, 3.0)


def sweep_grid(base: CostModel) -> list[CostModel]:
    """基準コストを 0〜3 倍に振ったモデル列を返す。"""
    return [base.scaled(k) for k in SWEEP_MULTIPLIERS]
