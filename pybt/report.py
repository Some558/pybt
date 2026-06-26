"""SuiteResult を Markdown レポートに整形。

TV から数値を貼り戻していた表を、そのまま機械生成する。
Codex ピストン (C/H=0) へ渡す一次成果物。
"""
from __future__ import annotations

from .suite import SuiteResult


def _row(label: str, m: dict) -> str:
    def f(x, suf=""):
        return "—" if x != x else f"{x:.2f}{suf}"  # NaN は —
    return (f"| {label} | {f(m['pf'])} | {f(m['dd'], '%')} | "
            f"{f(m['winrate'], '%')} | {m['trades']} | {f(m['ret'], '%')} |")


_HEAD = "| 区分 | PF | 最大DD | 勝率 | 取引数 | リターン |\n|---|---|---|---|---|---|"


def to_markdown(r: SuiteResult, meta: dict) -> str:
    lines: list[str] = []
    lines.append(f"# 検証レポート — {r.name}")
    lines.append("")
    lines.append("| 項目 | 値 |\n|---|---|")
    for k, v in meta.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## 1. ベースライン")
    lines.append(_HEAD)
    lines.append(_row("baseline", r.baseline))
    lines.append("")

    lines.append("## 2. コスト感度 (0〜3倍)")
    lines.append(_HEAD)
    for lbl, m in r.cost:
        lines.append(_row(lbl, m))
    lines.append("")

    lines.append("## 3. 方向")
    lines.append(_HEAD)
    for lbl, m in r.direction:
        lines.append(_row(lbl, m))
    lines.append("")

    lines.append("## 4. パラメータ近傍 (尖り=過剰最適化 / 台地=頑健)")
    lines.append(_HEAD)
    for lbl, m in r.rr:
        lines.append(_row(lbl, m))
    lines.append("")

    lines.append("## 5. 期間分割 OOS")
    lines.append(_HEAD)
    for lbl, m in r.oos:
        lines.append(_row(lbl, m))
    lines.append("")

    v = r.verdict
    mark = "🟢 durable の可能性" if v["durable"] else "🔴 durable とは言えない"
    lines.append("## 判定 (一次ヒューリスティック)")
    lines.append(f"**{mark}**")
    lines.append("")
    for reason in v["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("> ⚠ これは一次フィルタ。最終判断は Codex ピストン (C/H=0) で。")
    return "\n".join(lines)
