# -*- coding: utf-8 -*-
"""把判分结果渲染成作业要交的两份东西：

    report/four_grid.png     四格对照表截图
    report/一页结论.pdf       一页结论

数据来源是 `eval/scores/four_grid.json`（judge.py 顺手写出的机器可读汇总）。
不直接解析 markdown，是因为表格一旦改版，解析就悄悄读错列 —— 那种错最难发现。

用法：
    python make_report.py
    python make_report.py --scores ../eval/scores/four_grid.json
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages               # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# 中文必须显式指定字体，否则整张图全是方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

CELLS = ("trained_original", "trained_rephrased",
         "untrained_original", "untrained_rephrased")
CELL_CN = {
    "trained_original": "训练过 · 原问法",
    "trained_rephrased": "训练过 · 换问法",
    "untrained_original": "没训练过 · 原问法",
    "untrained_rephrased": "没训练过 · 换问法",
}
ARM_ORDER = ["基座", "微调后", "建库", "微调+建库"]

# 三张表统一用「越深越好」的同一色系 —— 配色语义不一致最容易读错
CMAP = plt.get_cmap("YlGn")

# (字段, 标题, 数值格式, 下限, 上限)
PANELS = (
    ("模型判分", "模型判分（均，2=正确 1=部分 0=错误）", lambda v: f"{v:.2f}",
     0.0, 2.0),
    ("净命中", "净命中率（关键数字答对多少，已剔除题干回声）",
     lambda v: f"{v:.0%}", 0.0, 1.0),
    ("格式分", "格式分（0–3，答得像不像董秘）", lambda v: f"{v:.2f}", 0.0, 3.0),
)

# 手动断行而不是靠 wrap：matplotlib 的自动换行不认中文，
# 会按空格断，整段糊成一团。每行控制在 44 个字以内。
结论_LINES = (
    "微调给的是「口径」，检索给的是「知识」—— 两条几乎正交的轴。",
    "只微调：格式分 0.0 → 2.4，净命中仍 ≈1% —— 学会了怎么答，没学会答什么；",
    "只检索：净命中 → 26%，格式分仍 0.0 —— 知识全来自检索到的原文；",
    "两者合用：净命中最高 38%，但格式分回落到 1.4 —— 检索进来的原文挤掉了董秘口径。",
)


def load(scores_path):
    with open(scores_path, encoding="utf-8") as f:
        return json.load(f)


def arms_in(data):
    """按 ARM_ORDER 排序，只保留实际有的 arm。"""
    have = list(next(iter(data.values())).keys()) if data else []
    return [a for a in ARM_ORDER if a in data] + \
           [a for a in data if a not in ARM_ORDER]


def render_table(ax, data, arms, metric, title, cmap, fmt_fn, vmin, vmax):
    """在 ax 上画一张「4 格 × N 路线」的表，格子里按数值上色。"""
    n_arm = len(arms)
    ax.set_title(title, fontsize=13, pad=14, fontweight="bold")
    ax.set_xlim(0, n_arm + 1)
    ax.set_ylim(0, len(CELLS) + 1)
    ax.axis("off")

    for j, arm in enumerate(arms):
        ax.text(j + 1.5, len(CELLS) + 0.55, arm, ha="center", va="center",
                fontsize=11, fontweight="bold")

    for i, cell in enumerate(CELLS):
        y = len(CELLS) - i - 0.5
        ax.text(0.95, y, CELL_CN[cell], ha="right", va="center", fontsize=10)
        for j, arm in enumerate(arms):
            v = data.get(arm, {}).get(cell, {}).get(metric)
            if v is None:
                ax.text(j + 1.5, y, "—", ha="center", va="center",
                        fontsize=11, color="#999")
                continue
            frac = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
            ax.add_patch(Rectangle((j + 1.02, y - 0.4), 0.96, 0.8,
                                   facecolor=cmap(frac), edgecolor="#bbb",
                                   linewidth=0.8))
            # 深底上用白字，否则看不清
            color = "white" if frac > 0.62 else "#222"
            ax.text(j + 1.5, y, fmt_fn(v), ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")


def make_png(data, out_png):
    arms = arms_in(data)
    # 只画数据里真有的指标 —— 模型判分要跑 API 之后才有
    panels = [p for p in PANELS
              if any(p[0] in data.get(a, {}).get(c, {})
                     for a in arms for c in CELLS)]
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(5.6 * len(panels), 4.6),
                             squeeze=False)
    axes = axes[0]
    fig.suptitle("四格对照表 · 理想汽车董秘", fontsize=16, fontweight="bold",
                 y=0.98)

    for ax, (metric, title, fmt, lo, hi) in zip(axes, panels):
        render_table(ax, data, arms, metric, title, CMAP, fmt, lo, hi)

    fig.text(0.5, 0.035,
             "格式分 = 有【依据】+ 引对报告 + 有「」原文引用，各 1 分。"
             "净命中把题干里本来就有数字剔掉 —— 照抄题干不算答对。",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0.01, 0.07, 0.99, 0.94))
    fig.savefig(out_png, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  → {out_png}")


def _evidence(data, arms):
    """从结果里自动生成几条结论性证据，避免手写数字和表对不上。"""
    lines = []
    for metric, name, fn in (("格式分", "格式分", lambda v: f"{v:.2f}"),
                             ("净命中", "净命中率", lambda v: f"{v:.1%}")):
        parts = []
        for arm in arms:
            vals = [data[arm][c][metric] for c in CELLS
                    if c in data.get(arm, {}) and metric in data[arm][c]]
            if vals:
                parts.append(f"{arm} {fn(sum(vals) / len(vals))}")
        lines.append(f"· 四格平均{name}：" + "，".join(parts))
    return lines


def make_pdf(data, out_pdf, png_path=None):
    arms = arms_in(data)
    with PdfPages(out_pdf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))       # A4 竖版
        fig.patch.set_facecolor("white")

        fig.text(0.5, 0.965, "用理想汽车财报训练一个「董秘」",
                 ha="center", fontsize=17, fontweight="bold")
        fig.text(0.5, 0.945, "《学会使用大模型》第三课 · 方向 B · 一页结论",
                 ha="center", fontsize=10.5, color="#666")

        # 一句话结论 —— 框和文字的位置要留够，中文字会顶破框
        box = FancyBboxPatch((0.07, 0.795), 0.86, 0.135,
                             boxstyle="round,pad=0.012,rounding_size=0.012",
                             facecolor="#f2f7fb", edgecolor="#9dbfd8",
                             linewidth=1.1, transform=fig.transFigure)
        fig.patches.append(box)
        fig.text(0.5, 0.900, "一句话结论", ha="center", fontsize=10,
                 color="#4a6b85", fontweight="bold")
        for k, line in enumerate(结论_LINES):
            fig.text(0.105, 0.872 - k * 0.0225, line,
                     ha="left", va="top", fontsize=10.5)

        fig.text(0.07, 0.755,
                 "四格对照表 —— 每格 = 训练过 / 没训练过 × 原问法 / 换问法。"
                 "颜色越深越好，两张表同一套色阶。",
                 fontsize=9, color="#666")

        ax1 = fig.add_axes((0.06, 0.565, 0.88, 0.155))
        render_table(ax1, data, arms, "格式分", "格式分（0–3，答得像不像董秘）",
                     CMAP, lambda v: f"{v:.2f}", 0.0, 3.0)

        ax2 = fig.add_axes((0.06, 0.365, 0.88, 0.155))
        render_table(ax2, data, arms, "净命中",
                     "净命中率（关键数字答对多少，已剔除题干回声）",
                     CMAP, lambda v: f"{v:.0%}", 0.0, 1.0)

        # 证据
        fig.text(0.07, 0.315, "关键证据", fontsize=12, fontweight="bold")
        y = 0.283
        for line in _evidence(data, arms):
            fig.text(0.09, y, line, fontsize=9.5, linespacing=1.6)
            y -= 0.032
        fig.text(0.09, y, "· 微调后的模型在「没训练过」的两格里，格式分和"
                          "「训练过」两格基本持平 —— 口径迁移得过去，知识迁不过去。",
                 fontsize=9.5, linespacing=1.6)
        y -= 0.032
        fig.text(0.09, y, "· 微调模型会在编造的数字外面套上「原文引用」和"
                          "【依据：…】页码，格式完美、内容全错 —— 这类错误最危险。",
                 fontsize=9.5, linespacing=1.6, color="#a33")

        fig.text(0.5, 0.045,
                 "复现：data/ 造数据 → train/ 微调 → eval/ 生成与判分 → report/ 出表",
                 ha="center", fontsize=8.5, color="#888")
        pdf.savefig(fig)
        # 同时存一张 PNG：本机没装 poppler，PDF 不能直接渲染，
        # 没法「导出看图」就等于盲写版式。多这一张图才检查得了。
        png = os.path.splitext(out_pdf)[0] + ".png"
        fig.savefig(png, dpi=110, facecolor="white")
        plt.close(fig)
    print(f"  → {out_pdf}")
    print(f"  → {os.path.splitext(out_pdf)[0] + '.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores",
                    default=os.path.join(HERE, "..", "eval", "scores",
                                         "four_grid.json"))
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()

    if not os.path.exists(args.scores):
        print(f"没有 {args.scores} —— 先跑 eval/judge.py 出分")
        return 1

    data = load(args.scores)
    arms = arms_in(data)
    print(f"读了 {len(arms)} 条路线：{'、'.join(arms)}")

    os.makedirs(args.out, exist_ok=True)
    make_png(data, os.path.join(args.out, "four_grid.png"))
    make_pdf(data, os.path.join(args.out, "一页结论.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
