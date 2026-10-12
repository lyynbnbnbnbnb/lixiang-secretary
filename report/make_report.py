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
    ("格式分", "格式分（0–2，答得像不像董秘：有依据 + 有引用）",
     lambda v: f"{v:.2f}", 0.0, 2.0),
    ("引用分", "引用分（0–3，引文逐字回原文查：报告名 + 页码 + 引文真）",
     lambda v: f"{v:.2f}", 0.0, 3.0),
)


def avg_of(data, arm, metric):
    """某条路线的四格平均；缺数据返回 None。"""
    vals = [data[arm][c][metric] for c in CELLS
            if c in data.get(arm, {}) and metric in data[arm][c]]
    return sum(vals) / len(vals) if vals else None


def conclusion_lines(data, diag=None):
    """一句话结论 —— 数字全部从数据算，不允许手抄（手抄必和表对不上）。

    手动断行而不是靠 wrap：matplotlib 的自动换行不认中文，
    会按空格断，整段糊成一团。每行控制在 46 个字以内。
    """
    def g(arm, m):
        v = avg_of(data, arm, m)
        return v if v is not None else 0.0

    lines = [
        "微调给的是「口径」，检索给的是「知识」—— 两条几乎正交的轴。",
        f"只微调：格式分 {g('基座', '格式分'):.2f} → {g('微调后', '格式分'):.2f}，"
        f"净命中仍 ≈{g('微调后', '净命中'):.0%} —— 学会了怎么答，没学会答什么；",
        f"只检索：净命中 → {g('建库', '净命中'):.0%}（基座完全没微调过），"
        f"知识全来自检索到的原文；",
        f"两者合用：净命中最高 {g('微调+建库', '净命中'):.0%}，"
        f"但格式分回落到 {g('微调+建库', '格式分'):.2f} —— 检索挤掉了董秘口径。",
    ]
    if diag:
        d = diag.get("微调后", {})
        if d.get("格式满分条数"):
            lines.append(
                f"引用拆开才看得见：微调后 {d['格式满分条数']} 条格式满分，"
                f"其中 {d['格式满分且引文编造']} 条的「引文」原文里查无。")
    return tuple(lines)


def load(scores_path):
    with open(scores_path, encoding="utf-8") as f:
        return json.load(f)


def load_diag(scores_path):
    """引用维度的诊断（编造引文、格式满分但引文假）——判分器顺手写出的。"""
    p = os.path.join(os.path.dirname(scores_path), "four_grid_diag.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


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
             "格式分(0–2) = 有【依据】+ 有「」引用，只看有没有；"
             "引用分(0–3) = 报告名对 + 页码对 + 引文逐字可查 —— "
             "引文编造在这里扣分；净命中已剔除题干里本来就有的数字。",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0.01, 0.07, 0.99, 0.94))
    fig.savefig(out_png, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  → {out_png}")


def _evidence(data, arms, diag=None):
    """从结果里自动生成几条结论性证据，避免手写数字和表对不上。"""
    lines = []
    for metric, name, fn in (("格式分", "格式分(0–2)", lambda v: f"{v:.2f}"),
                             ("净命中", "净命中率", lambda v: f"{v:.1%}"),
                             ("引用分", "引用分(0–3)", lambda v: f"{v:.2f}")):
        parts = []
        for arm in arms:
            v = avg_of(data, arm, metric)
            if v is not None:
                parts.append(f"{arm} {fn(v)}")
        if parts:
            lines.append(f"· 四格平均{name}：" + "，".join(parts))
    if diag:
        d = diag.get("微调后", {})
        if d.get("有引文条数"):
            lines.append(
                f"· 微调后带「」引文的 {d['有引文条数']} 条里，"
                f"引文共 {d['引文总数']} 句 —— 原文里逐字查无 "
                f"{d['引文未命中']} 句（编造或改写）。")
        d2 = diag.get("微调+建库", {})
        if d2.get("有引文条数"):
            lines.append(
                f"· 微调+建库的引文 {d2['引文总数']} 句里，逐字可查 "
                f"{d2['引文逐字']} 句 —— 检索让「引用」第一次变成真的。")
    return lines


def make_pdf(data, out_pdf, png_path=None, diag=None):
    arms = arms_in(data)
    concl = conclusion_lines(data, diag)
    with PdfPages(out_pdf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))       # A4 竖版
        fig.patch.set_facecolor("white")

        fig.text(0.5, 0.968, "用理想汽车财报训练一个「董秘」",
                 ha="center", fontsize=17, fontweight="bold")
        fig.text(0.5, 0.948, "《学会使用大模型》第三课 · 方向 B · 一页结论",
                 ha="center", fontsize=10.5, color="#666")

        # 一句话结论 —— 框和文字的位置要留够，中文字会顶破框
        box = FancyBboxPatch((0.07, 0.760), 0.86, 0.175,
                             boxstyle="round,pad=0.012,rounding_size=0.012",
                             facecolor="#f2f7fb", edgecolor="#9dbfd8",
                             linewidth=1.1, transform=fig.transFigure)
        fig.patches.append(box)
        fig.text(0.5, 0.907, "一句话结论", ha="center", fontsize=10,
                 color="#4a6b85", fontweight="bold")
        for k, line in enumerate(concl):
            fig.text(0.105, 0.876 - k * 0.0225, line,
                     ha="left", va="top", fontsize=9.6)

        fig.text(0.07, 0.735,
                 "四格对照表 —— 每格 = 训练过 / 没训练过 × 原问法 / 换问法。"
                 "颜色越深越好，三张表同一套色阶。",
                 fontsize=9, color="#666")

        ax1 = fig.add_axes((0.06, 0.578, 0.88, 0.108))
        render_table(ax1, data, arms, "格式分",
                     "格式分（0–2，答得像不像董秘）", CMAP,
                     lambda v: f"{v:.2f}", 0.0, 2.0)

        ax2 = fig.add_axes((0.06, 0.426, 0.88, 0.108))
        render_table(ax2, data, arms, "净命中",
                     "净命中率（关键数字答对多少，已剔除题干回声）",
                     CMAP, lambda v: f"{v:.0%}", 0.0, 1.0)

        ax3 = fig.add_axes((0.06, 0.274, 0.88, 0.108))
        render_table(ax3, data, arms, "引用分",
                     "引用分（0–3，引文逐字回原文查：编造在这里扣分）",
                     CMAP, lambda v: f"{v:.2f}", 0.0, 3.0)

        # 证据
        fig.text(0.07, 0.245, "关键证据", fontsize=12, fontweight="bold")
        y = 0.215
        for line in _evidence(data, arms, diag):
            fig.text(0.09, y, line, fontsize=9.0, linespacing=1.6)
            y -= 0.0255
        fig.text(0.09, y, "· 微调后的模型在「没训练过」的两格里，格式分和"
                          "「训练过」两格基本持平 —— 口径迁移得过去，知识迁不过去。",
                 fontsize=9.0, linespacing=1.6)
        y -= 0.0255
        fig.text(0.09, y, "· 格式满分 ≠ 引文为真：微调模型会把编造的内容包进"
                          "漂亮的「」引用里 —— 三列拆开才看得见。",
                 fontsize=9.0, linespacing=1.6, color="#a33")

        fig.text(0.5, 0.035,
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
    diag = load_diag(args.scores)
    arms = arms_in(data)
    print(f"读了 {len(arms)} 条路线：{'、'.join(arms)}"
          f"{'' if diag else '（无引用诊断文件）'}")

    os.makedirs(args.out, exist_ok=True)
    make_png(data, os.path.join(args.out, "four_grid.png"))
    make_pdf(data, os.path.join(args.out, "一页结论.pdf"), diag=diag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
