# -*- coding: utf-8 -*-
"""
第 5 步（判分器）：给四格评测集判分，出对照表。

评测集 `eval_grid.jsonl` 有四个格子：
    trained_original     trained_rephrased
    untrained_original   untrained_rephrased

本脚本吃「生成结果」文件，吐两种分数：

  ①【客观】关键数字命中 —— 自动比对参考答案里带量纲的数字（金额/百分比）
     在模型回答里出现了没有，金额先折算到同一量纲再比。
     完全确定、可复现、不用花钱，也骗不了。财报问答的答案大多是数字，所以这条很管用。
     额外报一个**净命中率**：把题干里本来就有的数字剔掉 —— 提问常自带数字，
     照抄题干不算答对。
  ②【主观】模型判分 —— 让第三方模型对照参考答案判 0/1/2。
     能判"方向对不对"这类数字之外的东西。

两种都报，是因为它们各有盲区：数字全中但结论说反了 → 客观给满分、主观给 0。

用法：
    python judge.py --preds 结果/base.jsonl --preds 结果/sft.jsonl
    python judge.py --preds 结果/sft.jsonl --label "微调后"
    python judge.py --selftest                  # 先验证判分器本身靠不靠谱

生成结果文件的格式（每行一条）：
    {"qa_id": "...", "cell": "...", "prediction": "模型回答的原文"}
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
from build_qa import call_llm, parse_json                      # noqa: E402
from common import key_figures, match_figures                  # noqa: E402

GRID = os.path.join(HERE, "..", "data", "eval_grid.jsonl")
CELLS = ("trained_original", "trained_rephrased",
         "untrained_original", "untrained_rephrased")
CELL_CN = {
    "trained_original": "训练过 · 原问法",
    "trained_rephrased": "训练过 · 换问法",
    "untrained_original": "没训练过 · 原问法",
    "untrained_rephrased": "没训练过 · 换问法",
}

# 附在 four_grid.md 末尾 —— 表是给老师看的，列的含义得写清楚
FOOTNOTE = """
---

### 列的含义

| 列 | 怎么算 |
|---|---|
| **模型判分(均)** | 第三方模型对照参考答案判 0/1/2 的均分（2=正确，1=部分正确，0=错误或编造） |
| **正确率** | 判 2 的比例 |
| **关键数字命中** | 参考答案里带量纲的关键数字（金额 / 百分比）答对了多少。金额折算到同一量纲后再比——「2,760,242 千元」与「276.0 亿元」算同一个数。裸年份不计入 |
| **净命中** | 把**题干里本来就出现的数字**剔掉之后的命中率 |
| **格式分(0–3)** | 答得像不像董秘：① 有 `【依据：…】` ② 引对了报告 ③ 有 `「…」` 原文引用，各 1 分 |

**为什么要单独看「净命中」**：投资者提问常自带数字
（「资本承诺 63 亿元，相比去年 64 亿元…」），模型把题干复述一遍就能拿分，
那不是知识。两个数差距越大，说明成绩里"照抄题干"的成分越多。

**格式分是这张表里唯一能把四条路线区分开的客观列** —— 数字命中四条都接近 0，
区别全在「答得像不像董秘」上。微调真正学到的东西就落在这里。

> 复算提示：早期版本的「数字命中率」是拿字符串直接比数字，
> 实测 174 个"命中"里有 164 个（94%）是年份，等于没在量知识。
> 换成带量纲的关键数字之后，这把尺子才量得准。
"""

J_SYS = "你是严谨的评分员。只输出 JSON，不要任何解释。"

J_TPL = """你在给一次财报问答评测打分。

下面是一个投资者提问、一份**参考答案**，以及**模型的实际回答**。
请判断模型回答是否复现了参考答案里的关键事实。

评分标准：
- **2 = 正确**：关键事实（数字、结论）与参考答案一致
- **1 = 部分正确**：方向对，但有数字出入、或只答对一半
- **0 = 错误**：关键数字错了、答非所问、说不知道，或编造了参考答案里没有的事实

注意：
- 模型回答的措辞、格式与参考答案不同**不影响**得分，只看事实对不对
- 数字允许换算（如「520 亿元」与「52,006,683,587.88 元」算一致）
- 模型多说了参考答案之外的内容，只要不影响核心事实，**不扣分**；但如果是编的数字，判 0

只输出 JSON：{{"score": 2, "理由": "一句话说明"}}

投资者提问：{question}

参考答案：{reference}

模型回答：{prediction}"""


def strip_cite(s):
    """去掉【依据：…】和「」引用标记，只留事实内容。"""
    s = re.sub(r"【[^】]*】", "", s or "")
    return re.sub(r"[「」]", "", s)


def _fmt_fig(f):
    """把 (kind, value) 写成人能读的样子，只用在明细 CSV 里。"""
    kind, v = f
    return f"{v:g}%" if kind == "pct" else f"{v:,.0f} 元"


def numeric_score(prediction, reference, question=""):
    """客观：参考答案的关键数字在模型回答里出现了多少。

    返回 dict；参考答案里没有带量纲的数字时返回 None。

    **为什么要算「净命中率」**：投资者提问经常自带数字
    （「资本承诺 63 亿元，相比去年 64 亿元…」），模型把题干复述一遍就能拿分，
    那不是知识。实测老口径下 174 个"命中"里有 164 个（94%）是年份、
    剩下的大多是照抄题干。所以两个率一起报：命中率看表面成绩，
    净命中率把题干里已出现的数字剔掉，看真答对多少。

    **为什么不复用旧口径**：旧口径拿 numbers_in 直接比字符串，
    而参考答案是繁体「千元」、模型答简体「亿元」，同一个事实对不上。
    key_figures 会把量纲折算干净再比。
    """
    ref = key_figures(strip_cite(reference))
    if not ref:
        return None
    pred = key_figures(strip_cite(prediction))
    hit = match_figures(ref, pred)
    echo = match_figures(hit, key_figures(strip_cite(question)))
    net = hit - echo
    return {
        "n": len(ref),
        "hit": len(hit),
        "echo": len(echo),
        "net": len(net),
        "ratio": len(hit) / len(ref),
        "net_ratio": len(net) / len(ref),
        "missing": "、".join(sorted(_fmt_fig(f) for f in ref - hit)),
    }


def format_score(prediction, doc):
    """格式分（0–3）：董秘口径里可以客观判定的三个特征，每满足一个得 1 分。

    1. 有出处 —— 出现 `【依据：…】`
    2. 引对报告 —— 出处里写的是这道题来源的那份报告，不是别的年份
    3. 有原文引用 —— 出现 `「…」`，这是本项目约定的照抄繁体原文的方式

    **为什么需要这一列**：数字命中四条路线都接近 0（那是共同的短板），
    真正区分它们的是「答得像不像董秘」。微调的成果全落在这三个特征上，
    没有这一列，四格表就是一片零，什么也说明不了。
    """
    s = prediction or ""
    return (int(bool(re.search(r"【依据[:：]", s)))
            + int(bool(doc) and doc in s)
            + int("「" in s and "」" in s))


def judge_one(question, reference, prediction):
    """主观：让模型判 0/1/2。"""
    if not prediction or not prediction.strip():
        return {"score": 0, "理由": "模型没有输出"}
    out = call_llm(J_SYS, J_TPL.format(
        question=question,
        reference=strip_cite(reference)[:1200],
        prediction=strip_cite(prediction)[:1200]), temperature=0.0)
    d = parse_json(out)
    return {"score": int(d.get("score", 0)),
            "理由": str(d.get("理由", "")).strip()}


def load_grid():
    """按 (qa_id, cell) 建索引。

    注意不能只用 qa_id —— 同一条题在原问法和换问法两个格子里各出现一次，
    用 qa_id 当键会把一半吃掉。
    """
    grid = {}
    for line in open(GRID, encoding="utf-8"):
        r = json.loads(line)
        grid[(r["qa_id"], r["cell"])] = r
    return grid


def run(preds_path, label, grid, out_dir):
    preds = [json.loads(l) for l in open(preds_path, encoding="utf-8")]
    print(f"  {label}：{len(preds)} 条预测")

    rows = []
    for p in preds:
        ref = grid.get((p["qa_id"], p["cell"]))
        if not ref:
            continue
        q = ref.get("question_eval") or ref["question"]
        num = numeric_score(p.get("prediction", ""), ref["answer"], q)
        j = judge_one(q, ref["answer"], p.get("prediction", ""))
        rows.append({
            "qa_id": p["qa_id"], "cell": p["cell"], "label": label,
            "问题": q, "参考答案": ref["answer"],
            "模型回答": p.get("prediction", ""),
            "模型判分": j["score"], "判分理由": j["理由"],
            "关键数字命中率": round(num["ratio"], 3) if num else "",
            "净命中率": round(num["net_ratio"], 3) if num else "",
            "命中/总数": f'{num["hit"]}/{num["n"]}' if num else "",
            "其中来自题干": num["echo"] if num else "",
            "格式分": format_score(p.get("prediction", ""), ref.get("doc", "")),
            "遗漏数字": num["missing"] if num else "",
        })

    os.makedirs(out_dir, exist_ok=True)
    detail = os.path.join(out_dir, f"score_{label}.csv")
    import csv
    with open(detail, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows, detail


def summarize(rows, label):
    """按四格汇总。"""
    print(f"\n【{label}】")
    print(f"  {'格子':<22}{'条数':>5}{'模型判分(均)':>13}"
          f"{'正确率(判2)':>12}{'关键数字命中':>14}{'净命中':>9}{'格式分':>8}")
    summary = {}
    for cell in CELLS:
        sub = [r for r in rows if r["cell"] == cell]
        if not sub:
            continue
        n = len(sub)
        avg = sum(r["模型判分"] for r in sub) / n
        acc = sum(1 for r in sub if r["模型判分"] == 2) / n
        ratios = [r["关键数字命中率"] for r in sub
                  if r["关键数字命中率"] != ""]
        nets = [r["净命中率"] for r in sub if r["净命中率"] != ""]
        hit_rate = sum(ratios) / len(ratios) if ratios else 0.0
        net_rate = sum(nets) / len(nets) if nets else 0.0
        fmt = sum(r["格式分"] for r in sub) / n
        summary[cell] = (n, avg, acc, hit_rate, net_rate, fmt)
        print(f"  {CELL_CN[cell]:<22}{n:>5}{avg:>13.2f}"
              f"{acc:>11.0%}{hit_rate:>13.1%}{net_rate:>9.1%}{fmt:>8.2f}")
    return summary


def selftest(grid, out_dir, sample=0):
    """先验证判分器本身：拿参考答案当预测，应该全对；随机配错，应该全错。"""
    os.makedirs(out_dir, exist_ok=True)
    keys = list(grid)
    if sample and sample < len(keys):       # 取样跑，省时间
        step = len(keys) / sample
        keys = [keys[int(i * step)] for i in range(sample)]
    same = os.path.join(out_dir, "_selftest_same.jsonl")
    cross = os.path.join(out_dir, "_selftest_cross.jsonl")
    with open(same, "w", encoding="utf-8") as f:
        for k in keys:
            f.write(json.dumps({"qa_id": k[0], "cell": k[1],
                                "prediction": grid[k]["answer"]},
                               ensure_ascii=False) + "\n")
    with open(cross, "w", encoding="utf-8") as f:
        for j, k in enumerate(keys):         # 拿别的题的答案冒充
            other = grid[keys[(j + 37) % len(keys)]]
            f.write(json.dumps({"qa_id": k[0], "cell": k[1],
                                "prediction": other["answer"]},
                               ensure_ascii=False) + "\n")
    print("自检 1：把参考答案当模型回答（应接近满分）")
    rows, _ = run(same, "selftest_same", grid, out_dir)
    summarize(rows, "自检·参考答案当预测")
    print("\n自检 2：拿别的题答案冒充（应接近 0 分）")
    rows, _ = run(cross, "selftest_cross", grid, out_dir)
    summarize(rows, "自检·张冠李戴")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", action="append", default=[],
                    help="预测文件，可给多个")
    ap.add_argument("--label", action="append", default=[],
                    help="每个预测文件的名字，与 --preds 一一对应")
    ap.add_argument("--out", default=os.path.join(HERE, "scores"))
    ap.add_argument("--selftest", action="store_true",
                    help="先跑自检，确认判分器靠谱")
    ap.add_argument("--selftest-n", type=int, default=20,
                    help="自检取样条数，0=全跑")
    args = ap.parse_args()

    grid = load_grid()
    print(f"评测集：{len(grid)} 条")
    if args.selftest:
        return selftest(grid, args.out, args.selftest_n)
    if not args.preds:
        print("没给 --preds。先跑 --selftest 验证判分器，"
              "或在 Colab 里用 generate.py 生成预测。")
        return 1

    labels = args.label or [os.path.splitext(os.path.basename(p))[0]
                            for p in args.preds]
    all_summary = {}
    for path, label in zip(args.preds, labels):
        rows, detail = run(path, label, grid, args.out)
        all_summary[label] = summarize(rows, label)
        print(f"  明细 → {detail}")

    # 机器可读的汇总 —— report/make_report.py 靠它出图和 PDF，
    # 免得那边去解析 markdown 表格
    dump = {label: {cell: {"条数": v[0], "模型判分": round(v[1], 3),
                           "正确率": round(v[2], 4),
                           "关键数字命中": round(v[3], 4),
                           "净命中": round(v[4], 4),
                           "格式分": round(v[5], 3)}
                    for cell, v in s.items()}
            for label, s in all_summary.items()}
    with open(os.path.join(args.out, "four_grid.json"), "w",
              encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)

    # 汇总成一张对照表
    table = os.path.join(args.out, "four_grid.md")
    with open(table, "w", encoding="utf-8") as f:
        f.write("# 四格对照表\n\n")
        for label, s in all_summary.items():
            f.write(f"## {label}\n\n")
            f.write("| 格子 | 条数 | 模型判分(均) | 正确率 | "
                    "关键数字命中 | 净命中 | 格式分(0–3) |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for cell in CELLS:
                if cell not in s:
                    continue
                n, avg, acc, hit_rate, net_rate, fmt = s[cell]
                f.write(f"| {CELL_CN[cell]} | {n} | {avg:.2f} | "
                        f"{acc:.0%} | {hit_rate:.1%} | {net_rate:.1%} | "
                        f"{fmt:.2f} |\n")
            f.write("\n")
        f.write(FOOTNOTE)
    print(f"\n对照表 → {table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
