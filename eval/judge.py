# -*- coding: utf-8 -*-
"""
第 5 步（判分器）：给四格评测集判分，出对照表。

**格式 / 事实 / 引用三个维度分开评**（2026-10-12 按批改意见拆分）：

| 维度 | 分值 | 评什么 | 裁判 |
|---|---|---|---|
| 格式 | 0–2 | ① 有【依据：…】 ② 有「…」引用 —— 只判「像不像」，不判真假 | 规则 |
| 事实 | 无独立分 | 净命中率（关键数字）+ 第三方模型判 0/1/2 | 规则 + 模型 |
| 引用 | 0–3 | ① 依据里报告名对 ② 页码对（±1） ③ 所有「」引文能在原文逐字溯源 | 规则 |

**为什么必须把引用从格式里拆出来**：微调模型会在编造的数字外面套上
看起来毫无破绽的「原文引用」和【依据】页码 —— 正文写 207.8%、引文写 203.1%，
格式分照样满分。原来「引对报告」混在格式分里，编造的引文没人查。
现在引文逐字回原文查（`data/quote_check.py`），编造直接扣引用分。

评测集 `eval_grid.jsonl` 有四个格子：
    trained_original     trained_rephrased
    untrained_original   untrained_rephrased

用法：
    python judge.py --preds preds/base.jsonl --preds preds/sft.jsonl \
                    --label 基座 --label 微调后
    python judge.py --selftest                  # 先验证判分器本身靠不靠谱
    python judge.py --preds x.jsonl --no-llm    # 只跑客观+引用，不花 API

生成结果文件的格式（每行一条）：
    {"qa_id": "...", "cell": "...", "prediction": "模型回答的原文"}
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
from build_qa import call_llm, parse_json                      # noqa: E402
from common import key_figures, match_figures                  # noqa: E402
from quote_check import (Corpus, extract_quotes, load_pages,   # noqa: E402
                         parse_citations)

GRID = os.path.join(HERE, "..", "data", "eval_grid.jsonl")
PAGES = os.path.join(HERE, "..", "data", "pages.jsonl")
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
| **格式分(0–2)** | ① 有 `【依据：…】` ② 有 `「…」` 原文引用，各 1 分。**只判有没有、像不像，不判真假** |
| **引用分(0–3)** | ① 【依据】里的报告名 == 题目来源报告 ② 依据页码与来源页相差 ≤1 ③ 所有 `「」` 引文都能在 1,449 页财报里**逐字**查到（只容忍空白/标点差异；写成简体算口径缺陷，不算逐字），各 1 分 |

**三个维度为什么要拆开**：微调模型会把编造的数字包进漂亮的 `「」` 引用和
`【依据】` 页码里 —— 格式分满分，引文却一个字都查不到。原来两者混在一列，
这种错误反而得了高分。拆开后：格式分只量「口径」，
引用分只量「引文真不真」，事实分只量「数字对不对」。**谁骗不了谁。**

「引用分 ③」把简体引文判为 0（口径要求照抄繁体一字不改），但明细表把
`引文仅繁简` 单独记一列 —— 那是口径缺陷，不是编造，两件事不能混。

**为什么要单独看「净命中」**：投资者提问常自带数字
（「资本承诺 63 亿元，相比去年 64 亿元…」），模型把题干复述一遍就能拿分，
那不是知识。两个数差距越大，说明成绩里"照抄题干"的成分越多。

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
    """事实（客观）：参考答案的关键数字在模型回答里出现了多少。

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


def format_score(prediction):
    """格式分（0–2）：董秘口径的外壳，只看有没有、不看真假。

    1. 有出处 —— 出现 `【依据：…】`
    2. 有原文引用 —— 出现非空 `「…」`（纯话术「材料未涉及」不算）

    **为什么不再判「引对报告」**：那是引文的真假问题，已归到引用分 ①。
    格式分只管"答得像不像董秘"——这是微调真正学到的东西，
    也是四条路线唯一天然拉开差距的一列。
    """
    s = prediction or ""
    return (int(bool(re.search(r"【依据[:：]", s)))
            + int(bool(extract_quotes(s))))


def citation_score(prediction, doc, page, corpus, min_seg=6):
    """引用分（0–3）：引文是不是真的，逐条回原文查。

    1. **报告名对** —— 【依据】块里的报告名全部等于题目来源报告（至少写了一个）
    2. **页码对** —— 依据页码与来源页相差 ≤1（至少写了一个页码）
    3. **引文真** —— 所有「」引文都能在 1,449 页原文里逐字查到
       （只容忍空白/标点差异；写成简体算口径缺陷，不算逐字，单独记列）

    同时还出几个诊断数字：引文落在被引页 / 落在题目来源页 / 未命中示例 ——
    「格式满分、引文全是编的」这种病例靠它们定位。
    """
    cites = parse_citations(prediction)
    docs = [d for c in cites for d in c["docs"]]
    pages = [p for c in cites for p in c["pages"]]
    c1 = int(bool(docs) and all(d == doc for d in docs))
    c2 = int(bool(pages) and any(a - 1 <= page <= b + 1 for a, b in pages))

    quotes = extract_quotes(prediction)
    checks = [corpus.check_quote(q, min_seg=min_seg) for q in quotes]
    judged = [c for c in checks if c["判定"] != "空"]
    n_verbatim = sum(1 for c in judged if c["判定"] in ("逐字", "去标点"))
    n_zh = sum(1 for c in judged if c["判定"] == "繁简")
    n_miss = sum(1 for c in judged if c["判定"] == "未命中")
    c3 = int(bool(judged) and n_verbatim == len(judged))

    cited_pages = {(doc, p) for c in cites for a, b in c["pages"]
                   for p in range(a, b + 1)}
    n_on_cited = n_on_src = 0
    miss_example = ""
    for c in judged:
        hit = {corpus.meta[i] for seg in c["段"] for i in seg["页"]}
        if hit & cited_pages:
            n_on_cited += 1
        if (doc, page) in hit:
            n_on_src += 1
        if c["判定"] == "未命中" and not miss_example:
            miss_example = c["段"][0]["段"][:60] if c["段"] else ""

    return {
        "引用分": c1 + c2 + c3,
        "报告名对": c1,
        "页码对": c2,
        "引文真": c3,
        "依据": "；".join(c["raw"] for c in cites)[:120],
        "引文数": len(judged),
        "引文逐字": n_verbatim,
        "引文仅繁简": n_zh,
        "引文未命中": n_miss,
        "引文落在被引页": n_on_cited,
        "引文落在题目页": n_on_src,
        "未命中引文示例": miss_example,
    }


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


def load_corpus():
    """全量 1,449 页 —— 引文溯源就在这上面查。"""
    pages = load_pages(PAGES)
    return Corpus(pages)


def run(preds_path, label, grid, out_dir, corpus, workers=8, use_llm=True):
    preds = [json.loads(l) for l in open(preds_path, encoding="utf-8")]
    print(f"  {label}：{len(preds)} 条预测")

    # 键必须是 (qa_id, cell)：同一条题在原/换问法两格各出现一次
    jobs = []
    for p in preds:
        ref = grid.get((p["qa_id"], p["cell"]))
        if not ref:
            continue
        jobs.append((p["qa_id"], p["cell"], ref, p.get("prediction", "")))

    # 主观判分跑 API，一条几秒——串行 140 条要十分钟，并发一下
    if use_llm:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            verdicts = list(ex.map(
                lambda j: judge_one(j[2].get("question_eval")
                                    or j[2]["question"], j[2]["answer"], j[3]),
                jobs))
    else:
        verdicts = [{"score": "", "理由": ""}] * len(jobs)

    rows = []
    for (qa_id, cell, ref, pred), j in zip(jobs, verdicts):
        q = ref.get("question_eval") or ref["question"]
        num = numeric_score(pred, ref["answer"], q)
        cit = citation_score(pred, ref.get("doc", ""), ref.get("page", 0),
                             corpus)
        rows.append({
            "qa_id": qa_id, "cell": cell, "label": label,
            "问题": q, "参考答案": ref["answer"], "模型回答": pred,
            "模型判分": j["score"], "判分理由": j["理由"],
            "关键数字命中率": round(num["ratio"], 3) if num else "",
            "净命中率": round(num["net_ratio"], 3) if num else "",
            "命中/总数": f'{num["hit"]}/{num["n"]}' if num else "",
            "其中来自题干": num["echo"] if num else "",
            "格式分(0–2)": format_score(pred),
            "引用分(0–3)": cit["引用分"],
            "依据": cit["依据"], "报告名对": cit["报告名对"],
            "页码对": cit["页码对"], "引文真": cit["引文真"],
            "引文数": cit["引文数"], "引文逐字": cit["引文逐字"],
            "引文仅繁简": cit["引文仅繁简"], "引文未命中": cit["引文未命中"],
            "引文落在被引页": cit["引文落在被引页"],
            "引文落在题目页": cit["引文落在题目页"],
            "未命中引文示例": cit["未命中引文示例"],
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
    """按四格汇总。返回 (每格汇总, 诊断)。"""
    print(f"\n【{label}】")
    print(f"  {'格子':<22}{'条数':>5}{'模型判分(均)':>13}"
          f"{'正确率(判2)':>12}{'关键数字命中':>14}{'净命中':>9}"
          f"{'格式(0–2)':>10}{'引用(0–3)':>10}")
    summary = {}
    for cell in CELLS:
        sub = [r for r in rows if r["cell"] == cell]
        if not sub:
            continue
        n = len(sub)
        scored = [r["模型判分"] for r in sub if r["模型判分"] != ""]
        avg = sum(scored) / len(scored) if scored else float("nan")
        acc = (sum(1 for r in scored if r == 2) / len(scored)
               if scored else float("nan"))
        ratios = [r["关键数字命中率"] for r in sub
                  if r["关键数字命中率"] != ""]
        nets = [r["净命中率"] for r in sub if r["净命中率"] != ""]
        hit_rate = sum(ratios) / len(ratios) if ratios else 0.0
        net_rate = sum(nets) / len(nets) if nets else 0.0
        fmt = sum(r["格式分(0–2)"] for r in sub) / n
        cit = sum(r["引用分(0–3)"] for r in sub) / n
        summary[cell] = (n, avg, acc, hit_rate, net_rate, fmt, cit)
        avg_s = f"{avg:.2f}" if scored else "—"
        acc_s = f"{acc:.0%}" if scored else "—"
        print(f"  {CELL_CN[cell]:<22}{n:>5}{avg_s:>13}"
              f"{acc_s:>12}{hit_rate:>13.1%}{net_rate:>9.1%}"
              f"{fmt:>10.2f}{cit:>10.2f}")

    diag = diagnose(rows)
    if diag["有引文条数"]:
        print(f"  ── 引用真实性 ── 有「」引文的 {diag['有引文条数']} 条 /"
              f" 引文共 {diag['引文总数']} 句：逐字 {diag['引文逐字']}、"
              f"仅繁简 {diag['引文仅繁简']}、"
              f"**原文查无 {diag['引文未命中']}**（编造或改写）")
        print(f"  ── 格式满分但引文编造 ── 格式满分 {diag['格式满分条数']} 条，"
              f"其中 {diag['格式满分且引文编造']} 条引文在原文里查无")
    return summary, diag


def diagnose(rows):
    """汇总引用维度最说明问题的几个数：编造、口径缺陷（简体）、页码错位。"""
    with_q = [r for r in rows if r["引文数"] > 0]
    fmt_full = [r for r in rows if r["格式分(0–2)"] == 2]
    return {
        "有引文条数": len(with_q),
        "引文总数": sum(r["引文数"] for r in rows),
        "引文逐字": sum(r["引文逐字"] for r in rows),
        "引文仅繁简": sum(r["引文仅繁简"] for r in rows),
        "引文未命中": sum(r["引文未命中"] for r in rows),
        "引文落在被引页": sum(r["引文落在被引页"] for r in rows),
        "引文落在题目页": sum(r["引文落在题目页"] for r in rows),
        "格式满分条数": len(fmt_full),
        "格式满分且引文编造": sum(1 for r in fmt_full
                                  if r["引文未命中"] > 0),
        "有引文但无一逐字": sum(1 for r in with_q
                                if r["引文逐字"] == 0),
        "引用分0但有引文": sum(1 for r in with_q if r["引用分(0–3)"] == 0),
    }


def selftest(grid, out_dir, corpus, sample=0, use_llm=True):
    """先验证判分器本身：拿参考答案当预测，应该全对；随机配错，应该全错。

    顺带把**参考答案自己的引用分**报出来 —— 语料里就有多少引文不是逐字的
    （表格改写成叙述之类），那是这个指标的天花板，先亮出来再谈模型分。
    """
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
    print("自检 1：把参考答案当模型回答（数字应接近满分，引用分是天花板）")
    rows, _ = run(same, "selftest_same", grid, out_dir, corpus, use_llm=use_llm)
    summarize(rows, "自检·参考答案当预测")
    print("\n自检 2：拿别的题答案冒充（应接近 0 分）")
    rows, _ = run(cross, "selftest_cross", grid, out_dir, corpus,
                  use_llm=use_llm)
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
    ap.add_argument("--workers", type=int, default=8,
                    help="主观判分的并发数（本地跑从 8 起）")
    ap.add_argument("--no-llm", action="store_true",
                    help="跳过主观判分，只算客观+引用（不花 API，调试用）")
    args = ap.parse_args()

    grid = load_grid()
    print(f"评测集：{len(grid)} 条")
    print("装载语料（引文溯源用）…", flush=True)
    corpus = load_corpus()
    print(f"语料：{len(corpus.meta)} 页")
    if args.selftest:
        return selftest(grid, args.out, corpus, args.selftest_n,
                        use_llm=not args.no_llm)
    if not args.preds:
        print("没给 --preds。先跑 --selftest 验证判分器，"
              "或在 Colab 里用 generate.py 生成预测。")
        return 1

    labels = args.label or [os.path.splitext(os.path.basename(p))[0]
                            for p in args.preds]
    all_summary, all_diag = {}, {}
    for path, label in zip(args.preds, labels):
        rows, detail = run(path, label, grid, args.out, corpus,
                           workers=args.workers, use_llm=not args.no_llm)
        all_summary[label], all_diag[label] = summarize(rows, label)
        print(f"  明细 → {detail}")

    # 机器可读的汇总 —— report/make_report.py 靠它出图和 PDF，
    # 免得那边去解析 markdown 表格
    dump = {label: {cell: {"条数": v[0], "模型判分": _r(v[1]),
                           "正确率": _r(v[2]),
                           "关键数字命中": round(v[3], 4),
                           "净命中": round(v[4], 4),
                           "格式分": round(v[5], 3),
                           "引用分": round(v[6], 3)}
                    for cell, v in s.items()}
            for label, s in all_summary.items()}
    with open(os.path.join(args.out, "four_grid.json"), "w",
              encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)

    # 引用维度的诊断单独一个文件：编造率、口径缺陷、页码错位
    with open(os.path.join(args.out, "four_grid_diag.json"), "w",
              encoding="utf-8") as f:
        json.dump(all_diag, f, ensure_ascii=False, indent=2)

    # 汇总成一张对照表
    table = os.path.join(args.out, "four_grid.md")
    with open(table, "w", encoding="utf-8") as f:
        f.write("# 四格对照表\n\n")
        for label, s in all_summary.items():
            f.write(f"## {label}\n\n")
            f.write("| 格子 | 条数 | 模型判分(均) | 正确率 | "
                    "关键数字命中 | 净命中 | 格式分(0–2) | 引用分(0–3) |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for cell in CELLS:
                if cell not in s:
                    continue
                n, avg, acc, hit_rate, net_rate, fmt, cit = s[cell]
                avg_s = f"{avg:.2f}" if avg == avg else "—"   # NaN 自己不等于自己
                acc_s = f"{acc:.0%}" if acc == acc else "—"
                f.write(f"| {CELL_CN[cell]} | {n} | {avg_s} | "
                        f"{acc_s} | {hit_rate:.1%} | {net_rate:.1%} | "
                        f"{fmt:.2f} | {cit:.2f} |\n")
            f.write("\n")
            d = all_diag.get(label)
            if d and d["有引文条数"]:
                f.write(f"**引用真实性**（{label}）：有「」引文 {d['有引文条数']} 条、"
                        f"引文共 {d['引文总数']} 句 —— 逐字 {d['引文逐字']}、"
                        f"仅繁简差异 {d['引文仅繁简']}、"
                        f"**原文里查无 {d['引文未命中']}**（编造或改写）。\n\n"
                        f"格式满分（有【依据】有引用）共 {d['格式满分条数']} 条，"
                        f"其中 {d['格式满分且引文编造']} 条的引文在原文里查无"
                        f"——**这就是「只看格式分会被骗」的直接证据**。\n\n")
        f.write(FOOTNOTE)
    print(f"\n对照表 → {table}")
    return 0


def _r(v):
    """NaN（--no-llm 跑出来的）写回 JSON 时换成 null。"""
    return None if v != v else round(v, 3)


if __name__ == "__main__":
    sys.exit(main())
