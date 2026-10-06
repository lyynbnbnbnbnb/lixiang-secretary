# -*- coding: utf-8 -*-
"""
第 3 步（续）：生成换问法，拼出四格对照的评测集。

四格设计：

                    原问法                  换问法
    训练过的报告     cell 1 (trained_original)   cell 2 (trained_rephrased)
    没训练过的报告   cell 3 (untrained_original) cell 4 (untrained_rephrased)

  · cell 1 / 2 用训练集里抽出的探针题。**它们本来就进训练集**——
    这是刻意的：原问法考"记住没有"，换问法考"是背题面还是记住了事实"。
  · cell 3 / 4 用测试池（2025 年报 + 2026 中报），绝不进训练集。

换问法只换问法、不换事实：仍然问同一件事，但换措辞和句式，
且**不得比原问题泄露更多答案数字**（生成后再自动查一遍，泄露的就重试）。

用法：
    python build_paraphrase.py --probe 30      # 训练池抽 30 条做探针
    python build_paraphrase.py --dry-run

输出：
    eval_grid.jsonl   四格评测集（每行带 cell 标签）
    final_test.jsonl  同步写入换问法列（方便人工过一眼）
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import is_leading, read_csv_any                    # noqa: E402
from build_qa import call_llm, parse_json                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

P_SYS = "你在改写投资者提问的措辞。只输出 JSON，不要任何解释。"

P_TPL = """下面是一条投资者对理想汽车财报的提问，以及它对应的原文段落。

请把它**换个问法**再问一遍。要求：

1. **问的是同一件事**，答案不变，原文段落仍然能回答它
2. **换措辞、换句式**：改变用词、语序、提问角度。比如
   「是多少」→「具体数字是多少」「达到什么水平」「表现如何」
   口语化的可以改正式，正式的也可以改口语
3. **不得比原问题泄露更多答案数字**——原问题里没有的数字，改写后也不能出现
4. 仍然是投资者会问的话，一个问句，用简体中文
5. 长度控制在 40 字以内

只输出 JSON：{{"question": "改写后的问题"}}

原文段落（仅供你确认答案没变，不要在问题里复述它）：
{text}

原问题：{question}"""


def rephrase(rec, tries=3):
    """生成换问法，并确保它没比原问题泄露更多数字。"""
    for _ in range(tries):
        out = call_llm(P_SYS, P_TPL.format(
            text=(rec.get("source_text") or "")[:1200],
            question=rec["question"]), temperature=0.9)
        q = str(parse_json(out).get("question", "")).strip()
        if not q or q == rec["question"]:
            continue
        # 改写后不能比原问题更泄露答案
        if is_leading(q, rec["answer"]) and not is_leading(rec["question"],
                                                           rec["answer"]):
            continue
        return q
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=30,
                    help="训练池抽多少条做探针（cell 1/2）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 训练池的探针题：优先挑非引导性的，换问法才有意义
    train = [json.loads(l) for l in
             open(os.path.join(HERE, "final_train.jsonl"), encoding="utf-8")]
    test = [json.loads(l) for l in
            open(os.path.join(HERE, "final_test.jsonl"), encoding="utf-8")]

    clean = [r for r in train if not r["leading"]]
    probes = clean[:args.probe]
    print(f"训练池 {len(train)} 条，其中非引导性 {len(clean)} 条，"
          f"取前 {len(probes)} 条做探针")
    print(f"测试池 {len(test)} 条（全部要做换问法）")
    print(f"共需生成 {len(probes)+len(test)} 条换问法\n")

    if args.dry_run:
        for r in probes[:5]:
            print(f"  [探针] {r['question']}")
        return 0

    grid = []

    for i, r in enumerate(probes, start=1):
        q2 = rephrase(r)
        print(f"  探针 {i}/{len(probes)}  {r['question'][:34]}… → {q2[:34] or '（失败）'}")
        if q2:
            grid.append({**r, "cell": "trained_original",
                         "question_eval": r["question"]})
            grid.append({**r, "cell": "trained_rephrased", "question_eval": q2})

    for i, r in enumerate(test, start=1):
        # 测试集记录里没有原文，用答案+问题改写即可
        q2 = rephrase(r)
        print(f"  测试 {i}/{len(test)}  {r['question'][:34]}… → {q2[:34] or '（失败）'}")
        grid.append({**r, "cell": "untrained_original",
                     "question_eval": r["question"]})
        if q2:
            grid.append({**r, "cell": "untrained_rephrased",
                         "question_eval": q2})

    with open(os.path.join(HERE, "eval_grid.jsonl"), "w",
              encoding="utf-8") as f:
        for r in grid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    c = Counter(r["cell"] for r in grid)
    print("\n四格评测集：")
    for k in ("trained_original", "trained_rephrased",
              "untrained_original", "untrained_rephrased"):
        print(f"  {k:<22}{c[k]:>4} 条")
    print(f"\n写出 eval_grid.jsonl（共 {len(grid)} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
