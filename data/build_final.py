# -*- coding: utf-8 -*-
"""
第 3 步：应用人工复核结果，冻结训练集 / 测试集。

三条规则，按顺序执行：

  规则 1 · 人工复核
      读 qc_sample.csv 的「人审-处理」和「人审-备注」。
      备注里出现「删除 / 删 / 不宜作测试 / 重复」的一律剔除——
      因为填表时处理列常常漏改，而备注写的是真实判断。

  规则 2 · 测试集不得含引导性提问（硬约束）
      问题里已经报出答案数字的题，在测试集里一律剔除。
      理由：四格对照测的是「模型知不知道这个事实」；题干自带答案的话，
      模型可能是照抄题干而不是真记住了，结果会虚高。
      训练集里**不**剔除——现实中投资者就是这么问的。

  规则 3 · 去重
      问题完全相同或高度雷同的，只留一条。

用法：
    python build_final.py --dry-run     # 只报告会删哪些，不写文件
    python build_final.py

输出：
    final_train.jsonl   训练集（喂给 LoRA）
    final_test.jsonl    测试集（冻结，绝不进训练）
    final_report.md     这次筛选的完整记录
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import is_leading, is_half_refusal, read_csv_any   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DOCS = {"理想汽车_2025年报", "理想汽车_2026中报"}
DROP_WORDS = ("删除", "删掉", "不宜作测试", "重复", "改写")


def load_human_review():
    """读人工复核表，返回 {qa_id: (是否剔除, 原因)}。"""
    path = os.path.join(HERE, "qc_sample.csv")
    if not os.path.exists(path):
        print("没找到 qc_sample.csv，跳过人工复核")
        return {}, None
    rows, enc = read_csv_any(path)
    verdict = {}
    for r in rows:
        qid = r.get("qa_id", "").strip()
        if not qid:
            continue
        action = (r.get("人审-处理") or "").strip()
        note = (r.get("人审-备注") or "").strip()
        if action in ("删除", "改写"):
            verdict[qid] = (True, f"人工复核：{action}（{note or '无备注'}）")
        elif any(w in note for w in DROP_WORDS):
            verdict[qid] = (True, f"人工复核备注：{note}")
    return verdict, enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(os.path.join(HERE, "qa.jsonl"),
                                        encoding="utf-8")]
    kept = [r for r in recs if r["keep"]]

    verdict, enc = load_human_review()
    print(f"人工复核表编码：{enc}，读到 {len(verdict)} 条剔除意见")

    train, test, log = [], [], []

    # ---- 规则 1 + 2
    seen_q = {}
    for r in kept:
        qid, is_test = r["qa_id"], r["doc"] in TEST_DOCS
        q_norm = re.sub(r"[^\w]", "", r["question"])

        if qid in verdict:                                   # 规则 1
            log.append((qid, "剔除", verdict[qid][1]))
            continue
        if is_test and is_leading(r["question"], r["answer"]):  # 规则 2
            log.append((qid, "剔除",
                        "测试集不得含引导性提问（题干已报出答案数字）"))
            continue
        if q_norm in seen_q:                                 # 规则 3
            log.append((qid, "剔除", f"与 {seen_q[q_norm]} 重复"))
            continue
        seen_q[q_norm] = qid

        (test if is_test else train).append(r)
        log.append((qid, "保留", ""))

    for r in train + test:
        r["leading"] = is_leading(r["question"], r["answer"])
        r["half_refusal"] = is_half_refusal(r["answer"])

    # ---- 报告
    n_lead_tr = sum(1 for r in train if r["leading"])
    n_half_tr = sum(1 for r in train if r["half_refusal"])
    print(f"\n训练集 {len(train)} 条  （其中引导性提问 {n_lead_tr} 条，"
          f"半拒绝 {n_half_tr} 条 —— 训练集保留，不影响）")
    print(f"测试集 {len(test)} 条  （引导性提问 0 条，已强制剔除）")

    print("\n测试集按报告分布：")
    for d, n in Counter(r["doc"] for r in test).most_common():
        print(f"   {d:<20}{n:>3} 条")

    print("\n本次剔除明细：")
    for qid, act, why in log:
        if act == "剔除":
            print(f"   {qid}\n      {why}")

    if args.dry_run:
        print("\n（dry-run，未写文件）")
        return 0

    for name, data in (("final_train.jsonl", train), ("final_test.jsonl", test)):
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps({
                    "qa_id": r["qa_id"],
                    "doc": r["doc"], "page": r["page_start"],
                    "section": r["section"],
                    "question": r["question"], "answer": r["answer"],
                    "has_table": r["has_table"],
                    "leading": r["leading"],
                    "half_refusal": r["half_refusal"],
                    # 留原文：生成换问法时要参照它确认答案没变，
                    # 做 RAG 对照时也要用它当检索语料
                    "source_text": r.get("source_text", ""),
                }, ensure_ascii=False) + "\n")

    with open(os.path.join(HERE, "final_report.md"), "w", encoding="utf-8") as f:
        f.write("# 训练 / 测试集冻结记录\n\n")
        f.write("| | 条数 |\n|---|---|\n")
        f.write(f"| 生成问答对 | {len(recs)} |\n")
        f.write(f"| 自动筛选后 | {len(kept)} |\n")
        f.write(f"| **训练集** | **{len(train)}** |\n")
        f.write(f"| **测试集** | **{len(test)}** |\n\n")
        f.write("## 剔除规则\n\n")
        f.write("1. **人工复核** —— 读 `qc_sample.csv` 的备注与处理列\n")
        f.write("2. **测试集不得含引导性提问** —— 题干已报出答案数字的题，\n"
                "   会让四格对照虚高（模型可能照抄题干而非真记住）。训练集不剔除\n")
        f.write("3. **去重** —— 问题雷同的只留一条\n\n")
        f.write("## 本次剔除明细\n\n| qa_id | 原因 |\n|---|---|\n")
        for qid, act, why in log:
            if act == "剔除":
                f.write(f"| {qid} | {why} |\n")
        f.write(f"\n## 测试集构成\n\n| 报告 | 条数 |\n|---|---|\n")
        for d, n in Counter(r["doc"] for r in test).most_common():
            f.write(f"| {d} | {n} |\n")

    print(f"\n写出 final_train.jsonl（{len(train)}）"
          f"、final_test.jsonl（{len(test)}）、final_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
