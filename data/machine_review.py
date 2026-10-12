# -*- coding: utf-8 -*-
"""机器复核：给「自然」这一维度补一层独立复核，并把人工抽检的结论回填进总表。

**为什么要这个脚本**：批改要求"人工抽检"在提交物里看得见。现在的状态是——
60 条抽检表里 `人审-有据 / 人审-数字 / 人审-处理` 三列人工填好了，
但「自然」这列**没审过**；而 342 条总表 `qc_review.csv` 的人审列
设计上是留空待填的，打开容易被误判成"没做抽检"。

处理办法（不冒充人审，各是各的）：

  1. **机器复核**：用与原始质检**不同的提示词**（独立复核员口径），
     对全量 342 条重判「自然」（1=像真实董秘答投资者问，0=生硬/模板腔/问得琐碎）。
     结论进 `机器复核-自然` 列 —— 列名写清楚了，谁也不会把它当人审。
  2. **回填**：把抽检表里**人工已填**的三列（有据/数字/处理/备注）
     按 qa_id 回填进 342 条总表，并加「抽检」列标出哪 60 条被抽到。
     人审结论还是人工的结论，只是让总表自己说得清楚。
  3. `人审-自然` 列**保持为空** —— 那是留给本人的，机器不做主。

用法：
    python machine_review.py            # 复核 + 回填（走 API，有磁盘缓存）
    python machine_review.py --no-llm   # 不调 API，只用已有复核结果回填
输出：
    machine_review.csv   342 条复核明细（证据在 data/cache/，可离线核）
    qc_sample.csv        60 条抽检表，追加「机器复核-自然 / 理由」两列
    qc_review.csv        342 条总表，追加抽检标记与复核列、回填人审三列
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_qa as B                                              # noqa: E402
from common import read_csv_any                                   # noqa: E402

# 独立复核口径：与 build_qa 的 QC_TPL 不同的问题、不同的角度，
# 不是把原质检再问一遍 —— 那只会把原判定的噪声复制一份。
REV_SYS = "你是独立第三方数据复核员。只输出 JSON，不要任何解释。"
REV_TPL = """你是独立第三方数据复核员，正在抽查一批"投资者问 → 董秘答"的合成数据，
判断回答**自不自然**（不看数字对错，那是别的环节）。

自然 = 1：问题是投资者真会关心的（业绩、交付、收入利润、毛利率、渠道、
研发、战略、风险），回答给出了实质信息。
自然 = 0：满足三者之一 ——
  ① 问的是与投资判断无关的琐碎细节（人事制度、培训体系、员工福利、
     各职能人数这类）；
  ② 空泛的模板腔，或整条只在拒绝作答、没有任何实质信息；
  ③ 明显答非所问。

**注意：本项目的口径要求回答先给结论、再用「」引用繁体原文佐证、
末尾标注【依据：《…》第 N 页】。这是规定的格式 ——
「引用原文」「标注依据页」本身不算生硬，不要因此判 0。**

只输出 JSON：{{"自然": 1, "理由": "一句话说明"}}

投资者提问：{question}

董秘回答：{answer}"""


def review_one(rec):
    out = B.call_llm(REV_SYS, REV_TPL.format(question=rec["question"],
                                             answer=rec["answer"]),
                     temperature=0.0)
    d = B.parse_json(out)
    return int(d.get("自然", 0)), str(d.get("理由", "")).strip()


def load_review_cache(path):
    """已有复核结果（qa_id → (自然, 理由)），离线回填时用。"""
    if not os.path.exists(path):
        return {}
    rows, _ = read_csv_any(path)
    return {r["qa_id"]: (r["机器复核-自然"], r["机器复核-理由"]) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true",
                    help="不调 API，直接用 machine_review.csv 里的既有结果回填")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    qa = {r["qa_id"]: r for r in
          (json.loads(l) for l in open(os.path.join(HERE, "qa.jsonl"),
                                       encoding="utf-8"))}
    order = [json.loads(l)["qa_id"] for l in
             open(os.path.join(HERE, "qa.jsonl"), encoding="utf-8")]
    out_csv = os.path.join(HERE, "machine_review.csv")

    if args.no_llm:
        cache = load_review_cache(out_csv)
        print(f"离线回填：已有复核结果 {len(cache)} 条")
        verdicts = {k: cache.get(k, ("", "")) for k in order}
    else:
        print(f"独立复核 {len(order)} 条（并发 {args.workers}）…")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(lambda q: review_one(qa[q]), order))
        verdicts = dict(zip(order, results))

        with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["qa_id", "报告", "页码", "问题", "董秘回答",
                        "原质检-自然(模型)", "机器复核-自然", "机器复核-理由",
                        "与原质检一致"])
            for qid in order:
                r, reason = verdicts[qid]
                old = int(qa[qid]["qc"].get("自然", 0))
                w.writerow([qid, qa[qid]["doc"], qa[qid]["page_start"],
                            qa[qid]["question"], qa[qid]["answer"], old,
                            r, reason, int(int(r) == old)])
        print(f"复核明细 → {out_csv}")

    n1 = sum(1 for r, _ in verdicts.values() if str(r) == "1")
    n0 = sum(1 for r, _ in verdicts.values() if str(r) == "0")
    disagree = [q for q in order
                if str(verdicts[q][0]) in ("0", "1")
                and int(verdicts[q][0]) != int(qa[q]["qc"].get("自然", 0))]
    print(f"独立复核：自然 {n1} 条 / 生硬 {n0} 条；"
          f"与原质检（模型）不一致 **{len(disagree)}** 条")

    # ---- 1) 抽检表追加两列（人工填的三列一字不动）
    sample_path = os.path.join(HERE, "qc_sample.csv")
    sample, _ = read_csv_any(sample_path)
    for r in sample:
        r["机器复核-自然"], r["机器复核-理由"] = verdicts.get(
            r["qa_id"], ("", ""))
    with open(sample_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sample[0].keys()))
        w.writeheader()
        w.writerows(sample)
    print(f"抽检表追加「机器复核-自然/理由」→ {sample_path}"
          f"（{len(sample)} 条，人工列未动）")

    # ---- 2) 总表：标抽检 + 回填人审三列 + 追加复核列
    review, _ = read_csv_any(os.path.join(HERE, "qc_review.csv"))
    by_id = {r["qa_id"]: r for r in sample}
    n_back = 0
    for r in review:
        s = by_id.get(r["qa_id"])
        r["抽检"] = "是" if s else ""
        r["人审-数字"] = s.get("人审-数字", "") if s else ""
        for col in ("人审-有据", "人审-处理", "人审-备注"):
            if s and str(s.get(col, "")).strip():
                r[col] = s[col]
                n_back += 1
        r["机器复核-自然"], r["机器复核-理由"] = verdicts.get(
            r["qa_id"], ("", ""))
    with open(os.path.join(HERE, "qc_review.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(review[0].keys()))
        w.writeheader()
        w.writerows(review)
    print(f"总表 → {os.path.join(HERE, 'qc_review.csv')}："
          f"标出抽检 {len(sample)} 条，回填人工结论 {n_back} 格，"
          f"`人审-自然` 保持为空（留给本人）")

    if disagree:
        print("\n与原质检不一致的例子（前 5 条）：")
        for q in disagree[:5]:
            print(f"  · {q}：原质检自然={qa[q]['qc']['自然']} → "
                  f"复核自然={verdicts[q][0]}｜{verdicts[q][1][:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
