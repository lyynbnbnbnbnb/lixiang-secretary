# -*- coding: utf-8 -*-
"""来源页审计：把「每条问答都能追回原文哪一页」从口号变成一张可核对的表。

批改意见要求补"来源页"。数据里其实每条都写了 `doc / page_start / section`，
回答里也带 `【依据：《…》第 N 页】`，但**没人逐条核对过它们是不是真的对得上** ——
"写是写了"和"核过了"是两回事。本脚本把后者做出来：

对 `qa.jsonl` 全部 342 条 + `eval_grid.jsonl` 全部 140 条，逐条检查三件事：

  1. **报告名** —— 回答里【依据】写的报告名是否与记录的来源报告逐字一致
  2. **页码** —— 依据页码与来源页（±1）是否对得上
  3. **引文** —— 每个 `「」` 引文能不能在**来源页原文**里逐字找到
     （只在全库找到、来源页找不到的会单独标出来：那说明引文真但出处追错页）

和判分器共用 `data/quote_check.py` 的同一套溯源逻辑，两边永远不会各说各话。

用法：
    python audit_sources.py                 # 全量 342 + 140
    python audit_sources.py --limit 10      # 抽查前 10 条
输出：
    source_audit.csv    逐条审计明细（两批数据都在，用「来源」列区分）
    source_audit.md     摘要表（数字直接写进笔记本「第 0 步」和 README）
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from quote_check import (Corpus, extract_quotes, load_pages,  # noqa: E402
                         parse_citations)


def audit_one(rec, corpus, source, page_lo, page_hi):
    """一条问答的溯源审计。返回明细 dict。"""
    ans = rec.get("answer", "")
    doc = rec.get("doc", "")
    cites = parse_citations(ans)
    docs = [d for c in cites for d in c["docs"]]
    pages = [p for c in cites for p in c["pages"]]

    doc_ok = int(bool(docs) and all(d == doc for d in docs))
    # 依据页码与来源页比，两边都允许 ±1；引用区间与来源区间有交集即算对上
    page_ok = int(bool(pages) and any(
        not (b < page_lo - 1 or a > page_hi + 1) for a, b in pages))

    rows, n_v = [], 0
    n_zh = n_miss = n_on_src = 0
    miss_example = ""
    for q in extract_quotes(ans):
        res = corpus.check_quote(q)
        if res["判定"] == "空":
            continue
        rows.append(res)
        hit = {corpus.meta[i] for seg in res["段"] for i in seg["页"]}
        on_src = any(h[0] == doc and page_lo <= h[1] <= page_hi for h in hit)
        n_on_src += int(on_src)
        if res["判定"] in ("逐字", "去标点"):
            n_v += 1
        elif res["判定"] == "繁简":
            n_zh += 1
        else:
            n_miss += 1
            if not miss_example:
                miss_example = res["段"][0]["段"][:60] if res["段"] else ""

    n_q = len(rows)
    return {
        "来源": source,
        "qa_id": rec.get("qa_id", ""),
        "报告": doc,
        "页码": f"{page_lo}" if page_lo == page_hi else f"{page_lo}-{page_hi}",
        "依据原文": "；".join(c["raw"] for c in cites)[:120],
        "报告名一致": doc_ok,
        "页码一致": page_ok,
        "引文数": n_q,
        "引文逐字": n_v,
        "引文仅繁简": n_zh,
        "引文查无": n_miss,
        "引文都在来源页": (int(n_on_src == n_q) if n_q else ""),
        "查无引文示例": miss_example,
    }


def summarize(rows, name):
    n = len(rows)
    with_q = [r for r in rows if r["引文数"]]
    tot_q = sum(r["引文数"] for r in rows)
    s = {
        "批": name, "条数": n,
        "有依据": sum(1 for r in rows if r["依据原文"]),
        "报告名一致": sum(r["报告名一致"] for r in rows),
        "页码一致": sum(r["页码一致"] for r in rows),
        "有引文": len(with_q),
        "引文总数": tot_q,
        "引文逐字": sum(r["引文逐字"] for r in rows),
        "引文仅繁简": sum(r["引文仅繁简"] for r in rows),
        "引文查无": sum(r["引文查无"] for r in rows),
        "引文都在来源页": sum(1 for r in with_q
                              if r["引文都在来源页"] == 1),
    }
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("装载语料…", flush=True)
    corpus = Corpus(load_pages(os.path.join(HERE, "pages.jsonl")))
    print(f"语料 {len(corpus.meta)} 页")

    qa = [json.loads(l) for l in open(os.path.join(HERE, "qa.jsonl"),
                                      encoding="utf-8")]
    grid = [json.loads(l) for l in open(os.path.join(HERE, "eval_grid.jsonl"),
                                        encoding="utf-8")]
    if args.limit:
        qa, grid = qa[:args.limit], grid[:args.limit]

    rows = [audit_one(r, corpus, "生成集(342)", r["page_start"], r["page_end"])
            for r in qa]
    rows += [audit_one(r, corpus, "评测集(140)", r["page"], r["page"])
             for r in grid]

    out_csv = os.path.join(HERE, "source_audit.csv")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    s_qa = summarize([r for r in rows if r["来源"] == "生成集(342)"],
                     "生成集(342)")
    s_gr = summarize([r for r in rows if r["来源"] == "评测集(140)"],
                     "评测集(140)")

    lines = ["# 来源页审计", "",
             "每条问答逐条核对：依据里的报告名/页码是否与来源一致、",
             "每个 `「」` 引文能否在来源页原文里逐字找到。",
             "明细见 `source_audit.csv`（判分器与审计共用同一套溯源逻辑）。", ""]
    for s in (s_qa, s_gr):
        lines += [f"## {s['批']}", "",
                  "| 指标 | 数值 |", "|---|---|",
                  f"| 条数 | {s['条数']} |",
                  f"| 有【依据】 | {s['有依据']} |",
                  f"| 依据报告名与来源一致 | **{s['报告名一致']} / {s['条数']}** |",
                  f"| 依据页码与来源一致(±1) | **{s['页码一致']} / {s['条数']}** |",
                  f"| 带「」引文的条数 | {s['有引文']} |",
                  f"| 引文总数 | {s['引文总数']} |",
                  f"| 引文在原文逐字可查 | **{s['引文逐字']}** |",
                  f"| 引文仅繁简差异 | {s['引文仅繁简']} |",
                  f"| 引文在原文查无 | **{s['引文查无']}** |",
                  f"| 引文全部落在来源页的条数 | {s['引文都在来源页']} / {s['有引文']} |",
                  ""]
        print(f"【{s['批']}】报告名一致 {s['报告名一致']}/{s['条数']}，"
              f"页码一致 {s['页码一致']}/{s['条数']}，"
              f"引文 {s['引文总数']} 句：逐字 {s['引文逐字']} / "
              f"繁简 {s['引文仅繁简']} / 查无 {s['引文查无']}")

    # 查无的引文按条数最多的报告排个序 —— 一眼看出问题集中在哪几份报告
    miss = [r for r in rows if r["引文查无"]]
    if miss:
        from collections import Counter
        c = Counter(r["报告"] for r in miss)
        lines += ["## 引文查无：集中在哪几份报告（该列常见的是表格页）", "",
                  "| 报告 | 条数 |", "|---|---|"]
        lines += [f"| {d} | {n} |" for d, n in c.most_common()]
        lines += ["", "> 表格页的原文是一格一格的数字，模型把表格改写成叙述句后",
                  "> 再套上 `「」`，就会出现「内容对、逐字查无」——",
                  "> 这类在明细里与真编造混在一起，看示例列区分。", ""]

    out_md = os.path.join(HERE, "source_audit.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n明细 → {out_csv}\n摘要 → {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
