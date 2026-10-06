# -*- coding: utf-8 -*-
"""
把 data/raw/ 里的财报 PDF 逐页解析成 JSONL —— 每页一条记录，带页码。

为什么必须逐页保留页码：
    方向 B 的作业要求「答案注明出自哪份报告哪一页」，页码是硬约束，
    后面切块、生成问答、评测引用都靠它。

用法：
    python parse.py                 # 解析全部
    python parse.py --only 2025年报  # 只解析指定的

输出：
    data/pages.jsonl   每页一条：{doc, year, doc_type, page, text, chars}
    data/toc.json      每份 PDF 的书签目录（章 -> 起始页）
    data/parse_report.md  解析概况（页数、字数、可能的扫描页）
"""
import argparse
import json
import os
import re
import sys

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

# 页眉页脚：纯页码行、"第 x 页"之类，抽完就丢掉
JUNK = re.compile(r"^\s*[-—–\s]*(?:第\s*)?\d{1,4}\s*(?:页)?[-—–\s]*$")


def clean_page(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip() and not JUNK.match(ln)]
    text = "\n".join(lines)
    text = re.sub(r"[ \t]{2,}", " ", text)          # 多余空格
    text = re.sub(r"\n{3,}", "\n\n", text)          # 多余空行
    return text.strip()


def parse_pdf(path: str):
    doc = pymupdf.open(path)
    pages, scanned = [], []
    for i, page in enumerate(doc, start=1):
        text = clean_page(page.get_text("text"))
        if len(text) < 20:                          # 几乎没字 —— 可能是扫描页/纯图页
            scanned.append(i)
        pages.append({"page": i, "text": text, "chars": len(text)})
    toc = doc.get_toc()                             # [[level, title, page], ...]
    n = doc.page_count
    doc.close()
    return pages, toc, n, scanned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只解析文件名含该字符串的报告")
    args = ap.parse_args()

    pdfs = sorted(f for f in os.listdir(RAW) if f.lower().endswith(".pdf"))
    if args.only:
        pdfs = [f for f in pdfs if args.only in f]
    if not pdfs:
        print("data/raw/ 里没有 PDF，先跑 download.py")
        return 1

    out_path = os.path.join(HERE, "pages.jsonl")
    all_toc, summary = {}, []

    with open(out_path, "w", encoding="utf-8") as out:
        for name in pdfs:
            stem = os.path.splitext(name)[0]                 # 理想汽车_2025年报
            m = re.match(r"理想汽车_(\d{4})(年报|中报)", stem)
            year, doc_type = (m.group(1), m.group(2)) if m else ("unknown", "unknown")
            pages, toc, n, scanned = parse_pdf(os.path.join(RAW, name))

            for p in pages:
                out.write(json.dumps({
                    "doc": stem, "year": year, "doc_type": doc_type,
                    "page": p["page"], "chars": p["chars"], "text": p["text"],
                }, ensure_ascii=False) + "\n")

            all_toc[stem] = [{"level": lv, "title": t, "page": pg} for lv, t, pg in toc]
            total_chars = sum(p["chars"] for p in pages)
            summary.append((stem, n, total_chars, len(scanned), scanned[:8]))
            print(f"{stem}: {n} 页 / {total_chars:,} 字 / 近空白页 {len(scanned)}")

    with open(os.path.join(HERE, "toc.json"), "w", encoding="utf-8") as f:
        json.dump(all_toc, f, ensure_ascii=False, indent=2)

    with open(os.path.join(HERE, "parse_report.md"), "w", encoding="utf-8") as f:
        f.write("# 财报解析概况\n\n")
        f.write("| 报告 | 页数 | 字数 | 近空白页数 | 前几个空白页 |\n|---|---|---|---|---|\n")
        for stem, n, c, ns, head in summary:
            f.write(f"| {stem} | {n} | {c:,} | {ns} | {head} |\n")
        f.write(f"\n合计 {sum(s[1] for s in summary):,} 页 / "
                f"{sum(s[2] for s in summary):,} 字\n")

    print(f"\n写出 {out_path}")
    print(f"写出 {os.path.join(HERE, 'toc.json')} 和 parse_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
