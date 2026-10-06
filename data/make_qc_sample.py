# -*- coding: utf-8 -*-
"""
第 6 步：生成人工抽检表。

作业要求「人工抽检，删掉编造的」——但 264 条全审太慢。这里分层抽 60 条：
    训练池 45 条 + 测试池 15 条，每池内按「表格块 / 叙述块」分层，跨报告分散。

并且做一层自动预检：**把答案里出现、原文里找不到的数字标出来**。
数字背串是这套数据最高频的失败模式，先标出来，人只需确认那几个。

注意：标出来 ≠ 错。答案里的数字可能是由原文数字**推算**出来的
（如用收入/成本算毛利率），那属于合理作答，人工判断即可。

用法：
    python make_qc_sample.py                 # 默认抽 60 条
    python make_qc_sample.py --n 80 --seed 1

输出：
    qc_sample.csv    抽检表（可直接在 Excel 里填）
    qc_sample.md     同样内容的可读版（对着屏幕审更舒服）
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DOCS = {"理想汽车_2025年报", "理想汽车_2026中报"}

# 千分位必须成组，否则相邻数字会被粘连成一个大数
NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def norm_num(s):
    """把数字归一化：去千分位、全角转半角、去尾零，便于比对。"""
    s = s.translate(str.maketrans("０１２３４５６７８９，．", "0123456789,."))
    s = s.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def numbers_in(text):
    """提取文本里的数字，只保留 3 位以上的（页码、序号之类的短数字不算）。"""
    out = set()
    for m in NUM.finditer(text):
        t = norm_num(m.group(0))
        if len(t.replace(".", "")) >= 3:
            out.add(t)
    return out


def check_numbers(answer, source):
    """返回 (引用里对不上的数字, 正文里原文没有的数字)。

    引用（「」内）的数字必须能在原文找到；正文里的数字可能是**推算**出来的
    （如用收入/成本算毛利率），所以分开报，让人一眼看出该重点看哪边。
    """
    src_nums = numbers_in(source)

    ans = re.sub(r"【[^】]*】", "", answer)        # 去掉【依据…】——页码会被误判成数字
    # 每段引用单独处理，否则相邻引用的数字会首尾粘连
    quoted_nums = set()
    for part in re.findall(r"「([^」]*)」", ans):
        quoted_nums |= numbers_in(part)
    body_nums = numbers_in(re.sub(r"「[^」]*」", "", ans))

    missing_q = sorted(quoted_nums - src_nums)
    missing_b = sorted(body_nums - src_nums - quoted_nums)
    return missing_q, missing_b


def stratified_pick(pool, k, rng):
    """在池子里按「表格 / 叙述」分层抽 k 条，尽量跨报告分散。"""
    by = defaultdict(list)
    for r in pool:
        by[r["has_table"]].append(r)
    for v in by.values():
        rng.shuffle(v)

    picked, tables = [], sorted(by, key=lambda x: -len(by[x]))
    # 按两类实际占比分配名额
    total = sum(len(v) for v in by.values())
    for i, key in enumerate(tables):
        quota = round(k * len(by[key]) / total) if i < len(tables) - 1 else k - len(picked)
        picked += by[key][:quota]
    # 补足 / 截断，顺便打散顺序
    rest = [r for v in by.values() for r in v if r not in picked]
    rng.shuffle(rest)
    picked += rest[:max(0, k - len(picked))]
    rng.shuffle(picked)
    return picked[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="抽检条数")
    ap.add_argument("--test-ratio", type=float, default=0.25, help="测试池占比")
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    recs = [json.loads(l) for l in open(os.path.join(HERE, "qa.jsonl"),
                                        encoding="utf-8")]
    kept = [r for r in recs if r["keep"]]
    train = [r for r in kept if r["doc"] not in TEST_DOCS]
    test = [r for r in kept if r["doc"] in TEST_DOCS]

    n_test = round(args.n * args.test_ratio)
    sample = stratified_pick(train, args.n - n_test, rng) + \
             stratified_pick(test, n_test, rng)
    rng.shuffle(sample)

    rows = []
    for i, r in enumerate(sample, start=1):
        src = r.get("source_text", "")
        miss_q, miss_b = check_numbers(r["answer"], src)
        flags = []
        if miss_q:
            flags.append("引用里有原文查不到的数字：" + "、".join(miss_q))
        if miss_b:
            flags.append("正文数字原文没有（可能是推算）：" + "、".join(miss_b))
        rows.append({
            "序号": i,
            "qa_id": r["qa_id"],
            "池": "测试池" if r["doc"] in TEST_DOCS else "训练池",
            "报告": r["doc"],
            "页码": r["page_start"],
            "章节": r["section"],
            "块类型": "表格" if r["has_table"] else "叙述",
            "问题": r["question"],
            "董秘回答": r["answer"],
            "原文摘录": src,
            "自动预检": "；".join(flags) if flags else "无异常",
            "人审-有据": "", "人审-数字": "", "人审-处理": "", "人审-备注": "",
        })

    with open(os.path.join(HERE, "qc_sample.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(HERE, "qc_sample.md"), "w", encoding="utf-8") as f:
        f.write("# 人工抽检表（%d 条，seed=%d）\n\n" % (len(rows), args.seed))
        f.write("审核要点：① 引用「」里的数字能否在原文摘录里找到 "
                "② 正文有没有原文没有的信息 ③ 问得像不像真实投资者\n\n")
        f.write("在 CSV 里填 → `人审-有据`（2/1/0）· `人审-数字`（对/错/推算）"
                "· `人审-处理`（保留/改写/删除）\n\n---\n\n")
        for r in rows:
            f.write(f"## {r['序号']}. {r['qa_id']}\n\n")
            f.write(f"`{r['池']}` · 《{r['报告']}》第 {r['页码']} 页 · "
                    f"{r['章节']} · {r['块类型']}块\n\n")
            f.write(f"**问**：{r['问题']}\n\n")
            f.write(f"**答**：{r['董秘回答']}\n\n")
            f.write("<details><summary>原文摘录</summary>\n\n```\n")
            f.write(r["原文摘录"][:1400])
            f.write("\n```\n\n</details>\n\n")
            flag = r["自动预检"]
            mark = "✅" if flag == "无异常" else "⚠️"
            f.write(f"{mark} **自动预检**：{flag}\n\n---\n\n")

    n_flag = sum(1 for r in rows if r["自动预检"] != "无异常")
    n_q = sum(1 for r in rows if "引用里" in r["自动预检"])
    print(f"抽检表：{len(rows)} 条（训练池 {len(rows)-n_test} / 测试池 {n_test}）")
    print(f"  表格块 {sum(1 for r in rows if r['块类型']=='表格')} 条 / "
          f"叙述块 {sum(1 for r in rows if r['块类型']=='叙述')} 条")
    print(f"  自动预检有提示的 {n_flag} 条，其中【引用数字对不上】{n_q} 条"
          f" ← 这几条优先看")
    print(f"\n写出 qc_sample.csv（Excel 里填）和 qc_sample.md（对着屏幕审）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
