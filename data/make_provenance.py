# -*- coding: utf-8 -*-
"""造数据留痕：把「一条问答是怎么生出来的」从 API 缓存里原样还原出来。

批改意见要求补"造数据"的留痕 —— 光有 `build_qa.py` 脚本和 342 条成果
不算留痕，中间那一步（模型到底收到了什么、返回了什么）看不见。
好在每次 API 调用都落盘在 `data/cache/`（键 = 请求载荷的 sha256），
**原始返回一直是留着的**，只是没人把它摆出来。

本脚本挑 3 条代表（叙述块/表格块/测试池各一），对每条完整重放：

    原文 → 提问提示词 → 原始返回 → 作答提示词 → 原始返回
         → 质检提示词 → 原始返回 → 入库记录（含保留/剔除决定）

并做**重放核对**：用缓存内容重新解析出的问题/答案/质检，
必须与 `qa.jsonl` 里的逐字一致 —— 对不上就是链断了，宁可报错也不装。

用法：
    python make_provenance.py
输出：
    provenance_report.md    三链全文（每条都给缓存文件名，可点开对）
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_qa as B                                              # noqa: E402

# 三条代表：覆盖 叙述/表格 两种块、训练池/测试池两个池
SAMPLES = [
    ("理想汽车_2024年报_p9_0075_q1",
     "叙述块 · 训练池（抽检表序号 1）"),
    ("理想汽车_2021中报_p10_0005_q1",
     "表格块 · 训练池（正文 203.1% 是推算数字）"),
    ("理想汽车_2026中报_p7_0107_q3",
     "叙述块 · 测试池（绝不进训练集）"),
]


def lookup(system, user, temperature):
    """找这一问的缓存文件。返回 (文件名, 原始 content, 是否存在)。"""
    path = B.cache_path(system, user, temperature)
    name = os.path.basename(path)
    if not os.path.exists(path):
        return name, None, False
    return name, json.load(open(path, encoding="utf-8"))["content"], True


def code_block(text, lang=""):
    return f"```{lang}\n{text}\n```"


def render_sample(rec, chunk, i):
    doc, page = chunk["doc"], chunk["page_start"]
    q_index = int(rec["qa_id"].rsplit("_q", 1)[1])

    q_user = B.Q_TPL.format(n=3, doc=doc, page=page,
                            section=chunk["section"], text=chunk["text"])
    q_file, q_raw, q_ok = lookup(B.Q_SYS, q_user, 0.7)
    a_user = B.A_TPL.format(doc=doc, page=page, section=chunk["section"],
                            text=chunk["text"], question=rec["question"])
    a_file, a_raw, a_ok = lookup(B.A_SYS, a_user, 0.3)
    c_user = B.QC_TPL.format(doc=doc, page=page, text=chunk["text"],
                             question=rec["question"], answer=rec["answer"])
    c_file, c_raw, c_ok = lookup(B.QC_SYS, c_user, 0.0)

    # 重放核对：解析缓存里的原始返回，必须与 qa.jsonl 逐字一致
    ok_q = ok_a = ok_c = False
    questions_all = []
    if q_ok:
        questions_all = B.parse_json(q_raw).get("questions", [])
        ok_q = (len(questions_all) >= q_index
                and questions_all[q_index - 1].strip() == rec["question"].strip())
    if a_ok:
        ok_a = (B.parse_json(a_raw).get("answer", "").strip()
                == rec["answer"].strip())
    if c_ok:
        d = B.parse_json(c_raw)
        ok_c = all(
            (str(d.get(k)) == str(rec["qc"].get(k)))
            for k in ("有据", "自然", "简体", "编造"))
    replay = "✅ 三项与 qa.jsonl 逐字一致" if (ok_q and ok_a and ok_c) \
        else "⚠️ 重放对不上（缓存缺失或链断，需人工查）"

    keep = "保留" if rec["keep"] else f"剔除（{rec['drop_reason']}）"
    md = [f"## 样本 {i}：`{rec['qa_id']}`", "",
          f"> {SAMPLES[i-1][1]} · 《{doc}》第 {page} 页 · "
          f"章节：{chunk['section']} · 块类型："
          f"{'表格' if chunk['has_table'] else '叙述'}", "",
          f"**重放核对：{replay}**", ""]

    md += ["### ① 原文（块文本，就是提示词里塞进去的那一段）", "",
           f"来源：`chunks.jsonl` → `{rec['chunk_id']}`（{chunk['chars']} 字）", "",
           code_block(chunk["text"][:1200]), ""]

    md += ["### ② 提问（提问者角色）", "",
           f"缓存：`data/cache/{q_file}`"
           + ("" if q_ok else " —— **缺失**"), "",
           "<details><summary>提示词全文</summary>", "",
           code_block(q_user), "", "</details>", "",
           "原始返回：" if q_ok else "原始返回缺失",
           code_block(q_raw if q_ok else "—", "json") if q_ok else "",
           f"本样本用的是第 {q_index} 问：**{rec['question']}**", ""]

    md += ["### ③ 作答（董秘角色，temperature=0.3）", "",
           f"缓存：`data/cache/{a_file}`"
           + ("" if a_ok else " —— **缺失**"), "",
           "<details><summary>提示词全文</summary>", "",
           code_block(a_user), "", "</details>", "",
           "原始返回：", code_block(a_raw if a_ok else "—", "json"), ""]

    md += ["### ④ 质检（质检员角色，temperature=0）", "",
           f"缓存：`data/cache/{c_file}`"
           + ("" if c_ok else " —— **缺失**"), "",
           "<details><summary>提示词全文</summary>", "",
           code_block(c_user), "", "</details>", "",
           "原始返回：", code_block(c_raw if c_ok else "—", "json"),
           "",
           f"**入库结论**：{keep}"
           f"（规则见 `build_qa.py` 的 `classify()`；质检原始判定："
           f"有据={rec['qc']['有据']} 自然={rec['qc']['自然']} "
           f"简体={rec['qc']['简体']} 编造={rec['qc']['编造']}）", "",
           "---", ""]
    return "\n".join(md), (ok_q and ok_a and ok_c)


def main():
    qa = {r["qa_id"]: r for r in
          (json.loads(l) for l in open(os.path.join(HERE, "qa.jsonl"),
                                       encoding="utf-8"))}
    chunks = {}
    for l in open(os.path.join(HERE, "chunks.jsonl"), encoding="utf-8"):
        c = json.loads(l)
        chunks[c["chunk_id"]] = c

    n_cache = len([f for f in os.listdir(B.CACHE) if f.endswith(".json")])
    head = [
        "# 造数据留痕：从原文到问答对，一步不少",
        "",
        "本报告把 342 条问答里的代表样本**完整重放**一遍：模型收到了什么提示词、",
        "原样返回了什么。所有原始返回都躺在 `data/cache/`（每次 API 调用一个文件，",
        f"共 {n_cache} 个），键 = 请求载荷的 sha256 前 32 位 —— **可离线复核，",
        "不依赖任何后续接口**。",
        "",
        "生成链：`data/build_qa.py`，三角色同一模型靠提示词分工；",
        "114 个原文块 × 每块 3 问 → 342 条 → 逐条质检 → 自动筛选 264 条。",
        "",
        f"缓存总数 **{n_cache}** 个 = 提问调用（114）+ 作答调用（342）+ 质检调用（342）"
        "+ 换问法生成（build_paraphrase）与判分调用等。",
        "",
        "> 换问法的生成留痕在 `data/build_paraphrase.py` + 同一套 cache；",
        "> 抽检与审计见 `qc_sample.csv` / `source_audit.csv`。",
        "",
        "---",
        "",
    ]
    parts, all_ok = [], True
    for i, (qid, _) in enumerate(SAMPLES, start=1):
        rec = qa[qid]
        md, ok = render_sample(rec, chunks[rec["chunk_id"]], i)
        parts.append(md)
        all_ok = all_ok and ok

    foot = [
        "## 怎么自己核",
        "",
        "1. 打开上面任意一个 `data/cache/<哈希>.json`，里面的 `content` 就是模型当时的原始返回；",
        "2. 想重算哈希：",
        "",
        code_block(
            "cd data && python -X utf8 -c \"\n"
            "import json, sys; sys.path.insert(0,'.')\n"
            "import build_qa as B\n"
            "rec = json.loads(open('qa.jsonl',encoding='utf-8').readline())\n"
            "chunk = next(json.loads(l) for l in open('chunks.jsonl',encoding='utf-8')\n"
            "             if json.loads(l)['chunk_id']==rec['chunk_id'])\n"
            "user = B.A_TPL.format(doc=chunk['doc'], page=chunk['page_start'],\n"
            "                      section=chunk['section'], text=chunk['text'],\n"
            "                      question=rec['question'])\n"
            "print(B.cache_path(B.A_SYS, user, 0.3))  # 应打到上面 ③ 的那个文件\"",
            "bash"),
        "",
        "3. 两次调用之间没有任何人工改写 —— 重放核对（本报告每节的 ✅）就是查这个的：",
        "   用缓存里的原始返回重新解析，得到的问题/答案/质检与 `qa.jsonl` 逐字一致。",
        "",
    ]
    out = os.path.join(HERE, "provenance_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(head + parts + foot))
    print(f"写出 {out}（{'全部重放核对通过' if all_ok else '有重放对不上的样本！'}）")
    return 0 if all_ok else 3


if __name__ == "__main__":
    sys.exit(main())
