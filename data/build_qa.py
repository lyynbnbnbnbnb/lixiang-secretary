# -*- coding: utf-8 -*-
"""
第 2 步：从财报原文生成「投资者提问 → 董秘作答」问答对，并逐条质检。

流水线（对应课程里的三角色分工，同一个便宜模型靠提示词分工）：
    切块 → 提问者（投资者视角，每块 3 问）
         → 董秘作答（只依据本块，先结论、再「」引原文、末尾注依据）
         → 质检（有据 0/1/2、自然 0/1，是否编造）

口径约定（用户选定）：
    原文保持繁体（港交所中文版原貌，引用时一字不改）
    问题与问答的正文用简体中文

用法：
    python build_qa.py --dry-run              # 只看切块结果，不调 API
    python build_qa.py --limit 3              # 只跑 3 个块（先验证通不通）
    python build_qa.py                        # 全量跑
    python build_qa.py --docs 理想汽车_2021年报 理想汽车_2022年报

环境变量：
    DEEPSEEK_API_KEY    必填（不要写进代码）
    DEEPSEEK_MODEL      可选，默认 deepseek-chat
    DEEPSEEK_BASE       可选，默认 https://api.deepseek.com

输出：
    chunks.jsonl       切块结果
    qa.jsonl           全部问答对（含质检分数）
    qc_review.csv      抽检记录（含人工复核留空列，交付物之一）
    cache/             API 响应缓存（重跑不重复花钱）
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

DEFAULT_SECTIONS = r"摘要|業務回顧|业务回顾|管理層討論|管理层讨论"

API_BASE = os.environ.get("DEEPSEEK_BASE", "https://api.deepseek.com")
API_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# 表格页：纯数字行占比超过这个值就标记为表格块（标记但不丢弃）
TABLE_LINE = re.compile(r"^[\s(（]*-?[\d,，.]+[)）%]?[\s]*$")


# ---------------------------------------------------------------- 切块

def build_sections(toc_doc):
    """把书签目录变成 [(起始页, 标题), ...]，按起始页排序。"""
    return sorted([(t["page"], t["title"]) for t in toc_doc], key=lambda x: x[0])


def section_of(page_no, sections):
    cur = "（无章节）"
    for start, title in sections:
        if start <= page_no:
            cur = title
        else:
            break
    return cur


def is_table(text):
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 4:
        return False
    num = sum(1 for l in lines if TABLE_LINE.match(l))
    return num / len(lines) > 0.25


def split_long_page(text, target):
    """一页太长就按段落劈成 ~target 字的块。"""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        paras = [l.strip() for l in text.splitlines() if l.strip()]
    out, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) > target * 1.4:
            out.append(buf.strip())
            buf = p
        else:
            buf = (buf + "\n" + p).strip()
    if buf.strip():
        out.append(buf.strip())
    return out


def chunk_report(pages, toc_doc, target=650, section_pat=DEFAULT_SECTIONS):
    """把一个报告的正文页切成带页码、带章节的块。"""
    sections = build_sections(toc_doc)
    pat = re.compile(section_pat)
    chunks, buf, buf_pages, buf_sec = [], "", [], ""

    def flush():
        nonlocal buf, buf_pages, buf_sec
        if buf.strip() and buf_pages:
            chunks.append({
                "section": buf_sec,
                "page_start": min(buf_pages),
                "page_end": max(buf_pages),
                "text": buf.strip(),
            })
        buf, buf_pages = "", []

    for pg in pages:
        text, page_no = pg["text"], pg["page"]
        sec = section_of(page_no, sections)
        if not pat.search(sec):            # 只要叙事章节：业务回顾 / 管理层讨论 / 摘要
            flush()
            continue
        if len(text) < 40:                 # 空页 / 纯图页
            continue
        if len(text) > target * 1.7:       # 长页劈开，每块仍属于这一页
            flush()
            for piece in split_long_page(text, target):
                chunks.append({
                    "section": sec, "page_start": page_no,
                    "page_end": page_no, "text": piece,
                })
            continue
        if len(text) < 400 and buf:        # 短页与上一块合并
            buf = (buf + "\n" + text).strip()
            buf_pages.append(page_no)
        else:
            flush()
            buf, buf_pages, buf_sec = text, [page_no], sec
        if len(buf) >= target:
            flush()
    flush()

    for c in chunks:                      # chunk_id / doc 等在 main 里统一补
        c["chars"] = len(c["text"])
        c["has_table"] = is_table(c["text"])
    return chunks


# ---------------------------------------------------------------- API

def get_api_key():
    """密钥来源，按优先级：环境变量 → data/.deepseek_key → data/.env。
    两条路都不写进代码，且都被 .gitignore 挡住。
    文件名允许带 .txt 后缀 —— Windows 记事本会自动加。"""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key.strip()

    import glob
    candidates = sorted(glob.glob(os.path.join(HERE, ".deepseek_key*"))) + \
                 sorted(glob.glob(os.path.join(HERE, "*.deepseek_key*")))
    candidates += [os.path.join(HERE, ".env")]

    for path in candidates:
        if not os.path.isfile(path):
            continue
        for line in open(path, encoding="utf-8-sig"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 兼容 "DEEPSEEK_API_KEY=sk-xxx" 和 "sk-xxx" 两种写法
            val = line.split("=", 1)[1] if "=" in line else line
            val = val.strip().strip('"').strip("'")
            if val:
                return val
    raise RuntimeError(
        "没找到 DeepSeek 密钥。任选一种：\n"
        f"  1) 建文件 {os.path.join(HERE, '.deepseek_key')}，里面只写一行 key\n"
        f"  2) 建文件 {os.path.join(HERE, '.env')}，写 DEEPSEEK_API_KEY=sk-xxx\n"
        "  3) 设置环境变量 DEEPSEEK_API_KEY")


def call_llm(system, user, retries=4, temperature=0.7):
    """带磁盘缓存 + 重试的 DeepSeek 调用。"""
    key = get_api_key()
    payload = {
        "model": API_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    ck = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    cpath = os.path.join(CACHE, ck + ".json")
    if os.path.exists(cpath):
        return json.load(open(cpath, encoding="utf-8"))["content"]

    last = None
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json=payload, timeout=180,
            )
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            os.makedirs(CACHE, exist_ok=True)
            json.dump({"content": content}, open(cpath, "w", encoding="utf-8"),
                      ensure_ascii=False)
            return content
        except Exception as e:                                  # noqa: BLE001
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"调用失败（重试 {retries} 次）：{last}")


def parse_json(text):
    """模型偶尔会包 ```json 围栏或加解释，尽量抠出 JSON。"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.M).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for pat in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pat, t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    raise ValueError(f"解析不出 JSON：{t[:200]}")


# ---------------------------------------------------------------- 三个角色

Q_SYS = "你是正在阅读港股财报的个人投资者。只输出 JSON，不要任何解释。"

Q_TPL = """你是理想汽车（港股 02015）的个人投资者，正在读公司刚发布的财报。

请根据下面这一段财报原文，以投资者的口吻提 {n} 个问题。

**第一步，先判断这段原文的类型：**
- **A. 叙述性内容**（业务回顾、管理层讨论等成段文字）：可以问事实、原因、趋势
- **B. 数据表格**（只有科目名和数字，没有成段说明）：**只能问能由表内数字直接算出或读出的事实性问题**（如"某某科目是多少""同比变动多少"），**绝对不要问原因、解释、预测**——表格里没有这些东西，问了也只能答"材料未涉及"

**第二步，提 {n} 个问题：**
1. 每个问题都必须能**用这一段的内容实际回答出来**，不能是这一段没写的
2. **问题必须一律使用简体中文书写，不得出现繁体字**（原文是繁体，但你的问题要写简体）
3. 提问要**像真实投资者会关心的**：业绩、交付量、收入利润、毛利率、成本费用、
   产品与车型、渠道门店、研发投入、战略与风险。**不要问人事制度、培训体系、
   员工福利这类与投资判断无关的细节**
4. 角度尽量不同（但必须都在本段能答的范围内）
5. 不要问「这一段讲了什么」这类元问题

只输出 JSON：{{"questions": ["问题1", "问题2", "问题3"]}}

原文（《{doc}》第 {page} 页，章节：{section}）：
{text}"""

A_SYS = "你是理想汽车的董事会秘书，正在回答投资者提问。只输出 JSON，不要任何解释。"

A_TPL = """你是理想汽车（港股 02015）的董事会秘书，正在回答投资者提问。

规则（必须严格遵守）：
1. **只能依据下面提供的原文作答**。绝对不要使用原文之外的知识，包括你自己知道的任何关于理想汽车的信息
2. **回答的正文一律用简体中文书写，不得出现繁体字**；唯一例外是「」里引用的原文——
   **引用必须照抄繁体原文，一字不改**
3. 结构：先用一句话给结论，再引原文佐证，末尾注明依据
4. 依据格式固定为：【依据：《{doc}》第 {page} 页】——**报告名必须原样照抄，
   如上例所示，不要做简繁转换，不要增删任何字**
5. 如果原文信息不足以回答，直接写「材料未涉及」，**不要编造、不要含糊其辞**
6. 语气专业克制，像真实的董秘答投资者问，不要写成"这段文字讲了……"

只输出 JSON：{{"answer": "……"}}

原文（《{doc}》第 {page} 页，章节：{section}）：
{text}

投资者提问：{question}"""

QC_SYS = "你是问答数据质检员。只输出 JSON，不要任何解释。"

QC_TPL = """你是问答数据质检员。请对照原文，检查下面这条投资者问答。

评分标准：
- **有据**：2 = 完全有据，引用准确；1 = 大意对但有小的出入（数字或措辞不准）；0 = 原文里找不到依据，或与原文矛盾
- **自然**：1 = 自然，像真实董秘答投资者问；0 = 生硬、模板腔，或问的是与投资无关的琐碎细节（如培训体系、员工福利）
- **简体**：问题与回答的正文是否**全部使用简体中文**（「」内引用的繁体原文不算）。
  正文里出现任何繁体字就填 false
- **编造**：回答里是否出现了原文完全没有的信息
- **引用名**：依据里写的报告名是否与下方给定的报告名**逐字一致**（不得简繁转换）

只输出 JSON：{{"有据": 2, "自然": 1, "简体": true, "编造": false, "引用名一致": true, "理由": "一句话说明"}}

给定的报告名：《{doc}》

原文（《{doc}》第 {page} 页）：
{text}

投资者提问：{question}

董秘回答：{answer}"""


def make_questions(chunk, n=3):
    out = call_llm(Q_SYS, Q_TPL.format(
        n=n, doc=chunk["doc"], page=chunk["page_start"],
        section=chunk["section"], text=chunk["text"]))
    qs = parse_json(out).get("questions", [])
    return [str(q).strip() for q in qs if str(q).strip()][:n]


def make_answer(chunk, question):
    out = call_llm(A_SYS, A_TPL.format(
        doc=chunk["doc"], page=chunk["page_start"], section=chunk["section"],
        text=chunk["text"], question=question), temperature=0.3)
    return str(parse_json(out).get("answer", "")).strip()


def make_qc(chunk, question, answer):
    out = call_llm(QC_SYS, QC_TPL.format(
        doc=chunk["doc"], page=chunk["page_start"], text=chunk["text"],
        question=question, answer=answer), temperature=0.0)
    d = parse_json(out)
    return {
        "有据": int(d.get("有据", 0)),
        "自然": int(d.get("自然", 0)),
        "简体": bool(d.get("简体", False)),
        "编造": bool(d.get("编造", True)),
        "引用名一致": bool(d.get("引用名一致", True)),
        "理由": str(d.get("理由", "")).strip(),
    }


# ---------------------------------------------------------------- 筛选

REFUSAL = ("材料未涉及", "材料未提及", "材料未披露", "原文未涉及")


def classify(rec):
    """判断这条问答值不值得进训练集。返回 (keep, 原因)。

    不要一刀切删掉所有含「材料未涉及」的回答 —— 先给数据再说明
    「其余部分原文没有」是**好的**董秘回答（诚实、有据）。
    要删的是**整条只是在拒绝作答**的那种。
    """
    ans = rec.get("answer", "")
    if not ans:
        return False, "回答为空"
    head = ans[:45]
    if any(w in head for w in REFUSAL) and len(ans) < 120:
        return False, "整条只在说材料未涉及"
    if any(w in head for w in REFUSAL):
        return False, "开头即拒绝作答"
    if rec["qc"].get("编造"):
        return False, "质检判定编造"
    if rec["qc"].get("有据", 0) == 0:
        return False, "质检判定无据"
    if not rec["qc"].get("简体", True):
        return False, "正文混入繁体"
    if not rec["qc"].get("引用名一致", True):
        return False, "依据里的报告名不一致"
    return True, ""


# ---------------------------------------------------------------- 主流程

def process_chunk(chunk):
    """一个块：提 3 问 → 逐个作答 → 逐个质检。返回问答对列表。"""
    records = []
    try:
        questions = make_questions(chunk, n=3)
    except Exception as e:                                       # noqa: BLE001
        print(f"  [提问失败] {chunk['chunk_id']}: {e}")
        return records
    for qi, q in enumerate(questions, start=1):
        rec = {k: chunk[k] for k in
               ("chunk_id", "doc", "year", "doc_type", "section",
                "page_start", "page_end", "chars", "has_table")}
        rec["qa_id"] = f"{chunk['chunk_id']}_q{qi}"
        rec["question"] = q
        try:
            rec["answer"] = make_answer(chunk, q)
            rec["qc"] = make_qc(chunk, q, rec["answer"])
        except Exception as e:                                   # noqa: BLE001
            rec["answer"], rec["qc"] = "", {"有据": 0, "自然": 0, "简体": False,
                                            "编造": True, "引用名一致": False,
                                            "理由": f"出错：{e}"}
        records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*", default=None, help="只处理这些报告")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个块（试跑用）")
    ap.add_argument("--sample", action="store_true",
                    help="试跑时跨报告均匀取样，而不是只取开头几个块")
    ap.add_argument("--workers", type=int, default=6, help="并发数")
    ap.add_argument("--target", type=int, default=650, help="每块目标字数")
    ap.add_argument("--sections", default=DEFAULT_SECTIONS, help="章节正则")
    ap.add_argument("--dry-run", action="store_true", help="只切块，不调 API")
    args = ap.parse_args()

    pages_all = [json.loads(l) for l in
                 open(os.path.join(HERE, "pages.jsonl"), encoding="utf-8")]
    toc = json.load(open(os.path.join(HERE, "toc.json"), encoding="utf-8"))

    docs = sorted({p["doc"] for p in pages_all})
    if args.docs:
        docs = [d for d in docs if d in args.docs]

    all_chunks = []
    for doc in docs:
        pages = [p for p in pages_all if p["doc"] == doc]
        for c in chunk_report(pages, toc[doc], target=args.target,
                              section_pat=args.sections):
            c["_doc"] = doc
            m = re.match(r"理想汽车_(\d{4})(年报|中报)", doc)
            c.update(doc=doc, year=m.group(1), doc_type=m.group(2))
            c.pop("_doc")
            c["chunk_id"] = f"{doc}_p{c['page_start']}_{len(all_chunks):04d}"
            all_chunks.append(c)

    with open(os.path.join(HERE, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    n_tab = sum(1 for c in all_chunks if c["has_table"])
    print(f"切块：{len(all_chunks)} 块 / 共 {sum(c['chars'] for c in all_chunks):,} 字 "
          f"/ 其中疑似表格块 {n_tab}")
    if args.dry_run:
        for c in all_chunks[:8]:
            print(f"\n[{c['chunk_id']}] {c['section']} p{c['page_start']} "
                  f"{c['chars']}字 表格={c['has_table']}")
            print(c["text"][:160].replace("\n", " ") + " …")
        print(f"\n（dry-run 结束，共 {len(all_chunks)} 块）")
        return 0

    if args.limit:
        if args.sample and args.limit < len(all_chunks):
            step = len(all_chunks) / args.limit          # 跨报告均匀取样
            all_chunks = [all_chunks[int(i * step)] for i in range(args.limit)]
            print(f"试跑模式：均匀取样 {args.limit} 个块")
        else:
            all_chunks = all_chunks[:args.limit]
            print(f"试跑模式：只处理前 {args.limit} 个块")

    print(f"开始调用 API（并发 {args.workers}）…")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_chunk, c): c for c in all_chunks}
        for i, fut in enumerate(as_completed(futs), start=1):
            recs = fut.result()
            results += recs
            if i % 10 == 0 or i == len(futs):
                print(f"  块 {i}/{len(futs)}  已生成 {len(results)} 条问答  "
                      f"{time.time()-t0:.0f}s")

    results.sort(key=lambda r: r["qa_id"])
    for r in results:                       # 打上 keep 标记（保留全部，便于留痕）
        keep, why = classify(r)
        r["keep"], r["drop_reason"] = keep, why
        r["source_text"] = next(
            (c["text"] for c in all_chunks if c["chunk_id"] == r["chunk_id"]), "")

    with open(os.path.join(HERE, "qa.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    kept = [r for r in results if r["keep"]]
    with open(os.path.join(HERE, "train_sft.jsonl"), "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps({
                "qa_id": r["qa_id"], "doc": r["doc"], "page": r["page_start"],
                "section": r["section"], "question": r["question"],
                "answer": r["answer"], "has_table": r["has_table"],
                "qc": r["qc"],
            }, ensure_ascii=False) + "\n")

    # 抽检记录：模型打分 + 人工复核留空列（交付物之一）
    csv_path = os.path.join(HERE, "qc_review.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qa_id", "报告", "页码", "章节", "问题", "董秘回答",
                    "有据(模型)", "自然(模型)", "简体(模型)", "编造(模型)",
                    "引用名一致", "模型理由",
                    "自动筛选", "丢弃原因",
                    "人审-有据", "人审-自然", "人审-处理", "人审-备注"])
        for r in results:
            qc = r["qc"]
            w.writerow([r["qa_id"], r["doc"], r["page_start"], r["section"],
                        r["question"], r["answer"],
                        qc["有据"], qc["自然"],
                        "是" if qc.get("简体") else "否",
                        "是" if qc["编造"] else "否",
                        "是" if qc.get("引用名一致") else "否",
                        qc["理由"],
                        "保留" if r["keep"] else "丢弃", r["drop_reason"],
                        "", "", "", ""])

    ok = sum(1 for r in results if r["qc"]["有据"] == 2)
    fab = sum(1 for r in results if r["qc"]["编造"])
    print(f"\n完成：{len(results)} 条问答，用时 {time.time()-t0:.0f}s")
    print(f"  质检：有据满分 {ok} / 小出入 "
          f"{sum(1 for r in results if r['qc']['有据']==1)} / 无据或编造 "
          f"{sum(1 for r in results if r['qc']['有据']==0)}，编造 {fab} 条")
    print(f"  自动筛选：保留 {len(kept)} 条 / 丢弃 {len(results)-len(kept)} 条")
    import collections
    for reason, n in collections.Counter(
            r["drop_reason"] for r in results if not r["keep"]).most_common():
        print(f"      {reason}: {n}")
    tab = sum(1 for r in kept if r["has_table"])
    print(f"  保留的里面：表格块 {tab} 条 / 叙述块 {len(kept)-tab} 条")
    print(f"\n写出 qa.jsonl（全部）、train_sft.jsonl（保留的）、{csv_path}")
    print("下一步：人工抽检 qc_review.csv，填「人审-」几列")
    return 0


if __name__ == "__main__":
    sys.exit(main())
