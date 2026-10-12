# -*- coding: utf-8 -*-
"""引文溯源：把回答里的「」引用和【依据…】拆开，逐条回财报原文里查。

**为什么单独成模块**：判分器（`eval/judge.py` 的引用维度）和溯源审计
（`data/audit_sources.py`）都要做同一件事——「这句引用到底是不是原文」。
两边各写一套的话，判分和审计迟早对不上，批改那边一问就露馅。

查证分四档，从严到宽：

| 档 | 规则 | 含义 |
|---|---|---|
| 逐字 | 只去空白后就命中 | 真·一字不差（PDF 解析的换行不影响） |
| 去标点 | 再去掉标点后命中 | 只差标点/全半角，属实抄 |
| 繁简 | 转成简体后命中 | 引的是简体（口径要求照抄繁体，算缺陷但**不是编造**） |
| 未命中 | 都不命中 | 原文里找不到 → 编造/改写 |

「逐字」和「去标点」都算**真实存在于原文**（判"编造"用这个口径）；
「繁简」单独报——它是口径错误，不是捏造。
"""
import bisect
import json
import re

try:
    import zhconv
except ImportError:                       # 没装就退化成"不查繁简档"
    zhconv = None

# 归一化时统一丢掉的字符：空白、标点、括号等格式性内容
_PUNCT = set(
    "「」『』【】《》〈〉（）()［］[]｛｝{}＜＞<>"
    "，,。.、；;：:！!？?…—－~～·•｜|/＼\\"
    "\"'“”‘’′″　 \t\r\n\x00​﻿"
)


def norm_space(s):
    """第一档：只去空白。PDF 换行、空格不该算「不逐字」。"""
    return re.sub(r"\s+", "", s or "")


def norm_all(s):
    """第二档：去空白 + 去标点。全半角、中英标点差异一并抹平。"""
    return "".join(ch for ch in (s or "") if ch not in _PUNCT)


def to_simplified(s):
    return zhconv.convert(s, "zh-cn") if zhconv else s


# ---------------------------------------------------------------- 拆引用

CITE_BLOCK = re.compile(r"【([^】]{0,300})】")
_DOC_IN = re.compile(r"《([^》]{1,80})》")
# 没写《》的情况（如「依据：理想汽车_2025年报 第 6 页」）也能兜住
_DOC_LOOSE = re.compile(r"([一-鿿A-Za-z0-9_]{2,40}?(?:年报|中报))")
_PAGE_IN = re.compile(r"第\s*(\d{1,4})\s*(?:[-–~～至]\s*(\d{1,4})\s*)?页")
# 没写【】、只写「依据：……」的情况。窗口取到句号/分号/换行为止，
# 免得把后文的页码也吞进来。
_BARE_CITE = re.compile(r"依据[:：]\s*([^。；\n]{1,90})")

# 「材料未涉及」这类固定话术不是引用，别拿去查原文
_REFUSAL = ("材料未涉及", "材料未提及", "材料未披露", "原文未涉及",
            "材料中没有", "材料未涵盖", "无法回答")


def _cite_of(inner):
    docs = _DOC_IN.findall(inner)
    if not docs:
        docs = _DOC_LOOSE.findall(inner)
    pages = [(int(a), int(b) if b else int(a))
             for a, b in _PAGE_IN.findall(inner)]
    return {"raw": inner.strip(), "docs": docs, "pages": pages}


def parse_citations(text):
    """拆出【…】块里的依据，以及**漏写括号**的「依据：…」。

    返回 [{"raw": 依据原文, "docs": [...], "pages": [(起, 讫), ...]}, ...]
    只保留确实带报告名或页码的块——【注】之类的说明块不算依据。

    漏写【】的情况实测有 3 条（「……。依据：《…》第 5 页」），
    格式分该扣就扣（口径要求带【依据：…】），但**引用分查的是内容**，
    不能因为少个括号就说人家没写依据。
    """
    text = text or ""
    out = []
    for m in CITE_BLOCK.finditer(text):
        c = _cite_of(m.group(1))
        if c["docs"] or c["pages"]:
            out.append(c)
    for m in _BARE_CITE.finditer(CITE_BLOCK.sub(" ", text)):
        c = _cite_of(m.group(1))
        if c["docs"] or c["pages"]:
            out.append(c)
    return out


def is_refusal_quote(q):
    """「材料未涉及」这类话术不是引用。"""
    n = norm_all(q)
    return any(n and (n in norm_all(r) or norm_all(r) in n) for r in _REFUSAL)


def extract_quotes(text):
    """取出所有非空且非话术的「」引文，按出现顺序。"""
    qs = [q.strip() for q in re.findall(r"「([^」]{0,600})」", text or "")]
    return [q for q in qs if q and not is_refusal_quote(q)]


# ---------------------------------------------------------------- 查原文

_ELLIPSIS = re.compile(r"…+|\.{3,}|·{3,}|。{3,}|\*{3,}")


def split_quote(q, min_seg=6):
    """含省略号的引文按省略号拆段，太短的段丢掉。

    太短的段（如「本公司」「截至」）在任何一页都能查到，留着只会把
    「查得到」灌水成必然事件。丢掉几段就如实报几段，不装看不见。
    """
    segs = [s.strip() for s in _ELLIPSIS.split(q)]
    keep = [s for s in segs if len(norm_all(s)) >= min_seg]
    n_short = len([s for s in segs if s and len(norm_all(s)) < min_seg])
    return keep, n_short


class Corpus:
    """把 pages.jsonl 的全量页面装进来，提供「这句话在哪几页」。

    一次装好反复查：1449 页各自归一化后拼成一个大字符串（\x00 分隔，
    引文里不可能出现），查一句就是一次 C 级扫描 + 二分定位页号——
    判分要对几千句引文做这件事，按页 for 循环会慢一两个数量级。
    """

    def __init__(self, pages):
        self.meta = [(p["doc"], p["page"]) for p in pages]
        self.space = [norm_space(p["text"]) for p in pages]
        self.plain = [norm_all(p["text"]) for p in pages]
        self.zh = [norm_all(to_simplified(p["text"])) for p in pages] \
            if zhconv else None
        self._big, self._offs = self._join(self.plain)
        if self.zh is not None:
            self._big_zh, self._offs_zh = self._join(self.zh)

    @staticmethod
    def _join(texts):
        big = "\x00".join(texts)
        offs, pos = [], 0
        for t in texts:
            offs.append(pos)
            pos += len(t) + 1
        return big, offs

    @staticmethod
    def _pages_of(big, offs, s):
        out, i = set(), big.find(s)
        while i != -1:
            out.add(bisect.bisect_right(offs, i) - 1)
            i = big.find(s, i + 1)
        return out

    def locate(self, seg):
        """一段引文在哪几页、命中在哪一档。返回 (档位, 页号集合)。"""
        n2 = norm_all(seg)
        if n2:
            pages = self._pages_of(self._big, self._offs, n2)
            if pages:
                n1 = norm_space(seg)
                whitespace_exact = any(n1 in self.space[i] for i in pages)
                return ("逐字" if whitespace_exact else "去标点"), pages
        if self.zh is not None:
            n3 = norm_all(to_simplified(seg))
            if n3:
                pages = self._pages_of(self._big_zh, self._offs_zh, n3)
                if pages:
                    return "繁简", pages
        return "未命中", set()

    def check_quote(self, quote, min_seg=6):
        """查一句引文：先按省略号拆段，再逐段定档，整句取最差档。"""
        segs, n_short = split_quote(quote, min_seg=min_seg)
        rows = []
        for s in segs:
            band, pages = self.locate(s)
            rows.append({"段": s, "档": band, "页": sorted(pages)})
        if not rows:
            return {"判定": "空", "段": [], "跳过短段": n_short}
        bands = [r["档"] for r in rows]
        if any(b == "未命中" for b in bands):
            verdict = "未命中"
        elif all(b == "逐字" for b in bands):
            verdict = "逐字"
        elif all(b in ("逐字", "去标点") for b in bands):
            verdict = "去标点"
        else:
            verdict = "繁简"
        return {"判定": verdict, "段": rows, "跳过短段": n_short}


# 真实存在（不算编造）的两档
REAL = ("逐字", "去标点")

REFUSAL_WORDS = ("材料未涉及", "材料未提及", "材料未披露", "原文未涉及")


def load_pages(pages_path):
    with open(pages_path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]
