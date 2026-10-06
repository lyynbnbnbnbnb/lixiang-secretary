# -*- coding: utf-8 -*-
"""共用小工具：编码兼容读 CSV、大数字提取、引导性提问检测。

单独成文件是因为这几件事在好几个脚本里都要用：
build_qa.py / make_qc_sample.py / build_final.py
"""
import csv
import io
import re

# 千分位必须成组，否则相邻数字会被粘连成一个大数
NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
YEARS = {str(y) for y in range(2015, 2027)}


def read_csv_any(path):
    """读 CSV，自动识别编码。

    Excel（中文版）另存为 CSV 时会写成 GBK，把我们写出的 utf-8-sig 覆盖掉，
    所以这里不能假定编码。
    """
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return list(csv.DictReader(io.StringIO(raw.decode(enc)))), enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"认不出 {path} 的编码")


def _norm(s):
    s = s.translate(str.maketrans("０１２３４５６７８９，．", "0123456789,."))
    s = s.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def numbers_in(text):
    """提取 3 位以上的数字（页码、序号之类的短数字不算）。"""
    out = set()
    for m in NUM.finditer(text):
        t = _norm(m.group(0))
        if len(t.replace(".", "")) >= 3:
            out.add(t)
    return out


def big_numbers(text):
    """提取"有信息量"的数字：百分比，以及 5 位以上、不是年份的整数。

    用来判"问题里是不是已经把答案数字报出来了"——年份和页码不算线索。
    """
    out = set()
    for m in re.findall(r"\d[\d,]*(?:\.\d+)?%?", text):
        t = m.replace(",", "")
        if t.endswith("%") and len(t) >= 4:
            out.add(t)
        elif t.isdigit() and len(t) >= 5 and t not in YEARS:
            out.add(t)
    return out


# 金额单位 → 倍率。统一折算到「元」再比，长单位必须排在前面，
# 否则「亿元」会被「元」先截走，只剩一个孤零零的「亿」。
_UNIT_SCALE = {"元": 1.0, "千元": 1e3, "万元": 1e4,
               "百万元": 1e6, "亿元": 1e8}
_TRAD = str.maketrans("億萬", "亿万")
_FIG = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(百万元|亿元|万元|千元|元|%)")


def key_figures(text):
    """提取带量纲的关键数字，金额一律折算成「元」。

    返回 {(kind, value)}，kind 取 "yuan" 或 "pct"。

    **为什么不复用 numbers_in / big_numbers**：它们只看数字本身，不看量纲。
    参考答案是财报原貌的繁体「千元」（如 2,760,242 千元），模型却习惯答
    简体「亿元」（如 1,066.8 亿元）——同一个事实，字符串对不上。
    实测下来，老口径下 174 个"命中"里 164 个（94%）是年份，等于没在量知识。

    裸年份（2015–2026）不带单位，天然被排除在外，不用再特判。
    """
    out = set()
    for m in _FIG.finditer(text.translate(_TRAD)):
        raw, unit = m.group(1), m.group(2)
        try:
            v = float(raw.replace(",", ""))
        except ValueError:               # 形如 "1,23" 的畸形千分位
            continue
        if unit == "%":
            out.add(("pct", round(v, 2)))
        else:
            out.add(("yuan", v * _UNIT_SCALE[unit]))
    return out


def match_figures(ref, pred, rel=0.01, pct_tol=0.05):
    """ref 里的关键数字有多少能在 pred 里找到，返回命中的那个子集。

    金额按**相对误差**判（默认 1%）—— 换算和四舍五入必然带来小偏差：
    106,683,100 千元 = 1,066.831 亿元，模型写「1,066.8 亿元」，差 0.003%。
    百分比按**绝对差**判（默认 0.05 个百分点）。

    注意它偏宽：模型输出里数字越多，蒙中的机会越大。所以判分时还要
    另外统计「有多少是照着题干抄的」，两个数一起看才有意义。
    """
    pv = [v for k, v in pred if k == "yuan"]
    pp = [v for k, v in pred if k == "pct"]
    hit = set()
    for kind, v in ref:
        if kind == "pct":
            ok = any(abs(v - w) <= pct_tol for w in pp)
        else:
            ok = any(abs(v - w) <= max(abs(v), abs(w)) * rel for w in pv)
        if ok:
            hit.add((kind, v))
    return hit


def is_leading(question, answer):
    """问题里已经含了答案里的关键数字 → 引导性提问。

    为什么在意：测试集里出现这种题，模型可能是照着题干复述而不是真的记住了，
    四格对照会虚高。训练集里则宽容得多——现实中投资者就是这么问的。
    """
    return bool(big_numbers(question) & big_numbers(answer))


def is_half_refusal(answer):
    """答案里夹着「材料未涉及」——部分是有效信息，部分是拒绝作答。"""
    return any(w in answer for w in
               ("材料未涉及", "材料未提及", "材料未披露", "原文未涉及"))
