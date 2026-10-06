# -*- coding: utf-8 -*-
"""
下载理想汽车（港股 02015 / Li Auto）历年财报 PDF（中文版）。

数据源：港交所披露易 HKEXnews
  1) 用 prefix.do 把股票代码换成内部 stockId
  2) 用 titleSearchServlet.do 拉「财务报表」类别下的年报 / 中报列表
  3) 逐个下载到 data/raw/

用法：
    python download.py                # 下载 2021 年至今的全部年报 + 中报
    python download.py --from 2023    # 只下 2023 年及以后的
    python download.py --list         # 只列清单不下
"""
import argparse
import json
import os
import re
import sys
import time

import requests

BASE = "https://www1.hkexnews.hk"
STOCK_CODE = "02015"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": f"{BASE}/search/titlesearch.xhtml",
}
# t1code=40000 财务报表/ESG资料；t2code=40100 年报，40200 中期/半年报
DOC_TYPES = {"40100": "年报", "40200": "中报"}
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")


def get_stock_id(code: str) -> str:
    r = requests.get(
        f"{BASE}/search/prefix.do",
        params={"callback": "cb", "lang": "ZH", "type": "A", "name": code, "market": "SEHK"},
        headers=HEADERS, timeout=30,
    )
    r.raise_for_status()
    m = re.search(r"\((\{.*\})\)", r.text, re.S)
    if not m:
        raise RuntimeError(f"解析 prefix.do 失败: {r.text[:200]}")
    for item in json.loads(m.group(1)).get("stockInfo", []):
        if item["code"].lstrip("0") == code.lstrip("0"):
            return str(item["stockId"])
    raise RuntimeError(f"没找到股票代码 {code}")


def list_reports(stock_id: str, t2code: str, from_date: str, to_date: str):
    r = requests.get(
        f"{BASE}/search/titleSearchServlet.do",
        params={
            "sortDir": 0, "sortByOptions": "DateTime", "category": 0,
            "market": "SEHK", "stockId": stock_id, "documentType": -1,
            "fromDate": from_date, "toDate": to_date, "title": "",
            "searchType": 1, "t1code": 40000, "t2Gcode": -2, "t2code": t2code,
            "rowRange": 100, "lang": "zh",
        },
        headers=HEADERS, timeout=30,
    )
    r.raise_for_status()
    rows = json.loads(json.loads(r.text)["result"])
    out = []
    for row in rows:
        title = row.get("TITLE", "")
        year = re.search(r"(20\d{2})", title)
        link = row.get("FILE_LINK") or row.get("FILE_INFO", {}).get("fileLink", "")
        if not link:
            continue
        out.append({
            "year": year.group(1) if year else "unknown",
            "doc_type": DOC_TYPES[t2code],
            "date": row.get("DATE_TIME", ""),
            "title_cn": title,
            "url": BASE + link if link.startswith("/") else link,
        })
    return out


def download(url: str, dest: str) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print(f"  已存在，跳过（{os.path.getsize(dest)/1e6:.1f} MB）")
        return True
    r = requests.get(url, headers=HEADERS, timeout=120, stream=True)
    r.raise_for_status()
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
    if not open(tmp, "rb").read(5).startswith(b"%PDF"):
        os.remove(tmp)
        raise RuntimeError("下载到的不是 PDF")
    os.replace(tmp, dest)
    print(f"  完成 {os.path.getsize(dest)/1e6:.1f} MB")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_year", default="2021", help="起始年份，默认 2021")
    ap.add_argument("--to", dest="to_year", default=time.strftime("%Y"), help="结束年份")
    ap.add_argument("--list", action="store_true", help="只列清单，不下载")
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    stock_id = get_stock_id(STOCK_CODE)
    print(f"理想汽车 stockId = {stock_id}\n")

    from_date = f"{args.from_year}0101"
    to_date = f"{args.to_year}1231"

    reports = []
    for t2code in DOC_TYPES:
        reports += list_reports(stock_id, t2code, from_date, to_date)
    reports.sort(key=lambda x: (x["doc_type"], x["year"]))

    manifest = []
    for rep in reports:
        name = f"理想汽车_{rep['year']}{rep['doc_type']}.pdf"
        rep["file"] = name
        print(f"[{rep['year']} {rep['doc_type']}] 公告日 {rep['date']}")
        print(f"  {rep['url']}")
        if not args.list:
            download(rep["url"], os.path.join(RAW, name))
        manifest.append(rep)

    with open(os.path.join(HERE, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n共 {len(manifest)} 份，清单写入 data/manifest.json")


if __name__ == "__main__":
    sys.exit(main())
