# -*- coding: utf-8 -*-
"""把 Colab 需要的东西打成一个 zip，省得在网页上一个个传。

打包内容按 generate.py 期望的目录结构摆好：
    lixiang/data/final_train.jsonl   训练集
    lixiang/data/eval_grid.jsonl     四格评测集（140 条）
    lixiang/data/pages.jsonl         检索语料（1431 页）
    lixiang/eval/generate.py         生成器

用法：
    python make_colab_bundle.py
输出：
    train/lixiang_colab.zip
"""
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ITEMS = [
    ("data/final_train.jsonl", "lixiang/data/final_train.jsonl"),
    ("data/eval_grid.jsonl", "lixiang/data/eval_grid.jsonl"),
    ("data/pages.jsonl", "lixiang/data/pages.jsonl"),
    ("eval/generate.py", "lixiang/eval/generate.py"),
]


def main():
    out = os.path.join(HERE, "lixiang_colab.zip")
    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for src_rel, arc in ITEMS:
            src = os.path.join(ROOT, src_rel)
            if not os.path.exists(src):
                print(f"缺文件：{src}")
                return 1
            z.write(src, arc)
            size = os.path.getsize(src)
            total += size
            print(f"  + {arc:<38}{size/1024:>8.0f} KB")
    print(f"\n写出 {out}（{os.path.getsize(out)/1e6:.1f} MB，"
          f"压缩前 {total/1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
