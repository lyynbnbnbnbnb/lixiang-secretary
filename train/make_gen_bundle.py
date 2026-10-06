# -*- coding: utf-8 -*-
"""打一个"只做生成"的 Colab 上传包：语料 + 已训好的 LoRA 适配器。

跟 make_colab_bundle.py 的区别：
  make_colab_bundle.py  只给「完整跑一遍」（含训练），不含适配器
  本脚本                给「复用已训好的适配器，只补生成一条路线」，含适配器

为什么要单独一个包：sft_rag 路线需要「基座 + LoRA + 检索」三样同时在场，
而模型已经训好了，重训纯属浪费 Colab 配额。把适配器一起打包，用户在
Colab 上只需要传一个文件。

用法：
    python make_gen_bundle.py
输出：
    train/lixiang_gen_bundle.zip（约 70 MB）
"""
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 和 make_colab_bundle.py 保持同一套目录结构，generate.py 靠它定位数据
ITEMS = [
    ("data/final_train.jsonl", "lixiang/data/final_train.jsonl"),
    ("data/eval_grid.jsonl", "lixiang/data/eval_grid.jsonl"),
    ("data/pages.jsonl", "lixiang/data/pages.jsonl"),
    ("eval/generate.py", "lixiang/eval/generate.py"),
]

ADAPTER_DIR = os.path.join(HERE, "lora_adapter")

# 适配器里这些已经是压缩过的二进制，再 deflate 一遍白花时间
STORE_EXT = (".safetensors", ".bin", ".pt", ".gguf")


def main():
    if not os.path.isdir(ADAPTER_DIR):
        print(f"缺适配器目录：{ADAPTER_DIR}\n"
              f"（从 lixiang_results.zip 里把 lora_adapter/ 解压到 train/ 下）")
        return 1

    out = os.path.join(HERE, "lixiang_gen_bundle.zip")
    total = 0
    with zipfile.ZipFile(out, "w") as z:
        for src_rel, arc in ITEMS:
            src = os.path.join(ROOT, src_rel)
            if not os.path.exists(src):
                print(f"缺文件：{src}")
                return 1
            z.write(src, arc, compress_type=zipfile.ZIP_DEFLATED)
            size = os.path.getsize(src)
            total += size
            print(f"  + {arc:<38}{size/1024:>8.0f} KB")

        for name in sorted(os.listdir(ADAPTER_DIR)):
            src = os.path.join(ADAPTER_DIR, name)
            if not os.path.isfile(src):
                continue
            arc = f"lixiang/lora_adapter/{name}"
            ct = (zipfile.ZIP_STORED if name.endswith(STORE_EXT)
                  else zipfile.ZIP_DEFLATED)
            z.write(src, arc, compress_type=ct)
            size = os.path.getsize(src)
            total += size
            print(f"  + {arc:<38}{size/1024:>8.0f} KB")

    print(f"\n写出 {out}（{os.path.getsize(out)/1e6:.1f} MB，"
          f"压缩前 {total/1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
