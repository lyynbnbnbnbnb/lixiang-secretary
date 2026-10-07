# -*- coding: utf-8 -*-
"""在 Colab 里跑「带检索」的路线（rag / sft_rag）。

**为什么要有这个文件**：完整命令行有一百多个字符，粘贴到 Colab 时极易被
插进一个换行。一旦断开，`--out` 那截就被 IPython 当成 Python 代码解析，
报出莫名其妙的 `NameError: name 'out' is not defined`。
把参数写死在脚本里，用户只需粘一行极短的命令。

用法（Colab）：
    python run_sft_rag.py all             # 两条都跑（rag → sft_rag），约 50 分钟
    python run_sft_rag.py sft_rag         # 只跑 sft_rag
    python run_sft_rag.py rag 3           # 只跑 rag 前 3 条（试跑）

**两条都跑、一次挂机**，是因为它们是同一轮里唯一还没跑对的两条：
上一轮的 rag 被截断 bug 污染了，sft_rag 则是第一次跑。

跑完自动打包，不用再手动 zip。
"""
import json
import os
import subprocess
import sys

GEN = "/content/lixiang/eval/generate.py"
ADAPTER = "/content/lixiang/lora_adapter"
PREDS = "/content/lixiang/eval/preds"
OUTZIP = "/content/lixiang_rag_arms.zip"

USAGE = "用法：python run_sft_rag.py {all|rag|sft_rag} [试跑条数]"


def run_arm(arm, limit=0, batch=4):
    cmd = [sys.executable, GEN, "--arm", arm, "--load-4bit", "--out", PREDS,
           "--batch", str(batch)]
    if arm == "sft_rag":
        cmd += ["--adapter", ADAPTER]
    if limit:
        cmd += ["--limit", str(limit)]

    # 显存碎片化会让长提示词的生成更容易 OOM，这个开关让分配器能挪动已分配的块
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

    print(f"\n{'#' * 78}\n# {arm}（batch={batch}）\n{'#' * 78}")
    print("执行：", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print(f"\n{arm} 生成失败，退出码 {r.returncode}")
        print("若是显存不足（报 out of memory）：")
        print("  再跑一次，命令末尾加个 2 降 batch：")
        print(f"    python run_sft_rag.py {arm} 0 2")
        print("  （中间那个 0 表示不限条数）")
        return None

    path = f"{PREDS}/{arm}.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    n = len(rows)
    n_cite = sum("【依据" in x["prediction"] for x in rows)
    n_quote = sum("「" in x["prediction"] and "」" in x["prediction"]
                  for x in rows)
    print(f"\n{arm}：{n} 条")
    print(f"格式自检：【依据】{n_cite}/{n}   「」引用 {n_quote}/{n}"
          f"   （sft_rag 应接近满格；rag 因为没微调，低是正常的）")
    return n


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("all", "rag", "sft_rag"):
        print(USAGE)
        return 2
    which = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    arms = ["rag", "sft_rag"] if which == "all" else [which]

    for arm in arms:
        if run_arm(arm, limit, batch) is None:
            return 1

    # 顺手打包，省得再手动敲 zip
    import zipfile
    names = [f"eval/preds/{a}.jsonl" for a in arms]
    with zipfile.ZipFile(OUTZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            p = f"/content/lixiang/{n}"
            if os.path.exists(p):
                z.write(p, n)
    print(f"\n{'=' * 78}")
    print(f"打包完成 → {OUTZIP}（{os.path.getsize(OUTZIP)/1024:.0f} KB）")
    print("下一步：")
    print("  from google.colab import files")
    print(f"  files.download('{OUTZIP}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
