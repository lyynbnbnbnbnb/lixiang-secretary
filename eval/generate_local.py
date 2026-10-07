# -*- coding: utf-8 -*-
"""在本机 CPU 上跑 RAG 两条路线（rag / sft_rag）。

**为什么有这个脚本**：Colab 免费配额反复出问题，而这两条路线必须跑出来，
四格表才完整。本机 16 核 / 16 GB，纯 CPU 推 1.5B 模型虽慢，但可以挂后台
无人值守跑完。

和 generate.py 的三处关键差别，都是 CPU 逼出来的：

  1. **先拼提示词、再加载生成模型**。16 GB 内存里，向量模型（0.6B fp32，
     2.4 GB）和生成模型（1.5B fp32，6 GB）不能同时在场再叠上注意力矩阵。
     GPU 上无所谓，CPU 上必须先放掉前者。（分词器很小，可以先加载来套模板。）
  2. **一条一条生成（batch=1）**。CPU 上没有 Flash Attention ——
     注意力矩阵是实打实的 batch × 头数 × 长度² 的 fp32 数组。
     5700 token、12 头、batch 1 就要 1.6 GB。
  3. **上下文默认截到 3500 字、topk 默认 5**。CPU 上 prefill 是绝对大头，
     提示词长度直接决定要跑几小时。取舍写在 README 里。

用法：
    python generate_local.py --arm rag --benchmark      # 先测一条，估时间
    python generate_local.py --arm rag                  # 全量 140 条
    python generate_local.py --arm sft_rag --adapter ../train/lora_adapter
"""
import argparse
import gc
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "..", "data")
GRID = os.path.join(DATA, "eval_grid.jsonl")


def build_prompts(model_name, emb_model, topk, ctx_chars, limit=0):
    """拼好全部提示词就立刻放掉检索器 —— CPU 内存要省着用。

    分词器很小，先加载它来套 chat 模板；生成模型留到检索器释放之后再加载。
    """
    import generate as G
    from transformers import AutoTokenizer

    grid = [json.loads(l) for l in open(GRID, encoding="utf-8")]
    if limit:
        grid = grid[:limit]

    print("加载分词器…", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.time()
    retriever = G.build_retriever("hybrid", emb_model=emb_model)
    print(f"检索器就绪（{time.time()-t0:.0f}s），开始检索…", flush=True)

    prompts, metas = [], []
    t0 = time.time()
    for i, r in enumerate(grid):
        q = r.get("question_eval") or r["question"]
        hits = retriever.search(q, topk=topk)
        ctx = "\n\n".join(f"【{h['doc']} 第 {h['page']} 页】\n{h['text']}"
                          for h in hits)
        prompts.append(G.chat(tok, G.SYSTEM_RAG.format(context=ctx[:ctx_chars]),
                              q))
        metas.append((r["qa_id"], r["cell"]))
        if (i + 1) % 25 == 0:
            print(f"  检索 {i+1}/{len(grid)}（{time.time()-t0:.0f}s）",
                  flush=True)

    del retriever                                  # ★ 放掉向量模型
    gc.collect()
    try:                # GPU 上还要把缓存还给显存，6 GB 卡经不起占着不还
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                         # noqa: BLE001
        pass
    print(f"检索完成，向量模型已释放。共 {len(prompts)} 条提示词\n", flush=True)
    return tok, prompts, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["rag", "sft_rag"])
    ap.add_argument("--model",
                    default=os.path.join(HERE, "..", "train", "models",
                                         "Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--emb-model",
                    default=os.path.join(HERE, "..", "train", "models",
                                         "Qwen3-Embedding-0.6B"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--ctx-chars", type=int, default=3500)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    ap.add_argument("--benchmark", action="store_true",
                    help="只跑 1 条并估算总耗时，不写结果")
    ap.add_argument("--out", default=os.path.join(HERE, "preds"))
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16"])
    args = ap.parse_args()

    if args.arm == "sft_rag" and not args.adapter:
        print("sft_rag 需要 --adapter 指定 LoRA 目录")
        return 2

    limit = 1 if args.benchmark else args.limit
    tok, prompts, metas = build_prompts(args.model, args.emb_model,
                                        args.topk, args.ctx_chars, limit)

    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    # ★ 有 GPU 就用 GPU。这台机器本来就有 RTX 3060，之前一直没用到，
    #   是因为装的是 torch 的 CPU-only 轮子（`torch.__version__` 带 +cpu），
    #   `torch.cuda.is_available()` 恒为 False —— 和有没有显卡无关。
    #   换上 cuXXX 轮子之后这一行才真正有意义。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(os.cpu_count())
    print(f"加载 {args.model}（{device} / {args.dtype}）…", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    if args.adapter:
        from peft import PeftModel
        print(f"挂载 LoRA：{args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    # from_pretrained 只把权重放在 CPU，必须显式搬过去；
    # 漏了这行不会报错，只会在 CPU 上慢慢跑 —— 正是这个项目栽过的那类坑。
    model = model.to(device).eval()
    print(f"模型就绪（{time.time()-t0:.0f}s，{device}）\n", flush=True)

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"{args.arm}.jsonl")
    part = out + ".part"

    # 逐条落盘 + 断点续跑。原先只在 140 条全跑完后才写文件，
    # 中途被 kill（内存不足、关机、手滑）就是 2.4 小时全白跑 —— 实测栽过一次。
    start = 0
    if not args.benchmark and os.path.exists(part):
        with open(part, encoding="utf-8") as fh:
            start = sum(1 for _ in fh)
        print(f"续跑：{part} 已有 {start} 条，从第 {start + 1} 条接着跑",
              flush=True)

    fout = None if args.benchmark else open(part, "a", encoding="utf-8")
    outs, t0 = [], time.time()
    for i, p in enumerate(prompts):
        if i < start:
            continue
        enc = tok(p, return_tensors="pt", truncation=True,
                  max_length=8192).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        txt = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        outs.append(txt)

        if fout is not None:
            fout.write(json.dumps({"qa_id": metas[i][0], "cell": metas[i][1],
                                   "prediction": txt},
                                  ensure_ascii=False) + "\n")
            fout.flush()

        done = i + 1
        el = time.time() - t0
        progress = done - start               # 续跑时本轮真正新生成的条数
        eta = el / progress * (len(prompts) - done)
        print(f"  {done}/{len(prompts)}  {el:.0f}s 已用"
              f"  预计还需 {eta/60:.1f} 分钟", flush=True)
        if progress == 1:
            print(f"  ── 样本 ──\n  {txt[:200]}")
            print(f"  ── 【依据】{'有' if '【依据' in txt else '无'}  "
                  f"「」引用{'有' if '「' in txt else '无'} ──\n", flush=True)

    if args.benchmark:
        per = (time.time() - t0) / len(outs)
        print(f"\n单条 {per:.0f} 秒 → 140 条约 {per*140/3600:.1f} 小时")
        return 0

    fout.close()
    os.replace(part, out)                     # 全跑完才从 .part 转正
    lines = [json.loads(l) for l in open(out, encoding="utf-8")]
    n_cite = sum("【依据" in r["prediction"] for r in lines)
    print(f"\n写出 {out}（{len(lines)} 条，【依据】{n_cite}/{len(lines)}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
