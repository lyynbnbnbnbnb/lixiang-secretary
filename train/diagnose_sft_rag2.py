# -*- coding: utf-8 -*-
"""诊断第二轮：把 4bit/fp16 × system/上下文位置 的 2×2 补全，且用 hybrid 检索。

第一轮（diagnose_sft_rag.py）的结论：单独换 system、单独换精度，格式都还在。
但它用的是 BM25 检索，而真实预跑用的是 hybrid（向量 + BM25）。
没被覆盖的格子恰好就是真实跑的那一种组合，所以这轮把它补上。

四格：
  A  4bit · system=SYSTEM_RAG · hybrid 上下文     （≈ 第一轮 ②，用来确认复现）
  B  fp16 · system=SYSTEM_RAG · hybrid 上下文     （← 真实预跑用的就是这一格）
  C  fp16 · system=SYSTEM     · 上下文挪到 user   （候选改法）
  D  4bit · system=SYSTEM     · 上下文挪到 user   （候选改法）

判读：
  A 好、B 坏  → 是**精度**问题，sft_rag 必须用 4bit 基座跑
  B 好      → 预跑另有原因（那就要回看预跑日志里有没有「挂载 LoRA」那行）
  C/D 好    → 换个写法就能救，且和精度无关

用法（Colab）：
    python diagnose_sft_rag2.py
"""
import json
import sys

import torch

EVAL = "/content/lixiang/eval"
DATA = "/content/lixiang/data"
ADAPTER = "/content/lixiang/lora_adapter"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

sys.path.insert(0, EVAL)
import generate as G                                              # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer,     # noqa: E402
                          BitsAndBytesConfig)


def load(load_4bit):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb if load_4bit else None,
        torch_dtype=torch.float16, device_map="auto")
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, ADAPTER)
    if hasattr(m, "gradient_checkpointing_disable"):
        m.gradient_checkpointing_disable()
    m.config.use_cache = True
    m.eval()
    return m, tok


def ask(m, tok, system, user, max_new=192):
    text = tok.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def verdict(text):
    return (f"【依据】{'有' if '【依据' in text else '无'}  "
            f"「」{'有' if '「' in text and '」' in text else '无'}  "
            f"简体开头{'是' if text[:1] in '0123456789' or '一' <= text[:1] <= '鿿' else '?'}  "
            f"长度{len(text)}")


def main():
    grid = [json.loads(l) for l in
            open(f"{DATA}/eval_grid.jsonl", encoding="utf-8")]
    item = grid[0]
    q = item.get("question_eval") or item["question"]
    print(f"诊断题：{q}\n")

    print("建 hybrid 检索（和真实预跑一致，首次要向量化 1431 页）…")
    retriever = G.build_retriever("hybrid")
    hits = retriever.search(q, topk=8)
    ctx = "\n\n".join(f"【{h['doc']} 第 {h['page']} 页】\n{h['text']}"
                      for h in hits)[:6000]
    print("检索到的页：", [f"{h['doc']} p{h['page']}" for h in hits])
    print()

    def with_rag_ctx():
        return G.SYSTEM_RAG.format(context=ctx)

    def with_user_ctx():
        return f"【财报原文片段】\n{ctx}\n\n投资者提问：{q}"

    for tag, load_4bit, sys_maker in [
            ("A  4bit · 上下文在 system（≈ 第一轮②，确认复现）", True,
             with_rag_ctx),
            ("B  fp16 · 上下文在 system（← 真实预跑用的这格）", False,
             with_rag_ctx),
            ("C  fp16 · 上下文放 user", False, with_user_ctx),
            ("D  4bit · 上下文放 user", True, with_user_ctx),
    ]:
        print(f"加载 {'4bit' if load_4bit else 'fp16'} 基座 + 适配器 …")
        m, tok = load(load_4bit)
        user = q if sys_maker is with_rag_ctx else with_user_ctx()
        out = ask(m, tok, sys_maker(), user)
        print("=" * 74)
        print(f"### {tag}")
        print(f"    格式：{verdict(out)}")
        print(out[:400])
        print()
        del m
        torch.cuda.empty_cache()

    print("=" * 74)
    print("判读：")
    print("  A 好 B 坏        → 精度问题，sft_rag 必须用 4bit 基座")
    print("  B 也好           → 预跑另有原因，回头看预跑日志有没有「挂载 LoRA：」那行")
    print("  C / D 好         → 把上下文挪到 user 就能救，与精度无关")
    return 0


if __name__ == "__main__":
    sys.exit(main())
