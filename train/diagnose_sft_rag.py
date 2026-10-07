# -*- coding: utf-8 -*-
"""诊断：为什么 sft_rag 跑出来没有董秘格式？

现象：`generate.py --arm sft_rag` 的输出是繁体财报腔、没有【依据】、没有「」引用、
答非所问（问 2021 中报却答 2022 年）。而上一轮 notebook 里的 `sft` 路线
（同一份适配器）格式完全正常。

两处嫌疑：
  A. **基座精度**：上一轮三条路线跑的都是 4bit 量化基座（训练时就是这么加载的），
     而 generate.py 默认 fp16。适配器是在 4bit 基座上训出来的。
  B. **system 提示词**：训练时 system 是短的那句 SYSTEM，而 sft_rag 给的是
     带 6000 字上下文的 SYSTEM_RAG。SFT 模型对提示词格式极敏感，
     换个没见过的 system 就可能整个失效。

本脚本用四组对照把这些变量逐个隔离。跑完看哪一组恢复了董秘格式。

用法（Colab，会话里应已有 /content/lixiang）：
    python diagnose_sft_rag.py
"""
import json
import os
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


def build_context(q, topk=8):
    """用 BM25 取片段就够诊断了 —— 快，且不依赖 GPU 上的向量模型。"""
    r = G.build_retriever("bm25")
    hits = r.search(q, topk=topk)
    return "\n\n".join(f"【{h['doc']} 第 {h['page']} 页】\n{h['text']}"
                       for h in hits)[:6000]


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
    """只看格式 —— 数字对不对是另一回事，这里要诊断的是格式为什么丢了。"""
    return (f"【依据】{'有' if '【依据' in text else '无'}  "
            f"「」引用{'有' if '「' in text and '」' in text else '无'}  "
            f"简体{'是' if not any(c in text for c in '個為與這') else '否(繁体)'}  "
            f"长度{len(text)}")


def main():
    grid = [json.loads(l) for l in
            open(f"{DATA}/eval_grid.jsonl", encoding="utf-8")]
    item = grid[0]
    q = item.get("question_eval") or item["question"]
    print(f"诊断题：{q}")
    print(f"（来自 {item['doc']} 第 {item['page']} 页）\n")

    ctx = build_context(q)
    print(f"检索到 {len(ctx)} 字上下文\n")

    print("加载 4bit 基座 + 适配器 …")
    m4, tok = load(True)

    cases = [
        ("① 4bit · system=SYSTEM（训练时原样）· 无上下文",
         G.SYSTEM, q),
        ("② 4bit · system=SYSTEM_RAG · 上下文在 system（现在跑的做法）",
         G.SYSTEM_RAG.format(context=ctx), q),
        ("③ 4bit · system=SYSTEM · 上下文放 user（改法）",
         G.SYSTEM, f"【财报原文片段】\n{ctx}\n\n投资者提问：{q}"),
    ]
    for tag, system, user in cases:
        out = ask(m4, tok, system, user)
        print("=" * 74)
        print(f"### {tag}")
        print(f"    格式：{verdict(out)}")
        print(out[:360])
        print()

    del m4
    torch.cuda.empty_cache()

    print("加载 fp16 基座 + 适配器 …")
    m16, tok16 = load(False)
    out = ask(m16, tok16, G.SYSTEM, q)
    print("=" * 74)
    print("### ④ fp16 · system=SYSTEM · 无上下文（隔离精度的影响）")
    print(f"    格式：{verdict(out)}")
    print(out[:360])
    print()
    print("=" * 74)
    print("怎么读这四组：")
    print("  ① 有格式、④ 没格式  → 是**精度**问题，sft_rag 必须用 4bit 基座")
    print("  ① 有格式、② 没格式  → 是**system 提示词**问题，上下文要挪到 user")
    print("  ① 没格式            → 适配器压根没生效，另一回事")
    print("  ①④ 都有格式、②③ 都没 → 上下文本身把模型带跑了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
