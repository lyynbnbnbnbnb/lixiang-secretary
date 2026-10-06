# -*- coding: utf-8 -*-
"""
第 5 步（生成器）：在 Colab 上跑模型，产出四格评测集的答案。

四条路线（arm），对着同一套 140 条题各答一遍：

  base      原始基座，不给上下文              —— 训练前，模型知道多少
  sft       基座 + LoRA 适配器，不给上下文     —— 微调后，写进权重的有多少
  rag       基座 + 检索到的原文片段            —— 「知识库 + 没微调的小模型」
  sft_rag   基座 + LoRA + 检索到的原文片段     —— 「知识库 + 微调过的小模型」

前三条各自缺一半：sft 会答但数字是编的，rag 拿得到原文但不会答。
sft_rag 才是完整方案——口径来自微调，事实来自检索。

判分不在本脚本做——判分用 eval/judge.py，两边分开才能换模型重跑。

用法（Colab）：
    python generate.py --arm base
    python generate.py --arm sft     --adapter /content/lora_out
    python generate.py --arm rag
    python generate.py --arm sft_rag --adapter /content/lora_out

输出：
    preds/{arm}.jsonl     每行 {qa_id, cell, prediction}，直接喂给 judge.py
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

SYSTEM = "你是理想汽车（港股 02015）的董事会秘书，正在回答投资者提问。"

SYSTEM_RAG = """你是理想汽车（港股 02015）的董事会秘书，正在回答投资者提问。

只依据下面提供的财报原文片段作答，不要使用片段之外的知识。
如果片段里没有足够信息，直接回答「材料未涉及」。

【财报原文片段】
{context}"""


# ---------------------------------------------------------------- 检索（只用标准库）

def bigrams(text):
    """中文字符二元组 —— 不依赖分词库的轻量索引方式。"""
    t = re.sub(r"[^\w一-鿿]", "", text)
    return [t[i:i + 2] for i in range(len(t) - 1)]


class BM25:
    """极简 BM25，字符二元组，纯标准库。语料只有一千多页，够用。"""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.toks = [bigrams(d["text"]) for d in docs]
        self.lens = [len(t) for t in self.toks]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        df = Counter()
        for t in self.toks:
            df.update(set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5))
                    for w, c in df.items()}
        self.tf = [Counter(t) for t in self.toks]

    def search(self, query, topk=5):
        q = bigrams(query)
        scores = []
        for i in range(len(self.docs)):
            s = 0.0
            for w in q:
                f = self.tf[i].get(w)
                if not f:
                    continue
                s += self.idf.get(w, 0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            scores.append((s, i))
        scores.sort(reverse=True)
        return [self.docs[i] for _, i in scores[:topk]]


class EmbedIndex:
    """向量检索。用课程推荐的 Qwen3-Embedding-0.6B。

    和 BM25 是互补的：向量管意思相近，BM25 管字面必须出现。
    作业 A 方向要求的就是「向量 + BM25」，所以 RAG 对照也要两条都上。

    两个容易踩的点（都踩过）：
      1. **模型必须搬到 GPU**，否则 1431 页在 CPU 上要跑几十分钟
      2. **Qwen3-Embedding 取最后一个 token 做池化**，不是第一个。
         取 first_hidden_state[:, 0] 不会报错，但向量质量是坏的
    """

    # 课程里提过：Qwen3 系列在查询侧加英文指令前缀，官方称还能涨 1–5%
    QUERY_PREFIX = ("Instruct: Retrieve passages from a Chinese company's "
                    "financial report that answer the question\nQuery: ")

    def __init__(self, docs, model_name="Qwen/Qwen3-Embedding-0.6B",
                 cache_dir=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.docs = docs
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"加载向量模型 {model_name}（{self.device}）…")
        self.tok = AutoTokenizer.from_pretrained(model_name,
                                                 trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name,
                                               trust_remote_code=True)
        self.model = self.model.to(self.device).eval()      # ← 别漏这句

        cache = os.path.join(cache_dir or HERE,
                             f"_emb_cache_{len(docs)}.pt")
        if os.path.exists(cache):
            print(f"读向量缓存 {cache}")
            self.emb = torch.load(cache).to(self.device)
        else:
            print(f"向量化 {len(docs)} 页 …")
            self.emb = self._encode([d["text"][:1200] for d in docs],
                                    batch=64)
            torch.save(self.emb.cpu(), cache)
            print(f"向量已缓存到 {cache}")

    def _encode(self, texts, batch=64):
        torch = self.torch
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], padding=True, truncation=True,
                           max_length=512,
                           return_tensors="pt").to(self.device)
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state
            # 取每个序列最后一个非 padding token
            last = enc["attention_mask"].sum(dim=1) - 1
            v = h[torch.arange(h.size(0), device=self.device), last]
            out.append(torch.nn.functional.normalize(v, dim=-1))
            if (i // batch) % 5 == 0:
                print(f"  {min(i+batch, len(texts))}/{len(texts)}")
        return torch.cat(out)

    def search(self, query, topk=5):
        torch = self.torch
        enc = self.tok([self.QUERY_PREFIX + query], padding=True,
                       truncation=True, max_length=512,
                       return_tensors="pt").to(self.device)
        with torch.no_grad():
            h = self.model(**enc).last_hidden_state
        q = h[0, enc["attention_mask"].sum() - 1]
        q = torch.nn.functional.normalize(q, dim=-1)
        sim = self.emb @ q
        idx = torch.topk(sim, min(topk, len(self.docs))).indices.tolist()
        return [self.docs[i] for i in idx]


class Hybrid:
    """混合检索：两路各出一份排名，用 RRF（倒数排名融合）合并。

    RRF 只用到名次、不用分数，所以不用去调两边的分数量纲。
    """

    def __init__(self, *retrievers, k=60):
        self.rs, self.k = retrievers, k

    def search(self, query, topk=5):
        fused, seen = {}, {}
        for r in self.rs:
            for rank, d in enumerate(r.search(query, topk=topk * 4)):
                key = (d["doc"], d["page"])
                fused[key] = fused.get(key, 0) + 1.0 / (self.k + rank + 1)
                seen[key] = d
        order = sorted(fused, key=fused.get, reverse=True)[:topk]
        return [seen[k] for k in order]


def load_corpus():
    """检索语料 = 全部报告逐页正文（不是只取叙事章节，要公平）。"""
    pages = [json.loads(l) for l in
             open(os.path.join(DATA, "pages.jsonl"), encoding="utf-8")]
    return [{"doc": p["doc"], "page": p["page"], "text": p["text"]}
            for p in pages if len(p["text"]) > 80]


def build_retriever(kind="bm25"):
    docs = load_corpus()
    print(f"检索语料：{len(docs)} 页")
    if kind == "bm25":
        return BM25(docs)
    try:
        emb = EmbedIndex(docs)
    except Exception as e:                                   # noqa: BLE001
        print(f"向量模型加载失败（{e}）—— 退回 BM25 单独跑，不影响流程")
        return BM25(docs)
    return emb if kind == "embed" else Hybrid(BM25(docs), emb)


# ---------------------------------------------------------------- 生成

def load_model(model_name, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # T4(Turing) 无 bf16 硬件支持，用 fp16
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
             else torch.float16)
    print(f"加载 {model_name} …（{dtype}）")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # 解码器批量生成必须左填充
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True)
    if adapter:
        from peft import PeftModel
        print(f"挂载 LoRA：{adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    # 梯度检查点和 KV cache 互斥；训练态下直接生成会出复读
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    return model, tok


def chat(tok, system, user):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)


def generate(model, tok, prompts, batch=8, max_new=192):
    """贪心解码，单次确定性推理（对应课程里「全部单次确定性推理」的口径）。

    max_new 默认 192：基座不会主动输出 EOS，设太大会一路生成到上限；
    模型一旦退化就会把同一个 token 重复 max_new 次（实测 512 → 复读 511 遍）。
    """
    import torch
    if tok.padding_side != "left":     # 右填充会让批量生成从 pad 之后续写
        tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=3072).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new,
                                 do_sample=False, temperature=None,
                                 top_p=None,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(chunk)):
            n_in = enc["input_ids"].shape[1]
            outs.append(tok.decode(gen[j][n_in:], skip_special_tokens=True).strip())
        print(f"  {min(i+batch, len(prompts))}/{len(prompts)}")
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["base", "sft", "rag", "sft_rag"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="基座模型（和微调时用的要保持一致）")
    ap.add_argument("--adapter", default=None,
                    help="LoRA 适配器目录（sft / sft_rag 用）")
    ap.add_argument("--topk", type=int, default=8, help="rag 检索几页")
    ap.add_argument("--retriever", default="hybrid",
                    choices=["bm25", "embed", "hybrid"],
                    help="rag 用哪种检索（默认混合，对应作业 A 的「向量 + BM25」）")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（试跑）")
    ap.add_argument("--out", default=os.path.join(HERE, "preds"))
    args = ap.parse_args()

    grid = [json.loads(l) for l in
            open(os.path.join(DATA, "eval_grid.jsonl"), encoding="utf-8")]
    if args.limit:
        grid = grid[:args.limit]
    print(f"{args.arm}：{len(grid)} 条题")

    # 两条"带检索"的路线共用一套上下文注入，两条"带适配器"的共用一次挂载
    use_rag = args.arm in ("rag", "sft_rag")
    use_adapter = args.arm in ("sft", "sft_rag")

    retriever = None
    if use_rag:
        retriever = build_retriever(args.retriever)

    model, tok = load_model(args.model,
                            args.adapter if use_adapter else None)

    prompts, metas = [], []
    for r in grid:
        q = r.get("question_eval") or r["question"]
        if use_rag:
            hits = retriever.search(q, topk=args.topk)
            ctx = "\n\n".join(
                f"【{h['doc']} 第 {h['page']} 页】\n{h['text']}"
                for h in hits)
            sys_p = SYSTEM_RAG.format(context=ctx[:6000])
        else:
            sys_p = SYSTEM
        prompts.append(chat(tok, sys_p, q))
        metas.append({"qa_id": r["qa_id"], "cell": r["cell"]})

    preds = generate(model, tok, prompts, batch=args.batch)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.arm}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for m, p in zip(metas, preds):
            f.write(json.dumps({**m, "prediction": p},
                               ensure_ascii=False) + "\n")
    n_empty = sum(1 for p in preds if not p.strip())
    print(f"\n写出 {path}（{len(preds)} 条，空回答 {n_empty} 条）")
    print(f"下一步：把 {args.arm}.jsonl 下载下来，跑 "
          f"python judge.py --preds {args.arm}.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
