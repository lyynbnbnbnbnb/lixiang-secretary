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
import time
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
      1. **模型必须搬到 GPU**。CPU 上 fp32 跑 1431 页不是「几十分钟」——
         实测 batch=64 时一批就要好几分钟，全程以小时计。而且 0.6B fp32
         常驻 2.4 GB，批大了还要再叠注意力矩阵，16 GB 机器容易换页、
         越跑越慢。`EMB_BATCH` 因此压到 32。
      2. **Qwen3-Embedding 取最后一个 token 做池化**，不是第一个。
         取 first_hidden_state[:, 0] 不会报错，但向量质量是坏的
    """

    # 向量化的批大小。别调大：注意力矩阵是 batch × 头数 × 512² 的 fp32 数组，
    # batch=64 时单这一项就 1 GB，本机实测直接把空闲内存吃到 1.5 GB 并开始换页。
    # 16 是下限附近：再小一批要跑 40 秒出头，但批间开销开始咬人。
    EMB_BATCH = 16

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
        print(f"加载向量模型 {model_name}（{self.device}）…", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_name,
                                                 trust_remote_code=True)
        # ★★ 必须显式给 dtype。config.json 里写着 torch_dtype=bfloat16，
        #    不指定就按 bf16 加载 —— GPU 上没问题，本机 CPU 是灾难：
        #    i7-10870H（Comet Lake）只有 AVX2，没有原生 bf16，torch 走模拟
        #    实现，实测 batch=32 一批要 5 分钟以上（全量 1431 页 >3.5 小时）。
        #    fp32 实测快好几倍。这个坑不报错、不掉精度，只是慢，最难查。
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        try:
            self.model = AutoModel.from_pretrained(model_name,
                                                   trust_remote_code=True,
                                                   dtype=dtype)
        except TypeError:                   # transformers 4.x 里叫 torch_dtype
            self.model = AutoModel.from_pretrained(model_name,
                                                   trust_remote_code=True,
                                                   torch_dtype=dtype)
        self.model = self.model.to(self.device).eval()      # ← 别漏这句

        self.cache = os.path.join(cache_dir or HERE,
                                  f"_emb_cache_{len(docs)}.pt")
        if os.path.exists(self.cache):
            print(f"读向量缓存 {self.cache}", flush=True)
            self.emb = torch.load(self.cache).to(self.device)
        else:
            print(f"向量化 {len(docs)} 页（batch={self.EMB_BATCH}）…",
                  flush=True)
            self.emb = self._encode([d["text"][:1200] for d in docs],
                                    batch=self.EMB_BATCH)
            # 一律按 fp32 落盘：GPU 上模型是 bf16，算出来的向量也是 bf16，
            # 直接存下去，下次换设备/换 dtype 加载就会和查询向量对不上
            torch.save(self.emb.float().cpu(), self.cache)
            print(f"向量已缓存到 {self.cache}", flush=True)

    def _encode(self, texts, batch=None):
        """CPU 上这一步以小时计，所以每一批都必须看得见进度。

        为了这件事踩过两次：
          · **print 必须 flush**。输出被 `tee` / 重定向接走时 stdout 是
            块缓冲，不加 flush 的话跑两小时日志里也只有「向量化 …」一行，
            完全看不出是在跑还是卡死了 —— 我第一次就是这么误杀了它。
          · **要逐段落盘**。原先只在全部跑完后才 save，中途 Ctrl-C / 关机
            / 我手贱 kill，几小时全废。现在每 5 批写一次 .part，
            下次自动从断点续跑。
        """
        torch = self.torch
        batch = batch or self.EMB_BATCH
        part = self.cache + ".part"

        out, start = [], 0
        if os.path.exists(part):
            start, saved = torch.load(part)
            out = list(saved)
            print(f"  续跑：已编码 {start}/{len(texts)} 页", flush=True)

        t0 = time.time()
        for i in range(start, len(texts), batch):
            enc = self.tok(texts[i:i + batch], padding=True, truncation=True,
                           max_length=512,
                           return_tensors="pt").to(self.device)
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state
            # 取每个序列最后一个非 padding token
            last = enc["attention_mask"].sum(dim=1) - 1
            v = h[torch.arange(h.size(0), device=self.device), last]
            out.append(torch.nn.functional.normalize(v, dim=-1))

            done = min(i + batch, len(texts))
            if len(out) % 5 == 0 or done == len(texts):
                el = time.time() - t0
                per = el / max(1, done - start)
                print(f"  {done}/{len(texts)}  已用 {el/60:.1f} 分"
                      f"  预计还需 {per*(len(texts)-done)/60:.1f} 分",
                      flush=True)
                torch.save((done, out), part)

        emb = torch.cat(out)
        if os.path.exists(part):
            os.remove(part)                 # 全跑完才删，中途 kill 还能续
        return emb

    def search(self, query, topk=5):
        torch = self.torch
        enc = self.tok([self.QUERY_PREFIX + query], padding=True,
                       truncation=True, max_length=512,
                       return_tensors="pt").to(self.device)
        with torch.no_grad():
            h = self.model(**enc).last_hidden_state
        q = h[0, enc["attention_mask"].sum() - 1]
        q = torch.nn.functional.normalize(q, dim=-1)
        # 两边 dtype 不一定相同：缓存的向量是 fp32，而 GPU 上模型可能是
        # bf16，算出来的 q 就是 bf16，直接 `self.emb @ q` 会报
        # "addmv input tensors must have the same dtype"。
        # 统一到 fp32 再点积，1431×1024 的转换开销可以忽略。
        sim = self.emb.float() @ q.float()
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


def build_retriever(kind="bm25", emb_model=None):
    """emb_model 可以指本地目录 —— 离线跑时必须给，
    否则会去 HuggingFace 现场下载（本机走不了外网时会卡住）。"""
    docs = load_corpus()
    print(f"检索语料：{len(docs)} 页")
    if kind == "bm25":
        return BM25(docs)
    try:
        emb = EmbedIndex(docs, model_name=emb_model or "Qwen/Qwen3-Embedding-0.6B")
    except Exception as e:                                   # noqa: BLE001
        # ★ 这不是「不影响流程」：hybrid 退化成纯 BM25 就换了一个实验条件，
        #   跑出来的「建库」一列和 README 写的不是一回事。必须喊出来。
        print("!" * 72)
        print(f"!! 向量模型加载失败：{e}")
        print("!! 已退回 **纯 BM25**（hybrid/embed 名不副实）。")
        print("!! 这样跑出来的结果不能标成「建库 = hybrid」，请先修模型再重跑。")
        print(f"!! 检查：bash {os.path.join(HERE, '..', 'train', 'download_models.sh')}")
        print("!" * 72)
        if kind != "bm25":
            raise RuntimeError("向量模型不可用，拒绝以 BM25 冒充 hybrid") from e
        return BM25(docs)
    return emb if kind == "embed" else Hybrid(BM25(docs), emb)


# ---------------------------------------------------------------- 生成

def load_model(model_name, adapter=None, load_4bit=False):
    """加载基座（可选 4bit 量化）+ 可选挂载 LoRA。

    **为什么需要 load_4bit**：适配器是在 4bit 量化基座上训出来的（QLoRA）。
    拿它去挂 fp16 基座，量化和未量化的权重有偏差，微调出来的那点浅层行为
    可能就散了 —— 实测 sft_rag 在 fp16 下丢了整个董秘格式（【依据】和「」引用
    全没了），换成 4bit 就正常。

    更关键的是**可比性**：四条路线应当只差"有没有 adapter"这一个变量。
    上一轮的 base/sft/rag 都是在 notebook 里用 4bit 基座跑的，
    所以带 adapter 的那条也必须用 4bit，否则对比里混进了精度差异。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # T4(Turing) 无 bf16 硬件支持，用 fp16
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
             else torch.float16)
    print(f"加载 {model_name} …（{dtype}"
          f"{'，4bit 量化' if load_4bit else '，不量化'}）")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # 解码器批量生成必须左填充
    quant = None
    if load_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, quantization_config=quant)
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


def generate(model, tok, prompts, batch=8, max_new=192, max_len=8192):
    """贪心解码，单次确定性推理（对应课程里「全部单次确定性推理」的口径）。

    max_new 默认 192：基座不会主动输出 EOS，设太大会一路生成到上限；
    模型一旦退化就会把同一个 token 重复 max_new 次（实测 512 → 复读 511 遍）。

    **max_len 曾经写死 3072，是个隐蔽的坑**：RAG 的系统提示词里塞了 6000 字
    原文片段，加上模板约 5500 token，超过 3072 就会被 `truncation=True`
    **从后面截断 —— 而提问正好在最末尾**。模型看不到问题，只拿到一大段财报，
    就顺着往下写，表现为「没有董秘格式、答非所问、繁体原文腔」。
    整个过程**不报任何错**，只是结果不对。base 和 sft 因为 prompt 短躲过了，
    所以看起来像"只有带检索的两条路线坏了"，很容易误判成检索的问题。

    现在放到 8192，并在真的截断时打印警告 —— 这类错误不能再悄悄地发生。
    """
    import torch
    if tok.padding_side != "left":     # 右填充会让批量生成从 pad 之后续写
        tok.padding_side = "left"
    outs, n_truncated = [], 0
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(model.device)
        # 顶到上限就说明被截了 —— 位置在最末尾，丢的多半正是提问
        hit_cap = (enc["attention_mask"].sum(dim=1) >= max_len).sum().item()
        if hit_cap:
            n_truncated += hit_cap
            print(f"  ⚠️ 第 {i}–{i+len(chunk)} 条里有 {hit_cap} 条提示词顶到 "
                  f"{max_len} token 上限，提问可能已被截掉，这一批结果不可信")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new,
                                 do_sample=False, temperature=None,
                                 top_p=None,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(chunk)):
            n_in = enc["input_ids"].shape[1]
            outs.append(tok.decode(gen[j][n_in:], skip_special_tokens=True).strip())
        print(f"  {min(i+batch, len(prompts))}/{len(prompts)}")
        if i == 0:
            # 第一批就亮出结果：格式不对的话两分钟内就能发现并按停止键，
            # 不用等满 25 分钟。批量生成慢，早看一眼省一小时的返工。
            print("  ── 第一批样本（看格式对不对）──")
            print(f"  {outs[0][:220]}")
            print(f"  ── 【依据】{'有' if '【依据' in outs[0] else '无'}  "
                  f"「」引用{'有' if '「' in outs[0] else '无'} ──", flush=True)
    if n_truncated:
        print(f"\n⚠️ 共 {n_truncated} 条被截断 —— 请调大 --max-len 或调小 --topk")
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["base", "sft", "rag", "sft_rag"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="基座模型（和微调时用的要保持一致）")
    ap.add_argument("--adapter", default=None,
                    help="LoRA 适配器目录（sft / sft_rag 用）")
    ap.add_argument("--load-4bit", action="store_true",
                    help="基座按 4bit 量化加载。适配器是在 4bit 基座上训的，"
                         "且上一轮 base/sft/rag 三条也都是 4bit —— "
                         "带 adapter 的路线必须跟着用，否则对比混进精度差异")
    ap.add_argument("--topk", type=int, default=8, help="rag 检索几页")
    ap.add_argument("--retriever", default="hybrid",
                    choices=["bm25", "embed", "hybrid"],
                    help="rag 用哪种检索（默认混合，对应作业 A 的「向量 + BM25」）")
    # batch 默认 4 而不是 8：T4 是 Turing 架构，没有 Flash Attention，
    # 注意力只能算全矩阵，显存 ≈ batch × 头数 × 长度²。RAG 的提示词约 5700
    # token，batch=8 时那一项就要 6 GB 以上，直接 OOM（实测报过）。
    # 降到 4 就回到和旧版 max_length=3072 时相当的占用。
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=8192,
                    help="提示词长度上限。RAG 的提示词约 5500 token，"
                         "低于这个数就会被从后面截断、把提问切掉")
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
                            args.adapter if use_adapter else None,
                            load_4bit=args.load_4bit)

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

    # 提示词都拼完了，检索器就没用了 —— 它的向量模型还占着 1.2 GB 显存，
    # 生成阶段正是最缺显存的时候。T4 没有 Flash Attention，注意力矩阵
    # 随提示词长度**平方**增长，能省一点是一点。
    if retriever is not None:
        del retriever
        import torch
        torch.cuda.empty_cache()
        print("已释放检索器占用的显存")

    preds = generate(model, tok, prompts, batch=args.batch,
                     max_len=args.max_len)

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
