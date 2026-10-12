# -*- coding: utf-8 -*-
"""生成微调 notebook（一份代码，三个平台都能跑）。

用脚本生成而不是手写 .ipynb，是因为 notebook 本质是 JSON，手写转义容易出错；
而且平台差异只集中在三个格子，共用一份内容改起来才不会漏。

**2026-10-12 改版**（批改意见：所交笔记本未保存运行输出）：
  · 平台自适应 —— 同一个文件在 Colab / Kaggle / **本机**都能直接跑：
    自动认环境、认数据位置、认 dtype、按显存定批量。
  · 新增「第 0 步 · 数据与抽检」—— 训练集统计、来源页示例、
    人工抽检摘要、溯源审计结果直接打印在笔记本里（数字都从文件读，不手抄）。
  · 本文件生成后用 `run_notebook_local.py` 在本机真执行、真存输出。

用法：
    python make_notebook.py                  # 生成自适应版 + Kaggle 版
    python make_notebook.py --exec-note local   # 在头部标注"输出为本机实跑"
输出：
    train/lixiang_lora_sft.ipynb         （主交付件，本机执行后带真实输出）
    train/lixiang_lora_sft_kaggle.ipynb  （同样的格子，文件名区分）
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MD, CODE = "markdown", "code"


def md(text):
    return (MD, text.strip("\n"))


def code(text):
    return (CODE, text.strip("\n"))


def build_cells(exec_note=""):
    note = (f"\n> **本次运行输出**：{exec_note}\n" if exec_note else "")
    return [

    md(f"""
# 用理想汽车财报微调一个「董秘」· LoRA SFT

《学会使用大模型》第三课 · 方向 B。
把 212 条「投资者问 → 董秘答」喂给一个小模型，看它能记住多少、又漏掉什么。
{note}
**这个 notebook 一个文件三个平台都能跑**（Colab / Kaggle / 本机），
第二个格子里会自动认出环境、找到数据、按显卡定批量 —— 不用改任何一行。

**本机也能跑**：1.5B + 4bit QLoRA，6 GB 显存就够（RTX 3060 实测峰值 3.2 GB）。

## 流程一览

| 步骤 | 干什么 | 产出 |
|---|---|---|
| 1、0 | 认环境、先看数据（来源页 / 抽检 / 溯源审计） | 数据统计 + 示例 |
| 3–5 | 配置、加载基座 + LoRA、组装训练数据 | 可训练参数占比 |
| 6 | **微调前**：先问三题 | 训练前对照 |
| 6.5 | 先跑 `base` / `rag` 两条路线（趁 LoRA 还是零初始化） | `preds/base.jsonl`、`preds/rag.jsonl` |
| 7 | 训练（212 条 × 2 轮） | loss 曲线、真实输出 |
| 8 | **微调后**：再问同样三题 + 复读自检 | 训练前后对照 |
| 9 | 跑 `sft` / `sft_rag` 两条路线 | `preds/sft.jsonl`、`preds/sft_rag.jsonl` |
| 10 | 存 LoRA 适配器 | `lora_adapter/` |

四条路线跑完，回到本地 `eval/judge.py` 判分，出四格对照表。
"""),

    md("""
## 1 · 认环境、找数据

三种环境都能自动适配：

| 平台 | 数据从哪来 |
|---|---|
| **本机**（Windows/Linux） | 从当前目录往上找仓库根（认 `data/pages.jsonl`） |
| **Colab** | 上传 `train/lixiang_colab.zip`（弹窗选文件） |
| **Kaggle** | 挂数据集或把 zip 传到工作区，脚本自己找 |

依赖安装也只在该装的平台装：本机已经装好了就跳过。
"""),

    code("""
import os, sys, glob, json, platform, subprocess, time

# ★ 6 GB 显存跑 4bit + 长上下文生成，必须先开这个分配器选项，
#   否则显存碎片会把进程拖到假死（本项目的踩坑记录六）。
#   必须在 import torch 之前设置。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

IN_COLAB = os.path.isdir("/content") and not os.path.isdir("/kaggle")
IN_KAGGLE = os.path.isdir("/kaggle")
ON_LOCAL = not IN_COLAB and not IN_KAGGLE
PLATFORM = "Colab" if IN_COLAB else ("Kaggle" if IN_KAGGLE else "本机")

# ---------- 找项目根：往上走，找带 data/pages.jsonl 的那一层 ----------
def find_root(start):
    d = os.path.abspath(start)
    for _ in range(6):
        if os.path.exists(os.path.join(d, "data", "pages.jsonl")):
            return d
        d = os.path.dirname(d)
    return None

ROOT = None
if ON_LOCAL:
    ROOT = find_root(os.getcwd())
else:
    for cand in ["/content/lixiang", "/kaggle/working/lixiang"]:
        if os.path.exists(os.path.join(cand, "data", "pages.jsonl")):
            ROOT = cand

# ---------- 该平台的依赖与数据准备 ----------
if IN_COLAB and ROOT is None:
    print("Colab：请在弹出的窗口里选 train/lixiang_colab.zip（约 1 MB）")
    from google.colab import files
    up = files.upload()
    name = list(up)[0]
    import zipfile
    with zipfile.ZipFile(name) as z:
        z.extractall("/content")
    ROOT = find_root("/content/lixiang") or "/content/lixiang"

if IN_KAGGLE and ROOT is None:
    import shutil, zipfile
    def _find_kaggle():
        for depth in range(1, 4):
            for p in glob.glob("/kaggle/input/" + "*/" * depth
                               + "lixiang/data/pages.jsonl"):
                return ("dir", os.path.dirname(os.path.dirname(p)))
        for p in (glob.glob("/kaggle/input/*/lixiang_colab.zip")
                  + glob.glob("/kaggle/input/*/*/lixiang_colab.zip")
                  + glob.glob("/kaggle/working/lixiang_colab.zip")):
            return ("zip", p)
        return (None, None)
    kind, src = _find_kaggle()
    if kind is None:
        raise SystemExit("没找到数据 —— 请确认右侧 Data 面板挂上了 lixiang 数据集")
    WORK = "/kaggle/working/lixiang"
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    if kind == "zip":
        with zipfile.ZipFile(src) as z:
            z.extractall("/kaggle/working")
    else:
        shutil.copytree(src, WORK)
    ROOT = WORK

assert ROOT, "没找到项目根目录（往上 6 层里没有 data/pages.jsonl）"
DATA, EVAL = f"{ROOT}/data", f"{ROOT}/eval"
MODELS = f"{ROOT}/train/models"
LOCAL_BASE = f"{MODELS}/Qwen2.5-1.5B-Instruct"
LOCAL_EMB = f"{MODELS}/Qwen3-Embedding-0.6B"
print(f"平台：{PLATFORM}   项目根：{ROOT}")

# ---------- 装依赖（按平台；本机跳过） ----------
if IN_COLAB:
    print(subprocess.run("pip install -q transformers peft accelerate "
                         "bitsandbytes", shell=True).returncode == 0
          and "依赖就绪" or "依赖安装有问题")
elif IN_KAGGLE:
    print(subprocess.run("pip install -q accelerate bitsandbytes",
                         shell=True).returncode == 0 and "依赖就绪"
          or "依赖安装有问题")
else:
    print("本机运行：跳过 pip（依赖已就绪）")

import torch
print(f"torch {torch.__version__} / CUDA 可用 {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"显卡：{p.name}，显存 {p.total_memory/2**30:.1f} GB")
print(f"Python {platform.python_version()}")
sys.path.insert(0, DATA)
"""),

    md("""
## 0 · 先看数据：来源页、人工抽检、溯源审计

**先把要喂给模型的数据本身看一遍**，再看模型。这里的三样东西
对应批改意见里要补的「造数据 / 来源页 / 人工抽检」：

- 每条问答都带来源页（`doc / page_start / section`），回答里带 `【依据：《…》第 N 页】`
- 60 条人工抽检的结论（`qc_sample.csv`，三列人工填的 + 一列机器复核的）
- 溯源审计：**每条引文逐字回 1,449 页原文里查过**
  （`source_audit.csv`，342 条生成集 + 140 条评测集全量）

造数据的完整生成链（原文 → 提示词 → 模型原始返回 → 问答对）
留痕在 `data/provenance_report.md`，原始返回可离线复核。
"""),

    code("""
def show(title):
    print("\\n" + "=" * 78 + f"\\n{title}\\n" + "=" * 78)

show("① 数据集规模")
train_raw = [json.loads(l) for l in open(f"{DATA}/final_train.jsonl", encoding="utf-8")]
test_raw = [json.loads(l) for l in open(f"{DATA}/final_test.jsonl", encoding="utf-8")]
grid = [json.loads(l) for l in open(f"{DATA}/eval_grid.jsonl", encoding="utf-8")]
print(f"训练集 {len(train_raw)} 条 | 测试集 {len(test_raw)} 条 | 四格评测集 {len(grid)} 条")
print("四格分布：", {c: sum(1 for r in grid if r["cell"] == c)
                   for c in sorted({r["cell"] for r in grid})})

show("② 三条问答的来源页示例（训练集）")
for r in train_raw[:3]:
    print(f"《{r['doc']}》第 {r['page']} 页 · {r['section']} · "
          f"{'表格' if r['has_table'] else '叙述'}块")
    print(f"  问：{r['question'][:60]}")
    print(f"  答：{r['answer'][:110]}…")
    print(f"  原文：{r['source_text'][:90].strip()}…")
    print("-" * 74)

# ---------- 人工抽检与机器复核摘要（数字从文件读，不手抄） ----------
from common import read_csv_any          # data/common.py，兼容 GBK
show("③ 人工抽检（60 条）与机器复核（全量 342 条）")
qc, _ = read_csv_any(f"{DATA}/qc_sample.csv")
n = len(qc)
def filled(col):
    return sum(1 for r in qc if str(r.get(col, "")).strip())
print(f"抽检表 {n} 条：人审-有据 已填 {filled('人审-有据')}、"
      f"人审-数字 已填 {filled('人审-数字')}、人审-处理 已填 {filled('人审-处理')}、"
      f"备注 {filled('人审-备注')}")
n_you = sum(1 for r in qc if r["人审-有据"] == "2")
n_calc = sum(1 for r in qc if r["人审-数字"] == "推算")
n_machine = sum(1 for r in qc if str(r.get("机器复核-自然", "")) == "1")
print(f"  有据=2 的 {n_you}/{n}；正文数字属推算的 {n_calc} 条；"
      f"机器复核判「自然」 {n_machine}/{n}（该列明确标注为模型复核，非人审）")
if os.path.exists(f"{DATA}/machine_review.csv"):
    mr, _ = read_csv_any(f"{DATA}/machine_review.csv")
    keep = {json.loads(l)["qa_id"]: json.loads(l)["keep"]
            for l in open(f"{DATA}/qa.jsonl", encoding="utf-8")}
    kept = [r for r in mr if keep.get(r["qa_id"])]
    ok = sum(1 for r in kept if str(r.get("与原质检一致", "")) == "1")
    print(f"全量机器复核 342 条：与原始质检一致率 "
          f"{100*ok/len(kept):.0f}%（保留池 {len(kept)} 条）")

show("④ 溯源审计：引文逐字回原文查（source_audit.csv）")
aud, _ = read_csv_any(f"{DATA}/source_audit.csv")
for batch in ["生成集(342)", "评测集(140)"]:
    sub = [r for r in aud if r["来源"] == batch]
    nq = sum(int(r["引文数"]) for r in sub)
    nv = sum(int(r["引文逐字"]) for r in sub)
    nm = sum(int(r["引文查无"]) for r in sub)
    nd = sum(1 for r in sub if str(r["报告名一致"]) == "1")
    print(f"{batch}：依据报告名与来源一致 {nd}/{len(sub)}，"
          f"「」引文 {nq} 句 → 逐字可查 {nv}、原文查无 {nm}")
print("\\n详细留痕：source_audit.md（审计摘要）· provenance_report.md（生成链）")
"""),

    md("""
## 3 · 配置

超参是拿实验换来的，别乱改（下面注释写了每个数怎么来的）：

- **2 轮**：212 条小样本，3 轮开始背题（loss 掉到 0.1 以下就该警惕）
- **lr 1e-4**：2e-4 + alpha=2r 实测把模型练成复读机，减半
- **alpha = r**：原来 2r 让等效步长翻倍，也是复读的来源之一

生成批量按显存自动定：提示词带 6000 字财报原文的两条路线最吃显存。
**批量不是越大越快** —— RTX 3060 6 GB 实测（带检索、4,700+ token 提示词）：

| 批量 | 单条耗时 | 140 条预计 | 峰值显存 |
|---|---|---|---|
| 1 | 6 s | 14.8 分钟 | 2.19 GB |
| 4 | 4 s | 8.8 分钟 | 2.95 GB |
| 8 | 11 s | 25.6 分钟 | 5.07 GB ← 顶到显存上限，反而变慢 |

所以 6 GB 档取 4：既快又留 2 GB 余量。
"""),

    code("""
# 基座模型。课程用的是 2B 档（Qwen3.5-2B）；这里用同档位、一定能下载到的
# Qwen2.5-1.5B-Instruct。本机有本地权重就直接用本地目录（离线也能跑）。
MODEL = LOCAL_BASE if os.path.exists(LOCAL_BASE) else "Qwen/Qwen2.5-1.5B-Instruct"
EMB_MODEL = LOCAL_EMB if os.path.exists(LOCAL_EMB) else "Qwen/Qwen3-Embedding-0.6B"
OUT_DIR = f"{ROOT}/train/_out" if ON_LOCAL else "/kaggle/working/lora_out"

MAX_LEN  = 1024      # 问题和回答都不长，1024 够
EPOCHS   = 2         # 212 条小样本：3 轮就会开始背串，2 轮够学格式
LR       = 1e-4      # 2e-4 + alpha=2r 实测把模型练成复读机，减半
LORA_R   = 16
LORA_ALPHA = 16      # = r，缩放 1.0。原来 2r 让等效步长翻倍
BATCH    = 2
GRAD_ACC = 4
SEED     = 3407

# 训练和评测必须用同一个 system prompt，否则模型学到的格式对不上
SYSTEM = "你是理想汽车（港股 02015）的董事会秘书，正在回答投资者提问。"

import torch
DTYPE = (torch.bfloat16
         if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
         else (torch.float16 if torch.cuda.is_available() else torch.float32))

# 生成批量：带检索的长提示词（4,700+ token）最吃显存。
# 6 GB 档取 rag=4 是本机实测出来的：batch 4 峰值 2.95 GB / 8.8 分钟；
# 拉到 8 反而顶到 5.07 GB、退回慢速内存，单条从 4s 涨到 11s。
VRAM = (torch.cuda.get_device_properties(0).total_memory / 2**30
        if torch.cuda.is_available() else 0)
if VRAM >= 24:   BATCH_PLAIN, BATCH_RAG = 16, 8
elif VRAM >= 14: BATCH_PLAIN, BATCH_RAG = 8, 4
elif VRAM >= 10: BATCH_PLAIN, BATCH_RAG = 8, 2
elif VRAM >= 5:  BATCH_PLAIN, BATCH_RAG = 8, 4
else:            BATCH_PLAIN, BATCH_RAG = 2, 1
print(f"基座 {MODEL}")
print(f"{EPOCHS} 轮 / lr {LR} / r={LORA_R} / 精度 {DTYPE}")
print(f"生成批量：不带检索 {BATCH_PLAIN} 条 / 带检索 {BATCH_RAG} 条"
      f"（按 {VRAM:.1f} GB 显存自适应）")
"""),

    md("""
## 4 · 加载基座模型 + LoRA（4bit QLoRA）

T4 是 Turing 架构、RTX 3060 是 Ampere：两者都按各自的硬件能力选 dtype，
统一走 4bit nf4 量化 —— **适配器是在 4bit 基座上训的，判分时的四条路线
也必须全用 4bit**，否则对比里混进了精度差异（实测 fp16 基座会把
董秘格式整个弄丢）。
"""),

    code("""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=DTYPE, bnb_4bit_use_double_quant=True)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"        # 解码器批量生成必须左填充

model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto", torch_dtype=DTYPE)

# 试试 unsloth 的加速版 LoRA，不行就退回标准 peft —— 两条路结果一样
USE_UNSLOTH = False
try:
    from unsloth import FastLanguageModel
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=SEED)
    USE_UNSLOTH = True
    print("已挂载 LoRA（unsloth 加速版）")
except Exception as e:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    print(f"已挂载 LoRA（peft 标准版；unsloth 不可用：{e}）")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"可训练参数 {trainable:,} / {total:,} = {trainable/total:.2%}")
print(f"精度 {DTYPE} / padding_side={tok.padding_side}")

def to_inference():
    \"\"\"把模型切回推理态。

    ★ 这一步是必须的，漏了就出复读 ★

    prepare_model_for_kbit_training() 会打开梯度检查点，训练结束后模型仍停在
    训练态。此时直接调 .generate() 会输出乱码（同一 token 重复到上限），
    而**权重其实是好的** —— 表现为「训练前正常、训练后复读」。
    梯度检查点和 KV cache 互斥，所以 use_cache 也要一起恢复。
    \"\"\"
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    print("已切回推理态（梯度检查点关闭 / use_cache 打开 / eval）")
"""),

    md("""
## 5 · 组装训练数据

**关键一点：只对「回答」算 loss，问题部分的 label 设成 -100。**
不然模型会花力气去学怎么提问，而我们只想要它学怎么答。
"""),

    code("""
import json
from torch.utils.data import Dataset

def encode(rec):
    '''把一条问答编成 input_ids；labels 把问题部分挖掉（-100）'''
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user",   "content": rec["question"]}],
        tokenize=False, add_generation_prompt=True)
    answer = rec["answer"] + tok.eos_token
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(answer, add_special_tokens=False)["input_ids"]
    ids = (p + a)[:MAX_LEN]
    return {"input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": ([-100] * len(p) + a)[:MAX_LEN]}

class QADataset(Dataset):
    def __init__(self, recs):
        self.rows = [encode(r) for r in recs]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        return self.rows[i]

ds = QADataset(train_raw)
print(f"训练集 {len(ds)} 条；样本长度前 5 条：",
      [len(ds[i]['input_ids']) for i in range(5)])
"""),

    md("""
## 6 · 微调前：先问三题

这是全流程最直观的一屏 —— 训练前模型**什么都不知道**。
记住这三条问题，训练结束后会再问一遍。
"""),

    code("""
@torch.no_grad()
def ask(questions, max_new=220):
    outs = []
    for q in questions:
        text = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user",   "content": q}],
            tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        outs.append(tok.decode(gen[0][enc['input_ids'].shape[1]:],
                               skip_special_tokens=True).strip())
    return outs

DEMO_Q = [r["question"] for r in train_raw[:3]]

probe_before = ask(["1+1等于几？只回答数字。"], max_new=16)
print(f"★ 体温计（训练前）= {probe_before[0][:60]!r}\\n")

before = ask(DEMO_Q)
for q, a in zip(DEMO_Q, before):
    print(f"【问】{q}\\n【答】{a}\\n{'-'*70}")
"""),

    md("""
## 6.5 · 先跑 `base` 和 `rag` 两条路线

**顺序很重要。** 这两条路线用的都是「原始基座」，
所以趁 LoRA 还没开始训练、B 矩阵还是零初始化的时候跑最干净。

（上一版是训练完之后用 `disable_adapter()` 把 LoRA 关掉再跑，
实测**还原不干净**——base 路线出来的也是复读，整组对照就废了。）

带检索的提示词有 5,700 来 token，6 GB 显存只能一条一条生成；
跑完先把检索器**放掉**再训练 —— 它的向量模型占着 1.2 GB，
训练正缺显存。第 9 步要用时再懒加载一次（页面向量有缓存，很快）。
"""),

    code("""
import gc
sys.path.insert(0, EVAL)
import generate as G          # 和另外两条路线共用同一份加载/生成代码

os.makedirs(f"{EVAL}/preds", exist_ok=True)

def run_arm(arm, system_fn, batch, force=False, free_retriever=False):
    '''生成一条路线的预测。已有结果就跳过 —— 断了重跑不用从头发。'''
    path = f"{EVAL}/preds/{arm}.jsonl"
    if os.path.exists(path) and not force:
        n = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {arm}: 已有 {n} 条结果，跳过（要重跑就删掉这个文件）")
        return
    prompts, metas = [], []
    for r in grid:
        q = r.get("question_eval") or r["question"]
        prompts.append(G.chat(tok, system_fn(q), q))
        metas.append({"qa_id": r["qa_id"], "cell": r["cell"]})
    if free_retriever:
        # 提示词拼完，向量模型就没用了 —— 它占着 1.2 GB 显存，
        # 生成正是最缺显存的时候（6 GB 卡上这 1.2 GB 是生死线）
        release_retriever()
    t0 = time.time()
    try:
        outs = G.generate(model, tok, prompts, batch=batch, max_new=192)
    except RuntimeError as e:
        # 6 GB 卡上长上下文批量生成的兜底：显存被别的程序占满时，
        # CUDA 会在建 GPU 映射这步直接报 "device not ready"。
        # 与其整条路线作废，不如清理显存、降到 batch=1 慢速跑完 —— 如实打印。
        if "CUDA" not in str(e) and "memory" not in str(e).lower():
            raise
        print(f"  ⚠️ 批量 {batch} 撞上显存问题（{str(e)[:120]}），"
              f"清理后改用批量 1 重试（慢但稳）", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        outs = G.generate(model, tok, prompts, batch=1, max_new=192)
    with open(path, "w", encoding="utf-8") as f:
        for m, o in zip(metas, outs):
            f.write(json.dumps({**m, "prediction": o}, ensure_ascii=False) + "\\n")
    print(f"  {arm}: {len(outs)} 条 → {path}（{time.time()-t0:.0f}s）")

# 切回推理态：梯度检查点必须关掉，否则生成出来是乱码
to_inference()

# 检索器懒加载：跑 rag 时建、跑完放掉，第 9 步跑 sft_rag 时再建
# （页面向量有磁盘缓存 eval/_emb_cache_1431.pt，重建只要几秒）
_rt = {"r": None}
def get_retriever():
    if _rt["r"] is None:
        _rt["r"] = G.build_retriever("hybrid", emb_model=EMB_MODEL)
    return _rt["r"]

def release_retriever():
    _rt["r"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"检索器已释放；显存占用 {torch.cuda.memory_allocated()/2**30:.2f} GB")

def sys_rag(q):
    hits = get_retriever().search(q, topk=8)
    ctx = "\\n\\n".join(f"【{h['doc']} 第 {h['page']} 页】\\n{h['text']}" for h in hits)
    return G.SYSTEM_RAG.format(context=ctx[:6000])

print("=== base：原始基座，不给任何上下文 ===")
run_arm('base', lambda q: G.SYSTEM, BATCH_PLAIN)

print("=== rag：同一个基座 + 检索到的原文 ===")
run_arm('rag', sys_rag, BATCH_RAG, free_retriever=True)
print("\\nbase / rag 已落盘。接下来训练，训练只会影响 sft 那一条。")
"""),

    md("""
## 7 · 训练

212 条 × 2 轮 ≈ 53 步，T4 上几分钟，RTX 3060 上约 3 分钟。
重点看 **loss 一路下降**，以及可训练参数只有百分之零点几。
**loss 掉到 0.1 以下要警惕**——那是背题不是学格式，紧接着就会复读。
"""),

    code("""
from transformers import Trainer, TrainingArguments
from dataclasses import dataclass

@dataclass
class Collator:
    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        ids, att, lab = [], [], []
        for f in feats:
            pad = maxlen - len(f["input_ids"])
            ids.append(f["input_ids"] + [tok.pad_token_id] * pad)
            att.append(f["attention_mask"] + [0] * pad)
            lab.append(f["labels"] + [-100] * pad)
        return {"input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(att),
                "labels": torch.tensor(lab)}

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="linear",
    warmup_steps=10,
    logging_steps=5,
    save_strategy="no",
    optim="adamw_torch",
    bf16=(DTYPE == torch.bfloat16),
    fp16=(DTYPE == torch.float16),      # 和加载时一致，别用 is_bf16_supported 另判
    report_to="none",
    seed=SEED,
)

trainer = Trainer(model=model, args=args, train_dataset=ds,
                  data_collator=Collator())
stats = trainer.train()
print(f"\\n训练完成：loss {stats.training_loss:.3f}，"
      f"用时 {stats.metrics['train_runtime']:.0f}s")

# ★ 训完立刻先把适配器存一份。后面还有两条路线的长生成，
#   要是那一步崩了（显存被别的程序抢光），至少权重不会跟着丢 ——
#   实测崩过一次，训练成果只活在内存里，白训。
SAVE_DIR = f"{ROOT}/train/lora_adapter" if ON_LOCAL else "/content/lora_adapter"
model.save_pretrained(SAVE_DIR)
tok.save_pretrained(SAVE_DIR)
print(f"适配器已先保存一份 → {SAVE_DIR}")

del trainer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"训练后显存占用 {torch.cuda.memory_allocated()/2**30:.2f} GB")
"""),

    md("## 8 · 微调后：再问同样三题\n\n和上面那三屏对照着看。"),

    code("""
# ★ 训练完必须切回推理态 —— 漏了这一步就是「训练前正常、训练后复读」
to_inference()
print(f"  训练态={model.training}  "
      f"梯度检查点={getattr(model,'is_gradient_checkpointing',False)}"
      f"  use_cache={model.config.use_cache}")

# 体温计：一道和财报完全无关的常识题。
# 答得出 → 权重是好的，问题在推理态；也复读 → 权重真坏了。
probe = ask(["1+1等于几？只回答数字。"], max_new=16)
print(f"  ★ 体温计 = {probe[0][:60]!r}")

after = ask(DEMO_Q)
for q, b, a in zip(DEMO_Q, before, after):
    print(f"【问】{q}")
    print(f"【训练前】{b}")
    print(f"【训练后】{a}")
    print('=' * 74)

# ---- 崩塌自检：练崩了就别再白跑 140 条
from collections import Counter

def is_degenerate(t, thresh=8):
    '''同一小段反复出现 = 模型退化成复读机'''
    t = t.strip()
    if not t:
        return True
    grams = Counter(t[i:i + 4] for i in range(max(1, len(t) - 3)))
    return grams.most_common(1)[0][1] > thresh

if sum(is_degenerate(a) for a in after) >= 2:
    raise RuntimeError(
        "\\n训练后出现复读：模型练崩了，下面的 sft 预测不用跑了。\\n"
        "把 EPOCHS 降到 1、LR 减到 5e-5 再试一次。")
print("自检通过：训练后没有出现复读。")
"""),

    md("""
## 9 · `sft` / `sft_rag` 路线（带 LoRA）

- **sft** = 微调后 + 不给上下文 → 只看写进权重里有多少
- **sft_rag** = 微调后 + 检索到的原文 → 口径来自微调，事实来自检索

**为什么不把 LoRA 合并进权重再跑**（试过、放弃了）：`merge_and_unload()`
要把 nf4 主权重反量化、加上 LoRA 增量、再量化回去 —— 增量本身只有
百分之几的量级，我在这台机器上实测**重量化把增量抹掉了**：
合并后的模型输出和**没微调的基座逐字一样**（sft 与 base 的前三条回答完全相同），
董秘格式整个消失。所以这里老老实实挂 adapter 生成。
（合并能快 3 倍、省 1.2 GB，但那是个陷阱 —— 四组对照见 `_scratch/bench_peft.py`，
结论 = 别合并。）

`sft_rag` 是四条里唯一三条腿都齐的，也是结论的关键证据。
`sys_rag` 还是 6.5 格定义的那个（检索器此时会懒加载重建，几秒钟）。
"""),

    code("""
# 先跑长的那条（sft_rag 要现场检索 + 长上下文，是四条里最慢的），再跑 sft
run_arm('sft_rag', sys_rag, BATCH_RAG, free_retriever=True)
run_arm('sft', lambda q: G.SYSTEM, BATCH_PLAIN)
"""),

    md("""
## 10 · 四条路线落盘自检 + 保存适配器

判分在本地 `eval/judge.py` 做（判分和生成分开，才能换模型重判）。
这里先做个轻量自检：条数对不对、每条路线的**格式率**（有没有 `【依据】`、
有没有 `「」` 引用）——这就是四格表里"格式分"的原料。
"""),

    code("""
import re
print(f"{'路线':<10}{'条数':>5}{'有【依据】':>12}{'有「」引用':>12}")
for arm in ['base', 'sft', 'rag', 'sft_rag']:
    path = f"{EVAL}/preds/{arm}.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    n_cite = sum(1 for r in rows if re.search(r"【依据[:：]", r["prediction"]))
    n_q = sum(1 for r in rows if "「" in r["prediction"] and "」" in r["prediction"])
    print(f"{arm:<10}{len(rows):>5}{n_cite:>10}  {n_q:>10}  ")
    assert len(rows) == 140, f"{arm} 应该有 140 条，实际 {len(rows)} 条"
print("\\n四条路线各 140 条，全部落盘 ✓")
"""),

    code("""
# 保存 LoRA 适配器（训练完已先存过一份，这里再存一次保证是最新的）
model.save_pretrained(SAVE_DIR)
tok.save_pretrained(SAVE_DIR)
print(f"适配器已保存 → {SAVE_DIR}")

if IN_COLAB:
    import subprocess
    subprocess.run(f"cd /content && zip -qr lixiang_results.zip lora_adapter "
                   f"lixiang/eval/preds", shell=True)
    from google.colab import files
    files.download("/content/lixiang_results.zip")
elif IN_KAGGLE:
    print("Kaggle：结果留在 /kaggle/working/，Save Version 后从 Output 面板下载")
"""),

    md("""
## 跑完之后（回到本地判分）

```bash
cd eval
python judge.py --preds preds/base.jsonl    --label 基座 \\
                --preds preds/sft.jsonl     --label 微调后 \\
                --preds preds/rag.jsonl     --label 建库 \\
                --preds preds/sft_rag.jsonl --label 微调+建库
```

会出 `scores/four_grid.md`，就是作业要交的**四格对照表**。
格式分（0–2）、事实（净命中 + 模型判分）、引用分（0–3，引文逐字回原文查）
三个维度分开评 —— 微调模型会在编造的数字外面套上漂亮的「」引用，
只看格式分会被它骗过去。

### 四条路线各自缺什么

- `base`：既不知道数字，也不会用董秘的口径答 —— 只会说"我无法提供"
- `sft`：口径、引用格式、「该在这儿放个数字」全学会了，**但数字和引文是编的**
- `rag`：读得到原文，但基座只会照抄检索到的段落，不会组织成回答
- `sft_rag`：口径来自微调、事实来自检索 —— **这才是完整方案**

**结论**：微调学的是「怎么答」，知识得靠检索递进来，两者缺一不可。
"""),
    ]


def write_notebook(path, exec_note=""):
    cells = build_cells(exec_note)
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    for kind, src in cells:
        cell = {"cell_type": kind, "metadata": {},
                "source": src.splitlines(True)}
        if kind == CODE:
            cell["outputs"] = []
            cell["execution_count"] = None
        nb["cells"].append(cell)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    check = json.load(open(path, encoding="utf-8"))
    assert check["nbformat"] == 4 and len(check["cells"]) == len(cells)
    n_code = sum(1 for k, _ in cells if k == CODE)
    print(f"写出 {os.path.basename(path)}：{len(cells)} 格"
          f"（{n_code} 代码 / {len(cells)-n_code} 说明）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-note", default="",
                    help='标注运行输出来源，如 "本机 RTX 3060 实跑（2026-10-12）"')
    ap.add_argument("--only", default="",
                    help="只写某个文件：main / kaggle")
    args = ap.parse_args()

    if args.only != "kaggle":
        write_notebook(os.path.join(HERE, "lixiang_lora_sft.ipynb"),
                       args.exec_note)
    if args.only != "main":
        write_notebook(os.path.join(HERE, "lixiang_lora_sft_kaggle.ipynb"),
                       args.exec_note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
