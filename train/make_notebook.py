# -*- coding: utf-8 -*-
"""生成微调 notebook（Colab 版 + Kaggle 版）。

用脚本生成而不是手写 .ipynb，是因为 notebook 本质是 JSON，手写转义容易出错；
而且两个平台只差几个格子，共用一份内容改起来才不会漏。

用法：
    python make_notebook.py              # 两个平台都生成
    python make_notebook.py --platform kaggle
输出：
    train/lixiang_lora_sft.ipynb         （Colab）
    train/lixiang_lora_sft_kaggle.ipynb  （Kaggle）
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


# ---------------------------------------------------------------- 平台差异格

def cell_install(platform):
    if platform == "colab":
        return code("""
# unsloth 在 Colab 上一键装好，装不上也无所谓 —— 加载模型时会自动退回纯 peft
!pip install -q unsloth 2>/dev/null || echo "unsloth 装不上，将使用 peft 回退方案"
!pip install -q transformers peft datasets accelerate bitsandbytes
""")
    return code("""
# Kaggle 的镜像里 torch / transformers / peft 都带好了，基本不用装什么。
# unsloth 可选，装不上会自动退回纯 peft（结果一样，只是慢一点）
!pip install -q unsloth 2>/dev/null || echo "unsloth 装不上，将使用 peft 回退方案"
!pip install -q accelerate bitsandbytes
""".strip("\n"))


def cell_setup(platform):
    if platform == "colab":
        return code("""
import os, zipfile

if not os.path.exists('/content/lixiang'):
    from google.colab import files
    up = files.upload()                      # 点「选择文件」，选 lixiang_colab.zip
    name = list(up)[0]
    with zipfile.ZipFile(name) as z:
        z.extractall('/content')

os.system('find /content/lixiang -type f | head -20')
DATA, EVAL = '/content/lixiang/data', '/content/lixiang/eval'
""")
    return code("""
import os, glob, zipfile, shutil, sys

# 数据可能以三种形态存在，都认：
#   A. 作为 Dataset 上传，Kaggle 自动解压 → /kaggle/input/<名>/lixiang/data/…
#   B. 作为 Dataset 上传，保留 zip       → /kaggle/input/<名>/lixiang_colab.zip
#   C. 用 Data 面板传到工作区            → /kaggle/working/lixiang_colab.zip
WORK = '/kaggle/working/lixiang'

def find_data():
    for depth in range(1, 4):
        pat = '/kaggle/input/' + '*/' * depth + 'lixiang/data/pages.jsonl'
        for p in glob.glob(pat):
            return ('dir', os.path.dirname(os.path.dirname(p)))
    for p in (glob.glob('/kaggle/input/*/lixiang_colab.zip')
              + glob.glob('/kaggle/input/*/*/lixiang_colab.zip')
              + glob.glob('/kaggle/working/lixiang_colab.zip')):
        return ('zip', p)
    return (None, None)

kind, src = find_data()
if kind is None:
    print("没找到数据。请确认右侧 Data 面板里已经挂上了 lixiang 数据集。")
    print("现在 /kaggle 下有什么：")
    os.system('find /kaggle -maxdepth 4 -name "pages.jsonl" -o -maxdepth 4 -name "*.zip" 2>/dev/null | head')
    sys.exit(1)

print(f"数据形态：{kind}  →  {src}")
if os.path.exists(WORK):
    shutil.rmtree(WORK)                      # 重跑时清掉旧的
if kind == "zip":
    with zipfile.ZipFile(src) as z:
        z.extractall('/kaggle/working')
else:
    shutil.copytree(src, WORK)               # /kaggle/input 是只读的，得拷出来

os.system(f'find {WORK} -type f')
DATA, EVAL = f'{WORK}/data', f'{WORK}/eval'
assert os.path.exists(f'{DATA}/pages.jsonl') and os.path.exists(f'{EVAL}/generate.py')
print("\\n数据就位 ✓")
""".strip("\n"))


def cell_setup_gen():
    """只做生成用的上传格：包比 lixiang_colab.zip 多一个 lora_adapter/。"""
    return code("""
import os, zipfile

if not os.path.exists('/content/lixiang/lora_adapter'):
    from google.colab import files
    # 点「选择文件」，选 lixiang_gen_bundle.zip（约 77 MB，上传要等几分钟）
    up = files.upload()
    name = list(up)[0]
    with zipfile.ZipFile(name) as z:
        z.extractall('/content')

os.system('find /content/lixiang -type f | head -20')
DATA, EVAL = '/content/lixiang/data', '/content/lixiang/eval'
ADAPTER = '/content/lixiang/lora_adapter'
assert os.path.exists(f'{ADAPTER}/adapter_model.safetensors'), \\
    '适配器没解压出来 —— 确认上传的是 lixiang_gen_bundle.zip'
print('\\n数据 + 适配器就位 ✓')
""")


def cell_download(platform):
    if platform == "colab":
        return code("""
model.save_pretrained('/content/lora_adapter')
tok.save_pretrained('/content/lora_adapter')

!cd /content && zip -qr lixiang_results.zip lora_adapter lixiang/eval/preds
!ls -la /content/lixiang_results.zip /content/lixiang/eval/preds

from google.colab import files
files.download('/content/lixiang_results.zip')
""")
    return code("""
model.save_pretrained('/kaggle/working/lora_adapter')
tok.save_pretrained('/kaggle/working/lora_adapter')

!cd /kaggle/working && zip -qr lixiang_results.zip lora_adapter lixiang/eval/preds
!ls -la /kaggle/working/lixiang_results.zip
!ls -la /kaggle/working/lixiang/eval/preds

print('\\n打包完成。下载方式（二选一）：')
print('  ① 右上角 Save Version（或 Save & Run All）后，')
print('     在 Output 标签页里下载 lixiang_results.zip')
print('  ② 右侧 Data 面板展开 /kaggle/working，右键 lixiang_results.zip 下载')
""".strip("\n"))


def md_header(platform):
    if platform == "colab":
        return md("""
# 用理想汽车财报微调一个「董秘」· LoRA SFT

《学会使用大模型》第三课 · 方向 B。
把 212 条「投资者问 → 董秘答」喂给一个小模型，看它能记住多少、又漏掉什么。

**免费 T4 就能跑完**（模型 1.5B + 4bit QLoRA）。

---

## 怎么用

1. 菜单 `代码执行程序 → 更改运行时类型 → T4 GPU`
2. 从上往下依次运行每个格子（点格子左边的 ▶，或按 `Shift + Enter`）
3. 最后会打包出一个 zip，下载后拿回本地跑 `eval/judge.py` 判分

**要先准备好**：本地项目目录里的 `train/lixiang_colab.zip`（1 MB）。
""")
    return md("""
# 用理想汽车财报微调一个「董秘」· LoRA SFT（Kaggle 版）

《学会使用大模型》第三课 · 方向 B。
把 212 条「投资者问 → 董秘答」喂给一个小模型，看它能记住多少、又漏掉什么。

**免费 T4 就能跑完**（模型 1.5B + 4bit QLoRA）。

---

## 跑之前先做三件事，缺一不可

| # | 做什么 | 在哪 |
|---|---|---|
| 1 | **Accelerator 选 GPU T4 x2**（或 P100） | 右侧 `Settings` 面板 |
| 2 | **Internet 打开** ← 不打开就下载不了模型 | 右侧 `Settings` 面板 |
| 3 | **上传 `lixiang_colab.zip`** | 右侧 `Data` 面板 → Upload；或加成 Dataset |

第 2 条最容易漏。Kaggle 默认**关着网**，不开的话第 4 格会卡在下载模型那里报错。

---

## 怎么用

从上往下依次运行每个格子（点格子左边的 ▶，或按 `Shift + Enter`）。
最后结果打包在 `/kaggle/working/lixiang_results.zip`，下载后拿回本地跑 `eval/judge.py` 判分。
""")


def md_setup(platform):
    if platform == "colab":
        return md("## 2 · 上传数据\n\n点「选择文件」按钮，选 `lixiang_colab.zip`。")
    return md("""
## 2 · 布置数据

zip 有两种放法，脚本会自动找到：

- **右侧 `Data` 面板 → Upload**，传到 `/kaggle/working/`
- 或把 zip 加成 Kaggle Dataset，挂载到 `/kaggle/input/` 下

（`Data` 面板的上传是真正的拖拽 / 选文件，比 Colab 稳）
""")


def md_download(platform):
    head = "## 12 · 保存 LoRA 适配器 + 打包下载"
    if platform == "kaggle":
        head += "\n\nKaggle 没有 `files.download()`，结果留在 `/kaggle/working/` 里，"
        head += "用 Output 面板或 Save Version 取。"
    return md(head)


# ---------------------------------------------------------------- 共用的格子

CELLS_TAIL = [

md("""
## 跑完之后（回到本地）

把 `lixiang_results.zip` 解压，取出 `preds/` 里的四个 jsonl 放到本地 `eval/preds/` 下：

```bash
cd eval
python judge.py --preds preds/base.jsonl    --label 基座 \\
                --preds preds/sft.jsonl     --label 微调后 \\
                --preds preds/rag.jsonl     --label 建库 \\
                --preds preds/sft_rag.jsonl --label 微调+建库
```

会出 `scores/four_grid.md`，就是作业要交的**四格对照表**。

### 四条路线各自缺什么

- `base`：既不知道数字，也不会用董秘的口径答 —— 只会说"我无法提供"
- `sft`：口径、引用格式、「该在这儿放个数字」全学会了，**但数字是编的**
- `rag`：读得到原文，但基座只会照抄检索到的段落，不会组织成回答
- `sft_rag`：口径来自微调、事实来自检索 —— **这才是完整方案**

**结论**：微调学的是「怎么答」，知识得靠检索递进来，两者缺一不可。
"""),
]


def build_cells(platform):
    return [

    md_header(platform),

    md("## 1 · 安装依赖"),
    cell_install(platform),

    md_setup(platform),
    cell_setup(platform),

    md("## 3 · 配置"),

    code("""
# 基座模型。课程用的是 2B 档（Qwen3.5-2B）；这里用同档位、一定能下载到的
# Qwen2.5-1.5B-Instruct。想换更大的就改这一行（3B 也能塞进 T4）
MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"

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
print(f"GPU：{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '没有 GPU！'}")
print(f"模型 {MODEL} / {EPOCHS} 轮 / lr {LR} / r={LORA_R}")
"""),

    md("## 4 · 加载基座模型 + LoRA"),

    code("""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# T4 是 Turing 架构，**硬件不支持 bfloat16**。4bit 量化若用 bf16 计算，
# 再叠加 fp16 自混合精度训练，会静默产出垃圾权重 —— 上一版就是这么练崩的。
# 统一成 fp16：T4 原生支持。
DTYPE = torch.float16

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=DTYPE, bnb_4bit_use_double_quant=True)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"        # 解码器批量生成必须左填充

model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto",
    torch_dtype=DTYPE)

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

train_raw = [json.loads(l) for l in open(f'{DATA}/final_train.jsonl', encoding='utf-8')]
print(f"训练集 {len(train_raw)} 条")

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
print("样本长度：", [len(ds[i]['input_ids']) for i in range(5)])
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
## 6.5 · 先跑 base 和 rag 两条路线

**顺序很重要。** 这两条路线用的都是「原始基座」，
所以趁 LoRA 还没开始训练、B 矩阵还是零初始化的时候跑最干净。

（上一版是训练完之后用 `disable_adapter()` 把 LoRA 关掉再跑，
实测**还原不干净**——base 路线出来的也是复读，整组对照就废了。）
"""),

    code("""
import sys
sys.path.insert(0, EVAL)
os.chdir(EVAL)          # generate.py 按自己的位置找 ../data
import generate as G

grid = [json.loads(l) for l in open(f'{DATA}/eval_grid.jsonl', encoding='utf-8')]
os.makedirs(f'{EVAL}/preds', exist_ok=True)

def run_arm(arm, system_fn, force=False):
    '''生成一条路线的预测。已有结果就跳过 —— 断了重跑不用从头发。'''
    path = f'{EVAL}/preds/{arm}.jsonl'
    if os.path.exists(path) and not force:
        n = sum(1 for _ in open(path, encoding='utf-8'))
        print(f"  {arm}: 已有 {n} 条结果，跳过（要重跑就删掉这个文件）")
        return
    prompts, metas = [], []
    for r in grid:
        q = r.get("question_eval") or r["question"]
        prompts.append(G.chat(tok, system_fn(q), q))
        metas.append({"qa_id": r["qa_id"], "cell": r["cell"]})
    outs = G.generate(model, tok, prompts, batch=8, max_new=192)
    with open(path, 'w', encoding='utf-8') as f:
        for m, o in zip(metas, outs):
            f.write(json.dumps({**m, "prediction": o}, ensure_ascii=False) + "\\n")
    print(f"  {arm}: {len(outs)} 条 → {path}")

# 切回推理态：梯度检查点必须关掉，否则生成出来是乱码
to_inference()

print("=== base：原始基座，不给任何上下文 ===")
run_arm('base', lambda q: G.SYSTEM)

print("=== rag：同一个基座 + 检索到的原文 ===")
retriever = G.build_retriever('hybrid')      # 向量 + BM25 融合，对应作业 A 的要求

def sys_rag(q):
    hits = retriever.search(q, topk=8)
    ctx = "\\n\\n".join(f"【{h['doc']} 第 {h['page']} 页】\\n{h['text']}" for h in hits)
    return G.SYSTEM_RAG.format(context=ctx[:6000])

run_arm('rag', sys_rag)
print("\\nbase / rag 已落盘。接下来训练，训练只会影响 sft 那一条。")
"""),

    md("""
## 7 · 训练

212 条 × 2 轮 ≈ 53 步，T4 上几分钟。
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
    output_dir="/kaggle/working/lora_out",
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
"""),

    md("## 8 · 微调后：再问同样三题\n\n和上面那三屏对照着看。"),

    code("""
# ★ 训练完必须切回推理态 —— 漏了这一步就是「训练前正常、训练后复读」
to_inference()
print(f"  训练态={model.training}  梯度检查点={getattr(model,'is_gradient_checkpointing',False)}"
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
## 9 · sft / sft_rag 路线（带 LoRA）

base 和 rag 在第 6.5 格已经跑完了（那时还没训练，`B` 矩阵是零初始化）。

这里跑两条都用 LoRA 的路线：

- **sft** = 微调后 + 不给上下文 → 只看写进权重里有多少
- **sft_rag** = 微调后 + 检索到的原文 → 口径来自微调，事实来自检索

`sft_rag` 是四条里唯一三条腿都齐的，也是结论的关键证据。
`retriever` 和 `sys_rag` 都还是第 6.5 格定义的那个，直接复用。
"""),

    code("""
run_arm('sft', lambda q: G.SYSTEM)
run_arm('sft_rag', sys_rag)
"""),

    md_download(platform),
    cell_download(platform),
] + CELLS_TAIL


def build_gen_cells():
    """「只生成」notebook —— 复用已经训好的 LoRA，只补 sft_rag 这一条路线。

    为什么不直接重跑完整 notebook：模型上一轮已经训好了，重训纯属浪费
    Colab 配额（免费额度用完要等 12–24 小时）。生成才是耗时的那一步。

    这个 notebook 不碰训练，所有推理都走 generate.py 的命令行 ——
    和另外三条路线用的是**同一份加载代码**，避免两套实现跑出不同结果。
    """
    return [

    md("""
# 补跑 `sft_rag` 路线（只生成，不训练）

上一轮已经把 LoRA 训好并打包在 `lixiang_results.zip` 里了。
这个 notebook 复用那个适配器，只补跑第四条路线：

> **`sft_rag` = 微调后的模型 + 检索到的财报原文**

口径来自微调，事实来自检索 —— 四条路线里唯一三条腿都齐的一条。

**全程不训练**，只做生成，T4 上约 25–30 分钟。

### 跑之前

右上角 `Runtime` → `Change runtime type` → 确认是 **T4 GPU**。

> 免费版 Colab 的 GPU 配额是黑盒，用完会报「已达到使用量限额」且不说何时恢复。
> 真遇到了就等 12–24 小时再跑，代码本身没问题。
"""),

    md("## 1 · 安装依赖"),
    cell_install("colab"),

    md("""
## 2 · 上传数据 + 适配器

点下面这格的运行按钮，会弹出「选择文件」——
选本地的 **`train/lixiang_gen_bundle.zip`**（约 77 MB）。

> 是**点按钮选文件**，不是拖拽。上传 77 MB 要等几分钟，进度条走完再往下跑。
"""),
    cell_setup_gen(),

    md("""
## 3 · 先试跑 3 条（约 3 分钟）

**别跳过这格。** 跑全量要 25 分钟，先花 3 分钟确认三件事都搭对了：

1. 适配器挂上了吗（回答应该是董秘口吻，不是「我无法提供」）
2. 检索通了吗（回答里的数字应该能在原文片段里找到）
3. 输出格式对吗（应该有【依据：《…》第 N 页】）

三条里有任何一条不对，就**别往下跑**，先回来找我。

第一次跑会向量化 1431 页（1–2 分钟）并把结果缓存下来，
第 4 格会直接读缓存，不会重复算。
"""),
    code("""
!python /content/lixiang/eval/generate.py --arm sft_rag \\
    --adapter /content/lixiang/lora_adapter --limit 3 --out /content/preflight

import json
print('\\n' + '=' * 78)
for line in open('/content/preflight/sft_rag.jsonl', encoding='utf-8'):
    r = json.loads(line)
    print('-' * 78)
    print(f"{r['cell']}  {r['qa_id']}")
    print(r['prediction'][:500])
"""),

    md("""
## 4 · 跑满 140 条

上面三条看着没问题就往下跑。约 20–25 分钟，会打印进度。

跑完结果落在 `/content/lixiang/eval/preds/sft_rag.jsonl`。
"""),
    code("""
!python /content/lixiang/eval/generate.py --arm sft_rag \\
    --adapter /content/lixiang/lora_adapter

# 顺手自检：条数对不对、有没有出现复读
import json
p = '/content/lixiang/eval/preds/sft_rag.jsonl'
rows = [json.loads(l) for l in open(p, encoding='utf-8')]
print(f'\\n{len(rows)} 条')
assert len(rows) == 140, f'应该是 140 条，实际 {len(rows)} 条 —— 别下载，先告诉我'
print('条数正确 ✓')
print('\\n前两条长这样：')
for r in rows[:2]:
    print('-' * 78)
    print(f"{r['cell']}  {r['qa_id']}")
    print(r['prediction'][:300])
"""),

    md("## 5 · 打包下载"),
    code("""
!cd /content && zip -q lixiang_sft_rag.zip lixiang/eval/preds/sft_rag.jsonl
!ls -la /content/lixiang_sft_rag.zip

from google.colab import files
files.download('/content/lixiang_sft_rag.zip')
"""),

    md("""
## 跑完之后

把 `lixiang_sft_rag.zip` 里的 `sft_rag.jsonl` 放到本地 `eval/preds/` 下，
然后我在本地跑判分，四条路线一起出四格对照表。

**Colab 没有「暂停」**——关标签页 = 虚拟机销毁，训练结果和预测全没。
跑的时候别关标签页。
"""),
    ]


def write_notebook(platform, path, cells=None):
    cells = cells if cells is not None else build_cells(platform)
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    if platform == "colab":
        nb["metadata"]["colab"] = {"provenance": [], "toc_visible": True}
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
    ap.add_argument("--platform", choices=["colab", "kaggle", "both"],
                    default="both")
    ap.add_argument("--mode", choices=["train", "gen", "all"], default="all",
                    help="train=完整微调流程；gen=复用已训好的适配器只补生成")
    args = ap.parse_args()

    if args.mode in ("train", "all"):
        if args.platform in ("colab", "both"):
            write_notebook("colab",
                           os.path.join(HERE, "lixiang_lora_sft.ipynb"))
        if args.platform in ("kaggle", "both"):
            write_notebook("kaggle",
                           os.path.join(HERE, "lixiang_lora_sft_kaggle.ipynb"))

    if args.mode in ("gen", "all"):
        write_notebook("colab",
                       os.path.join(HERE, "lixiang_gen_only.ipynb"),
                       cells=build_gen_cells())
    return 0


if __name__ == "__main__":
    sys.exit(main())
