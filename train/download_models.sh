#!/usr/bin/env bash
# 下载本地推理要用的两个模型（断点续传，可反复跑）。
#
#   Qwen2.5-1.5B-Instruct   ~3.0 GB   生成用
#   Qwen3-Embedding-0.6B    ~1.1 GB   向量检索用
#
# 为什么不用 huggingface-cli：本机走代理，curl 更可控，而且 -C - 能续传 ——
# 下到一半关机也不怕。
#
# 用法：  bash train/download_models.sh

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Clash 的系统代理对 curl 不生效，必须显式给
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7897}"

# 远端字节数。HF 会 302 跳到 CDN，代理那一跳没有 content-length，
# 所以取**最后一条** content-length（= 最终对象的长度）。
remote_size() {
    curl -sIL --max-time 60 "$1" 2>/dev/null \
        | tr -d '\r' \
        | awk 'tolower($1) == "content-length:" { n = $2 } END { if (n) print n }'
}

local_size() { local n=0; [ -s "$1" ] && n="$(stat -c%s "$1" 2>/dev/null || echo 0)"; echo "$n"; }

fetch() {   # fetch <仓库> <本地目录> <文件...>
    local repo="$1" dir="$2"; shift 2
    mkdir -p "$dir"
    local base="https://huggingface.co/$repo/resolve/main"
    for f in "$@"; do
        local out="$dir/$f" want have
        want="$(remote_size "$base/$f")"
        have="$(local_size "$out")"

        # ★ 只看「文件非空」是不够的：下到一半关机，落盘的是非空但**截断**的文件，
        #   上一版就是这么把 98 MB 的缺口当成「已下载」跳过去了，而且
        #   transformers 只在加载时才报 "incomplete metadata"，看起来像模型坏了。
        #   必须和远端字节数比。
        if [ -n "$want" ] && [ "$have" = "$want" ]; then
            echo "  ✓ $f 已完整（$(du -h "$out" | cut -f1)），跳过"
            continue
        fi

        if [ "$have" -gt 0 ] 2>/dev/null; then
            echo "  ↓ $f 续传（已有 $have / 共 ${want:-?} 字节）…"
        else
            echo "  ↓ $f …"
        fi
        # -C - 续传：上次下了一半就接着下，不重来
        curl -L -C - --retry 3 --retry-delay 2 --max-time 7200 \
             -o "$out" "$base/$f"

        have="$(local_size "$out")"
        if [ -n "$want" ] && [ "$have" != "$want" ]; then
            echo "    ✗ $f 下载不完整（$have / $want 字节），重跑本脚本可续传"
            return 1
        fi
        echo "    ✓ $(du -h "$out" | cut -f1)"
    done
}

echo "=== Qwen2.5-1.5B-Instruct（生成用，约 3.0 GB）==="
fetch "Qwen/Qwen2.5-1.5B-Instruct" "models/Qwen2.5-1.5B-Instruct" \
      config.json generation_config.json merges.txt tokenizer.json \
      tokenizer_config.json vocab.json model.safetensors

echo
echo "=== Qwen3-Embedding-0.6B（向量检索用，约 1.1 GB）==="
fetch "Qwen/Qwen3-Embedding-0.6B" "models/Qwen3-Embedding-0.6B" \
      config.json generation_config.json merges.txt tokenizer.json \
      tokenizer_config.json vocab.json model.safetensors

echo
echo "=== 校验 ==="
for d in models/Qwen2.5-1.5B-Instruct models/Qwen3-Embedding-0.6B; do
    if [ -s "$d/model.safetensors" ]; then
        echo "  ✓ $d  $(du -h "$d/model.safetensors" | cut -f1)"
    else
        echo "  ✗ $d 权重缺失"
    fi
done
