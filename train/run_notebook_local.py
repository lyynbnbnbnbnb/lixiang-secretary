# -*- coding: utf-8 -*-
"""在本机真执行 notebook，并把**真实运行输出**存回 .ipynb 文件。

批改意见："所交笔记本未保存运行输出，不能只凭结论验收。"
手点 Jupyter 跑一遍也能有输出，但那样没法留时间戳、没法无人值守、
中断了输出也留不下来。这里用 nbclient 执行：

  · 逐格打印进度（第几格、已用多少分钟）；
  · 任何一格报错就**把已跑出的输出照样写盘**再退出（损失最小化）；
  · 显存分配器选项在起内核之前就设好 —— 6 GB 卡跑长上下文必须带。

用法：
    python run_notebook_local.py                      # 跑 train/lixiang_lora_sft.ipynb
    python run_notebook_local.py --timeout 5400       # 单格超时（秒），默认 1.5 小时
"""
import argparse
import os
import sys
import time

# ★ 必须在启动内核/导入 torch 之前设好（内核进程继承本进程的环境）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import nbformat                                                    # noqa: E402
from nbclient import NotebookClient                                # noqa: E402


class ProgressClient(NotebookClient):
    """带进度打印的执行器：长跑没人盯着，得能从日志看出跑到哪了。

    注意用 on_cell_executed（输出都收完了才算完）—— nbclient 的
    on_cell_complete 是"请求刚发出去"时调用的，拿它当完成会骗自己。
    """

    def __init__(self, *a, nb_path=None, **kw):
        super().__init__(*a, **kw)
        self._t0 = time.time()
        self._nb_path = nb_path

    def _save(self):
        """每跑完一格就落盘一次。

        ★ 被强杀时（系统内存压力、手滑、断电）连 except 都来不及跑，
        第一版只在结尾写盘，杀掉一次 = 一个多小时的输出全没。
        现在最多丢一格。
        """
        if not self._nb_path:
            return
        try:
            nbformat.write(self.nb, self._nb_path)
        except Exception as e:                                 # noqa: BLE001
            print(f"  （落盘失败：{e}）", flush=True)

    def on_cell_executed(self, cell, cell_index, execute_reply):
        el = time.time() - self._t0
        head = " ".join(cell.source.split())[:56]
        print(f"  [{el/60:5.1f} 分] 第 {cell_index + 1} 格完成：{head}…", flush=True)
        self._save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "lixiang_lora_sft.ipynb"))
    ap.add_argument("--timeout", type=int, default=5400,
                    help="单格超时秒数，默认 5400（1.5 小时，够最长的生成格）")
    ap.add_argument("--root", default=None,
                    help="内核工作目录，默认仓库根（笔记本从 cwd 往上找数据）")
    args = ap.parse_args()

    root = args.root or os.path.dirname(os.path.dirname(os.path.abspath(args.nb)))
    print(f"执行 {args.nb}\ncwd = {root}")
    nb = nbformat.read(args.nb, as_version=4)
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"共 {len(nb.cells)} 格（{n_code} 代码格），单格超时 {args.timeout}s\n")

    client = ProgressClient(nb, timeout=args.timeout, kernel_name="python3",
                            resources={"metadata": {"path": root}},
                            allow_errors=False, nb_path=args.nb)
    t0, failed = time.time(), None
    try:
        client.execute()
    except Exception as e:                                     # noqa: BLE001
        failed = e
        print(f"\n!! 执行中断：{type(e).__name__}: {str(e)[:400]}")

    # 不管成没成，先把输出写盘 —— 跑到第 9 格崩了，前 8 格的输出不能丢
    nbformat.write(nb, args.nb)
    el = time.time() - t0
    done = sum(1 for c in nb.cells
               if c.cell_type == "code" and c.get("outputs"))
    print(f"\n{'✓ 全部执行完成' if not failed else '× 中途中断'}："
          f"{done}/{n_code} 个代码格有输出，用时 {el/60:.1f} 分钟；输出已存回 {args.nb}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
