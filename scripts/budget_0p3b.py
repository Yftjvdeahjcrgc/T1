#!/usr/bin/env python3
"""T1 概率引擎 · 档位参数预算与「8 核 CPU 训练 / MX150 2GB 推理」平衡点测算。

回答三个问题：
  1. 0.3B 具体怎么凑？（Embedding / LSTM / head 各多少）
  2. 在 8 核 CPU + 32GB 内存上训得动吗？多快？占多少内存？
  3. 训完在 MX150 2GB 上推理要多少显存、多少 token/s？

默认只做**解析计算**（不建模型，本机 1.2GB 可用内存也能跑）。
加 `--bench` 会在当前机器上实测校准算力（需要能放下小模型的内存）。

校准基线（2026-10-03 实测，本机 i7-8550U 4 物理核 / torch 2.12.1+cpu）：
  V=8000 E=256 H=768 L=2（16.1M，稠密 14.0M）→ 4 线程 636 tok/s ≈ 53.5 GFLOPS 有效算力
  → LSTM 在 CPU 上只有约 18% 的峰值效率（循环 kernel 无法像大 GEMM 那样吃满 AVX）。

语料：T1/data/corpus/training_corpus_with_cmmlu.jsonl（54,132 条 / 106,611 token，已固化进 T1）。

用法：
  C:/Python314/python.exe T1/scripts/budget_0p3b.py
  C:/Python314/python.exe T1/scripts/budget_0p3b.py --target-cores 8 --core-ratio 2.0
  C:/Python314/python.exe T1/scripts/budget_0p3b.py --bench --threads 8   # 在训练机上实测
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 路径注入：只注入 T1 根目录，使 T1/src 成为 src 包（内含 t1 与固化的 rf_moe）。
# T1 完全自包含，结构与主项目一致；可单独拷到任意装有 torch 的机器运行。
_T1_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_T1_ROOT))

import torch
import torch.nn as nn

from src.t1.rfp_core import (
    T1_PRESETS,
    ProbabilityEngine,
    count_params,
    flops_per_token,
    infer_weight_bytes,
    suggest_batch,
    train_memory_bytes,
)

# 本机实测校准值（见模块 docstring）
BASE_GFLOPS = 53.5          # 本机 4 线程 LSTM 有效算力
BASE_CORES = 4              # 对应物理核数

# MX150 推理参数
MX150_BANDWIDTH_GBs = 48.0  # GDDR5 64-bit @ ~6008MHz
MX150_USABLE_GB = 1.6       # 2GB 标称，WDDM 下实际可用

CORPUS_TOKENS = 106_611     # T1/data/corpus/training_corpus_with_cmmlu.jsonl（已固化进 T1）


def target_gflops(base: float, core_ratio: float, freq_ratio: float,
                  par_eff: float) -> float:
    """把本机实测算力外推到目标机。

    core_ratio：物理核数比；freq_ratio：单核（频率/架构）比；par_eff：多核并行效率折扣。
    """
    return base * core_ratio * freq_ratio * par_eff


def bench_local(cfg: dict, seq: int, batch: int, threads: int, steps: int = 3) -> dict:
    """在本机实测一个配置的训练吞吐（小配置用；大模型会 OOM）。"""
    torch.set_num_threads(threads)
    model = ProbabilityEngine(**cfg)
    n = sum(p.numel() for p in model.parameters())
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, foreach=False)
    ids = torch.randint(0, cfg["vocab_size"], (batch, seq))
    out = model(ids[:, :-1])
    crit(out.reshape(-1, cfg["vocab_size"]), ids[:, 1:].reshape(-1)).backward()
    opt.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(ids[:, :-1])
        crit(out.reshape(-1, cfg["vocab_size"]), ids[:, 1:].reshape(-1)).backward()
        opt.step()
    dt = (time.perf_counter() - t0) / steps
    tokens = batch * (seq - 1)
    dense = count_params(cfg["vocab_size"], cfg["embed_dim"], cfg["hidden"],
                         cfg["num_layers"])["dense"]
    return {
        "params_M": n / 1e6,
        "sec_per_step": dt,
        "tokens_per_s": tokens / dt,
        "gflops": 6.0 * dense * tokens / dt / 1e9,
    }


# 额外参考档（不在 T1_PRESETS 里，仅作对照）
EXTRA = [
    ("[对照] 0.35B 极宽嵌入", 32000, 3072, 3072, 2, False),
    ("[对照] 0.30B 全堆LSTM", 16000, 2048, 3072, 3, False),
    ("[对照] 0.20B tied", 64000, 2048, 2048, 2, True),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-cores", type=int, default=8)
    ap.add_argument("--core-ratio", type=float, default=2.0, help="目标机/本机 物理核数比")
    ap.add_argument("--freq-ratio", type=float, default=1.3, help="单核频率·架构比")
    ap.add_argument("--par-eff", type=float, default=0.85, help="多核并行效率折扣")
    ap.add_argument("--base-gflops", type=float, default=BASE_GFLOPS)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=0, help="0=由 RAM 预算自动推")
    ap.add_argument("--ram-budget-gb", type=float, default=24.0)
    ap.add_argument("--optim", default="sgd", choices=["sgd", "sgd_momentum", "adamw"])
    ap.add_argument("--bench", action="store_true", help="在本机实测校准（小配置）")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    print("=" * 84)
    print("T1 概率引擎 · 档位预算（训练：8核CPU/32GB  ｜  推理：本机 MX150 2GB 不变）")
    print("=" * 84)

    # ---------- 1. 算力外推 ----------
    gf = args.base_gflops
    if args.bench:
        r = bench_local(dict(vocab_size=8000, embed_dim=256, hidden=768, num_layers=2),
                        seq=64, batch=16, threads=args.threads)
        gf = r["gflops"]
        print(f"\n[1] 本机实测：{r['params_M']:.1f}M 配置，{args.threads} 线程 → "
              f"{r['tokens_per_s']:.0f} tok/s ≈ {gf:.1f} GFLOPS")
    else:
        print(f"\n[1] 校准基线：本机 {BASE_CORES} 核实测 {args.base_gflops:.1f} GFLOPS（未重测）")
    tgf = target_gflops(gf, args.core_ratio, args.freq_ratio, args.par_eff)
    print(f"    外推目标机（{args.target_cores}核）：{gf:.1f} × 核数{args.core_ratio} "
          f"× 频率{args.freq_ratio} × 并行效率{args.par_eff} = **{tgf:.0f} GFLOPS**")
    print("    注：这是 LSTM 的实测口径（≈峰值 18%），别拿 CPU 理论 FLOPS 直接除。")

    # ---------- 2. 档位表 ----------
    cands = [(n, c["vocab"], c["embed"], c["hidden"], c["layers"], c["tied"])
             for n, c in T1_PRESETS.items()] + EXTRA
    print(f"\n[2] 档位预算（seq={args.seq}，fp32，opt={args.optim}，RAM 预算 {args.ram_budget_gb:.0f}GB）")
    hdr = (f"{'档位':<22}{'总参B':>7}{'稠密M':>8}{'GFLOP/tok':>11}{'batch':>7}"
           f"{'内存GB':>8}{'tok/s':>8}{'1epoch':>9}{'6h':>7}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name, v, e, h, l, tied in cands:
        c = count_params(v, e, h, l, tied)
        fpt = flops_per_token(v, e, h, l, tied)
        # 与 train_t1.py 的 _probe_batch_cpu 同口径（rfp_core.suggest_batch）
        b = args.batch if args.batch > 0 else suggest_batch(
            c["total"], v, args.seq, h, l, ram_gb=args.ram_budget_gb, opt=args.optim)
        mem = train_memory_bytes(c["total"], b, args.seq, v, h, l, opt=args.optim)
        tps = tgf * 1e9 / fpt
        ep = CORPUS_TOKENS / tps
        rows.append((name, c, fpt, b, mem, tps, ep))
        print(f"{name:<22}{c['total']/1e9:>7.3f}{c['dense']/1e6:>8.1f}{fpt/1e9:>11.3f}"
              f"{b:>7}{mem['total_MB']/1000:>8.2f}{tps:>8.0f}{ep/60:>8.1f}m{6*3600/ep:>7.1f}")

    # ---------- 3. 语料匹配 ----------
    print(f"\n[3] 语料匹配度（当前语料 {CORPUS_TOKENS:,} token）")
    print(f"{'档位':<22}{'token/param':>13}{'判定':>12}{'建议最小语料(2×)':>20}{'理想(20×)':>14}")
    for name, c, *_ in rows:
        tpp = CORPUS_TOKENS / c["total"]
        verdict = "严重不足" if tpp < 2 else ("偏少" if tpp < 10 else "匹配")
        print(f"{name:<22}{tpp:>13.3f}{verdict:>12}"
              f"{c['total'] * 2 / 1e6:>17.0f}M{c['total'] * 20 / 1e9:>12.1f}B")

    # ---------- 4. 推理侧 ----------
    print(f"\n[4] 推理侧（MX150 2GB，可用≈{MX150_USABLE_GB}GB，带宽 {MX150_BANDWIDTH_GBs:.0f}GB/s）")
    print(f"{'档位':<22}{'fp32MB':>9}{'fp16MB':>9}{'int8MB':>9}"
          f"{'fp16 tok/s':>12}{'int8 tok/s':>12}{'结论':>10}")
    for name, c, *_ in rows:
        f32 = infer_weight_bytes(c["total"], 32)
        f16 = infer_weight_bytes(c["total"], 16)
        i8 = infer_weight_bytes(c["total"], 8)
        # 自回归每 token 要读一遍全部权重 → 带宽上限 = BW / weight_bytes；实际打 5 折
        t16 = MX150_BANDWIDTH_GBs * 1e9 / (f16 * 1e6) * 0.5
        t8 = MX150_BANDWIDTH_GBs * 1e9 / (i8 * 1e6) * 0.5
        ok = "可行" if f16 / 1000 < MX150_USABLE_GB * 0.7 else (
            "仅INT8" if i8 / 1000 < MX150_USABLE_GB * 0.7 else "不可行")
        print(f"{name:<22}{f32:>9.0f}{f16:>9.0f}{i8:>9.0f}{t16:>12.0f}{t8:>12.0f}{ok:>10}")

    # ---------- 5. 结论 ----------
    print("\n[5] 结论（2026-10-03）")
    print("  · 参数量 ≠ 训练成本：Embedding 是查表（不花 FLOPs），LSTM + head 才按 token 计费。")
    print("  · 想要 0.3B 又想在 CPU 上训得快 → 参数往 Embedding 挪，稠密部分压到 ~130–200M。")
    print("  · head 的 H×V 是 CPU 训练最大单项成本，词表每翻倍，logits GEMM 与内存同翻倍。")
    print("  · 推理是**带宽受限**（每 token 读一遍权重）：0.3B fp16≈595MB → 上限 ~80 tok/s，"
          "实际 ~40；int8≈297MB → 实际 ~80。MX150 2GB 装得下。")
    print("  · 真正短板是**语料**：0.3B 配 10 万 token 只能记忆。要么扩语料，要么降档。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
