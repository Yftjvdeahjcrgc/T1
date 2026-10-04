#!/usr/bin/env python3
"""T1（特一）生成式训练骨架：训练 RPF 核心的概率引擎（自回归生成）。

对应 `T1架构设计_特一_v0.1.md` §6 / §7。本脚本只训练 **概率引擎**（生成器）；
规则引擎（综合名单）、变量引擎、因果引擎为确定性/约束层，不在本脚本训练预算内
（训练期以 L_rule / L_causal 两项正则把红线「内部化」进概率引擎）。

纪律红线沿用 trainbench（best-only / resume / 中断保 best / 按时长停）。

MX150 2GB 硬约束：
  - SGD（foreach=False，避免 fused 多张量更新 OOM）。
  - Embedding + 掩码（不用 EmbeddingBag；真实不等长数据上 CPU 反向会段错误）。
  - 默认 hidden=768（约 20M 级，稳妥）；要上 0.3B 用 --preset gen_0p3b
    （V=48000 E=2048 H=2048 L=3 → 297M，外部 8 核 CPU 训练机训练 / MX150 2GB 推理）。

显存防护（针对「显存不够」报错）：
  - 预检 `_vram_guard`：CUDA 下估算峰值（params+grads+logits+LSTM 激活），超 0.85×总显存
    时自动降 batch / seq，仍超则明确报错并点出根因（--hidden 过大 / --max-vocab 过大）。
  - `_probe_batch` 用真实 CE 反向 + representative seq 探测，余量 *0.6 留给 cudnn workspace。
  - 运行时 CUDA OOM 自动降 batch 重试（最多 8 次），仍爆则报错退出。
  - --max-seq 封顶 logits 峰值；--max-samples 供低 RAM / 快速冒烟子采样。

用法示例：
  python T1/scripts/train_t1.py --corpus T1/data/corpus/training_corpus_with_cmmlu.jsonl \
      --out T1/data/checkpoints/t1_gen_v01.pt --max-hours 6 --causal-weight 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import threading
import time
from pathlib import Path

# 路径注入：只注入 T1 根目录，使 T1/src 成为 src 包（内含 t1 与固化的 rf_moe）。
# T1 完全自包含，结构与主项目一致；可单独拷到任意装有 torch 的机器运行。
_T1_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_T1_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.t1.rfp_core import (
    T1_PRESETS,
    ProbabilityEngine,
    apply_verdict_torch,
    count_params,
    infer_weight_bytes,
    preset_report,
    suggest_batch,
    train_memory_bytes,
)
from src.rf_moe.services.composite_list import DEFAULT_POLICY, resolve


# --------------------------------------------------------------------------- #
# 中断保护（红线 5：中断保 best）
# --------------------------------------------------------------------------- #
_INTERRUPTED = False


def _install_signal_handler() -> None:
    def _handler(signum, frame):  # noqa: ARG001
        global _INTERRUPTED
        _INTERRUPTED = True
        print("\n[中断] 收到停止信号，将在当前 epoch 结束后保存退出（best 权重已保留）。")

    try:
        signal.signal(signal.SIGINT, _handler)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# 词表与数据
# --------------------------------------------------------------------------- #
def build_vocab_from_corpus(samples: list[dict], max_vocab: int = 0) -> list[str]:
    """构建词表；max_vocab>0 时只保留高频前 N 个（词表膨胀会让 Embedding 梯度块 OOM）。"""
    freq: dict[str, int] = {}
    for s in samples:
        for t in str(s.get("text", "")).split():
            freq[t] = freq.get(t, 0) + 1
    words = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    if max_vocab and max_vocab > 0 and len(words) > max_vocab:
        words = words[:max_vocab]
    return [w for w, _ in words]


def load_corpus(path: Path) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


class GenDataset(Dataset):
    """自回归语料：每条样本为 token id 序列（0 = <PAD>/<UNK>，训练时在 loss 中掩码）。"""

    def __init__(self, samples: list[dict], vocab: list[str], min_len: int = 4,
                 max_seq: int = 128):
        self.v2i = {w: i for i, w in enumerate(vocab)}
        self.seqs: list[list[int]] = []
        cap = max(min_len, max_seq)
        for s in samples:
            toks = [self.v2i.get(t, 0) for t in str(s.get("text", "")).split() if t in self.v2i]
            if len(toks) > cap:
                toks = toks[:cap]  # 截断：封顶 logits 显存峰值
            if len(toks) < min_len:
                toks = toks + [0] * (min_len - len(toks))
            self.seqs.append(toks)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.long)


def collate_gen(batch):
    lengths = [len(b) for b in batch]
    max_len = max(lengths)
    bsz = len(batch)
    padded = torch.zeros((bsz, max_len), dtype=torch.long)
    mask = torch.zeros((bsz, max_len), dtype=torch.bool)  # True=有效（非 pad）
    for i, b in enumerate(batch):
        padded[i, : lengths[i]] = b
        mask[i, : lengths[i]] = True
    return padded, mask


# --------------------------------------------------------------------------- #
# 规则正则 / 因果一致性（训练期把红线内部化）
# --------------------------------------------------------------------------- #
def rule_regularization(logits: torch.Tensor, target: torch.Tensor, hard_ids: set[int]) -> torch.Tensor:
    """对命中硬约束的目标 token 降权：最小化其在生成分布中的概率质量。

    返回标量惩罚（平均到每个有效位置）。
    """
    if not hard_ids:
        return torch.zeros((), device=logits.device)
    probs = torch.softmax(logits.float(), dim=-1)
    target_probs = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    hard_mask = torch.zeros_like(target, dtype=torch.bool)
    for hid in hard_ids:
        hard_mask = hard_mask | (target == hid)
    if not hard_mask.any():
        return torch.zeros((), device=logits.device)
    return target_probs[hard_mask].mean()


def causal_consistency_loss(
    logits: torch.Tensor,
    device: torch.device,
    hard_ids: set[int] | None = None,
) -> torch.Tensor:
    """结构因果一致性（因果引擎训练期注入）：果（生成分布）不得超因（规则允许分布）。

    「因」= 规则允许分布：硬约束 token 概率 0，其余 token 均匀基线；归一化。
    无因之果 = clamp(p_collab - p_allowed, 0) 的和。
    与综合名单「禁止 > 允许」同构——生成分布不得把概率堆到规则不允许的位置上。
    """
    probs = torch.softmax(logits.float(), dim=-1)
    allowed = torch.ones_like(probs)
    if hard_ids:
        for hid in hard_ids:
            if 0 <= hid < allowed.size(-1):
                allowed[..., hid] = 0.0
    denom = allowed.sum(dim=-1, keepdim=True).clamp(min=1.0)
    allowed = allowed / denom
    overflow = torch.clamp(probs - allowed, min=0.0)
    return overflow.sum(dim=-1).mean()


# --------------------------------------------------------------------------- #
# 训练主流程
# --------------------------------------------------------------------------- #
_TRUE_WORDS = frozenset({"auto", "true", "1", "yes", "on"})
_FALSE_WORDS = frozenset({"false", "0", "no", "off", "none", "null"})


def resolve_resume_path(raw: str, out_path: str) -> str:
    """把 `--resume` 的三种写法归一为 ckpt 路径（空串 = 不续训）。

    trainbench 与命令行对「续训」的传法不统一，这里一次性兼容：

      - `--resume`（无值，argparse 取 const）→ 用 `--out` 指向的 ckpt；
      - `--resume true|1|yes|on` → 同上（布尔开关式传参）；
      - `--resume path/to/ckpt.pt` → 用指定路径；
      - `--resume false|0|no|off|none` → 不续训；
      - 不传 → 不续训。

    Args:
        raw: `--resume` 的原始取值。
        out_path: `--out` 指定的产物路径。

    Returns:
        str：ckpt 路径；空串表示不续训。
    """
    value = (raw or "").strip()
    if not value:
        return ""
    low = value.lower()
    if low in _TRUE_WORDS:
        return out_path
    if low in _FALSE_WORDS:
        return ""
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="T1/data/corpus/training_corpus_with_cmmlu.jsonl")
    parser.add_argument("--epochs", type=int, default=10_000_000)
    parser.add_argument("--batch", type=int, default=0, help="0=自动探测")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out", default="T1/data/checkpoints/t1_gen_v01.pt")
    parser.add_argument("--max-hours", type=float, default=6.0, help="0=不限")
    parser.add_argument("--resume", nargs="?", const="auto", default="",
                        help="续训：不带值（或 auto/true/1）= 从 --out 指向的 ckpt 续训；"
                             "也可直接给路径；false/0/no = 不续训")
    parser.add_argument("--max-vocab", type=int, default=8000,
                        help="词表上限（按词频截断）；MX150 大词表下 Embedding 梯度块会 OOM")
    parser.add_argument("--max-seq", type=int, default=128,
                        help="单条序列最大长度（截断）；封顶 logits 张量 (B,T,V) 的显存峰值")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0=全量；>0=仅取前 N 条（低 RAM / 快速冒烟用）")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=768,
                        help="LSTM 隐层宽度；3072→逼近 0.124B（需更稳 torch 或更大机器）")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--tie-weights", action="store_true",
                        help="head 与 Embedding 共享权重（省 H×V 参数，计算量不变；需 embed==hidden）")
    parser.add_argument("--preset", default="", choices=[""] + list(T1_PRESETS),
                        help="档位预设，覆盖 --max-vocab/--embed-dim/--hidden/--num-layers；"
                             f"可选：{', '.join(T1_PRESETS)}")
    parser.add_argument("--threads", type=int, default=0,
                        help="CPU 线程数；0=自动（取物理核数）")
    parser.add_argument("--optim", default="sgd", choices=["sgd", "sgd_momentum", "adamw"],
                        help="优化器；AdamW 收敛快但多占 2N 状态（0.3B 下 ≈2.4GB）")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="梯度累积步数；显存/RAM 不够放大 batch 时用")
    parser.add_argument("--ram-budget-gb", type=float, default=24.0,
                        help="CPU 训练的内存预算（GB）；用于自动探测 batch（默认给系统留 8GB）")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--rule-weight", type=float, default=0.0,
                        help="硬约束内部化正则权重；>0 即把红线学进概率引擎")
    parser.add_argument("--causal-weight", type=float, default=0.0,
                        help="L_causal 权重；>0 即启用因果一致性正则")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-ids", default="", help="逗号分隔的硬约束 token id（规则正则用）")
    args = parser.parse_args()

    if args.preset:
        p = T1_PRESETS[args.preset]
        args.max_vocab = p["vocab"]
        args.embed_dim = p["embed"]
        args.hidden = p["hidden"]
        args.num_layers = p["layers"]
        print(f"[t1_train] 档位预设 {preset_report(args.preset)}")

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[t1_train] device={device}", flush=True)
    if device.type == "cpu" and args.threads > 0:
        torch.set_num_threads(args.threads)
    try:
        import os
        _phys = os.cpu_count() or 1
        if args.threads == 0 and device.type == "cpu":
            torch.set_num_threads(max(1, _phys))
    except Exception:  # noqa: BLE001
        pass
    print(f"[t1_train] CPU 线程={torch.get_num_threads()}", flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = load_corpus(Path(args.corpus))
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
        print(f"[t1_train] 子采样 --max-samples={args.max_samples}")
    vocab = build_vocab_from_corpus(samples, max_vocab=args.max_vocab)
    print(f"[t1_train] 语料={len(samples)} 条 | 词表={len(vocab)} 词 | max_seq={args.max_seq}", flush=True)
    vocab_size = len(vocab)
    print(f"[t1_train] 构建概率引擎并迁移到 {device} …", flush=True)

    n_train = int(len(samples) * 0.8)
    train_ds = GenDataset(samples[:n_train], vocab, max_seq=args.max_seq)
    val_ds = GenDataset(samples[n_train:], vocab, max_seq=args.max_seq)

    model = ProbabilityEngine(
        vocab_size, embed_dim=args.embed_dim, hidden=args.hidden,
        num_layers=args.num_layers, tie_weights=args.tie_weights,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    brk = count_params(vocab_size, args.embed_dim, args.hidden, args.num_layers,
                       args.tie_weights)
    corpus_tokens = (sum(len(s) for s in train_ds.seqs)
                     + sum(len(s) for s in val_ds.seqs))
    print(f"[t1_train] 概率引擎参数={n_params/1e6:.1f}M "
          f"(Embedding {brk['emb']/1e6:.1f}M / LSTM {brk['lstm']/1e6:.1f}M / "
          f"head {brk['head']/1e6:.1f}M) | 稠密 {brk['dense']/1e6:.1f}M", flush=True)
    print(f"[t1_train] 推理权重估算：fp32 {infer_weight_bytes(n_params, 32):.0f}MB / "
          f"fp16 {infer_weight_bytes(n_params, 16):.0f}MB / "
          f"int8 {infer_weight_bytes(n_params, 8):.0f}MB")
    tpp = corpus_tokens / max(n_params, 1)
    print(f"[t1_train] 语料 {corpus_tokens} token / 参数 {n_params} = {tpp:.2f} token-per-param")
    if tpp < 2.0:
        print(f"[t1_train] ⚠ 语料相对参数严重不足（<2 token/param）：模型会记忆而非泛化。"
              f"本档位建议 ≥{n_params * 2 / 1e6:.0f}M token（最低下限 2×），"
              f"理想 {n_params * 20 / 1e9:.1f}B token（Chinchilla 20×）。")

    hard_ids = set(int(x) for x in args.hard_ids.split(",") if x.strip()) if args.hard_ids else set()

    if args.optim == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                                foreach=False)
    elif args.optim == "sgd_momentum":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=1e-4, foreach=False)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0,
                              weight_decay=1e-4, foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=9999, eta_min=1e-5)
    gen_criterion = nn.CrossEntropyLoss(reduction="none")  # 手工掩码

    best_loss, epoch, elapsed = 0.0, 0, 0.0
    best_weights, best_opt_state, best_sched_state = None, None, None
    resume_path = resolve_resume_path(args.resume, args.out)
    if resume_path and not Path(resume_path).exists():
        print(f"[t1_train] ⚠ 续训 ckpt 不存在，改为从头训练：{resume_path}")
        resume_path = ""
    if resume_path:
        st = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(st["model_state"], strict=False)
        opt.load_state_dict(st["optimizer_state"])
        scheduler.load_state_dict(st["scheduler_state"])
        best_loss = float(st.get("best_loss", 0.0))
        epoch = int(st.get("epoch", 0))
        print(f"[t1_train] ↩ 续训：{resume_path} | epoch {epoch} | best_loss={best_loss:.4f}")

    batch_size = _probe_batch(model, vocab_size, device, args.batch, args.max_seq,
                              ram_gb=args.ram_budget_gb, opt=args.optim)
    batch_size = _vram_guard(model, batch_size, args.max_seq, vocab_size, device,
                             opt=args.optim)
    max_seconds = args.max_hours * 3600 if args.max_hours > 0 else None
    start_time = time.time()
    _install_signal_handler()
    _oom_halvings = 0

    while True:
        if _INTERRUPTED:
            print("[t1_train] 收到中断信号，收摊保存 best 后退出")
            break
        epoch += 1
        model.train()
        run_gen = run_rule = run_causal = 0.0
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                 collate_fn=collate_gen, drop_last=True)
        _oom_hit = False
        pending = 0
        effective_batch = max(1, batch_size * max(1, args.grad_accum))
        for ids, mask in train_loader:
            try:
                if pending == 0:
                    opt.zero_grad(set_to_none=True)
                ids, mask = ids.to(device), mask.to(device)
                logits = model(ids[:, :-1])              # (B, T-1, V)
                target = ids[:, 1:]                      # (B, T-1)
                tmask = mask[:, 1:]                      # (B, T-1)
                per_tok = gen_criterion(logits.reshape(-1, vocab_size), target.reshape(-1))
                per_tok = per_tok.reshape(target.shape)
                loss_gen = (per_tok * tmask).sum() / tmask.sum().clamp(min=1)

                loss = loss_gen
                if args.rule_weight > 0 and hard_ids:
                    lr_ = rule_regularization(logits.detach(), target, hard_ids)
                    loss = loss + args.rule_weight * lr_
                    run_rule += float(lr_.item())
                if args.causal_weight > 0:
                    lc_ = causal_consistency_loss(logits, device, hard_ids)
                    loss = loss + args.causal_weight * lc_
                    run_causal += float(lc_.item())

                (loss / max(1, args.grad_accum)).backward()
                pending += 1
                run_gen += loss_gen.item()
                if pending >= max(1, args.grad_accum):
                    opt.step()
                    pending = 0
            except RuntimeError as ex:
                opt.zero_grad(set_to_none=True)
                if "out of memory" in str(ex).lower() and device.type == "cuda":
                    torch.cuda.empty_cache()
                    if batch_size > 1 and _oom_halvings < 8:
                        batch_size = max(1, batch_size // 2)
                        _oom_halvings += 1
                        print(f"[t1_train] ⚠ CUDA OOM，自动降 batch→{batch_size} 重试"
                              f"（第 {_oom_halvings} 次）")
                        _oom_hit = True
                        break  # 本 epoch 重来（用更小的 batch）
                    print(f"[t1_train] ❌ CUDA OOM 且已降到 batch={batch_size} 仍爆显存。")
                    print("  仍显存不足：请降低 --hidden / --max-vocab / --max-seq，"
                          "或换更大显存机器。")
                    raise SystemExit(3)
                raise
        if _oom_hit:
            continue  # 跳过本 epoch 验证，直接以更小 batch 重跑
        if pending:
            opt.step()          # 末尾不足一个累积周期的余数
            pending = 0
        scheduler.step()

        # 验证（next-token CE，掩码）
        model.eval()
        val_loss_tot, val_tok = 0.0, 0
        with torch.no_grad():
            val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_gen, drop_last=True)
            for ids, mask in val_loader:
                ids, mask = ids.to(device), mask.to(device)
                logits = model(ids[:, :-1])
                target = ids[:, 1:]
                tmask = mask[:, 1:]
                per_tok = gen_criterion(logits.reshape(-1, vocab_size), target.reshape(-1)).reshape(target.shape)
                val_loss_tot += float((per_tok * tmask).sum().item())
                val_tok += int(tmask.sum().item())
        val_loss = val_loss_tot / max(val_tok, 1)

        if best_loss == 0.0 or val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
            best_opt_state = {k: v for k, v in opt.state_dict().items()}
            best_sched_state = {k: v for k, v in scheduler.state_dict().items()}

        elapsed = time.time() - start_time
        print(f"[epoch {epoch}] val_loss={val_loss:.4f} L_rule={run_rule:.4f} "
              f"L_causal={run_causal:.4f} time={elapsed/60:.1f}min")

        if max_seconds and elapsed >= max_seconds:
            print(f"[t1_train] 时间到 ({args.max_hours}h)，保存 best 退出")
            break

    _save(model, opt, scheduler, vocab, best_loss, epoch, elapsed, Path(args.out),
          best_weights, best_opt_state, best_sched_state)
    print(f"[t1_train] 已保存 -> {Path(args.out)} | best_val_loss={best_loss:.4f}")
    return 0


def _save(model, opt, scheduler, vocab, best_loss, epoch, elapsed, out_path: Path,
          best_weights, best_opt_state, best_sched_state) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(
        {
            "model_state": best_weights if best_weights is not None else model.state_dict(),
            "optimizer_state": best_opt_state if best_opt_state is not None else opt.state_dict(),
            "scheduler_state": best_sched_state if best_sched_state is not None else scheduler.state_dict(),
            "vocab": vocab,
            "vocab_size": model.head.out_features,
            "embed_dim": model.embed.embedding_dim,
            "hidden": model.lstm.hidden_size,
            "num_layers": model.lstm.num_layers,
            "pad_id": model.pad_id,
            "tie_weights": bool(model.head.weight.data_ptr() == model.embed.weight.data_ptr()),
            "best_loss": best_loss,
            "epoch": epoch,
            "elapsed_seconds": elapsed,
        },
        tmp,
    )
    tmp.replace(out_path)


def _probe_batch_cpu(model, vocab_size: int, max_seq: int, ram_gb: float,
                     opt: str = "sgd") -> int:
    """按 RAM 预算反推 batch（CPU 训练路径）。口径见 rfp_core.suggest_batch。

    占用 = 权重 + 梯度 + 优化器状态（固定）+ 每样本的 logits(T×V) 与 LSTM 激活。
    logits 是最大单项，故另给它 35% 的预算上限，避免掉进内存带宽瓶颈。
    """
    n = sum(p.numel() for p in model.parameters())
    b = suggest_batch(n, vocab_size, max_seq, model.lstm.hidden_size,
                      model.lstm.num_layers, ram_gb=ram_gb, opt=opt)
    print(f"[t1_train] (CPU) RAM 预算 {ram_gb:.0f}GB → 自动 batch={b}"
          f"（logits 单项已限 35% 预算，上限 256）")
    return b


def _estimate_peak_bytes(model, batch: int, seq: int, vocab: int) -> int:
    """峰值显存估算（fp32，MX150 2GB 预算用，单位 bytes）。

    仅供预检 / 诊断；不含 cudnn workspace（实测由 _probe_batch 的 OOM 兜底）。
      - params + grads + SGD(momentum=0) 状态 ≈ 3× 参数量
      - logits = batch × seq × vocab（训练期 .reshape(-1, V) 全词表 CE 即此尺寸）
      - LSTM 激活 ≈ 2 × layers × batch × seq × hidden（隐/单元状态）
    """
    p = sum(x.numel() for x in model.parameters())
    logits = batch * seq * vocab
    hidden = model.lstm.hidden_size
    nl = model.lstm.num_layers
    acts = 2 * nl * batch * seq * hidden
    return (p + p + p + logits + acts) * 4  # ×4: fp32


def _vram_guard(model, batch: int, seq: int, vocab: int, device, opt: str = "sgd") -> int:
    """CUDA 预检 / CPU 内存提示。opt 用于把优化器状态计入估算。

    直接回答「显存不够」：把超预算的根因（--hidden 过大 / --max-vocab 过大）显式说出来。
    CPU 路径只打印估算（占用系统 RAM，非显存）。
    """
    if device.type != "cuda":
        n = sum(x.numel() for x in model.parameters())
        mem = train_memory_bytes(n, batch, seq, vocab, model.lstm.hidden_size,
                                 model.lstm.num_layers, opt=opt)
        print(f"[t1_train] (CPU) 估算峰值≈{mem['total_MB']/1000:.2f}GB（占系统 RAM）："
              f"权重 {mem['weights_MB']:.0f}MB + 梯度 {mem['grads_MB']:.0f}MB + "
              f"优化器 {mem['opt_states_MB']:.0f}MB + logits {mem['logits_MB']:.0f}MB + "
              f"激活 {mem['activations_MB']:.0f}MB")
        return batch
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(total * 0.85)
    cur_seq = seq
    while True:
        est = _estimate_peak_bytes(model, batch, cur_seq, vocab)
        if est <= cap:
            print(f"[t1_train] ✓ VRAM 预算：总={total/1e6:.0f}MB 估算峰值={est/1e6:.0f}MB "
                  f"(batch={batch} seq={cur_seq} vocab={vocab})", flush=True)
            return batch
        if batch > 1:
            batch //= 2
            continue
        if cur_seq > 16:
            cur_seq = max(16, cur_seq // 2)
            continue
        print(f"[t1_train] ❌ 估算峰值 {est/1e6:.0f}MB 超 {cap/1e6:.0f}MB 上限"
              f"（0.85×总显存 {total/1e6:.0f}MB）。")
        print("  显存不足根因通常是 --hidden 过大（3072 在 2GB 下基本不可行）或"
              " --max-vocab 过大（词表膨胀使 Embedding/head/CE 爆显存）。")
        print("  请降低 --hidden（如 256 / 384）、--max-vocab（如 4000 / 8000），"
              "或换更大显存机器。")
        raise SystemExit(2)


def _probe_batch_cuda(model, vocab_size: int, max_seq: int, device, opt: str = "sgd") -> int:
    """按显存总量解析式反推 batch（CUDA 路径，**不做任何 GPU 试算**）。

    为什么不做经验探测：本机 MX150 2GB 走 WDDM，torch 2.13+cu126 / 驱动 582.66 下
    反复做大 batch 的 forward+backward 会偶发**原生访问冲突**（进程 exit 0xC0000005），
    该崩溃发生在驱动层、Python 抛不出异常也捕不到，会把整个训练进程直接打死。
    故改用 `_estimate_peak_bytes` 纯数学求解，再乘经验安全系数。
    """
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(total * 0.85)
    seq = max_seq
    b = 1
    while b < 4096 and _estimate_peak_bytes(model, b * 2, seq, vocab_size) <= cap:
        b *= 2
    # 经验系数 0.5：估算未含 CE 中间量（logits 梯度≈再加一份 logits）与 cudnn workspace
    b = max(1, int(b * 0.5))
    est = _estimate_peak_bytes(model, b, seq, vocab_size)
    print(f"[t1_train] (CUDA) 解析式 batch={b}（总显存 {total/1e6:.0f}MB，"
          f"估算峰值 {est/1e6:.0f}MB；已跳过 GPU 经验探测以避免 WDDM 原生崩溃）", flush=True)
    return b


def _probe_batch(model, vocab_size: int, device, manual: int, max_seq: int = 128,
                 ram_gb: float = 24.0, opt: str = "sgd") -> int:
    if manual > 0:
        return manual
    if device.type != "cuda":
        return _probe_batch_cpu(model, vocab_size, max_seq, ram_gb, opt)
    # 本机 MX150 2GB WDDM 在 CUDA 上偶发原生访问冲突（exit 0xC0000005=3221225477）：
    # 崩溃在驱动层、Python try/except 抓不住，而「反复做大 batch 试算」正是最易触发它的动作。
    # 故默认**不在 GPU 上做经验探测**，改为纯数学反推 batch（零 GPU 试算 → 零崩溃风险）。
    # 确需经验探测时设环境变量 T1_EMPIRICAL_PROBE=1（仅建议在驱动稳定的大显存机上使用）。
    if os.environ.get("T1_EMPIRICAL_PROBE", "0") != "1":
        return _probe_batch_cuda(model, vocab_size, max_seq, device, opt)
    # 用「真实训练形态」探测：representative seq（取 min(max_seq, 64) 兼顾稳定与代表性）
    # + 走真实 CE 反向，使峰值贴近实战（原 out.sum() 会漏掉 logits 梯度账）。
    seq = min(max_seq, 64)
    crit = nn.CrossEntropyLoss(reduction="mean")

    def _trial(batch: int) -> None:
        """跑一次 (batch, seq) 的 forward+backward。任意 RuntimeError 都视为「本档过大」，
        由调用方据此收缩；finally 强制清理张量与分配器状态，避免 OOM 污染 caching allocator
        后级联出 invalid resource handle（该错误不含 'out of memory' 字样，旧逻辑会直接 raise 崩溃）。"""
        tok = out = loss = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            tok = torch.randint(0, max(1, vocab_size), (batch, seq), device=device)
            out = model(tok)
            loss = crit(out.reshape(-1, vocab_size), tok.reshape(-1))
            loss.backward()
            model.zero_grad(set_to_none=True)
        finally:
            for _t in (tok, out, loss):
                if _t is not None:
                    del _t
            torch.cuda.empty_cache()

    # 倍增：从 1 向上找上限（不再从 2048 起步，否则小显存首探即 OOM 污染分配器）
    low, high = 1, 1
    while True:
        try:
            _trial(high)
            low = high
            if high >= 1 << 13:   # 8192 封顶，防止大卡无意义探到天量
                break
            high *= 2
        except RuntimeError:
            break
    # 二分：在 [low, high-1] 找最大可行 batch（任意 CUDA 错误都按「过大」收缩）
    best = low
    lo, hi = low, high - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            _trial(mid)
            best = mid
            lo = mid + 1
        except RuntimeError:
            hi = mid - 1
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    # 留足 cudnn workspace + 实战 seq 可能 > 探测 seq 的余量
    return max(1, int(best * 0.6))


if __name__ == "__main__":
    raise SystemExit(main())
