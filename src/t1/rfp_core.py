"""T1 RPF 核心：概率引擎（自回归生成）+ 规则引擎（按分析上下文选规则 + 实时约束）。

设计（对应 `T1架构设计_特一_v0.1.md` §3 / §4 / §4.3）：
  - 概率引擎：自回归语言模型（默认 LSTM 主干，MX150 安全；可换成 Transformer），
    给定上下文产出 next-token 的 logits，逐 token 拼出回答。
  - 规则引擎：把**综合名单**（硬 / 软 / 白）在每一步对 next-token logits 施加裁决，
    等价于一个实时 logits processor；且**依据 CLM 的分析上下文只加载本次相关规则子集**
    （`AnalysisContext` + `RuleSelector` + `make_domain_selector`），而非全量启动。
    生成能力在概率引擎，红线在规则引擎，二者解耦。

复用（不重写 / 已固化）：
  - `src.rf_moe.services.composite_list.resolve` —— 综合名单裁决（权威实现）。
    该模块已 **vendored 固化**进 `T1/src/rf_moe/services/composite_list.py`（纯标准库，
    零 numpy/torch 依赖），故 T1 整个文件夹**完全自包含**、可单独拷到任意装有 torch
    的机器运行，不再依赖 HEv6.0 主项目。
  - `logits_processor.apply_verdict_torch` 的等价纯函数实现本文件自带（`apply_verdict_torch`），
    未 vendored 主项目的 `logits_processor`（其依赖 torch，T1 不需要，避免无谓膨胀）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn

from src.rf_moe.services.composite_list import (
    DEFAULT_POLICY,
    CompositeListPolicy,
    CompositeVerdict,
    resolve,
)


# ---------------------------------------------------------------------------
# 分析上下文 + 规则子集化（G4：规则引擎按分析上下文只选本次相关规则，非全量启动）
# ---------------------------------------------------------------------------
@dataclass
class AnalysisContext:
    """CLM 字符引擎产出的【分析上下文】，同时是规则引擎"选规则"的唯一依据。

    - domain：领域标签（如 medical / finance / legal / general），决定本次加载哪个规则子集。
    - variables：抽取出的关键变量（供变量引擎 / 规则引擎做变量级约束）。
    - raw_text：CLM 的拆解结论原文，便于规则引擎做短语/Trie 级匹配。
    """

    domain: str = ""
    variables: list[str] = field(default_factory=list)
    raw_text: str = ""


# 规则选择器：把分析上下文映射为"本次要用的规则子集"（CompositeVerdict）。
# 取代了"把所有规则全量加载、每步扫一遍"的低效做法。
RuleSelector = Callable[[AnalysisContext], CompositeVerdict]


def merge_verdicts(*verdicts: Optional[CompositeVerdict]) -> CompositeVerdict:
    """合并多个裁决（静态全量 / Trie 动态 / 分析上下文子集）为一份 union 裁决。"""
    hard: set[int] = set()
    soft: set[int] = set()
    white: set[int] = set()
    for v in verdicts:
        if v is None:
            continue
        hard |= v.hard
        soft |= v.soft
        white |= v.white
    return resolve(hard=hard, soft=soft, white=white)


def make_domain_selector(
    rule_bank: dict[str, CompositeVerdict],
    fallback: Optional[CompositeVerdict] = None,
) -> RuleSelector:
    """从"领域 → 规则子集"的规则库构造选择器：只返回当前领域对应的规则，非全量。

    rule_bank：{domain: CompositeVerdict}。未知领域回退到 fallback（可为空裁决）。
    这正是 G4「规则引擎按分析上下文选规则、不把所有规则全量常驻」的可操作实现。
    """
    def _select(ctx: AnalysisContext) -> CompositeVerdict:
        return rule_bank.get(ctx.domain, fallback if fallback is not None else resolve())
    return _select


# ---------------------------------------------------------------------------
# 规则引擎：把综合名单裁决应用到 torch logits（纯函数，复用 composite_list.resolve）
# ---------------------------------------------------------------------------
def apply_verdict_torch(
    scores: torch.Tensor,
    verdict: CompositeVerdict,
    policy: CompositeListPolicy = DEFAULT_POLICY,
) -> torch.Tensor:
    """把综合名单裁决应用到 torch logits（语义与 logits_processor.apply_verdict_torch 一致）。

    顺序：硬 → -inf；软 → 降权；生效白 → 加成；软∩白 → clamp 至 <= 原始 logit
    （白名单不得抵消软约束惩罚 = 「禁止永远 > 允许」）。
    """
    original = scores.clone()
    dev = scores.device

    if verdict.hard:
        idx = torch.as_tensor(sorted(verdict.hard), dtype=torch.long, device=dev)
        scores[..., idx] = -torch.inf

    soft_apply = verdict.soft - verdict.hard
    if soft_apply and policy.soft_penalty:
        idx = torch.as_tensor(sorted(soft_apply), dtype=torch.long, device=dev)
        scores[..., idx] = scores[..., idx] - float(policy.soft_penalty)

    if verdict.white_effective and policy.whitelist_bonus:
        idx = torch.as_tensor(sorted(verdict.white_effective), dtype=torch.long, device=dev)
        scores[..., idx] = scores[..., idx] + float(policy.whitelist_bonus)

    overlap = verdict.soft_and_white
    if overlap and policy.clamp_soft_white:
        idx = torch.as_tensor(sorted(overlap), dtype=torch.long, device=dev)
        scores[..., idx] = torch.minimum(scores[..., idx], original[..., idx])

    return scores


# ---------------------------------------------------------------------------
# 概率引擎：自回归生成器（默认 LSTM 主干，MX150 安全）
# ---------------------------------------------------------------------------
class ProbabilityEngine(nn.Module):
    """T1 的**生成器**：自回归产出 next-token logits。

    结构：Embedding → LSTM（多层）→ 解码头（→ vocab logits）。
    选用 LSTM 而非 Transformer，是为在本机 MX150 2GB 下稳妥跑通；
    需要更强容量时，把 ``backend`` 换 Transformer、或调大 ``hidden`` 即可。

    参数量拆分（预算关键，见 `count_params`）：
      - Embedding(V, E)：**查表**，训练期只有被命中的行参与梯度，不计 FLOPs；
      - LSTM：每 token 都要算的**稠密**部分，是 CPU 训练的主要成本之一；
      - head(H→V)：GEMM，H×V 决定成本，通常是最大头。
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden: int = 768,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_id: int = 0,
        tie_weights: bool = False,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, vocab_size)
        if tie_weights:
            # 要求 hidden == embed_dim；省掉 H×V 的参数，但**计算量不变**
            # （每 token 仍要算 H×V 的 logits）。
            if hidden != embed_dim:
                raise ValueError(f"tie_weights 要求 hidden == embed_dim，当前 {hidden} != {embed_dim}")
            self.head.weight = self.embed.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """输入 (B, T) token id → 输出 (B, T, V) 每个位置的 next-token logits。"""
        x = self.embed(ids)
        out, _ = self.lstm(x)
        return self.head(out)

    def next_logits(self, ctx_ids: torch.Tensor) -> torch.Tensor:
        """给定 (B, T) 上下文，返回最后一位的 next-token logits (B, V)。"""
        return self.forward(ctx_ids)[..., -1, :]


# ---------------------------------------------------------------------------
# RPF 核心：概率引擎 + 规则引擎 封装（受约束生成）
# ---------------------------------------------------------------------------
class RuleConstrainedGenerator:
    """把概率引擎与规则引擎组合成「受约束的生成模型」（T1 的核心新机制）。

    生成循环每一步：
      logits = 概率引擎(ctx) → 规则引擎裁决(apply_verdict) → 采样 → 拼回 → 下一步。

    规则约束支持三种来源（按 union 合并施加）：
      - 静态全局 id 集合（hard_ids / soft_ids / whitelist_ids）：固定红线兜底；
      - 上下文相关裁决函数 ``verdict_fn(ctx_ids) -> CompositeVerdict``：Trie 动态红线；
      - **规则子集选择器** ``rule_selector(AnalysisContext) -> CompositeVerdict``：
        依据 CLM 的【分析上下文】只返回本次相关规则（G4：非全量启动）。
    """

    def __init__(
        self,
        engine: ProbabilityEngine,
        vocab_size: int,
        verdict_fn: Optional[Callable[[list[int]], CompositeVerdict]] = None,
        rule_selector: Optional[RuleSelector] = None,
        hard_ids: Optional[Iterable[int]] = None,
        soft_ids: Optional[Iterable[int]] = None,
        whitelist_ids: Optional[Iterable[int]] = None,
        policy: CompositeListPolicy = DEFAULT_POLICY,
        device: str = "cpu",
    ) -> None:
        self.engine = engine.to(device).eval()
        self.device = device
        self.vocab_size = vocab_size
        self.policy = policy
        self.verdict_fn = verdict_fn
        self.rule_selector = rule_selector
        if verdict_fn is None and rule_selector is None:
            self._static_verdict = resolve(
                hard=set(hard_ids or ()),
                soft=set(soft_ids or ()),
                white=set(whitelist_ids or ()),
            )
        else:
            self._static_verdict = None

    def _verdict(self, ctx_ids: list[int], analysis: Optional[AnalysisContext] = None) -> CompositeVerdict:
        parts: list[Optional[CompositeVerdict]] = []
        if self.verdict_fn is not None:
            parts.append(self.verdict_fn(ctx_ids))
        if self.rule_selector is not None and analysis is not None:
            parts.append(self.rule_selector(analysis))
        if not parts and self._static_verdict is not None:
            parts.append(self._static_verdict)
        return merge_verdicts(*parts)

    @torch.no_grad()
    def generate(
        self,
        prefix_ids: list[int],
        analysis: Optional[AnalysisContext] = None,
        max_len: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        max_ctx: int = 256,
        rng: Optional[random.Random] = None,
    ) -> list[int]:
        """自回归生成。返回生成的 token id 序列（不含 prefix）。

        ``analysis``：CLM 的【分析上下文】，用于驱动规则子集选择（rule_selector）。
        """
        rng = rng or random.Random(42)
        ids = list(prefix_ids)
        generated: list[int] = []
        for _ in range(max_len):
            ctx = ids[-max_ctx:]
            t = torch.tensor([ctx], dtype=torch.long, device=self.device)
            logits = self.engine.next_logits(t)[0]
            verdict = self._verdict(ctx, analysis)
            logits = apply_verdict_torch(logits, verdict, self.policy)
            # 全部候选被禁止（理论上不会，至少保留一个非禁止 token）→ 终止
            if not torch.isfinite(logits).any():
                break
            logits = logits / max(temperature, 1e-6)
            if top_k and top_k > 0:
                k = min(top_k, logits.size(-1))
                thr = torch.topk(logits, k).values.min()
                logits = torch.where(logits >= thr, logits, torch.tensor(-torch.inf, device=self.device))
            probs = torch.softmax(logits, dim=-1)
            nid = int(torch.multinomial(probs, num_samples=1).item())
            if nid == self.engine.pad_id:  # 生成 <PAD> 视为终止
                break
            generated.append(nid)
            ids.append(nid)
        return generated


# ---------------------------------------------------------------------------
# 档位预设与参数预算（权威出处；T1/scripts/budget_0p3b.py 直接消费这些函数）
# ---------------------------------------------------------------------------
# 设计原则（2026-10-03 定，针对「8 核 CPU / 32GB 训练 + MX150 2GB 推理」）：
#   1. **参数放 Embedding**（查表，训练期不花 FLOPs），别全堆 LSTM/head；
#   2. head 的 H×V 决定 GEMM 成本 → CPU 训练的主要瓶颈，V 别盲目翻倍；
#   3. E 与 H 尽量同量级（宽进窄出会造成信息瓶颈，生成质量受损）；
#   4. 推理只看总参数：fp16 = 2N 字节，MX150 2GB 上 0.3B ≈ 595MB，可行。
T1_PRESETS: dict[str, dict] = {
    # name: (vocab, embed, hidden, layers, tied) —— vocab 用 --max-vocab 再截断
    "edge_16m":     dict(vocab=8000,  embed=256,  hidden=768,  layers=2, tied=False),
    "gen_0p124b":   dict(vocab=8000,  embed=1024, hidden=3072, layers=2, tied=False),
    # 0.3B 主力档（能力优先）：E=H=2048 不失衡，词表 48k
    "gen_0p3b":     dict(vocab=48000, embed=2048, hidden=2048, layers=3, tied=False),
    # 0.3B 速度档（CPU 算力优先）：参数更多挪到 Embedding，稠密部分从 199M 降到 128M
    "gen_0p3b_fast": dict(vocab=64000, embed=2560, hidden=1280, layers=3, tied=False),
}


def count_params(vocab: int, embed: int, hidden: int, layers: int,
                 tied: bool = False) -> dict[str, int]:
    """按 ProbabilityEngine 结构解析参数量。

    LSTM 每层 4 个门，每门 W_ih(E×H) + W_hh(H×H) + 2 个 bias(H)。
    返回 total / emb(查表) / lstm / head / dense(每 token 都要算的部分)。
    """
    emb = vocab * embed
    lstm_first = 4 * (embed * hidden + hidden * hidden + 2 * hidden)
    lstm_rest = 4 * (hidden * hidden + hidden * hidden + 2 * hidden) * (layers - 1)
    lstm = lstm_first + lstm_rest
    head = 0 if tied else hidden * vocab + vocab
    return {
        "emb": emb,
        "lstm": lstm,
        "head": head,
        "dense": lstm + head,   # 训练算力的计费口径
        "total": emb + lstm + head,
    }


def flops_per_token(vocab: int, embed: int, hidden: int, layers: int,
                    tied: bool = False) -> float:
    """训练期每 token FLOPs（前向 2N + 反向 4N = 6N）。

    只对 dense 部分计费：Embedding 是查表（反向为 scatter-add，吃带宽不吃算力）。
    head 若 tied，计算量不变（仍要算 H×V 的 logits）。
    """
    return 6.0 * count_params(vocab, embed, hidden, layers, tied)["dense"]


def train_memory_bytes(total: int, batch: int, seq: int, vocab: int, hidden: int,
                       layers: int, opt: str = "sgd", dtype_bytes: int = 4) -> dict[str, float]:
    """训练期内存（默认 fp32），返回各分项 MB。

    - weights/grads：各 N×dtype（Embedding 梯度在 PyTorch 里是稠密的 V×E 全张量）
    - 优化器状态：SGD(momentum=0) 0；SGD+momentum 1N；AdamW 2N
    - logits：(B, T, V) —— 训练期最大块，V 大时主导内存
    - LSTM 激活：反向需保存每层每 timestep 的隐/单元状态 ≈ 2·L·B·T·H
    """
    b = dtype_bytes
    state_mult = {"sgd": 0.0, "sgd_momentum": 1.0, "adamw": 2.0}.get(opt, 0.0)
    parts = {
        "weights_MB": total * b / 1e6,
        "grads_MB": total * b / 1e6,
        "opt_states_MB": state_mult * total * b / 1e6,
        "logits_MB": batch * seq * vocab * b / 1e6,
        "activations_MB": 2 * layers * batch * seq * hidden * b / 1e6,
    }
    parts["total_MB"] = sum(parts.values())
    return parts


def suggest_batch(total: int, vocab: int, seq: int, hidden: int, layers: int,
                  ram_gb: float = 24.0, opt: str = "sgd", max_batch: int = 256,
                  dtype_bytes: int = 4) -> int:
    """CPU 训练的安全 batch：同时受 RAM 总预算与 logits 单项上限约束。

    logits(B, T, V) 通常是最大单项（V 一大就爆炸）。给它设 35% 的预算上限，
    避免把内存全喂给这一个张量——那样会掉进内存带宽瓶颈，CPU 训练反而更慢。
    另设 max_batch 上限：CPU 上 batch 过大对吞吐的边际收益很快消失。
    """
    b = dtype_bytes
    state_mult = {"sgd": 0.0, "sgd_momentum": 1.0, "adamw": 2.0}.get(opt, 0.0)
    fixed = b * total * (2 + state_mult)                    # 权重 + 梯度 + 优化器状态
    per_sample = b * (seq * vocab + 2 * layers * seq * hidden)
    budget = ram_gb * 1e9 * 0.85
    by_ram = int((budget - fixed) / max(per_sample, 1))
    by_logits = int((budget * 0.35) / max(b * seq * vocab, 1))
    return max(1, min(by_ram, by_logits, max_batch))


def infer_weight_bytes(total: int, bits: int = 16) -> float:
    """推理期权重占用（MB）。bits: 32=fp32 / 16=fp16 / 8=int8。"""
    return total * bits / 8 / 1e6


def preset_report(name: str) -> str:
    """一行摘要，供训练脚本启动打印。"""
    p = T1_PRESETS[name]
    c = count_params(p["vocab"], p["embed"], p["hidden"], p["layers"], p["tied"])
    return (f"{name}: V={p['vocab']} E={p['embed']} H={p['hidden']} L={p['layers']} "
            f"→ 总 {c['total']/1e9:.3f}B（Embedding {c['emb']/1e6:.1f}M / "
            f"LSTM {c['lstm']/1e6:.1f}M / head {c['head']/1e6:.1f}M）"
            f" | 稠密 {c['dense']/1e6:.1f}M | fp16 推理权重 "
            f"{infer_weight_bytes(c['total'], 16):.0f}MB")
