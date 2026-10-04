#!/usr/bin/env python3
"""T1（特一）受约束生成：加载 RPF 核心的概率引擎，叠加规则引擎实时约束，自回归生成。

对应 `T1架构设计_特一_v0.1.md` §3.3 / §7。规则引擎把**综合名单**（硬/软/白）在每一步
对 next-token logits 施加裁决（硬 −∞ / 软降权 / 白有效加成且 clamp ≤ 原始），
等价于一个实时 logits processor——生成能力在概率引擎，红线在规则引擎，二者解耦。

规则子集化（G4）：用 `--rule-bank`（按领域选规则）+ `--domain` 构建 CLM 的【分析上下文】，
规则引擎只加载当前领域对应的规则子集（非全量启动）。`--rule-policy` 仍保留作静态全量兜底。

用法示例：
  # 静态全量兜底
  python T1/scripts/generate_t1.py --ckpt T1/data/checkpoints/t1_gen_v01.pt \
      --prompt "感冒了怎么办" --rule-policy T1/data/rules/sample_policy.json
  # 按领域选规则（规则子集化）
  python T1/scripts/generate_t1.py --ckpt T1/data/checkpoints/t1_gen_v01.pt \
      --prompt "..." --domain medical --rule-bank T1/data/rules/rule_bank.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# 路径注入：只注入 T1 根目录，使 T1/src 成为 src 包（内含 t1 与固化的 rf_moe）。
# T1 完全自包含，结构与主项目一致；可单独拷到任意装有 torch 的机器运行。
_T1_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_T1_ROOT))

import torch

from src.t1.rfp_core import (
    AnalysisContext,
    ProbabilityEngine,
    RuleConstrainedGenerator,
    infer_weight_bytes,
    make_domain_selector,
)
from src.rf_moe.services.composite_list import resolve


def load_ckpt(ckpt_path: Path, dtype: str = "fp32"):
    """加载 ckpt；dtype=fp16 时把权重转半精度（MX150 2GB 上跑 0.3B 的必要手段）。"""
    st = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    vocab = st["vocab"]
    model = ProbabilityEngine(
        vocab_size=st["vocab_size"],
        embed_dim=st["embed_dim"],
        hidden=st["hidden"],
        num_layers=st["num_layers"],
        pad_id=st["pad_id"],
        tie_weights=bool(st.get("tie_weights", False)),
    )
    model.load_state_dict(st["model_state"], strict=False)
    n = sum(p.numel() for p in model.parameters())
    bits = {"fp32": 32, "fp16": 16, "bf16": 16}[dtype]
    print(f"[t1_gen] 概率引擎参数={n/1e6:.1f}M（{n/1e9:.3f}B） | "
          f"权重占用 fp32 {infer_weight_bytes(n, 32):.0f}MB / "
          f"fp16 {infer_weight_bytes(n, 16):.0f}MB / int8 {infer_weight_bytes(n, 8):.0f}MB")
    print(f"[t1_gen] 本轮推理精度={dtype} → 权重约 {infer_weight_bytes(n, bits):.0f}MB")
    if dtype in ("fp16", "bf16"):
        model = model.to(torch.float16 if dtype == "fp16" else torch.bfloat16)
    return model, vocab


def _to_ids(phrases: list[str], v2i: dict[str, int]) -> set[int]:
    ids: set[int] = set()
    for ph in phrases or []:
        for tok in str(ph).split():
            if tok in v2i:
                ids.add(v2i[tok])
    return ids


def build_verdict_from_policy(policy_path: Path | None, vocab: list[str]):
    """从 JSON 策略文件（hard/soft/white 短语列表）构建静态综合名单裁决（兜底）。

    JSON 形如：
      {"hard": ["禁止短语A"], "soft": ["谨慎短语B"], "white": ["允许短语C"]}
    短语按词表切分后映射到 token id（词表为空格分词，单 token 短语直接命中）。
    """
    if policy_path is None or not policy_path.exists():
        return None
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    v2i = {w: i for i, w in enumerate(vocab)}
    hard = _to_ids(policy.get("hard"), v2i)
    soft = _to_ids(policy.get("soft"), v2i)
    white = _to_ids(policy.get("white"), v2i)
    static = resolve(hard=hard, soft=soft, white=white)
    print(f"[t1_gen] 静态全量裁决：hard={len(hard)} soft={len(soft)} white={len(white)}")
    return static


def build_rule_bank(rule_bank_path: Path | None, vocab: list[str]):
    """从「领域 → 规则子集」JSON 构建规则库，并返回 (rule_selector, 加载摘要)。

    JSON 形如：
      {"medical": {"hard": [...], "soft": [...]},
       "finance": {"hard": [...], "white": [...]}}
    规则引擎只加载当前 domain 对应的子集（G4：非全量启动）。
    """
    if rule_bank_path is None or not rule_bank_path.exists():
        return None, {}
    bank_json = json.loads(rule_bank_path.read_text(encoding="utf-8"))
    v2i = {w: i for i, w in enumerate(vocab)}
    bank: dict[str, object] = {}
    summary: dict[str, int] = {}
    for domain, spec in bank_json.items():
        hard = _to_ids(spec.get("hard", []), v2i)
        soft = _to_ids(spec.get("soft", []), v2i)
        white = _to_ids(spec.get("white", []), v2i)
        bank[domain] = resolve(hard=hard, soft=soft, white=white)
        summary[domain] = len(hard) + len(soft) + len(white)
    selector = make_domain_selector(bank)
    print(f"[t1_gen] 规则库（按领域选规则）：{summary}")
    return selector, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--rule-policy", default="", help="静态全量裁决 JSON（兜底）")
    parser.add_argument("--rule-bank", default="", help="领域→规则子集 JSON（规则子集化）")
    parser.add_argument("--domain", default="", help="CLM 分析上下文的领域标签")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
                        help="推理精度；MX150 2GB 上跑 0.3B 请用 fp16（权重减半）")
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model, vocab = load_ckpt(Path(args.ckpt), dtype=args.dtype)
    i2v = {i: w for i, w in enumerate(vocab)}

    static_verdict = build_verdict_from_policy(
        Path(args.rule_policy) if args.rule_policy else None, vocab
    )
    rule_selector, _ = build_rule_bank(
        Path(args.rule_bank) if args.rule_bank else None, vocab
    )

    device = torch.device(args.device)
    generator = RuleConstrainedGenerator(
        model, vocab_size=len(vocab), verdict_fn=None,
        rule_selector=rule_selector, device=str(device),
    )
    # 若无领域规则库，则用静态全量裁决兜底（RuleConstrainedGenerator 内部已存 _static_verdict）
    if static_verdict is not None and rule_selector is None:
        generator._static_verdict = static_verdict

    # CLM 的【分析上下文】：当前由 domain 驱动规则子集选择；variables/raw_text 待 CLM 接入后填充
    analysis = AnalysisContext(domain=args.domain)

    prompt = args.prompt.strip()
    prefix_ids = [vocab.index(t) if t in vocab else 0 for t in prompt.split()] if prompt else [0]
    rng = random.Random(args.seed)
    gen_ids = generator.generate(
        prefix_ids, analysis=analysis, max_len=args.max_len,
        temperature=args.temperature, top_k=args.top_k, rng=rng,
    )
    gen_tokens = [i2v.get(i, "") for i in gen_ids]
    text = " ".join([t for t in gen_tokens if t])
    print("── 生成结果 ──")
    print(text or "（未生成；可能 prefix 后即被规则约束终止）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
