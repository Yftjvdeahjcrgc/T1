"""综合名单（Composite List）—— HEv7.0 权威约束裁决机制。

权威定义来源：`deliverables/HEv7.0_约束术语澄清_v0.1.md`（架构师 QT，2026-10-02）。
该澄清**覆盖并作废**了 v6.0 时期「纯约束 = 只有红线、没有绿线」的措辞。

术语（权威，勿再改写）
----------------------
- 硬约束（黑名单·硬）：直接禁止输出某字词。Trie 前缀命中 → 生成期 logits 置 -inf
  （`LogitsProcessor`）+ 输出期 Trie 校验 + 回滚。**绝对禁止**，不可被白名单或软约束覆盖。
- 软约束（黑名单·软）：**不删除**字词，仅**降低**其出现概率（负向偏置）。
  软约束本身不导致输出失败——约束侧视为「可回答，只是更不易出错」。
- 纯约束：**硬 + 软**两者结合 = 黑名单（含硬、软两档力度）。
  此处的「纯」指「只谈禁止这一件事的力度细分」，**并非**「只有红线、没有绿线」。
- 白名单（绿线）：明确「允许」的那部分。QT 判定：单独的白名单不是好的名单机制——
  AI 知道哪些答案好，却无法用排除法定位自己哪里错了，反而易产生幻觉。
- 综合名单：黑名单 + 白名单结合，但**禁止永远大于允许**。这是要落地的目标机制。

裁决规则（本模块的实现依据）
----------------------------
1. 命中硬约束 → -inf。白名单允许**直接失效**（硬约束绝对优先）。
2. 仅命中软约束 → 降权 `soft_penalty`，token 仍留在采样空间。
3. 命中软约束且同时命中白名单 → 白名单允许**不失效**，仍享 `whitelist_bonus` 加成；
   但净值被 clamp 至 **<= 原始 logit**，即白名单**不得抵消**软约束的惩罚——
   这是「禁止永远大于允许」在软档位的可操作体现。
4. 仅命中白名单 → 加 `whitelist_bonus`（绿线引导，不改变可回答性）。

本模块只做**纯逻辑裁决**（输入 id 集合 → 输出裁决结果），不依赖 numpy / torch，
以便在零第三方依赖环境下被 trainbench 与单元测试直接引用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Optional

__all__ = [
    "CompositeVerdict",
    "CompositeListPolicy",
    "DEFAULT_POLICY",
    "resolve",
]


@dataclass(frozen=True)
class CompositeListPolicy:
    """综合名单的数值策略。

    Attributes:
        soft_penalty: 软约束负向偏置（作用在 logits 上，非概率）。越大惩罚越重。
        whitelist_bonus: 白名单正向加成（作用在 logits 上）。用于绿线引导。
        clamp_soft_white: 是否把「同时命中软约束与白名单」的 token 净值
            clamp 到不超过其原始 logit。默认 True —— 关闭即违背
            「禁止永远大于允许」，仅调试对比时可临时关闭。
    """

    soft_penalty: float = 2.0
    whitelist_bonus: float = 1.0
    clamp_soft_white: bool = True


DEFAULT_POLICY: CompositeListPolicy = CompositeListPolicy()


@dataclass(frozen=True)
class CompositeVerdict:
    """一次综合名单裁决的结果（id 均为**合并空间**下标）。

    Attributes:
        hard: 硬约束命中（绝对禁止）→ 置 -inf。
        soft: 软约束命中（降概率，允许）→ 负向偏置。
        blacklist: 纯约束 = hard ∪ soft（黑名单全集）。
        white: 原始白名单命中（绿线）。
        white_effective: **生效**白名单 = white - hard（硬约束处白名单失效）。
        soft_and_white: 同时命中软约束与白名单 → 降权 + 加成，但净值不得超原始。
    """

    hard: FrozenSet[int]
    soft: FrozenSet[int]
    white: FrozenSet[int]

    @property
    def blacklist(self) -> FrozenSet[int]:
        """纯约束 = 硬 + 软（黑名单全集）。"""
        return self.hard | self.soft

    @property
    def white_effective(self) -> FrozenSet[int]:
        """生效白名单：硬约束命中处白名单直接失效。"""
        return self.white - self.hard

    @property
    def soft_and_white(self) -> FrozenSet[int]:
        """同时命中软约束与白名单：白名单不失效，但净值不得抵消软惩罚。"""
        return self.soft & self.white_effective

    @property
    def soft_only(self) -> FrozenSet[int]:
        """仅命中软约束（不在白名单）：纯降权。"""
        return self.soft - self.white_effective

    @property
    def white_only(self) -> FrozenSet[int]:
        """仅命中白名单（未受任何黑名单约束）：纯加成。"""
        return self.white_effective - self.soft

    def is_hard_blocked(self, token_id: int) -> bool:
        """该 token 是否被硬约束绝对禁止。"""
        return token_id in self.hard

    def describe(self) -> str:
        """人类可读摘要（日志 / 工作台展示用）。"""
        return (
            f"硬{len(self.hard)} 软{len(self.soft)} 白{len(self.white)} "
            f"(生效白{len(self.white_effective)}, 软白交集{len(self.soft_and_white)})"
        )


def resolve(
    hard: Optional[Iterable[int]] = None,
    soft: Optional[Iterable[int]] = None,
    white: Optional[Iterable[int]] = None,
) -> CompositeVerdict:
    """执行一次综合名单裁决。

    Args:
        hard: 硬约束命中的 token id（合并空间）。
        soft: 软约束命中的 token id。
        white: 白名单命中的 token id。

    Returns:
        CompositeVerdict：已按「禁止 > 允许」完成优先级裁决的不可变结果。

    Note:
        裁决不含数值偏置计算（那是 logits 应用侧的事，见 `logits_processor.py`
        与 `adapters/hf_constraint.py`）；本函数只回答「谁被禁、谁被降、谁被允许」。
    """
    return CompositeVerdict(
        hard=frozenset(hard or ()),
        soft=frozenset(soft or ()),
        white=frozenset(white or ()),
    )
