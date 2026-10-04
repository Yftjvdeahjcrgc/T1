"""T1（特一）隔离包：单 RPF 核心（受约束生成）+ CLM 分析 + TRA 反思 + 变量/因果确定性引擎。

T1 不修改现有 HEv6/7 任何代码，仅复用其已落地的规则引擎（composite_list /
logits_processor）、变量引擎（variable_engine）与因果一致性（CausalConsistencyLoss）。
"""
