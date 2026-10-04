"""T1 固化的约束引擎 services 子包（vendored）。

当前仅 `composite_list.py` 一个模块被 T1 引用。主项目该目录下其余模块
（logits_processor / variable_engine / rule_schema 等）**未** vendored 进 T1——
T1 的 rfp_core 自带与 `apply_verdict_torch` 等价的纯函数实现，不依赖它们。
"""
