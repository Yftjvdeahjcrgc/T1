"""T1 固化的约束引擎快照（vendored）。

仅含 `services/composite_list.py` —— 综合名单（硬/软/白）裁决的**纯标准库**实现，
是 T1 对主项目 HEv6.0 的唯一运行时依赖，已固化进 T1 使本文件夹**完全自包含**。

⚠️ 与主项目的关系：这是「快照」，不是软链接。若主项目
`src/rf_moe/services/composite_list.py` 后续升级，需**手动同步**到本文件；
T1 不会自动跟随主项目变化（T1 是 QT 个人定制固定版，不应静默吃到主项目的改动）。

来源（权威）：HEv6.0 `src/rf_moe/services/composite_list.py`
术语权威：`deliverables/HEv7.0_约束术语澄清_v0.1.md`
"""
