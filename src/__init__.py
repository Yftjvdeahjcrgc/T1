"""T1（特一）自包含包根。

结构与主项目 HEv6.0 一致：`T1/src/` 即 `src` 包，内含两个子包：
  - `t1/`     —— T1 RPF 核心（概率引擎 + 规则引擎）。
  - `rf_moe/` —— 固化的约束引擎快照（vendored，唯一外部依赖：composite_list）。

把 `T1/` 目录加入 sys.path 后，即可 `import src.t1...` / `import src.rf_moe...`，
无需 HEv6.0 主项目，可单独拷到任意装有 torch 的机器运行。
"""

__version__ = "0.1.0"
