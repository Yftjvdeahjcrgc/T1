# T1（特一）定制版

> **标注：T1** —— 本文件夹为「特一 / T1」定制模型的独立工作区，与 HEv6.0 主项目（HEv7.0/7.5
> 多专家 / RPF-MoE / WoE 路由 / AI 协作协议）**解耦**。T1 是 QT 个人定制特殊版，**无正式版本编号**，
> 直接写「特一 / T1」。等笔记本升级后，再把多专家 / RPF-MoE / WoE 路由 / AI 协作协议接回 7.5 / 100B 线。

## 定位

- **额定档位：0.3B**（gen_0p3b 预设 → 297M，稠密 199M，fp16 ≈ 595MB）。
- **训练**：外部 CPU 训练机（8 核 / 32GB）。本机（MX150 2GB）只做推理。
- **核心重构**：原「多专家那一块」合并为 **RPF 核心 = 规则引擎 + 概率引擎 = 受约束生成式模型**。
  - 概率引擎：自回归 token 生成（LSTM 起步，目标 0.3B）。
  - 规则引擎：综合名单（硬 −∞ / 软降权 / 白有效加成），每步实时 logits 约束。
- **保留**：CLM（字符引擎·分析）、TRA（推理控制器·确定性反思）。
- **从 7.5 提前拉下**：变量引擎 + 因果引擎。

## 目录结构（T1 相关的东西已全部归拢于此）

```
T1/
├── README.md                 # 本文件（标注 T1）
├── src/
│   ├── __init__.py           # src 包（结构与主项目一致）
│   ├── t1/
│   │   ├── __init__.py       # T1 隔离包说明
│   │   └── rfp_core.py       # RPF 核心：ProbabilityEngine / RuleConstrainedGenerator / AnalysisContext / T1_PRESETS
│   └── rf_moe/               # 固化的约束引擎快照（vendored）
│       ├── __init__.py
│       └── services/
│           ├── __init__.py
│           └── composite_list.py   # 综合名单裁决（唯一外部依赖，已固化进 T1）
├── scripts/
│   ├── train_t1.py           # 受约束生成式训练（概率引擎 + L_rule/L_causal 正则）
│   ├── generate_t1.py        # 受约束自回归生成（fp16/bf16 推理）
│   └── budget_0p3b.py        # 0.3B 档位参数预算与训练/推理平衡点测算
├── data/
│   ├── corpus/
│   │   └── training_corpus_with_cmmlu.jsonl   # 训练语料 54,132 条 / 106,611 token（8.1MB，已固化）
│   ├── checkpoints/          # T1 ckpt 落盘处（t1_gen_v01.pt / t1_gen_0p3b_v01.pt）
│   └── rules/
│       └── sample_policy.json                # 规则 policy 样本（generate --rule-policy 用）
└── docs/
    ├── T1架构设计_特一_v0.1.md                # T1 架构设计（说明书）
    └── T1_0.3B档位参数预算与训练推理平衡方案_v0.1.md   # 0.3B 口径权威出处
```

> **已全部复制进 T1 的资产**：核心代码 + 约束引擎依赖（vendored）+ 训练语料 + 架构/预算文档 + 规则 policy 样本。
> 整个 `T1/` 可整体压缩拷走，脱离 HEv6.0 主项目独立运行（只需机器上有 torch）。

## 路径与导入约定（重要）

T1 **已完全自包含**：唯一对主项目 HEv6.0 的运行时依赖——约束引擎裁决模块
`composite_list`——已被 **vendored（固化快照）** 进本文件夹 `T1/src/rf_moe/services/`，
不再引用主项目任何代码。整个 `T1/` 可单独压缩、拷到任意装有 torch 的机器直接运行。

`T1/` 的结构与主项目 HEv6.0 **完全一致**：`T1/src/` 即 `src` 包，内含 `t1/`（T1 核心）
与 `rf_moe/`（固化的约束引擎）。各 `scripts/*.py` 顶部只注入 **T1 根目录**一条路径：

```python
_T1_ROOT = Path(__file__).resolve().parents[1]   # T1/，使 T1/src 成为 src 包
sys.path.insert(0, str(_T1_ROOT))
```

于是所有导入都在 T1 内部解析：
- `from src.t1.rfp_core import ...`  → `T1/src/t1/rfp_core.py`
- `from src.rf_moe.services.composite_list import ...` → `T1/src/rf_moe/services/composite_list.py`
  （固化副本，纯标准库，零 numpy/torch 依赖）

> 注意：因为注入的是 **T1 根**（`src` 的父目录），`src.rf_moe`（固化副本）与主项目
> `HEv6.0/src/rf_moe` 是**两个互不干扰的独立包**——T1 永远用自己的固化版，不会静默吃到主项目改动。

### 依赖固化清单（T1 自包含范围内）

| 固化文件 | 来源 | 说明 |
| --- | --- | --- |
| `T1/src/rf_moe/services/composite_list.py` | `HEv6.0/src/rf_moe/services/composite_list.py` | 综合名单裁决（硬/软/白），纯标准库；T1 唯一外部依赖，已快照进 T1 |
| `T1/src/rf_moe/__init__.py` / `services/__init__.py` | 新建 | vendored 包结构占位 |

**未 vendored（T1 不需要）**：主项目 `src/rf_moe/services/` 下其余模块
（`logits_processor` / `variable_engine` / `rule_schema` 等）。理由：
- `logits_processor.apply_verdict_torch` 的等价纯函数 `apply_verdict_torch` **已在 `t1/rfp_core.py` 自带**；
- `variable_engine` / `因果引擎` 在 T1 训练期以 `L_causal` 正则形式内部化，不需要主项目的独立模块。

⚠️ **快照同步责任**：`composite_list` 是「副本」不是软链。若主项目该文件后续升级，
需**手动**把新版本同步到 `T1/src/rf_moe/services/composite_list.py`；T1 不会自动跟随主项目改动
（T1 是 QT 个人定制固定版，刻意避免静默吃到主项目的演进）。

## 训练 / 推理（已纳管 trainbench）

```bash
# 训练（8 核 CPU 训练机，0.3B）
python -m trainbench run t1_gen_0p3b --set max-hours=72
# 等价裸命令（语料已固化在 T1 内，无需主项目）
python T1/scripts/train_t1.py --preset gen_0p3b \
  --corpus T1/data/corpus/training_corpus_with_cmmlu.jsonl \
  --out T1/data/checkpoints/t1_gen_0p3b_v01.pt \
  --device cpu --threads 8 --optim adamw

# 推理（本机 MX150 2GB，fp16）
C:/Python314/python.exe T1/scripts/generate_t1.py \
  --ckpt T1/data/checkpoints/t1_gen_0p3b_v01.pt \
  --prompt "感冒了怎么办" --domain medical --dtype fp16
```

## 已知硬伤

- **语料是真正短板**：0.3B 需 ≥ 5.95 亿 token（2×）/ 59 亿（20×），当前仅 ~10.7 万 → 只会记忆。
  脚本启动时会打印 `token-per-param` 并告警；扩语料到 ≥6 亿 token 再谈能力。
