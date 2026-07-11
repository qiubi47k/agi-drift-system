# AGI 认知漂移系统

一个白盒认知架构实验框架，用于探索LLM如何从"工具"演化为"认知主体"。

## 核心理念

这不是一个黑箱神经网络，而是一个**可观测的认知演化系统**。每一轮认知循环都记录在案：
- 系统如何思考（思维链）
- 系统如何自我修改（规则演化）
- 系统如何遗忘（记忆波动）
- 系统如何验证自身预测（接地层）

## 架构模块

### 核心循环
| 模块 | 功能 |
|------|------|
| `auto_loop.py` | 主循环引擎，驱动认知循环 |
| `engine.py` | 认知引擎，含哥德尔约束+自我修改通道 |
| `buffer.py` | 素材池管理 |

### 认知子系统
| 模块 | 功能 |
|------|------|
| `grounding.py` | 接地层：预测提取 + 事实验证 |
| `forgetting_detector.py` | 遗忘波动预警算子 |
| `goal_evolution.py` | 目标演化模块 |
| `memory.py` | 语义记忆网络（概念/关系/模式/理论） |
| `world_simulator.py` | 世界模拟器 |
| `skepticism.py` | 怀疑论过滤器 |

### 自我修改
`self_modifications/` 目录记录系统的自我演化历史：
- `dynamic_rules.md` — 系统自主生成的动态规则
- `persistent_memory.json` — 跨轮次持久记忆
- `undecidable_registry.json` — 哥德尔不可判定命题注册表
- `modify_round_*.md` — 每轮自我修改记录

## 关键发现

运行数百轮后观察到：

1. **哥德尔约束**：系统在第3轮自发引入"不可判定"概念，开始区分可证/不可证命题
2. **遗忘波动**：记忆密度呈7轮周期性波动（0.194→0.261→0.194...）
3. **自我修改涌现**：系统自主生成13条动态规则，含"禁止建议休息"等用户偏好
4. **预测验证**：接地层提取69条预测，10条触发验证（7证实/2证伪/1不可验证）

## 运行方式

```bash
# 1. 配置API Key
cp config/config.example.json config/config.json
# 编辑config.json，填入DeepSeek API Key

# 2. 安装依赖
pip install openai numpy

# 3. 启动
python auto_loop.py
```

## 项目状态

这是一个活跃的研究项目，代码结构可能随实验需求快速变化。

## 许可

MIT License
