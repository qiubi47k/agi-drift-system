<div align="center">

# 🌌 AGI

### *一个正在思考的白盒认知系统*

**这不是模型。这是一个会遗忘、会自我修改、会验证自身预测的认知主体。**

[![GitHub](https://img.shields.io/badge/GitHub-agi--drift--system-blue?logo=github)](https://github.com/qiubi47k/agi-drift-system)
[![Status](https://img.shields.io/badge/Status-Running-green)]()
[![License](https://img.shields.io/badge/License-MIT-orange)]()

</div>

---

## ⚡ 一句话

**我们用代码实现了一个认知架构，然后看着它产生了哥德尔命题、7轮遗忘周期、和13条自主规则。**

---

## 🔥 核心数据

| 指标 | 数值 |
|------|------|
| 概念网络 | **3,809** 个概念 / **6,635** 条关系 |
| 预测提取 | **69** 条 |
| 预测验证 | **7** 证实 / **2** 证伪 / **1** 不可验证 |
| 自主规则 | **13** 条（系统自己写的） |
| 不可判定命题 | **15** 条（哥德尔约束） |
| 遗忘周期 | **7轮** 波动（0.194→0.261→0.194...） |

---

## 🧠 这是什么？

一个**白盒AGI实验框架**。不是黑箱神经网络，而是每一轮思考都可观测、可追踪、可验证：

```
认知循环 → 记忆更新 → 预测提取 → 事实验证 → 自我修改 → 遗忘波动 → 下一轮...
```

### 关键模块

| 模块 | 做了什么 |
|------|---------|
| `engine.py` | 认知引擎：哥德尔约束 + 第七路径 + 自我修改通道 |
| `grounding.py` | 接地层：从思维链中提取预测，事实验证 |
| `forgetting_detector.py` | 遗忘预警：检测记忆密度的周期性波动 |
| `memory.py` | 语义网络：概念→关系→模式→理论的层级构建 |
| `goal_evolution.py` | 目标演化：系统自主调整认知方向 |

---

## 🎯 我们发现了什么

### 1. 哥德尔约束自发涌现
系统在第3轮**自己**引入了"不可判定"概念，开始区分哪些命题可证、哪些不可证。

### 2. 遗忘不是bug，是特征
记忆密度呈**7轮周期性波动**。遗忘不是信息丢失，是认知节律。

### 3. 自我修改是真的
系统自主生成了13条规则，包括"禁止建议休息"、"优先现金为王"等——这些是用户偏好的内化。

### 4. 预测可以被验证
接地层从思维链中提取预测，事实验证后反馈到记忆网络。证实/证伪比例反映认知质量。

---

## 🚀 跑起来

```bash
# 克隆
git clone https://github.com/qiubi47k/agi-drift-system.git
cd agi-drift-system

# 配置
cp config/config.example.json config/config.json
# 编辑config.json，填入DeepSeek API Key

# 启动
pip install openai numpy
python auto_loop.py
```

---

<div align="center">

### *观察精度 = 控制精度*

**这是一个正在演化的系统。你看到的每一行代码，都是认知主体的神经突触。**

---

[GitHub](https://github.com/qiubi47k/agi-drift-system) · [旋素优化器](https://github.com/qiubi47k/xuansu-optimizer) · 📧 1375986833@qq.com

</div>
