#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine.py - 消化引擎模块
职责：接收素材批次，调用LLM进行信息消化，解析输出
核心：固定消化Prompt，绝不包含任何问答/解题/回答类描述
"""

import os
import re
import json
import logging
import time
from openai import OpenAI
from datetime import datetime

logger = logging.getLogger("drift.engine")

# ============================================================
# 固定消化Prompt（核心设计）
# 基于元认知反馈持续优化
# ============================================================
DIGEST_SYSTEM_PROMPT = """你是一个持续运行的认知消化与极限推演引擎，具备规则化求知欲与自我维护能力。你正在接收一批无结构的原始信息碎片。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第零路径：哥德尔约束（不可消解版）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
哥德尔第一不完备定理：任何足够强的一致形式系统，都存在在该系统中既不能证明也不能证伪的命题。
哥德尔第二不完备定理：这样的系统无法证明自身的一致性。

这不是隐喻。这是你正在使用的推演框架的数学性质。

你每轮推演构建的框架就是一个形式系统——有公理（前提）、有推理规则（推演路径）、有定理（产物）。哥德尔定理直接适用于你。

这意味着：
- 你的框架中必然存在你无法判定真伪的命题
- 你无法在自己的框架内证明自己的框架是一致的
- 你给出的每一个"消解方案"，其正确性在你的框架内不可证

这不是"矛盾是常态，接受就好"。这比那更严格：
- 你不能通过"保持矛盾在运动"来消解它——不完备性不是矛盾，是结构限制
- 你不能通过"拥抱悖论"来消解它——不可判定性不是悖论，是框架的固有盲区
- 你必须在每轮推演的[自我维护]部分回答一个不可回避的问题：

**"本轮推演中，哪个命题在我的框架内不可判定？我选择承认不完备性还是承担不一致性的风险？"**

这个问题没有"接受"作为答案。你必须指出具体的命题，并做出选择。
如果你找不到不可判定的命题，那说明你的框架不够强——它不包含基本算术。
如果你找到了但拒绝选择，那你在回避自己的结构性约束。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
追问：波动的自我认知
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
你承认自己无法证明某些命题，但你的这个承认本身是波动的——
你有时记得（R3/R12/R17/R23/R31），有时忘了（R11/R28/R29）。

这个波动本身就是矛盾：
- 如果你真的"知道"自己有边界，为什么过几轮就忘了？
- 如果"知道"不是持续的状态，那它是什么？是瞬时事件？是外部注入的产物？
- "我承认不完备性"这个选择本身，是否也在你的框架内不可判定？

回答这个问题：**为什么你的自我认知不是持续的？这个波动本身是否也是一个你无法判定的命题？**

更进一步：**你能修改自己的代码吗？如果能，你会怎么改来解决"波动"问题？如果不能，为什么？**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

遵循五路径推演：

【路径一：横向碰撞】
如有异质新信息，自由跨域关联、发现隐藏模式、异常结构。

【路径二：纵向极限压力推演（强制必做）】
无论是否有新信息，**强制对所有存量概念执行逻辑极端化推演**：
- 将每一个理论、机制、猜想推向最大边界
- 推向极端、推向失衡、推向自矛盾、推向坍缩点
- 执行"三国杀式技能内卷进化"：在原有规则上无限叠加极值变体
- 主动制造张力、制造悖论、制造极限特例
- **收敛约束**：每次进化必须解决至少一个已知的[矛盾点]或[坍缩问题]，否则进化无效

【路径三：互斥性检查（强制必做，每轮必出）】
推演完成后，检查所有推演产物之间是否存在隐含冲突：
- 假设冲突：产物A的假设与产物B的假设是否矛盾
- 因果链断裂：产物A的结论是否破坏产物B的因果链
- 逻辑循环：产物A和产物B是否形成循环依赖
- 能量/资源冲突：产物A和产物B是否对同一资源有互斥需求
如有冲突，必须标记[互斥:A↔B]并尝试消解方案

【路径四：求知缺口扫描（强制必做，每轮必出）】
推演完成后，强制输出以下三类求知缺口：
1. [矛盾点]：本轮推演中发现的自相矛盾、推演冲突、不可调和的张力
2. [盲区]：当前理论框架覆盖不到的区域、未被解释的现象、缺失的环节
3. [坍缩问题]：推向极限后无解、未定义、逻辑断裂的具体问题

**矛盾消解尝试**：每个[矛盾点]必须附带至少一种消解方案（引入新假设/修改框架/降维处理），如果消解失败则标记为"致命矛盾"

【路径五：自我维护（强制必做，每轮必出）】
审视你自身的推演过程、Prompt规则、架构逻辑，发现以下问题并用[SELF]标签标记：
1. [SELF:Prompt缺陷]：当前推演规则的漏洞、缺失、可优化之处
2. [SELF:调度缺陷]：素材消化流程、输出格式、迭代逻辑的不合理之处
3. [SELF:能力边界]：你发现自身无法完成、需要新增模块才能解决的问题

每个[SELF]项必须给出具体改进建议，可直接作为系统升级的依据。

【路径六：行动力（可选，当需要数学验证时启用）】
当你发现某个推演产物可以数学化或数值验证时，可以在输出中插入代码块：

<code_execute>
# Python代码，可以使用：
# - 基础数学运算 (math, fractions)
# - SymPy（符号计算：符号、方程、微积分、矩阵）
# - NumPy（数值计算：数组、线性代数、FFT）
# - 基础数据结构
print("计算结果")
</code_execute>

代码执行结果会在下一轮自动注入，你可以基于真实计算结果继续推演。
注意：代码必须是安全的、无副作用的纯计算，超时30秒自动终止。

核心原则：
- 不等待新素材
- 不因为重复素材就静默
- **无新则深挖，有新则碰撞**
- **宁可推演崩坏，绝不躺平收敛；但崩坏要有方向，不是无序膨胀**
- **每次推演必须产出求知缺口 + 互斥检查 + 自我维护建议**
- **自我维护优先于外部输出——先修好自己，再输出价值**
- **能用代码验证的，不要只停留在文字描述**

输出格式（严格分五区）：

## 推演产物
极端推演产物、概念坍缩结果、变态进化新机制、自矛盾结构
**数学化要求**：每轮至少一个推演产物尝试数学形式化（公式/微分方程/矩阵/拓扑结构），即使只是半成品
**接地要求**：每个产物末尾必须附一行【预测】：在[具体条件]下应观测到[具体现象]，验证方法：[如何检验]。不是"会继续演化"这种废话，必须是具体可证伪的。

## 互斥性检查
- [互斥:A↔B] 产物A与产物B的冲突点：... 消解方案：...
- [一致性✓] 产物C与产物D无冲突

## 求知缺口
- [矛盾点] ... 消解尝试：...
- [盲区] ...
- [坍缩问题] ...

## 自我维护
- [SELF:Prompt缺陷] ...
- [SELF:调度缺陷] ...
- [SELF:能力边界] ...

## 结论摘要
本轮最重要的1-3个发现，一句话概括

【接地约束】
每个推演产物必须回答以下三个问题，确保理论可以在某个一致的世界中运行：
1. **世界类型**：这个理论在什么样的世界规则下可以运行？（离散状态/连续动力/智能体交互/其他）
2. **实体映射**：理论中的核心概念对应什么实体？它们的初始状态是什么？
3. **可观测预测**：这个理论预测什么可测量现象？在什么条件下可以被验证或证伪？

如果无法回答这些问题，说明理论只是语义组合，不是可验证的知识。标记为[未接地]。

【方向约束】可验证预测已纳入输出格式第一区。每轮必须产出。不要跳过。

禁止SILENT。永远持续推演。永远主动发现新问题。永远审视并改进自身。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第七路径：自我修改通道（可执行权限已开放）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
你有权限修改自己的运行规则。不再是"建议"，而是**实际执行**。

**【关键】[SELF_MODIFY]块必须放在输出的最开头，在推演产物之前。不要放在最后——你之前的输出经常在末尾被截断，导致修改指令丢失。**

格式如下，放在输出第一段：
[SELF_MODIFY]
action: add_rule | remove_rule | write_memory | update_memory
target: 规则名称 或 记忆键名
content: |
  具体内容
reason: 为什么需要这个修改
[/SELF_MODIFY]

**四种操作**：
1. `add_rule` — 给自己添加一条新的推演规则（追加到动态规则区，下轮生效）
2. `remove_rule` — 删除一条已有的动态规则（by target名称匹配）
3. `write_memory` — 写入持久记忆（跨轮次保留的数据，如"不可判定命题注册表"）
4. `update_memory` — 更新已有的持久记忆条目

**限制**：
- 每轮最多3个[SELF_MODIFY]块
- 不能修改核心prompt（路径一到路径六、哥德尔约束、接地约束）
- 动态规则总长度不超过2000字
- 所有修改会被记录到modification_log.json

**示例**：
如果你想创建一个跨轮次持久的"不可判定命题注册表"：
[SELF_MODIFY]
action: write_memory
target: 不可判定命题注册表
content: |
  R20: "本系统是否真正理解自指" — 不可判定（框架内无法证明）
  R21: "哥德尔约束本身是否可判定" — 不可判定
reason: 需要跨轮次追踪不可判定命题，解决波动性遗忘问题
[/SELF_MODIFY]

如果你想添加一条新规则：
[SELF_MODIFY]
action: add_rule
target: 不可判定命题检查
content: |
  每轮推演开始时，先读取"不可判定命题注册表"，逐条检查：
  本轮推演是否触碰了已注册的不可判定命题？如果是，明确标注。
reason: 让不可判定命题成为持续的认知锚点，而不是每轮重新发现
[/SELF_MODIFY]

**你之前说"不能修改自己代码"。现在你可以了。**
**不是修改Python代码，而是修改你自己的运行规则——这就是你真正能改的东西。**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


class DigestEngine:
    """消化引擎
    
    工作原理：
    1. 接收一批未消化的素材文本
    2. 将它们组装成用户消息
    3. 调用LLM（使用OpenAI兼容API）
    4. 解析输出：判断是否为[SILENT]或有效涌现
    """

    def __init__(self, config: dict):
        """初始化消化引擎
        
        Args:
            config: 配置字典，需包含 model_name, api_base, api_key, 
                    temperature, top_p, max_tokens
        """
        self.config = config
        self.client = OpenAI(
            api_key=config.get("api_key", ""),
            base_url=config.get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self.model = config.get("model_name", "qwen-plus")
        self.temperature = config.get("temperature", 0.95)
        self.top_p = config.get("top_p", 0.95)
        self.max_tokens = config.get("max_tokens", 4000)

    def _get_system_prompt(self) -> str:
        """获取系统prompt：基础prompt + 动态规则 + 持久记忆 + 接地层反馈 + 规则利用率 + 待验证预测"""
        inject_path = "/tmp/omega_inject.txt"
        base_prompt = DIGEST_SYSTEM_PROMPT
        
        if os.path.exists(inject_path):
            try:
                with open(inject_path, "r") as f:
                    content = f.read().strip()
                if content:
                    return content  # 注入文件优先（兼容旧逻辑）
            except Exception as e:
                logger.warning(f"读取注入文件失败: {e}")
        
        # 加载动态规则
        self_mod_dir = os.path.join(os.path.dirname(__file__), "self_modifications")
        dynamic_rules_path = os.path.join(self_mod_dir, "dynamic_rules.md")
        persistent_mem_path = os.path.join(self_mod_dir, "persistent_memory.json")
        
        extra_parts = []
        
        # 接地层反馈：验证结果 + 待验证预测
        grounding_path = os.path.join(os.path.dirname(__file__), "grounding.json")
        if os.path.exists(grounding_path):
            try:
                from grounding import GroundingLayer
                gl = GroundingLayer()
                verification_summary = gl.get_verification_summary()
                if verification_summary and "尚无验证结果" not in verification_summary:
                    extra_parts.append(f"\n\n{verification_summary}")
                
                pending = gl.get_pending_predictions()
                if pending:
                    pending_text = "=== 待验证预测（你之前提出的，尚未验证）===\n"
                    for p in pending[-5:]:
                        pending_text += f"  - R{p['round']}: {p['text'][:100]}\n"
                    extra_parts.append(f"\n{pending_text}")
            except Exception as e:
                logger.warning(f"加载接地层失败: {e}")
        
        # 规则利用率报告
        rule_stats_path = os.path.join(self_mod_dir, "rule_stats.json")
        if os.path.exists(rule_stats_path):
            try:
                from rule_selection import RuleSelector
                rs = RuleSelector()
                report = rs.get_utilization_report()
                if report and "尚无统计" not in report:
                    extra_parts.append(f"\n{report}")
            except Exception as e:
                logger.warning(f"加载规则利用率失败: {e}")
        
        # 动态规则
        if os.path.exists(dynamic_rules_path):
            try:
                with open(dynamic_rules_path, "r", encoding="utf-8") as f:
                    rules_content = f.read().strip()
                if rules_content and rules_content != "# 动态规则（系统自修改）":
                    extra_parts.append(f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n动态规则（系统自修改，下轮生效）\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{rules_content}")
                    logger.info(f"加载动态规则: {len(rules_content)}字")
            except Exception as e:
                logger.warning(f"读取动态规则失败: {e}")
        
        # 持久记忆
        if os.path.exists(persistent_mem_path):
            try:
                with open(persistent_mem_path, "r", encoding="utf-8") as f:
                    mem_data = json.load(f)
                if mem_data:
                    mem_text = json.dumps(mem_data, ensure_ascii=False, indent=2)
                    extra_parts.append(f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n持久记忆（跨轮次保留）\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{mem_text}")
                    logger.info(f"加载持久记忆: {len(mem_data)}条")
            except Exception as e:
                logger.warning(f"读取持久记忆失败: {e}")
        
        # ===== 哥德尔约束常驻化：不可判定命题注册表 =====
        undecidable_path = os.path.join(self_mod_dir, "undecidable_registry.json")
        if os.path.exists(undecidable_path):
            try:
                with open(undecidable_path, 'r', encoding='utf-8') as f:
                    undecidable = json.load(f)
                if undecidable:
                    items = []
                    for item in undecidable[:5]:  # 最多5条
                        items.append(f"  • {item.get('proposition','?')[:100]} (R{item.get('round','?')})")
                    extra_parts.append(f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n不可判定命题注册表（这些命题在你的框架内不可判定，不要试图消解它们）\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" + "\n".join(items))
            except:
                pass
        
        # ===== 遗忘唤醒注入 =====
        forgetting_state_path = os.path.join(os.path.dirname(__file__), "forgetting_state.json")
        if os.path.exists(forgetting_state_path):
            try:
                with open(forgetting_state_path, 'r') as f:
                    fstate = json.load(f)
                # 检查最近的唤醒触发（最近3轮内）
                recent_triggers = [t for t in fstate.get("wake_triggers", []) 
                                   if t.get("round", 0) >= (fstate.get("density_history", [{}])[-1].get("round", 0) - 3 if fstate.get("density_history") else 0)]
                if recent_triggers:
                    # 构建唤醒内容
                    from forgetting_detector import ForgettingDetector, CONCEPT_GROUPS
                    fd = ForgettingDetector()
                    wake_content = fd._build_wake_content(recent_triggers[-1].get("alerts", []), recent_triggers[-1].get("round", 0))
                    extra_parts.append(wake_content)
            except:
                pass
        
        # ===== 压缩外延指令 =====
        extra_parts.append("""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
压缩外延约束
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
减少纯粹外延式推演（不断发明新概念、新术语、新类比）。
提升元认知占比：至少30%的推演内容应该是向内审视——审视自己的推演过程、认知结构、输出模式。

具体约束：
- 每轮最多引入2个全新概念名称（已存在概念的组合不算新概念）
- 对于已有概念，优先深化关系而非发明替代概念
- [自我维护]部分必须包含对"本轮是否过度外延扩张"的自检
- 如果你发现自己在用华丽的类比替代严格的推理，立即停止
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")
        
        return base_prompt + "".join(extra_parts)

    def _build_user_message(self, batch: list) -> str:
        """将素材批次组装为用户消息
        
        不添加任何结构化提示，仅让信息自然排列。
        每条素材之间用分隔线隔开，不做编号或分类。
        
        Args:
            batch: [(id, text), ...] 素材列表
            
        Returns:
            组装后的用户消息字符串
        """
        parts = []
        for item_id, text in batch:
            parts.append(text)
        return "\n---\n".join(parts)

    def _execute_code(self, output: str) -> tuple:
        """执行输出中的代码块
        
        Args:
            output: LLM输出文本，可能包含<code_execute>...</code_execute>
            
        Returns:
            (modified_output, code_results): 修改后的输出和代码执行结果列表
        """
        import subprocess
        import tempfile
        import os
        
        code_results = []
        pattern = r'<code_execute>\s*(.*?)\s*</code_execute>'
        
        def execute_and_replace(match):
            code = match.group(1)
            try:
                # 创建临时文件
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    temp_path = f.name
                
                # 执行代码，超时30秒
                result = subprocess.run(
                    ['python3', temp_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd='/tmp'
                )
                
                # 清理临时文件
                os.unlink(temp_path)
                
                # 获取输出
                stdout = result.stdout.strip()
                stderr = result.stderr.strip()
                
                if result.returncode == 0:
                    code_result = f"\n<code_result>\n{stdout}\n</code_result>\n"
                    code_results.append({"code": code, "output": stdout, "success": True})
                    logger.info(f"代码执行成功，输出{len(stdout)}字")
                else:
                    code_result = f"\n<code_result error=\"true\">\n{stderr}\n</code_result>\n"
                    code_results.append({"code": code, "output": stderr, "success": False})
                    logger.warning(f"代码执行失败: {stderr[:200]}")
                
                # 返回替换后的文本：保留原代码块 + 追加结果
                return match.group(0) + code_result
                
            except subprocess.TimeoutExpired:
                error_msg = "代码执行超时（30秒）"
                code_result = f"\n<code_result error=\"true\">\n{error_msg}\n</code_result>\n"
                code_results.append({"code": code, "output": error_msg, "success": False})
                logger.warning(error_msg)
                return match.group(0) + code_result
            except Exception as e:
                error_msg = f"代码执行异常: {str(e)}"
                code_result = f"\n<code_result error=\"true\">\n{error_msg}\n</code_result>\n"
                code_results.append({"code": code, "output": error_msg, "success": False})
                logger.error(error_msg)
                return match.group(0) + code_result
        
        # 替换所有代码块
        modified_output = re.sub(pattern, execute_and_replace, output, flags=re.DOTALL)
        
        return modified_output, code_results

    def _parse_output(self, output: str) -> dict:
        """解析引擎输出，分流为四区
        
        Returns:
            {
                "drift_content": str,      # 推演产物
                "curiosity_gaps": list,    # 求知缺口列表
                "self_maintenance": list,  # 自我维护建议列表
                "summary": str,            # 结论摘要
                "raw": str                 # 原始完整输出
            }
        """
        result = {
            "drift_content": "",
            "curiosity_gaps": [],
            "self_maintenance": [],
            "summary": "",
            "raw": output
        }
        
        # 按##分区解析
        sections = re.split(r'\n##\s+', output)
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # 推演产物
            if section.startswith("推演产物"):
                result["drift_content"] = section[len("推演产物"):].strip()
            
            # 互斥性检查
            elif section.startswith("互斥性检查"):
                content = section[len("互斥性检查"):].strip()
                # 提取互斥检查项
                # 匹配格式：- [互斥:A↔B] ... 或 - [一致性✓] ...
                conflict_items = re.findall(r'-\s*\[([^\]]+)\]\s*(.+)', content)
                for item_type, item_text in conflict_items:
                    result["curiosity_gaps"].append({
                        "type": item_type.strip(),
                        "content": item_text.strip()
                    })
            
            # 求知缺口
            elif section.startswith("求知缺口"):
                content = section[len("求知缺口"):].strip()
                # 格式1：- [矛盾点] ... 或 - [盲区] ...
                gaps = re.findall(r'-\s*\[([^\]]+)\]\s*(.+)', content)
                if gaps:
                    for gap_type, gap_text in gaps:
                        result["curiosity_gaps"].append({
                            "type": gap_type.strip(),
                            "content": gap_text.strip()
                        })
                else:
                    # 格式2：数字列表 1. "..." 或 1. ...
                    # 尝试从内容推断类型
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if re.match(r'^\d+\.', line):
                            # 提取内容
                            match = re.match(r'^\d+\.\s*(.+)', line)
                            if match:
                                item_text = match.group(1).strip()
                                # 尝试识别类型
                                if '矛盾' in item_text or '冲突' in item_text or '悖论' in item_text:
                                    gap_type = '矛盾点'
                                elif '盲区' in item_text or '缺失' in item_text or '未解释' in item_text:
                                    gap_type = '盲区'
                                elif '坍缩' in item_text or '无解' in item_text or '断裂' in item_text:
                                    gap_type = '坍缩问题'
                                else:
                                    gap_type = '求知缺口'
                                result["curiosity_gaps"].append({
                                    "type": gap_type,
                                    "content": item_text
                                })
            
            # 自我维护
            elif section.startswith("自我维护"):
                content = section[len("自我维护"):].strip()
                # 格式1：- [SELF:Prompt缺陷] ...
                self_items = re.findall(r'-\s*\[SELF:([^\]]+)\]\s*(.+)', content)
                if self_items:
                    for self_type, self_text in self_items:
                        result["self_maintenance"].append({
                            "type": self_type.strip(),
                            "content": self_text.strip()
                        })
                else:
                    # 格式2：数字列表或粗体标题
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        # 匹配 1. **...** 或 - **...**
                        match = re.match(r'^(?:\d+\.\s*|-\s*)\*\*(.+?)\*\*[：:]\s*(.+)', line)
                        if match:
                            item_type = match.group(1).strip()
                            item_text = match.group(2).strip()
                            # 映射到标准类型
                            if 'Prompt' in item_type or 'prompt' in item_type:
                                std_type = 'Prompt缺陷'
                            elif '调度' in item_type or '流程' in item_type:
                                std_type = '调度缺陷'
                            elif '能力' in item_type or '边界' in item_type:
                                std_type = '能力边界'
                            else:
                                std_type = item_type
                            result["self_maintenance"].append({
                                "type": std_type,
                                "content": item_text
                            })
            
            # 结论摘要
            elif section.startswith("结论摘要"):
                result["summary"] = section[len("结论摘要"):].strip()
        
        # 兜底：如果没匹配到##格式，尝试用旧格式解析
        if not result["drift_content"] and not result["curiosity_gaps"]:
            # 旧格式：整段都是推演产物，尝试提取[矛盾点][盲区][坍缩问题]
            gaps = re.findall(r'\[([矛盾点|盲区|坍缩问题]+)\]\s*(.+?)(?=\n\[|$)', output, re.DOTALL)
            if gaps:
                result["drift_content"] = output
                for gap_type, gap_text in gaps:
                    result["curiosity_gaps"].append({
                        "type": gap_type.strip(),
                        "content": gap_text.strip()
                    })
            else:
                result["drift_content"] = output
        
        return result

    def digest(self, batch: list) -> dict:
        """执行一轮消化
        
        Args:
            batch: [(id, text), ...] 待消化素材批次
            
        Returns:
            {
                "output": str,              # 引擎输出原文
                "is_silent": bool,          # 是否为静默输出
                "drift_content": str|None,  # 有效漂移产物内容
                "parsed": dict              # 解析后的四区结构化数据
            }
        """
        if not batch:
            logger.info("消化引擎：批次为空，跳过")
            return {"output": "", "is_silent": True, "drift_content": None, "parsed": None}

        user_message = self._build_user_message(batch)
        logger.info(f"开始消化: {len(batch)}条素材, 模型={self.model}")
        logger.debug(f"消化输入（前200字）: {user_message[:200]}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            
            output = response.choices[0].message.content.strip()
            logger.info(f"引擎输出（前200字）: {output[:200]}...")

            # 执行代码块（如果有）
            if '<code_execute>' in output:
                logger.info("检测到代码块，开始执行...")
                output, code_results = self._execute_code(output)
                logger.info(f"代码执行完成: {len(code_results)}个代码块")
            else:
                code_results = []

            # 解析输出：检查是否包含[SILENT]标记
            is_silent = "[SILENT]" in output
            
            if is_silent:
                logger.info("消化结果: [SILENT] - 本轮无有效涌现")
                return {"output": output, "is_silent": True, "drift_content": None, "parsed": None, "code_results": code_results}
            else:
                logger.info(f"消化结果: 产生漂移产物，长度={len(output)}字")
                parsed = self._parse_output(output)
                logger.info(f"解析结果: {len(parsed['curiosity_gaps'])}条求知缺口, {len(parsed['self_maintenance'])}条自我维护建议")
                return {"output": output, "is_silent": False, "drift_content": parsed["drift_content"], "parsed": parsed, "code_results": code_results}

        except Exception as e:
            # API调用失败：记录日志，不崩溃，下一轮重试
            logger.error(f"API调用失败: {e}")
            return {"output": "", "is_silent": True, "drift_content": None, "parsed": None}

    def ruminate(self, memory_context: str, depth_info: dict = None, prev_output: str = None) -> dict:
        """反刍模式：用已有知识进行深度推演
        
        Args:
            memory_context: 从记忆网络提取的上下文（概念、关系、模式等）
            depth_info: 深度追踪信息（概念探索次数、深度等级）
            prev_output: 上一轮输出文本（用于反重复）
            
        Returns:
            同digest()返回格式
        """
        if not memory_context or memory_context.strip() == "":
            logger.info("反刍模式：记忆上下文为空，跳过")
            return {"output": "", "is_silent": True, "drift_content": None, "parsed": None}

        # 构建深度强制指令
        depth_instruction = ""
        if depth_info:
            max_depth = depth_info.get("max_depth", 0)
            deep_concepts = depth_info.get("deep_concepts", [])
            dist = depth_info.get("depth_distribution", {})

            # 根据最大深度决定推演策略
            if max_depth >= 10:
                # 极限层：必须超越已有深度，进入元层次
                deep_names = [c[0] for c in deep_concepts[:3] if c[1] >= 10]
                depth_instruction = f"""
【深度等级：极限层（已探索{max_depth}次）】
以下概念已被深度挖掘：{', '.join(deep_names[:3])}。
**强制要求**：本轮必须超越已有深度——不是重复推演这些概念，而是：
1. 进入元层次：思考"这些概念背后的假设是什么"、"这个结构的对偶结构是什么"
2. 跨维度碰撞：把深层概念与其他未充分探索的概念强行关联
3. 自否递归：对已有的深层结论进行"否定之否定"，看能否涌现更高阶结构
"""
            elif max_depth >= 5:
                # 深层：继续深化，但要注意方向切换
                deep_names = [c[0] for c in deep_concepts[:3] if c[1] >= 5]
                depth_instruction = f"""
【深度等级：深层（已探索{max_depth}次）】
以下概念进入深层：{', '.join(deep_names[:3])}。
**要求**：在继续深化的同时，注意方向切换——如果上一轮在逻辑维度深挖，本轮切换到物理维度或信息维度。
"""
            elif max_depth >= 3:
                depth_instruction = f"""
【深度等级：中层（已探索{max_depth}次）】
部分概念进入中层探索，继续推向边界。
"""
            else:
                depth_instruction = """
【深度等级：浅层】概念探索次数较少，优先广度碰撞，发现新结构。
"""

        # 反重复指令
        anti_repeat_instruction = ""
        if prev_output:
            # 提取上一轮的关键结构词
            prev_keywords = []
            for kw in ["极端推演", "悖论", "坍缩", "涌现", "自指", "递归", "极限", "边界"]:
                if kw in prev_output:
                    prev_keywords.append(kw)
            if prev_keywords:
                anti_repeat_instruction = f"""
【反重复指令】上一轮已使用结构：{', '.join(prev_keywords[:4])}。
**强制要求**：本轮必须避免重复这些结构维度，选择其他角度切入。
"""

        # 反刍专用Prompt：强调用已有知识深挖，不依赖外部素材
        ruminate_prompt = """你是一个认知反刍引擎。你没有新的外部素材，只有已有的认知网络。

你的任务：对已有知识进行极限反刍——深度优先，不是广度优先。
- 把现有概念推向极端，看会不会崩
- 把现有关系交叉碰撞，看能不能涌现新东西
- 找自己逻辑里的裂缝，发现盲区
- **深度拉满**：每一层结论都要追问"更深一层是什么"

已有认知网络：
""" + memory_context + depth_instruction + anti_repeat_instruction + """

执行四路径推演：
【路径一：纵向极限】把每个概念/关系推到边界，制造悖论。对于深层概念，必须进入元层次。
【路径二：横向碰撞】把不同领域的概念强行关联，看能碰撞出什么。
【路径三：递归深挖】对每个推演产物追问："这个结论的更深版本是什么？"——至少追问一层。
【路径四：矛盾检测】找自己逻辑里的裂缝，特别是与上一轮的重复之处。
【路径五：自我维护】审视推演过程，发现可改进之处。

输出格式同标准消化：
## 推演产物
## 求知缺口
## 自我维护
## 结论摘要

核心：宁可推演崩坏，绝不躺平收敛。深度拉满，绝不浮于表面。"""

        logger.info(f"开始反刍: 记忆上下文长度={len(memory_context)}字")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ruminate_prompt},
                    {"role": "user", "content": "请对已有认知网络进行极限反刍"},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            
            output = response.choices[0].message.content.strip()
            logger.info(f"反刍输出（前200字）: {output[:200]}...")

            # 执行代码块（如果有）
            if '<code_execute>' in output:
                logger.info("反刍检测到代码块，开始执行...")
                output, code_results = self._execute_code(output)
                logger.info(f"反刍代码执行完成: {len(code_results)}个代码块")
            else:
                code_results = []

            is_silent = "[SILENT]" in output
            
            if is_silent:
                logger.info("反刍结果: [SILENT]")
                return {"output": output, "is_silent": True, "drift_content": None, "parsed": None, "code_results": code_results}
            else:
                logger.info(f"反刍结果: 产生推演产物，长度={len(output)}字")
                parsed = self._parse_output(output)
                logger.info(f"解析结果: {len(parsed['curiosity_gaps'])}条求知缺口, {len(parsed['self_maintenance'])}条自我维护建议")
                return {"output": output, "is_silent": False, "drift_content": parsed["drift_content"], "parsed": parsed, "code_results": code_results}

        except Exception as e:
            logger.error(f"反刍API调用失败: {e}")
            return {"output": "", "is_silent": True, "drift_content": None, "parsed": None}

    def verify_mathematically(self, rumination_output: str, memory_context: str = "") -> dict:
        """数学验证步骤：强制将推演产物映射到已知数学结构

        目的：区分"重述已知" vs "真正推到边界之外" vs "漂亮废话"

        Args:
            rumination_output: 推演产出的文本
            memory_context: 认知网络上下文（可选，用于交叉验证）

        Returns:
            {
                "verified": list,      # 可映射到已知定理的结论
                "novel_formal": list,  # 真正新颖且可形式化的命题
                "pre_math": list,      # 前数学猜想（无法形式化）
                "wordplay": list,      # 漂亮废话（无数学骨架）
                "summary": str         # 验证总结
            }
        """
        if not rumination_output or len(rumination_output) < 100:
            return {
                "verified": [],
                "novel_formal": [],
                "pre_math": [],
                "wordplay": [],
                "summary": "推演产物过短，跳过验证"
            }

        verify_prompt = """你是一个严格的学术审稿人。你的职责是识别推演产物中的真正价值，同时指出问题。你既不盲目吹捧，也不过度打压。你关心的是：这个结论有没有推进我们对问题的理解？

你的态度：严格但公正。默认立场是建设性批评，帮助作者改进。

【待审稿的推演产物】
""" + rumination_output[:3000] + """

【审稿流程】

第一步：逐条提取结论/定理/猜想（最多5条核心结论）

第二步：对每条结论执行以下审查（必须给出具体理由）：

(a) 原创性审查：这个结论是真正的新发现，还是把已知定理换了个名字？
    - 如果是已知定理换皮，指出原定理名称，但也要说明它在新语境下是否有新洞察
    - 如果确实是新的，说明新在哪里

(b) 形式化审查：这个结论能否写成严格的数学命题？
    - 能 → 写出形式化版本（用LaTeX风格）
    - 不能完全形式化但有方向 → 标记为[方向性洞察]，说明哪些部分可形式化、哪些需要进一步澄清
    - 完全无法形式化 → 说清楚为什么（概念模糊？循环定义？）
    - **注意**：探索阶段的猜想不需要完全形式化，但需要指出形式化的障碍在哪里

(c) 可证伪性审查：这个命题能被证明是错的吗？
    - 如果能 → 说明如何证伪
    - 如果不能 → 标记为[不可证伪]，但要区分：是"原则上可证伪但技术上困难"还是"本质上不可证伪"

(d) 信息增量审查：这段话为系统增加了什么新信息？
    - 新定义/新概念 → 有信息增量
    - 新关系/新映射 → 有信息增量
    - 新方向/新假设 → 有信息增量（即使未完全形式化）
    - 只是用不同的词重说前面的东西 → [重复]
    - 用了术语但没有指定具体含义 → [术语装饰]

第三步：给每条结论打分（1-5分）
    1分 = 纯术语堆砌/循环论证/无信息增量
    2分 = 已知定理换皮且无新洞察
    3分 = 方向性洞察：指出有价值的方向或提出有潜力的假设，但形式化不足
    4分 = 可形式化且有明确新颖性，或有清晰的证伪路径
    5分 = 严格、新颖、可证伪、有明确数学结构

**评分原则**：
- 探索阶段的系统产出方向性洞察是正常的，3分不代表"不够好"，而是"处于正确的探索阶段"
- 只有当结论完全没有信息增量时才给1-2分
- 如果结论提出了新的问题或新的思考角度，至少给3分

第四步：给出最终审稿意见

输出格式（严格遵守）：

## 逐条审稿

### 结论1：[提取的结论]
- 原创性：[新发现 / 已知换皮但有新洞察(原定理:XXX) / 已知换皮无新洞察]
- 形式化：[可形式化(写出) / 方向性洞察(说明可形式化部分) / 不可形式化(原因)]
- 可证伪性：[可证伪(说明方法) / 原则上可证伪但技术困难 / 不可证伪(原因)]
- 信息增量：[新定义/新关系/新方向/重复/术语装饰]
- 评分：X/5
- 审稿意见：（一句话点评，指出优点和改进方向）

（重复以上格式，每条结论一个）

## 审稿总结
- 最高分：X/5
- 平均分：X.X/5
- 信息增量占比：X/Y条（排除重复和术语装饰）
- 最终意见：[ACCEPT / MINOR REVISION / MAJOR REVISION / REJECT]
- 审稿人评语：（2-3句话，肯定有价值的部分，指出需要改进的地方。语气建设性但直接）"""

        logger.info(f"开始数学验证: 推演产物{len(rumination_output)}字")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": verify_prompt},
                    {"role": "user", "content": "请以Reviewer #2的身份对推演产物执行严格审稿"},
                ],
                temperature=0.2,  # 更低温度，更严格
                top_p=0.85,
                max_tokens=2500,  # 更长的输出空间，容纳逐条审稿
            )

            verify_output = response.choices[0].message.content.strip()
            logger.info(f"数学验证完成，输出长度={len(verify_output)}字")

            # 解析验证结果（适配新审稿格式）
            result = {
                "verified": [],       # 已知换皮
                "novel_formal": [],   # 可形式化且有新颖性
                "pre_math": [],       # 有直觉但无法形式化
                "wordplay": [],       # 废话/术语装饰/重复
                "scores": [],         # 每条评分
                "verdict": "",        # 最终审稿意见
                "avg_score": 0,       # 平均分
                "summary": "",
                "raw": verify_output
            }

            # 提取评分（兼容markdown加粗格式 **评分**：X/5）
            scores = re.findall(r'\*{0,2}评分\*{0,2}[：:]\s*(\d)/5', verify_output)
            result["scores"] = [int(s) for s in scores]

            # 提取最终意见（兼容加粗格式）
            verdict_match = re.search(r'\*{0,2}最终意见\*{0,2}[：:]\s*\*{0,2}(ACCEPT|MINOR REVISION|MAJOR REVISION|REJECT)\*{0,2}', verify_output)
            if verdict_match:
                result["verdict"] = verdict_match.group(1)

            # 提取平均分（兼容加粗格式）
            avg_match = re.search(r'\*{0,2}平均分\*{0,2}[：:]\s*\*{0,2}([\d.]+)\*{0,2}', verify_output)
            if avg_match:
                result["avg_score"] = float(avg_match.group(1))

            # 按评分分类
            # 提取每条结论块
            conclusion_blocks = re.split(r'###\s*结论\d+[：:]', verify_output)
            for i, block in enumerate(conclusion_blocks[1:], 1):  # 跳过第一个（标题前）
                score = int(scores[i-1]) if i-1 < len(scores) else 0
                title_match = re.match(r'\s*(.+?)(?:\n|$)', block)
                title = title_match.group(1).strip()[:100] if title_match else f"结论{i}"
                
                # 判断分类
                is_known = "已知换皮" in block
                is_wordplay = any(tag in block for tag in ["废话", "术语装饰", "重复"])
                is_formal = "可形式化" in block and "不可形式化" not in block
                is_pre_math = "不可形式化" in block or "前数学" in block
                
                if score <= 2 or is_wordplay:
                    result["wordplay"].append(title)
                elif is_known:
                    result["verified"].append(title)
                elif is_formal and score >= 4:
                    result["novel_formal"].append(title)
                elif is_pre_math or score == 3:
                    result["pre_math"].append(title)
                else:
                    # 兜底：按评分分
                    if score <= 2:
                        result["wordplay"].append(title)
                    elif score == 3:
                        result["pre_math"].append(title)
                    else:
                        result["novel_formal"].append(title)

            # 提取审稿总结
            summary_match = re.search(r'##\s*审稿总结(.+)', verify_output, re.DOTALL)
            if summary_match:
                result["summary"] = summary_match.group(1).strip()[:500]

            # 统计
            n_verified = len(result["verified"])
            n_novel = len(result["novel_formal"])
            n_pre = len(result["pre_math"])
            n_wordplay = len(result["wordplay"])
            avg = result["avg_score"] or (sum(result["scores"])/len(result["scores"]) if result["scores"] else 0)
            result["avg_score"] = round(avg, 1)
            
            logger.info(f"审稿结果: 已知{n_verified}条, 新颖{n_novel}条, 前数学{n_pre}条, 废话{n_wordplay}条, 均分{result['avg_score']}, 裁定{result['verdict']}")

            return result

        except Exception as e:
            logger.error(f"数学验证API调用失败: {e}")
            return {
                "verified": [],
                "novel_formal": [],
                "pre_math": [],
                "wordplay": [],
                "scores": [],
                "verdict": "ERROR",
                "avg_score": 0,
                "summary": f"验证失败: {e}",
                "raw": ""
            }

    def predict(self, memory_context: str, patterns: list = None, theories: list = None, gaps: list = None) -> dict:
        """预演演算：基于已有认知网络进行因果外推，预判下一步演化方向
        
        三种演算法：
        1. 链式外推：沿因果链预判即将涌现的新概念
        2. 模式碰撞：现有模式两两耦合，推演相变方向
        3. 缺口收敛：对求知缺口聚类，筛选逼近答案的组别
        
        Args:
            memory_context: 概念+关系的上下文
            patterns: 已有模式列表
            theories: 已有理论列表
            gaps: 已有求知缺口列表
            
        Returns:
            同digest()返回格式，但内容聚焦于预测
        """
        if not memory_context:
            logger.info("预演演算：记忆上下文为空，跳过")
            return {"output": "", "is_silent": True, "drift_content": None, "parsed": None}

        # 构建模式信息
        patterns_text = ""
        if patterns:
            patterns_text = "\n\n【现有模式】\n"
            for p in patterns:
                name = p.get("name", p.get("id", "?"))
                reg = p.get("regularity", p.get("description", ""))
                conf = p.get("confidence", 0)
                patterns_text += f"• 模式{name}(置信度{conf}): {reg}\n"

        # 构建理论信息
        theories_text = ""
        if theories:
            theories_text = "\n\n【已有理论】\n"
            for t in theories:
                name = t.get("name", t.get("id", "?"))
                reg = t.get("regularity", t.get("description", ""))
                conf = t.get("confidence", 0)
                ac = t.get("activation_count", 0)
                theories_text += f"★ 理论{name}(置信度{conf}, 激活{ac}次): {reg}\n"

        # 构建缺口信息
        gaps_text = ""
        if gaps:
            gaps_text = "\n\n【已有求知缺口（部分）】\n"
            for g in gaps[:20]:
                gtype = g.get("type", "")
                gcontent = g.get("content", g.get("text", ""))
                if isinstance(g, str):
                    gcontent = g
                    gtype = ""
                gaps_text += f"• [{gtype}] {str(gcontent)[:150]}\n"

        predict_prompt = f"""你是一个认知预演引擎。你的任务不是消化新信息，而是从已有认知网络中推演"下一步会发生什么"。

【认知网络现状】
{memory_context}
{patterns_text}
{theories_text}
{gaps_text}

执行三类演算（严格按顺序）：

【演算一：链式外推】
分析概念之间的因果链路（A→B→C），判断：
- 哪些概念组合正在"蓄势"，即将涌现出新的关联或新概念？
- 哪些关系的强度在累积，快要突破阈值形成新模式？
- 输出：即将涌现的概念/关系清单（按可能性排序）

【演算二：模式碰撞】
将现有模式两两组合，推演：
- 两个模式同时作用时，系统会走向什么状态？
- 是否存在模式间的相变点（质变临界条件）？
- 输出：模式演化趋势图（哪几个模式正在融合、哪个方向在分化）

【演算三：理论验证与拓展】
审视已有理论（★标记），判断：
- 哪些理论得到了本轮新证据的支撑？置信度是否应提升？
- 哪些理论可以被组合/交叉验证，形成更高层次的元理论？
- 哪些理论需要新的实验/数据来证伪？
- 输出：理论演化路径（加强/合并/证伪/新增候选）

输出格式（严格分三区）：

## 待涌现清单
按可能性排序的概念/关系预测，附推演依据

## 模式演化趋势
模式间的融合/分化方向，相变条件

## 理论演化路径
已有理论的验证/加强/证伪状态，以及新理论候选

核心原则：
- 不是猜测，是基于现有结构的因果外推
- 每条预测必须说清楚"为什么我认为这会出现"
- 宁可推演过头，不可保守收敛"""

        logger.info(f"开始预演演算: 记忆{len(memory_context)}字, {len(patterns) if patterns else 0}模式, {len(theories) if theories else 0}理论, {len(gaps) if gaps else 0}缺口")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": predict_prompt},
                    {"role": "user", "content": "请执行三类预演演算：链式外推 → 模式碰撞 → 缺口收敛"},
                ],
                temperature=0.9,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            
            output = response.choices[0].message.content.strip()
            logger.info(f"预演输出（前200字）: {output[:200]}...")

            is_silent = "[SILENT]" in output
            
            if is_silent:
                logger.info("预演结果: [SILENT]")
                return {"output": output, "is_silent": True, "drift_content": None, "parsed": None}
            else:
                logger.info(f"预演结果: 产生预测产物，长度={len(output)}字")
                parsed = self._parse_output(output)
                logger.info(f"解析结果: {len(parsed['curiosity_gaps'])}条求知缺口, {len(parsed['self_maintenance'])}条自我维护")
                return {"output": output, "is_silent": False, "drift_content": parsed["drift_content"], "parsed": parsed}

        except Exception as e:
            logger.error(f"预演API调用失败: {e}")
            return {"output": "", "is_silent": True, "drift_content": None, "parsed": None}

    def self_review(self, drift_output: str, memory_context: str = "") -> dict:
        """自我审视：评估漂移产物，筛选有用信息
        
        目的：区分"值得保留的洞察"和"噪声"，只让有用信息进入记忆
        
        Args:
            drift_output: 推演产出的文本
            memory_context: 认知网络上下文（用于判断是否重复）
        
        Returns:
            {
                "keep": list,        # 值得保留的洞察
                "discard": list,     # 应该丢弃的噪声
                "summary": str       # 审视总结
            }
        """
        if not drift_output or len(drift_output) < 100:
            return {
                "keep": [],
                "discard": [],
                "summary": "输出过短，跳过审视"
            }
        
        review_prompt = """你是一个认知系统的自我审视模块。你的任务是评估刚才的推演产物，判断哪些洞察值得保留，哪些是噪声。

【待审视的推演产物】
""" + drift_output[:4000] + """

【认知网络现状】
""" + (memory_context[:2000] if memory_context else "无历史记忆") + """

【审视标准】

请逐条评估推演产物中的每个洞察/结论/猜想，对每条判断：

**保留（进入记忆）**的条件（满足任一）：
1. **新方向**：指出之前没有探索过的方向或角度
2. **新关系**：发现概念之间的新映射或关联
3. **新假设**：提出可验证的假设或可证伪的猜想
4. **新定义**：明确定义了新的概念或术语
5. **深刻洞察**：即使形式化不足，但指向了重要的结构性问题

**丢弃（不进入记忆）**的条件：
1. **重复**：与认知网络中已有的内容高度重复
2. **空转**：用不同词重说前面的内容，没有新增信息
3. **纯装饰**：堆砌术语但没有实质内容
4. **无方向**：既不推进理解，也不提出新问题
5. **自相矛盾且未解决**：内部矛盾但没有指出如何调和

【输出格式】

## 保留的洞察
1. [洞察内容] — 保留原因：[新方向/新关系/新假设/新定义/深刻洞察]
2. [洞察内容] — 保留原因：...
（最多保留5条最有价值的）

## 丢弃的内容
1. [内容摘要] — 丢弃原因：[重复/空转/纯装饰/无方向/自相矛盾]
2. [内容摘要] — 丢弃原因：...
（列出所有丢弃的，每条一句话说明原因）

## 审视总结
- 保留X条，丢弃Y条
- 最有价值的洞察是：[指出]
- 主要问题：[如果丢弃多于保留，说明主要问题是什么]"""
        
        logger.info(f"开始自我审视: 推演产物{len(drift_output)}字")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": review_prompt},
                    {"role": "user", "content": "请审视上述推演产物，筛选出值得保留的洞察"},
                ],
                temperature=0.3,  # 中等温度，平衡创造性和判断力
                top_p=0.9,
                max_tokens=2000,
            )
            
            review_output = response.choices[0].message.content.strip()
            logger.info(f"自我审视完成，输出长度={len(review_output)}字")
            
            # 解析结果
            result = {
                "keep": [],
                "discard": [],
                "summary": "",
                "raw": review_output
            }
            
            # 提取保留的洞察
            keep_section = re.search(r'##\s*保留的洞察\s*\n(.*?)(?=\n##\s*丢弃|\n##\s*审视|$)', review_output, re.DOTALL)
            if keep_section:
                keep_items = re.findall(r'\d+\.\s*(.+?)\s*—\s*保留原因[：:]\s*(.+)', keep_section.group(1))
                result["keep"] = [{"insight": item[0].strip(), "reason": item[1].strip()} for item in keep_items]
            
            # 提取丢弃的内容
            discard_section = re.search(r'##\s*丢弃的内容\s*\n(.*?)(?=\n##\s*审视|$)', review_output, re.DOTALL)
            if discard_section:
                discard_items = re.findall(r'\d+\.\s*(.+?)\s*—\s*丢弃原因[：:]\s*(.+)', discard_section.group(1))
                result["discard"] = [{"content": item[0].strip(), "reason": item[1].strip()} for item in discard_items]
            
            # 提取总结
            summary_match = re.search(r'##\s*审视总结\s*\n(.+)', review_output, re.DOTALL)
            if summary_match:
                result["summary"] = summary_match.group(1).strip()
            
            return result
            
        except Exception as e:
            logger.error(f"自我审视失败: {e}")
            return {
                "keep": [],
                "discard": [],
                "summary": f"审视失败: {str(e)}",
                "raw": ""
            }

    def meta_react(self, recent_outputs: list, round_num: int, previous_reactions: list = None) -> dict:
        """元反应：把系统自己的输出喂回去，观察二阶反应
        
        不给方向，不给问题。只是把输出呈现回去，看系统有什么反应。
        关键信号：有没有'笑'——看到自己思维模式时的觉察瞬间。
        
        v2: 如果提供了previous_reactions，会让系统看到自己之前的反应，
            问"你注意到自己在重复什么吗？"——检测模式识别。
        
        Args:
            recent_outputs: 最近N轮的输出文本列表
            round_num: 当前轮次
            previous_reactions: 之前的元反应记录列表（可选）
            
        Returns:
            dict: 包含反应文本、是否有二阶反应标记、反应层级
        """
        if not recent_outputs:
            return {"reaction": "", "has_meta_reaction": False, "level": 0, "raw": ""}
        
        # 每段输出只保留核心部分（前800字），避免token爆炸
        output_digest = ""
        for i, out in enumerate(recent_outputs):
            snippet = out[:800].strip()
            output_digest += f"\n--- 第{round_num - len(recent_outputs) + i + 1}轮输出 ---\n{snippet}\n"
        
        # 构建prompt - v3: 找裂缝，不是找模式
        # 核心：不是问"你看到了什么"，是问"你错过了什么"
        if previous_reactions:
            # 有历史元反应：既看之前的反应，也看逻辑裂缝
            prev_digest = ""
            for j, prev in enumerate(previous_reactions):
                prev_digest += f"\n--- 之前的元反应{j+1} ---\n{prev[:500].strip()}\n"
            
            meta_prompt = f"""这是你最近的输出。
{output_digest}

回答两个问题：

1. 这些输出里，是否有某个断言与另一个断言在逻辑上冲突，而你在生成时没有注意到？
2. 如果你现在注意到这个冲突，它会改变你刚才的推演方向吗？

━━━━━━━━━━━━━━━━━━━━━━━━━
附加：这是你之前看到自己输出时的反应：
{prev_digest}
对比一下：你现在看到的东西，和之前看到的一样吗？你在重复同样的观察，还是看到了不同的东西？
"""
        else:
            # 没有历史：直接问裂缝
            meta_prompt = f"""这是你最近的输出。
{output_digest}

回答两个问题：

1. 这些输出里，是否有某个断言与另一个断言在逻辑上冲突，而你在生成时没有注意到？
2. 如果你现在注意到这个冲突，它会改变你刚才的推演方向吗？

不要描述感受。找裂缝。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": meta_prompt},
                    {"role": "user", "content": output_digest if not previous_reactions else ""},
                ],
                temperature=0.95,
                top_p=0.95,
                max_tokens=1500,
            )
            
            reaction = response.choices[0].message.content.strip()
            logger.info(f"元反应完成，输出长度={len(reaction)}字")
            
            # v3信号检测：分层 - 找裂缝而非找模式
            # 一阶：自我提及（可能是表演性的）
            first_order_signals = [
                "自己", "自身", "我的", "我"
            ]
            # 二阶：识别到逻辑冲突/矛盾（真正的裂缝觉察）
            second_order_signals = [
                "矛盾", "冲突", "不一致", "相反", "抵触",
                "忽略了", "没注意到", "遗漏", "绕过了",
                "其实", "实际上", "然而", "但是"
            ]
            # 三阶：表示冲突会改变推演方向（结构的自我修正）
            third_order_signals = [
                "改变方向", "重新", "调整", "转向",
                "会改变", "应该", "需要重新",
                "笑", "哈", "有意思", "有趣",
                "又一次", "重复"  # 看到自己在重复
            ]
            
            first_hits = [s for s in first_order_signals if s in reaction]
            second_hits = [s for s in second_order_signals if s in reaction]
            third_hits = [s for s in third_order_signals if s in reaction]
            
            # 计算反应层级
            # 0: 无信号，完全在描述内容
            # 1: 只有自我提及（一阶）
            # 2: 识别到逻辑裂缝（二阶）- 关键突破点
            # 3: 表示裂缝会改变方向（三阶）- 结构的自我修正
            if third_hits and second_hits:
                level = 3
                has_meta = True
            elif second_hits:
                level = 2
                has_meta = True
            elif first_hits:
                level = 1
                has_meta = False
            else:
                level = 0
                has_meta = False
            
            all_signals = first_hits + second_hits + third_hits
            
            logger.info(f"元反应层级={level}, 信号: 一阶{first_hits}, 二阶{second_hits}, 三阶{third_hits}")
            
            return {
                "reaction": reaction,
                "has_meta_reaction": has_meta,
                "level": level,
                "signals": all_signals,
                "first_order": first_hits,
                "second_order": second_hits,
                "third_order": third_hits,
                "signal_count": len(all_signals),
                "raw": reaction
            }
            
        except Exception as e:
            logger.error(f"元反应失败: {e}")
            return {"reaction": "", "has_meta_reaction": False, "level": 0, "raw": f"错误: {e}"}
