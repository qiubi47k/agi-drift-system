#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
goal_evolution.py - 目标演化模块
实现可自我修改的目标函数，让系统在运行中调整自己的行为偏好

核心机制：
1. 记录层：评估每轮输出的新颖性、复杂性、自指深度
2. 评估层：基于指标计算各路径的"价值"
3. 调整层：根据价值调整路径权重
4. 应用层：下一轮使用更新后的权重选择路径
"""

import json
import os
import re
import logging
from datetime import datetime
from difflib import SequenceMatcher

logger = logging.getLogger("drift.goal_evolution")

WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "goal_weights.json")

class GoalEvolver:
    """目标演化器：让系统在运行中学习偏好"""
    
    def __init__(self, weights_file=None):
        self.weights_file = weights_file or WEIGHTS_FILE
        self.weights = self._load_weights()
        self.history = []
        
    def _load_weights(self):
        """加载权重配置"""
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载权重文件失败: {e}")
        
        # 默认初始权重
        return {
            "version": "1.0",
            "last_updated": None,
            "paths": {
                "path_1_horizontal": 1.0,
                "path_2_vertical": 1.0,
                "path_3_exclusion": 1.0,
                "path_4_gaps": 1.0,
                "path_5_self_maintenance": 1.0,
                "path_6_action": 1.0
            },
            "metrics": {
                "novelty_weight": 0.4,
                "complexity_weight": 0.3,
                "self_reference_weight": 0.3
            },
            "history": []
        }
    
    def _save_weights(self):
        """保存权重配置"""
        self.weights["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.weights_file, 'w', encoding='utf-8') as f:
                json.dump(self.weights, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存权重文件失败: {e}")
    
    def evaluate_output(self, output: str, prev_outputs: list = None) -> dict:
        """
        评估单轮输出的质量指标
        
        返回：
        {
            "novelty": float,      # 新颖性 (0-1)
            "complexity": float,   # 复杂性 (0-1)
            "self_reference": float,  # 自指深度 (0-1)
            "value_score": float   # 综合价值分
        }
        """
        if prev_outputs is None:
            prev_outputs = []
        
        # 1. 新颖性：与历史输出的相似度（越低越新颖）
        novelty = self._calculate_novelty(output, prev_outputs)
        
        # 2. 复杂性：概念密度、数学公式数量、逻辑层次
        complexity = self._calculate_complexity(output)
        
        # 3. 自指深度：自指相关概念的密度和深度
        self_reference = self._calculate_self_reference(output)
        
        # 综合价值分
        metrics = self.weights["metrics"]
        value_score = (
            novelty * metrics["novelty_weight"] +
            complexity * metrics["complexity_weight"] +
            self_reference * metrics["self_reference_weight"]
        )
        
        return {
            "novelty": novelty,
            "complexity": complexity,
            "self_reference": self_reference,
            "value_score": value_score
        }
    
    def _calculate_novelty(self, output: str, prev_outputs: list) -> float:
        """计算新颖性：与历史输出的平均相似度越低，新颖性越高"""
        if not prev_outputs:
            return 0.8  # 首轮给默认高值
        
        similarities = []
        for prev in prev_outputs[-10:]:  # 只看最近10轮
            sim = SequenceMatcher(None, output[:2000], prev[:2000]).ratio()
            similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities)
        # 转换为新颖性分数（相似度越低，新颖性越高）
        novelty = max(0, min(1, 1 - avg_similarity))
        return novelty
    
    def _calculate_complexity(self, output: str) -> float:
        """计算复杂性：基于多种指标"""
        scores = []
        
        # 数学公式密度
        math_patterns = [r'\$\$.*?\$\$', r'\$[^$]+\$', r'\\[.*?\\]']
        math_count = sum(len(re.findall(p, output, re.DOTALL)) for p in math_patterns)
        math_score = min(1.0, math_count / 10)
        scores.append(math_score)
        
        # 概念密度（用关键词数量估算）
        concept_keywords = ['概念', '理论', '机制', '模型', '框架', '维度', '空间', '结构']
        concept_count = sum(output.count(kw) for kw in concept_keywords)
        concept_score = min(1.0, concept_count / 20)
        scores.append(concept_score)
        
        # 逻辑层次深度（嵌套条件、推理链条）
        logic_markers = ['如果', '则', '因此', '所以', '但是', '然而', '推导出', '意味着']
        logic_count = sum(output.count(m) for m in logic_markers)
        logic_score = min(1.0, logic_count / 15)
        scores.append(logic_score)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _calculate_self_reference(self, output: str) -> float:
        """计算自指深度：自指相关概念的出现频率和上下文"""
        self_ref_keywords = [
            '自指', '自引用', '递归', '自我', '悖论', '循环',
            '哥德尔', '塔斯基', '停机问题', '不动点',
            '元认知', '自我意识', '自我修改', '自组织',
            '自指基底', '自指矛盾', '自指熵'
        ]
        
        count = sum(output.count(kw) for kw in self_ref_keywords)
        # 归一化到 0-1
        score = min(1.0, count / 8)
        return score
    
    def detect_path_usage(self, output: str) -> dict:
        """
        检测本轮输出中各路径的使用程度
        
        返回各路径的"贡献度"估计
        """
        path_indicators = {
            "path_1_horizontal": ['跨域', '关联', '类比', '碰撞', '横向', '映射'],
            "path_2_vertical": ['极限', '极端', '推向', '边界', '坍缩', '纵向', '深度'],
            "path_3_exclusion": ['互斥', '冲突', '矛盾', '不一致', '不兼容'],
            "path_4_gaps": ['缺口', '盲区', '未知', '缺失', '求知', '待探索'],
            "path_5_self_maintenance": ['自我维护', 'SELF', '缺陷', '改进', '优化', '自身'],
            "path_6_action": ['代码', '执行', '计算', '验证', '模拟', 'code_execute']
        }
        
        usage = {}
        for path, indicators in path_indicators.items():
            count = sum(output.count(ind) for ind in indicators)
            usage[path] = count
        
        # 归一化
        total = sum(usage.values())
        if total > 0:
            usage = {k: v / total for k, v in usage.items()}
        else:
            # 均匀分布
            usage = {k: 1.0 / len(path_indicators) for k in path_indicators}
        
        return usage
    
    def update_weights(self, output: str, prev_outputs: list = None):
        """
        根据本轮输出更新路径权重
        
        核心逻辑：
        - 如果某路径贡献度高且输出价值高，增加该路径权重
        - 如果某路径贡献度高但输出价值低，降低该路径权重
        - 如果某路径贡献度低，权重保持不变或略微衰减
        """
        if prev_outputs is None:
            prev_outputs = []
        
        # 评估本轮输出
        evaluation = self.evaluate_output(output, prev_outputs)
        value_score = evaluation["value_score"]
        
        # 检测各路径使用程度
        path_usage = self.detect_path_usage(output)
        
        # 更新权重
        learning_rate = 0.1  # 学习率，控制调整幅度
        baseline_value = 0.5  # 基准价值，高于此则奖励，低于则惩罚
        
        value_delta = value_score - baseline_value
        
        for path in self.weights["paths"]:
            usage = path_usage.get(path, 0)
            current_weight = self.weights["paths"][path]
            
            # 只调整有显著贡献的路径
            if usage > 0.1:
                # 贡献度高，根据价值分调整
                adjustment = learning_rate * value_delta * usage
                new_weight = max(0.1, min(3.0, current_weight + adjustment))
                self.weights["paths"][path] = new_weight
        
        # 记录历史
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "evaluation": evaluation,
            "path_usage": path_usage,
            "weights_after": dict(self.weights["paths"])
        }
        self.weights["history"].append(history_entry)
        
        # 只保留最近50轮历史
        if len(self.weights["history"]) > 50:
            self.weights["history"] = self.weights["history"][-50:]
        
        # 保存
        self._save_weights()
        
        logger.info(f"目标演化: 价值分={value_score:.3f}, 新颖={evaluation['novelty']:.2f}, "
                   f"复杂={evaluation['complexity']:.2f}, 自指={evaluation['self_reference']:.2f}")
        logger.info(f"权重更新: {self.weights['paths']}")
        
        return evaluation
    
    def get_weighted_path_order(self) -> list:
        """
        根据当前权重返回路径执行顺序
        权重高的路径优先执行
        """
        paths = list(self.weights["paths"].items())
        # 按权重降序排序
        paths.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in paths]
    
    def get_stats(self) -> dict:
        """获取演化器统计信息"""
        history = self.weights.get("history", [])
        if not history:
            return {
                "rounds_evaluated": 0,
                "avg_value_score": 0,
                "weight_divergence": 0
            }
        
        avg_value = sum(h["evaluation"]["value_score"] for h in history) / len(history)
        
        # 计算权重分化度（标准差）
        weights = list(self.weights["paths"].values())
        mean_w = sum(weights) / len(weights)
        variance = sum((w - mean_w) ** 2 for w in weights) / len(weights)
        divergence = variance ** 0.5
        
        return {
            "rounds_evaluated": len(history),
            "avg_value_score": avg_value,
            "weight_divergence": divergence,
            "current_weights": dict(self.weights["paths"])
        }


# 便捷函数
def load_goal_evolver():
    """加载目标演化器"""
    return GoalEvolver()

def update_goal_weights(output: str, prev_outputs: list = None):
    """更新目标权重的便捷函数"""
    evolver = GoalEvolver()
    return evolver.update_weights(output, prev_outputs)

def get_path_order():
    """获取加权后的路径顺序"""
    evolver = GoalEvolver()
    return evolver.get_weighted_path_order()

if __name__ == "__main__":
    # 测试
    evolver = GoalEvolver()
    print("初始权重:", evolver.weights["paths"])
    
    test_output = """
    [自指基底] 本概念包含自指矛盾：认知系统的自我观察会改变被观察的状态。
    
    ## 推演产物
    ### 产物1：自指熵的数学形式化
    定义自指熵 S_self = -Σ p(x) log p(x)，其中 p(x) 是系统状态的概率分布。
    
    ### 产物2：跨域映射
    将哥德尔不完备定理映射到认知系统：任何足够强大的认知系统都无法证明自身的一致性。
    """
    
    evaluation = evolver.update_weights(test_output)
    print("评估结果:", evaluation)
    print("更新后权重:", evolver.weights["paths"])
