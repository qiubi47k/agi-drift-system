#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rule_selection.py - 规则选择压力模块
职责：对动态规则施加选择压力，淘汰无用规则，保留有效规则

核心机制：
1. 追踪每条规则的触发次数和效用
2. 连续N轮未触发的规则自动降级/删除
3. 检测规则之间的逻辑冲突
4. 限制规则总量，防止规则通胀
"""

import os
import json
import re
import logging
from datetime import datetime

logger = logging.getLogger("drift.rule_selection")

RULE_STATS_FILE = os.path.join(os.path.dirname(__file__), "self_modifications", "rule_stats.json")


class RuleSelector:
    """规则选择压力器"""
    
    def __init__(self):
        self.stats = self._load_stats()
        self.max_rules = 15  # 规则总量上限
        self.decay_threshold = 20  # 连续20轮未触发则淘汰
        self.conflict_threshold = 0.7  # 文本相似度超过0.7视为潜在冲突
    
    def _load_stats(self) -> dict:
        if os.path.exists(RULE_STATS_FILE):
            try:
                with open(RULE_STATS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"rules": {}, "last_cleanup_round": 0}
    
    def _save_stats(self):
        try:
            with open(RULE_STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存rule_stats失败: {e}")
    
    def track_rule_trigger(self, rule_name: str, round_num: int, effective: bool = True):
        """记录规则触发"""
        if rule_name not in self.stats["rules"]:
            self.stats["rules"][rule_name] = {
                "trigger_count": 0,
                "effective_count": 0,
                "last_triggered_round": 0,
                "created_round": round_num,
                "utility_score": 0.0
            }
        
        rule_stat = self.stats["rules"][rule_name]
        rule_stat["trigger_count"] += 1
        rule_stat["last_triggered_round"] = round_num
        if effective:
            rule_stat["effective_count"] += 1
        
        # 计算效用分：触发次数 * 有效率
        if rule_stat["trigger_count"] > 0:
            hit_rate = rule_stat["effective_count"] / rule_stat["trigger_count"]
            rule_stat["utility_score"] = rule_stat["trigger_count"] * hit_rate
        
        self._save_stats()
    
    def track_round_without_trigger(self, all_rule_names: list, round_num: int):
        """记录一轮中未触发的规则"""
        for name in all_rule_names:
            if name in self.stats["rules"]:
                # 如果本轮没有触发，不更新last_triggered_round
                pass
            else:
                # 新规则，初始化
                self.stats["rules"][name] = {
                    "trigger_count": 0,
                    "effective_count": 0,
                    "last_triggered_round": round_num,
                    "created_round": round_num,
                    "utility_score": 0.0
                }
        self._save_stats()
    
    def get_rules_to_eliminate(self, current_round: int) -> list:
        """获取应该被淘汰的规则"""
        to_eliminate = []
        
        for name, stat in self.stats["rules"].items():
            rounds_since_trigger = current_round - stat["last_triggered_round"]
            
            # 连续N轮未触发
            if rounds_since_trigger > self.decay_threshold and stat["trigger_count"] < 3:
                to_eliminate.append({
                    "name": name,
                    "reason": f"连续{rounds_since_trigger}轮未触发",
                    "trigger_count": stat["trigger_count"],
                    "action": "eliminate"
                })
            
            # 效用分太低
            elif stat["utility_score"] < 0.1 and current_round - stat["created_round"] > 10:
                to_eliminate.append({
                    "name": name,
                    "reason": f"效用分过低({stat['utility_score']:.2f})",
                    "trigger_count": stat["trigger_count"],
                    "action": "eliminate"
                })
        
        return to_eliminate
    
    def detect_conflicts(self, rules_text: str) -> list:
        """检测规则之间的潜在冲突"""
        # 将规则文本按段落分割
        rule_blocks = re.split(r'###\s*规则:\s*', rules_text)
        rule_blocks = [b.strip() for b in rule_blocks if b.strip()]
        
        conflicts = []
        
        for i, rule_a in enumerate(rule_blocks):
            for j, rule_b in enumerate(rule_blocks[i+1:], i+1):
                # 简单的文本相似度检测
                words_a = set(re.findall(r'[\u4e00-\u9fff]+', rule_a[:200]))
                words_b = set(re.findall(r'[\u4e00-\u9fff]+', rule_b[:200]))
                
                if not words_a or not words_b:
                    continue
                
                overlap = words_a & words_b
                similarity = len(overlap) / min(len(words_a), len(words_b))
                
                if similarity > self.conflict_threshold:
                    name_a = rule_a.split('\n')[0][:30]
                    name_b = rule_b.split('\n')[0][:30]
                    conflicts.append({
                        "rule_a": name_a,
                        "rule_b": name_b,
                        "similarity": similarity,
                        "shared_words": list(overlap)[:5]
                    })
        
        return conflicts
    
    def get_utilization_report(self) -> str:
        """生成规则利用率报告"""
        if not self.stats["rules"]:
            return "规则选择压力: 尚无统计数据"
        
        lines = ["=== 规则利用率报告 ==="]
        
        # 按效用分排序
        sorted_rules = sorted(self.stats["rules"].items(), 
                            key=lambda x: x[1].get("utility_score", 0), reverse=True)
        
        for name, stat in sorted_rules[:10]:
            score = stat.get("utility_score", 0)
            triggers = stat.get("trigger_count", 0)
            effective = stat.get("effective_count", 0)
            hit_rate = effective / triggers if triggers > 0 else 0
            lines.append(f"  {name[:25]:25s} | 效用:{score:.1f} | 触发:{triggers} | 有效:{effective} | 命中率:{hit_rate:.0%}")
        
        return "\n".join(lines)
    
    def cleanup(self, current_round: int) -> dict:
        """执行清理：淘汰无用规则，报告冲突"""
        eliminated = self.get_rules_to_eliminate(current_round)
        
        # 清理已淘汰规则的统计
        for item in eliminated:
            if item["action"] == "eliminate":
                del self.stats["rules"][item["name"]]
        
        self.stats["last_cleanup_round"] = current_round
        self._save_stats()
        
        return {
            "eliminated": eliminated,
            "remaining_rules": len(self.stats["rules"])
        }


if __name__ == "__main__":
    rs = RuleSelector()
    print(rs.get_utilization_report())
