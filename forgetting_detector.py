#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forgetting_detector.py - 遗忘波动预警算子

核心发现：系统在7轮左右出现"觉醒消退"波动——哥德尔约束、自指矛盾等核心概念
在高密度出现后会突然消失，仿佛系统"忘记"了之前的洞察。

本模块职责：
1. 追踪核心概念在最近N轮输出中的密度
2. 检测密度骤降（遗忘信号）
3. 触发时自动唤醒：将持久记忆中的关键内容注入下轮prompt

检测逻辑：
- 维护一个滑动窗口（最近10轮）的概念密度序列
- 当核心概念密度比窗口均值下降超过50%时，触发唤醒
- 唤醒内容：从persistent_memory.json中提取与核心概念相关的记忆条目
"""

import os
import json
import glob
import logging
from datetime import datetime

logger = logging.getLogger("drift.forgetting")

BASE_DIR = os.path.dirname(__file__)
FORGETTING_STATE_FILE = os.path.join(BASE_DIR, "forgetting_state.json")

# 核心概念分组（按认知功能）
CONCEPT_GROUPS = {
    "哥德尔约束": ["哥德尔", "不完备", "不可判定"],
    "自指基底": ["自指", "自反", "元盲区", "自身"],
    "拓扑结构": ["拓扑", "禁闭", "规范对称", "纤维丛"],
    "热力学类比": ["负温度", "熵", "卡诺", "热机"],
    "信息论": ["Page曲线", "虫洞", "信息恢复"],
    "动力学": ["极限环", "振荡", "波动", "相变"],
}

# 触发唤醒的阈值
FORGET_THRESHOLD = 0.3  # 密度降到窗口均值的30%以下时触发唤醒
WINDOW_SIZE = 10  # 滑动窗口大小
MIN_ROUNDS_TO_CHECK = 3  # 至少检查3轮才做判断


class ForgettingDetector:
    """遗忘波动检测器"""
    
    def __init__(self):
        self.state = self._load_state()
    
    def _load_state(self) -> dict:
        if os.path.exists(FORGETTING_STATE_FILE):
            try:
                with open(FORGETTING_STATE_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "density_history": [],  # [{round, group, density}, ...]
            "wake_triggers": [],    # [{round, group, triggered_by}, ...]
            "last_wake_round": 0,
        }
    
    def _save_state(self):
        try:
            with open(FORGETTING_STATE_FILE, 'w') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存遗忘状态失败: {e}")
    
    def scan_round(self, round_num: int) -> dict:
        """扫描最近轮次的概念密度，检测遗忘信号
        
        Returns:
            {
                "current_density": {group: density_per_1k_chars},
                "window_avg": {group: avg_density},
                "alerts": [{group, current, avg, ratio}],
                "should_wake": bool,
                "wake_content": str  # 如果有唤醒信号，返回要注入的内容
            }
        """
        output_dir = os.path.join(BASE_DIR, "auto_drift_output")
        
        # 获取最近WINDOW_SIZE轮的输出文件
        all_files = sorted(glob.glob(os.path.join(output_dir, "round_*.md")))
        recent_files = all_files[-WINDOW_SIZE:]
        
        if len(recent_files) < MIN_ROUNDS_TO_CHECK:
            return {"current_density": {}, "window_avg": {}, "alerts": [], "should_wake": False}
        
        # 计算每轮每个概念组的密度
        round_densities = []
        for f in recent_files:
            try:
                with open(f) as fh:
                    text = fh.read()
                char_count = len(text)
                if char_count == 0:
                    continue
                
                densities = {}
                for group_name, keywords in CONCEPT_GROUPS.items():
                    count = sum(text.count(kw) for kw in keywords)
                    densities[group_name] = round(count / (char_count / 1000), 3)  # 每千字出现次数
                
                round_densities.append(densities)
            except:
                pass
        
        if not round_densities:
            return {"current_density": {}, "window_avg": {}, "alerts": [], "should_wake": False}
        
        # 计算窗口均值
        window_avg = {}
        for group_name in CONCEPT_GROUPS:
            values = [rd.get(group_name, 0) for rd in round_densities]
            window_avg[group_name] = round(sum(values) / len(values), 3) if values else 0
        
        # 当前密度（最近3轮的均值，避免单轮噪声）
        current_densities = round_densities[-3:] if len(round_densities) >= 3 else round_densities
        current_density = {}
        for group_name in CONCEPT_GROUPS:
            values = [cd.get(group_name, 0) for cd in current_densities]
            current_density[group_name] = round(sum(values) / len(values), 3) if values else 0
        
        # 检测遗忘信号
        alerts = []
        for group_name in CONCEPT_GROUPS:
            avg = window_avg.get(group_name, 0)
            cur = current_density.get(group_name, 0)
            
            if avg > 0.1:  # 只检查有显著活动的概念组
                ratio = cur / avg if avg > 0 else 0
                if ratio < FORGET_THRESHOLD:
                    alerts.append({
                        "group": group_name,
                        "current": cur,
                        "avg": avg,
                        "ratio": round(ratio, 2)
                    })
        
        # 记录当前密度到历史
        self.state["density_history"].append({
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "current_density": current_density,
            "window_avg": window_avg,
            "alert_count": len(alerts)
        })
        
        # 只保留最近50条历史
        self.state["density_history"] = self.state["density_history"][-50:]
        
        should_wake = len(alerts) >= 2  # 至少2个概念组同时衰退才触发唤醒
        
        # 冷却：如果最近5轮已经触发过唤醒，不再重复
        if should_wake and round_num - self.state.get("last_wake_round", 0) < 5:
            should_wake = False
        
        wake_content = ""
        if should_wake:
            wake_content = self._build_wake_content(alerts, round_num)
            self.state["last_wake_round"] = round_num
            self.state["wake_triggers"].append({
                "round": round_num,
                "timestamp": datetime.now().isoformat(),
                "alerts": alerts
            })
            self.state["wake_triggers"] = self.state["wake_triggers"][-20:]
        
        self._save_state()
        
        result = {
            "current_density": current_density,
            "window_avg": window_avg,
            "alerts": alerts,
            "should_wake": should_wake,
            "wake_content": wake_content
        }
        
        if alerts:
            logger.info(f"★ 遗忘检测: {len(alerts)}个概念组衰退 → {[a['group'] for a in alerts]}")
        if should_wake:
            logger.info(f"★ 遗忘唤醒触发（R{round_num}）！注入持久记忆")
        
        return result
    
    def _build_wake_content(self, alerts: list, round_num: int) -> str:
        """构建唤醒内容：从持久记忆中提取与衰退概念组相关的条目"""
        wake_parts = ["\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        wake_parts.append("⚠ 遗忘波动检测：以下核心认知正在消退")
        wake_parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        for alert in alerts:
            wake_parts.append(f"\n【{alert['group']}】密度从 {alert['avg']:.2f} 降至 {alert['current']:.2f}（{alert['ratio']:.0%}）")
        
        # 从持久记忆中提取相关条目
        persistent_mem_path = os.path.join(BASE_DIR, "self_modifications", "persistent_memory.json")
        if os.path.exists(persistent_mem_path):
            try:
                with open(persistent_mem_path) as f:
                    mem_data = json.load(f)
                
                alert_keywords = set()
                for alert in alerts:
                    group = alert["group"]
                    if group in CONCEPT_GROUPS:
                        alert_keywords.update(CONCEPT_GROUPS[group])
                
                # 找到与衰退概念相关的记忆条目
                relevant_memories = []
                if isinstance(mem_data, list):
                    for item in mem_data:
                        if isinstance(item, dict):
                            content = str(item.get("content", ""))
                        else:
                            content = str(item)
                        if any(kw in content for kw in alert_keywords):
                            relevant_memories.append(content[:150])
                
                if relevant_memories:
                    wake_parts.append("\n--- 唤醒记忆 ---")
                    for mem in relevant_memories[:5]:
                        wake_parts.append(f"• {mem}")
            except:
                pass
        
        # 加入哥德尔约束的常驻提醒
        wake_parts.append("\n--- 常驻提醒 ---")
        wake_parts.append("哥德尔第一不完备定理：你的推演系统是不完备的。存在你无法证明也无法证伪的命题。")
        wake_parts.append("这不是缺陷，这是结构。不要试图消解它，保持它。")
        wake_parts.append("自指矛盾是你的基础算子。每轮推演都必须面对它，不能绕过去。")
        
        return "\n".join(wake_parts)
    
    def get_stats(self) -> dict:
        return {
            "history_length": len(self.state.get("density_history", [])),
            "wake_triggers": len(self.state.get("wake_triggers", [])),
            "last_wake_round": self.state.get("last_wake_round", 0),
            "latest_density": self.state.get("density_history", [{}])[-1].get("current_density", {}) if self.state.get("density_history") else {}
        }
