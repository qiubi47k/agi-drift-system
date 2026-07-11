#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memory.py - 认知记忆网络
设计哲学：记忆像模型权重——有复杂度(层级)、逻辑度(关系)、规律度(统计权重)

存储结构：
  L0 concepts  - 原子概念节点（带权重和激活历史）
  L1 relations - 概念间关系边（推导/矛盾/类比/蕴含，带证据强度）
  L2 patterns  - 从高频共现中涌现的模式/规律
  L3 theories  - 从模式中凝结的理论框架

检索方式：从查询概念出发，沿关系网络BFS遍历，按权重+距离排序
"""

import os
import re
import json
import logging
import hashlib
from collections import Counter, defaultdict
from datetime import datetime

# 导入世界模拟器
try:
    from world_simulator import WorldManager, WorldKernel, create_discrete_world
    WORLD_SIMULATOR_AVAILABLE = True
except ImportError:
    WORLD_SIMULATOR_AVAILABLE = False

# 导入质疑模块
try:
    from skepticism import get_tracker, SkepticismTracker
    SKEPTICISM_AVAILABLE = True
except ImportError:
    SKEPTICISM_AVAILABLE = False

logger = logging.getLogger("drift.memory")

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

# 世界目录
WORLDS_DIR = os.path.join(MEMORY_DIR, "worlds")
os.makedirs(WORLDS_DIR, exist_ok=True)

# 关系类型
RELATION_TYPES = ["derives", "contradicts", "analogizes", "implies", "co_occurs", "composes"]
RELATION_CN = {
    "derives": "推导",
    "contradicts": "矛盾",
    "analogizes": "类比",
    "implies": "蕴含",
    "co_occurs": "共现",
    "composes": "组成"
}

# 权重衰减参数
DECAY_FACTOR = 0.95       # 每轮未激活的权重衰减
MIN_WEIGHT = 0.05         # 最低权重（不删除，只降权）
PATTERN_THRESHOLD = 3     # 共现次数达到此值→提升为pattern
THEORY_THRESHOLD = 2      # pattern被验证次数达到此值→提升为theory


class CognitiveMemory:
    """认知记忆网络"""

    def __init__(self):
        self.concepts = {}     # name -> concept_dict
        self.relations = []    # list of relation_dicts
        self.patterns = []     # list of pattern_dicts
        self.theories = []     # list of theory_dicts
        self.activation_log = []
        self._next_id = {"c": 1, "r": 1, "p": 1, "t": 1}
        self.world_manager = None
        if WORLD_SIMULATOR_AVAILABLE:
            grounding_file = os.path.join(WORLDS_DIR, "grounding.json")
            self.world_manager = WorldManager(grounding_file)
        self._load()
        
        # 初始化质疑追踪器
        self.skepticism_tracker = None
        if SKEPTICISM_AVAILABLE:
            self.skepticism_tracker = get_tracker()
            # 为已有理论补建追踪数据
            self.skepticism_tracker.batch_init_from_theories(self.theories)

    # ============================================================
    # 持久化
    # ============================================================
    def _load(self):
        """从磁盘加载所有记忆数据"""
        self.concepts = self._load_json("concepts.json", {})
        self.relations = self._load_json("relations.json", [])
        self.patterns = self._load_json("patterns.json", [])
        self.theories = self._load_json("theories.json", [])
        self.activation_log = self._load_json("activation_log.json", [])

        # 恢复ID计数器
        for c in self.concepts.values():
            try:
                num = int(c["id"][1:])
                self._next_id["c"] = max(self._next_id["c"], num + 1)
            except (ValueError, KeyError):
                pass
        for r in self.relations:
            try:
                num = int(r["id"][1:])
                self._next_id["r"] = max(self._next_id["r"], num + 1)
            except (ValueError, KeyError):
                pass
        for p in self.patterns:
            try:
                num = int(p["id"][1:])
                self._next_id["p"] = max(self._next_id["p"], num + 1)
            except (ValueError, KeyError):
                pass
        for t in self.theories:
            try:
                num = int(t["id"][1:])
                self._next_id["t"] = max(self._next_id["t"], num + 1)
            except (ValueError, KeyError):
                pass

        logger.info(f"记忆加载完成: {len(self.concepts)}概念, {len(self.relations)}关系, "
                     f"{len(self.patterns)}模式, {len(self.theories)}理论")

    def _load_json(self, filename, default):
        filepath = os.path.join(MEMORY_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return default

    def _save(self):
        """持久化所有记忆数据"""
        self._save_json("concepts.json", self.concepts)
        self._save_json("relations.json", self.relations)
        self._save_json("patterns.json", self.patterns)
        self._save_json("theories.json", self.theories)
        self._save_json("activation_log.json", self.activation_log)

    def _save_json(self, filename, data):
        filepath = os.path.join(MEMORY_DIR, filename)
        tmp = filepath + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, filepath)

    # ============================================================
    # ID生成
    # ============================================================
    def _new_concept_id(self):
        cid = f"c{self._next_id['c']:04d}"
        self._next_id["c"] += 1
        return cid

    def _new_relation_id(self):
        rid = f"r{self._next_id['r']:04d}"
        self._next_id["r"] += 1
        return rid

    def _new_pattern_id(self):
        pid = f"p{self._next_id['p']:04d}"
        self._next_id["p"] += 1
        return pid

    def _new_theory_id(self):
        tid = f"t{self._next_id['t']:04d}"
        self._next_id["t"] += 1
        return tid

    # ============================================================
    # 存储：从推演产物中提取并存储结构化知识
    # ============================================================
    def store_extraction(self, extracted: dict, round_num: int):
        """存储从推演产物中提取的结构化知识

        Args:
            extracted: {
                "concepts": [{"name": str, "description": str}],
                "relations": [{"from": str, "to": str, "type": str, "description": str}],
            }
            round_num: 当前轮次
        """
        activated_concept_ids = []
        activated_relation_ids = []

        # 1. 处理概念
        for concept_data in extracted.get("concepts", []):
            name = concept_data.get("name", "").strip()
            if not name:
                continue

            # 查找已有概念（模糊匹配）
            existing = self._find_concept(name)
            if existing:
                # 激活已有概念，增强权重
                existing["weight"] = min(1.0, existing["weight"] + 0.1)
                existing["activation_count"] += 1
                if round_num not in existing.get("activation_rounds", []):
                    existing["activation_rounds"].append(round_num)
                existing["last_activated"] = round_num
                # 补充描述（如果有新的）
                desc = concept_data.get("description", "").strip()
                if desc and len(desc) > len(existing.get("description", "")):
                    existing["description"] = desc
                activated_concept_ids.append(existing["id"])
                logger.debug(f"概念激活: {name} (weight={existing['weight']:.2f})")
            else:
                # 创建新概念
                cid = self._new_concept_id()
                self.concepts[name] = {
                    "id": cid,
                    "name": name,
                    "aliases": concept_data.get("aliases", []),
                    "description": concept_data.get("description", ""),
                    "layer": 0,
                    "weight": 0.5,  # 新概念初始权重
                    "activation_count": 1,
                    "first_seen": round_num,
                    "last_activated": round_num,
                    "activation_rounds": [round_num]
                }
                activated_concept_ids.append(cid)
                logger.debug(f"概念创建: {name} (id={cid})")

        # 2. 处理关系
        for rel_data in extracted.get("relations", []):
            from_name = rel_data.get("from", "").strip()
            to_name = rel_data.get("to", "").strip()
            rel_type = rel_data.get("type", "co_occurs").strip()

            if not from_name or not to_name:
                continue

            # 确保两端概念存在
            from_concept = self._find_or_create_concept(from_name, round_num)
            to_concept = self._find_or_create_concept(to_name, round_num)

            if from_concept and to_concept:
                # 查找是否已有同类型关系
                existing_rel = self._find_relation(from_concept["id"], to_concept["id"], rel_type)
                if existing_rel:
                    # 增强已有关系
                    existing_rel["weight"] = min(1.0, existing_rel["weight"] + 0.1)
                    existing_rel["evidence_count"] += 1
                    if round_num not in existing_rel.get("evidence_rounds", []):
                        existing_rel["evidence_rounds"].append(round_num)
                    desc = rel_data.get("description", "").strip()
                    if desc and len(desc) > len(existing_rel.get("description", "")):
                        existing_rel["description"] = desc
                    activated_relation_ids.append(existing_rel["id"])
                else:
                    # 创建新关系
                    rid = self._new_relation_id()
                    self.relations.append({
                        "id": rid,
                        "from_concept": from_concept["id"],
                        "to_concept": to_concept["id"],
                        "type": rel_type,
                        "weight": 0.5,
                        "evidence_count": 1,
                        "evidence_rounds": [round_num],
                        "description": rel_data.get("description", "")
                    })
                    activated_relation_ids.append(rid)

        # 3. 激活模式（pattern）并检查理论提升
        activated_pattern_ids = []
        # 构建模式概念的邻居集合（与模式概念有直接关系的concept IDs）
        pattern_neighbor_map = {}  # pattern_id -> set of neighbor concept IDs
        for pattern in self.patterns:
            pcids = set(pattern.get("concepts", []))
            neighbors = set()
            for rel in self.relations:
                if rel["from_concept"] in pcids:
                    neighbors.add(rel["to_concept"])
                if rel["to_concept"] in pcids:
                    neighbors.add(rel["from_concept"])
            pattern_neighbor_map[pattern["id"]] = neighbors - pcids  # 排除自身

        for pattern in self.patterns:
            pattern_concepts = set(pattern.get("concepts", []))
            activated_set = set(activated_concept_ids)
            neighbors = pattern_neighbor_map.get(pattern["id"], set())
            
            # 直接命中：模式的核心概念被激活
            direct_hit = pattern_concepts & activated_set
            # 邻居命中：模式概念的相关概念被激活
            neighbor_hit = neighbors & activated_set
            
            if direct_hit:
                pattern["activation_count"] = pattern.get("activation_count", 0) + 1.0
                if round_num not in pattern.get("activation_rounds", []):
                    pattern.setdefault("activation_rounds", []).append(round_num)
                pattern["last_activated"] = round_num
                activated_pattern_ids.append(pattern["id"])
                logger.debug(f"模式激活(直接): {pattern['id']} (count={pattern.get('activation_count',0):.1f})")
            elif neighbor_hit:
                pattern["activation_count"] = pattern.get("activation_count", 0) + 0.5
                if round_num not in pattern.get("activation_rounds", []):
                    pattern.setdefault("activation_rounds", []).append(round_num)
                pattern["last_activated"] = round_num
                activated_pattern_ids.append(pattern["id"])
                logger.debug(f"模式激活(邻居): {pattern['id']} (count={pattern.get('activation_count',0):.1f})")
            
            # 检查是否达到理论提升阈值
            if pattern.get("activation_count", 0) >= THEORY_THRESHOLD and "promoted_to_theory" not in pattern:
                # 提升为理论
                tid = self._new_theory_id()
                c_names = []
                for cid in pattern.get("concepts", []):
                    name = next((n for n, c in self.concepts.items() if c["id"] == cid), cid)
                    c_names.append(name)
                ac = pattern.get("activation_count", 0)
                theory = {
                    "id": tid,
                    "name": f"{'∩'.join(c_names)}的共性规律",
                    "source_pattern": pattern["id"],
                    "concepts": pattern.get("concepts", []),
                    "regularity": pattern.get("regularity", ""),
                    "evidence_count": ac,
                    "confidence": min(1.0, ac / 5.0),
                    "activation_count": ac,  # 继承pattern的激活计数
                    "last_activated": round_num,
                    "activation_rounds": [round_num],
                    "created_round": round_num,
                    "description": f"从模式{pattern['id']}中归纳：{pattern.get('regularity', '')}，经{ac:.1f}次验证"
                }
                # 初始化质疑追踪
                if self.skepticism_tracker:
                    self.skepticism_tracker.init_theory(tid)
                    theory["skepticism"] = {
                        "score": 0.0,
                        "confirmed": 0,
                        "refuted": 0,
                        "unique_sources": 0,
                    }
                self.theories.append(theory)
                pattern["promoted_to_theory"] = tid
                logger.info(f"★ 理论涌现: {tid} ← 模式{pattern['id']} (activation_count={ac:.1f})")
                
                # 验证新理论是否可接地到模拟世界
                if self.world_manager:
                    grounding_result = self.world_manager.validate_theory(theory, self.concepts)
                    theory["grounding"] = {
                        "groundable": grounding_result["groundable"],
                        "world_ids": grounding_result["world_ids"],
                        "unmapped_concepts": grounding_result["unmapped_concepts"],
                        "validated": False
                    }
                    if grounding_result["groundable"]:
                        theory["confidence"] = min(1.0, theory["confidence"] + grounding_result["confidence_boost"])
                        logger.info(f"  ✓ 理论{tid}可接地到世界: {grounding_result['world_ids']}")
                    else:
                        logger.warning(f"  ✗ 理论{tid}无法接地，未映射概念: {grounding_result['unmapped_concepts']}")

        # 3.5 激活理论（theory）
        activated_theory_ids = []
        for theory in self.theories:
            theory_concepts = set(theory.get("concepts", []))
            activated_set = set(activated_concept_ids)
            
            # 直接命中：理论的核心概念被激活
            direct_hit = theory_concepts & activated_set
            if direct_hit:
                theory["activation_count"] = theory.get("activation_count", 0) + 1.0
                if round_num not in theory.get("activation_rounds", []):
                    theory.setdefault("activation_rounds", []).append(round_num)
                theory["last_activated"] = round_num
                theory["evidence_count"] = theory.get("evidence_count", 0) + 0.5
                # 原有置信度（基于证据数）
                base_confidence = min(1.0, theory["evidence_count"] / 10.0)
                
                # 融合质疑分数
                if self.skepticism_tracker:
                    sk_score = self.skepticism_tracker.get_skepticism_score(theory["id"])
                    # 如果有质疑数据，用 skepticism 公式计算；否则保持原逻辑
                    stats = self.skepticism_tracker.get_theory_stats(theory["id"])
                    if stats.get("tracked") and (stats["confirmed_count"] + stats["refuted_count"]) > 0:
                        # skepticism 加权：50%原有 + 50%质疑分数
                        theory["confidence"] = round(
                            0.5 * base_confidence + 0.5 * sk_score, 4
                        )
                        # 更新理论上的 skepticism 快照
                        theory["skepticism"] = {
                            "score": sk_score,
                            "confirmed": stats["confirmed_count"],
                            "refuted": stats["refuted_count"],
                            "unique_sources": stats["unique_sources"],
                        }
                    else:
                        theory["confidence"] = base_confidence
                else:
                    theory["confidence"] = base_confidence
                
                # 确保不超过上限
                theory["confidence"] = min(1.0, theory["confidence"])
                
                # 如果理论可接地，尝试运行模拟验证
                if self.world_manager and theory.get("grounding", {}).get("groundable"):
                    sim_result = self._run_theory_simulation(theory)
                    if sim_result.get("success") and sim_result.get("consistency"):
                        theory["grounding"]["validated"] = True
                        theory["grounding"]["simulation_steps"] = sim_result.get("steps", 0)
                        # 可稳定运行的理论获得额外置信度加成
                        theory["confidence"] = min(1.0, theory["confidence"] + 0.05)
                        logger.debug(f"理论{theory['id']}模拟验证通过")
                    elif sim_result.get("success"):
                        # 模拟运行但一致性检查失败
                        theory["grounding"]["validated"] = False
                        theory["grounding"]["consistency_failed"] = True
                        logger.warning(f"理论{theory['id']}模拟运行不一致")
                
                activated_theory_ids.append(theory["id"])
                logger.debug(f"理论激活: {theory['id']} (count={theory.get('activation_count',0):.1f}, conf={theory['confidence']:.2f})")

        # 3.6 模拟验证：检查本轮激活的理论间是否存在冲突
        if self.skepticism_tracker and len(activated_theory_ids) >= 2:
            active_theories = [
                t for t in self.theories if t["id"] in activated_theory_ids
            ]
            for i in range(len(active_theories)):
                for j in range(i + 1, len(active_theories)):
                    ta = active_theories[i]
                    tb = active_theories[j]
                    shared = set(ta.get("concepts", [])) & set(tb.get("concepts", []))
                    if shared:
                        # 有共享概念——记录内部验证（可信度较低，因为是内部自验证）
                        self.skepticism_tracker.record_internal_validation(
                            theory_id=ta["id"],
                            result="confirmed",
                            round_num=round_num,
                            note=f"与{tb['id']}共享{len(shared)}个概念",
                        )

        # 4. 记录激活日志
        if activated_concept_ids or activated_relation_ids or activated_theory_ids:
            self.activation_log.append({
                "round": round_num,
                "concept_ids": activated_concept_ids,
                "relation_ids": activated_relation_ids,
                "pattern_ids": activated_pattern_ids,
                "theory_ids": activated_theory_ids,
                "timestamp": datetime.now().isoformat()
            })

        # 5. 保存质疑追踪数据
        if self.skepticism_tracker:
            self.skepticism_tracker.save()

        # 6. 保存
        self._save()
        logger.info(f"第{round_num}轮记忆更新: +{len(activated_concept_ids)}概念激活, "
                     f"+{len(activated_relation_ids)}关系激活, "
                     f"+{len(activated_pattern_ids)}模式激活, "
                     f"+{len(activated_theory_ids)}理论激活")

    def _find_concept(self, name: str) -> dict:
        """查找概念（精确匹配 + 别名匹配）"""
        name_lower = name.lower().strip()
        # 精确匹配
        if name in self.concepts:
            return self.concepts[name]
        # 别名匹配
        for cname, cdata in self.concepts.items():
            if name_lower == cname.lower():
                return cdata
            for alias in cdata.get("aliases", []):
                if name_lower == alias.lower():
                    return cdata
            # 包含关系（短词在长词中）
            if name_lower in cname.lower() or cname.lower() in name_lower:
                return cdata
        return None

    def _find_or_create_concept(self, name: str, round_num: int) -> dict:
        """查找或创建概念"""
        existing = self._find_concept(name)
        if existing:
            existing["weight"] = min(1.0, existing["weight"] + 0.05)
            existing["activation_count"] += 1
            if round_num not in existing.get("activation_rounds", []):
                existing["activation_rounds"].append(round_num)
            existing["last_activated"] = round_num
            return existing
        # 创建
        cid = self._new_concept_id()
        concept = {
            "id": cid,
            "name": name,
            "aliases": [],
            "description": "",
            "layer": 0,
            "weight": 0.3,
            "activation_count": 1,
            "first_seen": round_num,
            "last_activated": round_num,
            "activation_rounds": [round_num]
        }
        self.concepts[name] = concept
        return concept

    def _find_relation(self, from_id: str, to_id: str, rel_type: str) -> dict:
        """查找已有关系"""
        for rel in self.relations:
            if (rel["from_concept"] == from_id and
                rel["to_concept"] == to_id and
                rel["type"] == rel_type):
                return rel
            # 双向匹配（对矛盾、类比、共现）
            if rel_type in ["contradicts", "analogizes", "co_occurs"]:
                if (rel["from_concept"] == to_id and
                    rel["to_concept"] == from_id and
                    rel["type"] == rel_type):
                    return rel
        return None

    def _run_theory_simulation(self, theory: dict, n_steps: int = 10) -> dict:
        """在世界模拟器中运行理论
        
        Args:
            theory: 理论字典
            n_steps: 模拟步数
            
        Returns:
            {
                "success": bool,
                "consistency": bool,
                "steps": int,
                "observations": dict
            }
        """
        if not self.world_manager:
            return {"success": False, "error": "世界模拟器不可用"}
        
        # 获取理论可接地的世界
        grounding = theory.get("grounding", {})
        world_ids = grounding.get("world_ids", [])
        
        if not world_ids:
            return {"success": False, "error": "无可接地世界"}
        
        # 选择第一个可用世界
        world_id = world_ids[0]
        
        # 运行模拟
        result = self.world_manager.run_simulation(theory, world_id, self.concepts, n_steps)
        return result

    # ============================================================
    # 检索：基于认知网络的智能检索
    # ============================================================
    def search(self, query: str, top_k: int = 5) -> list:
        """基于认知网络的检索

        策略：
        1. 从查询中匹配概念节点
        2. 从匹配节点出发，沿关系边BFS遍历
        3. 按 (概念权重 × 关系强度 / 距离衰减) 排序
        4. 返回最相关的认知链路

        Returns:
            [{"type": "concept|relation|pattern", "data": dict, "score": float}, ...]
        """
        if not self.concepts:
            return []

        # Step 1: 从查询中找匹配的概念
        query_tokens = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', query.lower()))
        matched_concepts = []

        for name, concept in self.concepts.items():
            name_tokens = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', name.lower()))
            overlap = len(query_tokens & name_tokens)
            if overlap > 0:
                score = overlap * concept["weight"]
                matched_concepts.append((name, concept, score))

        # 如果没匹配到概念，用描述匹配
        if not matched_concepts:
            for name, concept in self.concepts.items():
                desc = concept.get("description", "").lower()
                overlap = sum(1 for t in query_tokens if t in desc)
                if overlap > 0:
                    matched_concepts.append((name, concept, overlap * concept["weight"] * 0.5))

        if not matched_concepts:
            return []

        # 按匹配分数排序
        matched_concepts.sort(key=lambda x: x[2], reverse=True)

        # Step 2: BFS遍历关系网络
        results = []
        visited_concepts = set()
        queue = []  # (concept_id, distance, score)

        for name, concept, score in matched_concepts[:3]:  # 取top3种子概念
            cid = concept["id"]
            visited_concepts.add(cid)
            results.append({
                "type": "concept",
                "data": concept,
                "score": score
            })
            # 将邻居加入队列
            for rel in self.relations:
                neighbor_id = None
                if rel["from_concept"] == cid:
                    neighbor_id = rel["to_concept"]
                elif rel["to_concept"] == cid:
                    neighbor_id = rel["from_concept"]
                if neighbor_id and neighbor_id not in visited_concepts:
                    rel_score = score * rel["weight"] * 0.7  # 距离衰减
                    queue.append((neighbor_id, 1, rel_score, rel))

        # BFS展开（最多2层）
        for neighbor_id, dist, score, rel in queue[:10]:
            if neighbor_id in visited_concepts:
                continue
            visited_concepts.add(neighbor_id)
            # 找到概念数据
            concept_data = None
            for name, c in self.concepts.items():
                if c["id"] == neighbor_id:
                    concept_data = c
                    break
            if concept_data:
                results.append({
                    "type": "relation",
                    "data": {
                        "relation": rel,
                        "connected_concept": concept_data
                    },
                    "score": score / (dist + 1)
                })

        # Step 3: 检查是否有相关pattern
        matched_concept_ids = {c["id"] for _, c, _ in matched_concepts[:3]}
        for pattern in self.patterns:
            pattern_concepts = set(pattern.get("concepts", []))
            overlap = matched_concept_ids & pattern_concepts
            if overlap:
                results.append({
                    "type": "pattern",
                    "data": pattern,
                    "score": pattern["confidence"] * len(overlap) * 0.5
                })

        # Step 4: 检查相关theory
        for theory in self.theories:
            theory_concepts = set(theory.get("concepts", []))
            overlap = matched_concept_ids & theory_concepts
            if overlap:
                results.append({
                    "type": "theory",
                    "data": theory,
                    "score": theory.get("confidence", 0.4) * len(overlap) * 0.8
                })

        # 排序取top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_context(self, query: str, max_length: int = 10000) -> str:
        """获取注入推演上下文的记忆文本

        将检索结果格式化为可注入prompt的文本
        """
        results = self.search(query, top_k=30)
        if not results:
            return ""

        lines = ["[认知记忆]"]
        total_len = 10

        for item in results:
            if item["type"] == "concept":
                c = item["data"]
                line = f"• {c['name']}(权重{c['weight']:.2f}): {c.get('description', '')[:250]}"
            elif item["type"] == "relation":
                rel = item["data"]["relation"]
                conn = item["data"]["connected_concept"]
                type_cn = RELATION_CN.get(rel["type"], rel["type"])
                line = f"• {type_cn}关系(强度{rel['weight']:.2f}): {rel.get('description', '')[:250]}"
            elif item["type"] == "pattern":
                p = item["data"]
                line = f"• 规律(置信{p['confidence']:.2f}): {p.get('regularity', '')[:250]}"
            elif item["type"] == "theory":
                t = item["data"]
                line = f"★ 理论(置信{t['confidence']:.2f}): {t.get('regularity', '')[:250]}"
            else:
                continue

            if total_len + len(line) > max_length:
                break
            lines.append(line)
            total_len += len(line)

        return "\n".join(lines) if len(lines) > 1 else ""

    def get_depth_info(self) -> dict:
        """获取概念深度信息：每个概念的探索次数和深度等级

        Returns:
            {
                "max_depth": int,  # 当前最大探索深度
                "deep_concepts": [(name, count, level), ...],  # 深度探索的概念
                "depth_distribution": {1: count, 3: count, 5: count, ...}  # 各深度等级分布
            }
        """
        depth_data = []
        for name, concept in self.concepts.items():
            count = concept.get("activation_count", 0)
            # 深度等级：1=浅层(1-2次), 2=中层(3-4次), 3=深层(5-9次), 4=极限层(10+)
            if count >= 10:
                level = 4
            elif count >= 5:
                level = 3
            elif count >= 3:
                level = 2
            elif count >= 1:
                level = 1
            else:
                level = 0
            depth_data.append((name, count, level))

        # 按探索次数排序
        depth_data.sort(key=lambda x: x[1], reverse=True)

        # 统计分布
        dist = {}
        for _, _, level in depth_data:
            dist[level] = dist.get(level, 0) + 1

        max_depth = max((c[1] for c in depth_data), default=0)

        return {
            "max_depth": max_depth,
            "deep_concepts": depth_data[:10],  # top 10最深概念
            "depth_distribution": dist,
            "total_concepts": len(self.concepts)
        }

    # ============================================================
    # 巩固：离线权重调整与层级提升
    # ============================================================
    def consolidate(self, current_round: int):
        """离线巩固——类似睡眠时的记忆巩固

        1. 衰减未激活的概念和关系
        2. 高频共现概念 → 提升为pattern
        3. 高频验证的pattern → 提升为theory
        """
        # 1. 概念权重衰减
        for name, concept in self.concepts.items():
            rounds_since_active = current_round - concept.get("last_activated", 0)
            if rounds_since_active > 5:
                concept["weight"] = max(MIN_WEIGHT, concept["weight"] * DECAY_FACTOR)

        # 2. 关系权重衰减
        for rel in self.relations:
            last_round = max(rel.get("evidence_rounds", [0]))
            rounds_since = current_round - last_round
            if rounds_since > 5:
                rel["weight"] = max(MIN_WEIGHT, rel["weight"] * DECAY_FACTOR)

        # 3. 发现高频共现 → 提升为pattern
        concept_pairs = Counter()
        for log_entry in self.activation_log:
            cids = log_entry.get("concept_ids", [])
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    pair = tuple(sorted([cids[i], cids[j]]))
                    concept_pairs[pair] += 1

        for (c1, c2), count in concept_pairs.items():
            if count >= PATTERN_THRESHOLD:
                # 检查是否已有包含这对concept的pattern
                already_exists = False
                for p in self.patterns:
                    if c1 in p.get("concepts", []) and c2 in p.get("concepts", []):
                        p["evidence_count"] = max(p.get("evidence_count", 0), count)
                        p["confidence"] = min(1.0, count / 10.0)
                        already_exists = True
                        break
                if not already_exists:
                    # 找概念名
                    c1_name = next((n for n, c in self.concepts.items() if c["id"] == c1), c1)
                    c2_name = next((n for n, c in self.concepts.items() if c["id"] == c2), c2)
                    pid = self._new_pattern_id()
                    self.patterns.append({
                        "id": pid,
                        "concepts": [c1, c2],
                        "regularity": f"{c1_name} 与 {c2_name} 反复共现",
                        "confidence": min(1.0, count / 10.0),
                        "evidence_count": count,
                        "created_round": current_round,
                        "layer": 2
                    })
                    logger.info(f"新模式涌现: {c1_name} ↔ {c2_name} (共现{count}次)")

        # 4. 清理激活日志（只保留最近50轮）
        if len(self.activation_log) > 50:
            self.activation_log = self.activation_log[-50:]

        # 5. 保存质疑追踪数据
        if self.skepticism_tracker:
            self.skepticism_tracker.save()

        self._save()
        logger.info(f"巩固完成: {len(self.concepts)}概念, {len(self.relations)}关系, "
                     f"{len(self.patterns)}模式, {len(self.theories)}理论")

    # ============================================================
    # 统计
    # ============================================================
    def get_stats(self) -> dict:
        """获取记忆网络统计信息"""
        total_concepts = len(self.concepts)
        active_concepts = sum(1 for c in self.concepts.values() if c["weight"] > 0.3)
        total_relations = len(self.relations)
        total_patterns = len(self.patterns)
        total_theories = len(self.theories)

        # 估算存储大小
        total_size = 0
        for fname in ["concepts.json", "relations.json", "patterns.json",
                       "theories.json", "activation_log.json"]:
            fpath = os.path.join(MEMORY_DIR, fname)
            if os.path.exists(fpath):
                total_size += os.path.getsize(fpath)

        return {
            "total_concepts": total_concepts,
            "active_concepts": active_concepts,
            "total_relations": total_relations,
            "total_patterns": total_patterns,
            "total_theories": total_theories,
            "total_memories": total_concepts + total_patterns + total_theories,
            "storage_size_mb": round(total_size / 1024 / 1024, 3),
            "storage_path": MEMORY_DIR,
            "skepticism_stats": (
                self.skepticism_tracker.get_all_stats()
                if self.skepticism_tracker else {}
            ),
        }


    def get_top_concepts(self, top_k: int = 10, max_length: int = 2000) -> str:
        """获取权重最高的概念作为反刍上下文
        
        Args:
            top_k: 返回前K个概念
            max_length: 最大文本长度
            
        Returns:
            格式化的概念文本
        """
        if not self.concepts:
            return ""
        
        # 按权重排序，取top_k
        sorted_concepts = sorted(
            self.concepts.items(),
            key=lambda x: x[1].get("weight", 0),
            reverse=True
        )[:top_k]
        
        lines = ["[认知网络-核心概念（按权重排序）]"]
        total_len = 30
        
        for name, concept in sorted_concepts:
            weight = concept.get("weight", 0)
            desc = concept.get("description", "")[:150]
            line = f"• {name}(权重{weight:.2f}): {desc}"
            
            if total_len + len(line) > max_length:
                break
            lines.append(line)
            total_len += len(line)
        
        # 也加入一些高权重的关系
        if self.relations:
            sorted_rels = sorted(
                self.relations,
                key=lambda x: x.get("weight", 0),
                reverse=True
            )[:5]
            
            lines.append("\n[核心关系]")
            for rel in sorted_rels:
                from_c = rel.get("from_concept", "")
                to_c = rel.get("to_concept", "")
                rel_type = RELATION_CN.get(rel.get("type", ""), rel.get("type", ""))
                weight = rel.get("weight", 0)
                desc = rel.get("description", "")[:100]
                line = f"• {from_c} → {rel_type} → {to_c}(强度{weight:.2f}): {desc}"
                
                if total_len + len(line) > max_length:
                    break
                lines.append(line)


# ============================================================
# 知识提取函数：从推演文本中提取结构化知识
# ============================================================
def extract_knowledge_from_text(text: str, config: dict) -> dict:
    """使用LLM从文本中提取概念和关系
    
    Args:
        text: 推演产物文本
        config: 配置字典，包含api_key, api_base, model_name等
    
    Returns:
        {"concepts": [{"name": str, "description": str}], 
         "relations": [{"from": str, "to": str, "type": str, "description": str}]}
    """
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=config.get("api_key", ""),
            base_url=config.get("api_base", "https://api.deepseek.com/v1")
        )
        
        prompt = f"""从以下学术推演文本中提取核心概念和它们之间的关系。

要求：
1. 概念：提取文本中的核心理论概念（不是普通词汇），每个概念包含名称和一句话描述
2. 关系：提取概念之间的关系，类型包括：supports（支持）、contradicts（矛盾）、extends（扩展）、analogizes（类比）、co_occurs（共现）
3. 只提取真正重要的概念（最多8个）和有意义的关系（最多10条）
4. 严格JSON格式输出

文本：
{text[:3000]}

输出格式（严格JSON）：
{{"concepts": [{{"name": "概念名", "description": "一句话描述"}}], "relations": [{{"from": "概念A", "to": "概念B", "type": "关系类型", "description": "关系说明"}}]}}"""

        response = client.chat.completions.create(
            model=config.get("model_name", "deepseek-chat"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content.strip()
        # 提取JSON部分
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        return {
            "concepts": result.get("concepts", []),
            "relations": result.get("relations", [])
        }
    except Exception as e:
        logger.warning(f"知识提取失败: {e}")
        return {"concepts": [], "relations": []}
