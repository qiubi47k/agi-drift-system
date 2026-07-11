#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
world_simulator.py - 模拟世界接地层
设计哲学：任何理论必须能在某个一致的世界中实例化，否则只是语义组合

核心思想：
- 不硬套物理定律（太窄），而是建一个通用模拟世界
- 理论的意义在于它能在某个世界里运行并产生可观测预测
- 物理世界只是模拟世界的一个特例
"""

import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Callable, Any

logger = logging.getLogger("drift.world")


class WorldKernel:
    """世界内核 - 定义一个最小可计算世界
    
    一个世界由以下部分组成：
    - entities: 有状态的实体
    - rules: 实体间交互规则
    - evolution: 状态演化规则
    - observables: 可测量量
    """
    
    def __init__(self, world_id: str, name: str, description: str = ""):
        self.world_id = world_id
        self.name = name
        self.description = description
        self.entities: Dict[str, Dict] = {}  # name -> {state, properties, history}
        self.rules: List[Dict] = []  # 交互规则列表
        self.evolution_func: Optional[Callable] = None  # 演化函数
        self.observables: Dict[str, Callable] = {}  # 可观测量定义
        self.history: List[Dict] = []  # 状态历史
        self.step_count = 0
    
    def add_entity(self, name: str, initial_state: Dict, properties: Dict = None) -> None:
        """添加实体到世界"""
        self.entities[name] = {
            "state": initial_state.copy(),
            "properties": properties or {},
            "history": [initial_state.copy()]
        }
        logger.debug(f"世界{self.world_id}: 添加实体 {name}")
    
    def add_rule(self, rule_id: str, source: str, target: str, 
                 interaction: Callable, description: str = "") -> None:
        """添加交互规则"""
        self.rules.append({
            "id": rule_id,
            "source": source,
            "target": target,
            "interaction": interaction,
            "description": description
        })
    
    def set_evolution(self, evolve_func: Callable) -> None:
        """设置状态演化规则"""
        self.evolution_func = evolve_func
    
    def add_observable(self, name: str, measure_func: Callable) -> None:
        """添加可测量量"""
        self.observables[name] = measure_func
    
    def step(self, n_steps: int = 1) -> List[Dict]:
        """演化世界状态"""
        snapshots = []
        
        for _ in range(n_steps):
            # 1. 应用交互规则
            for rule in self.rules:
                if rule["source"] in self.entities and rule["target"] in self.entities:
                    source_state = self.entities[rule["source"]]["state"]
                    target_state = self.entities[rule["target"]]["state"]
                    
                    new_target_state = rule["interaction"](source_state, target_state)
                    self.entities[rule["target"]]["state"] = new_target_state
            
            # 2. 应用演化规则
            if self.evolution_func:
                for name, entity in self.entities.items():
                    new_state = self.evolution_func(name, entity["state"])
                    self.entities[name]["state"] = new_state
            
            # 3. 记录历史
            for name, entity in self.entities.items():
                entity["history"].append(entity["state"].copy())
            
            self.step_count += 1
            snapshots.append(self.get_snapshot())
        
        self.history.extend(snapshots)
        return snapshots
    
    def measure(self, entity_name: str, observable_name: str) -> Any:
        """测量实体的某个可观测量"""
        if entity_name not in self.entities:
            return None
        
        entity = self.entities[entity_name]
        
        if observable_name in self.observables:
            return self.observables[observable_name](entity["state"])
        
        # 默认返回状态中的值
        return entity["state"].get(observable_name)
    
    def get_snapshot(self) -> Dict:
        """获取当前状态快照"""
        return {
            "step": self.step_count,
            "entities": {name: entity["state"].copy() 
                        for name, entity in self.entities.items()}
        }
    
    def is_consistent(self) -> bool:
        """检查世界是否一致（无矛盾状态）"""
        for name, entity in self.entities.items():
            if not self._check_entity_consistency(entity):
                return False
        return True
    
    def _check_entity_consistency(self, entity: Dict) -> bool:
        """检查单个实体的一致性"""
        state = entity["state"]
        
        # 检查状态值是否有效（非NaN，非无穷大）
        for key, value in state.items():
            if isinstance(value, (int, float)):
                if value != value:  # NaN check
                    return False
                if abs(value) > 1e100:  # 无穷大检查
                    return False
        
        return True
    
    def reset(self) -> None:
        """重置世界到初始状态"""
        for entity in self.entities.values():
            if entity["history"]:
                entity["state"] = entity["history"][0].copy()
                entity["history"] = [entity["history"][0].copy()]
        
        self.history = []
        self.step_count = 0


class WorldRegistry:
    """世界注册表 - 管理多个可能的世界"""
    
    def __init__(self):
        self.worlds: Dict[str, WorldKernel] = {}
        self.grounding_map: Dict[str, Dict] = {}  # concept_name -> world mapping
    
    def register_world(self, world: WorldKernel) -> None:
        """注册一个世界"""
        self.worlds[world.world_id] = world
        logger.info(f"注册世界: {world.world_id} - {world.name}")
    
    def get_world(self, world_id: str) -> Optional[WorldKernel]:
        """获取世界"""
        return self.worlds.get(world_id)
    
    def map_concept_to_entity(self, concept_name: str, world_id: str, 
                             entity_name: str, mapping: Dict = None) -> None:
        """映射概念到世界的实体"""
        if concept_name not in self.grounding_map:
            self.grounding_map[concept_name] = {}
        
        self.grounding_map[concept_name][world_id] = {
            "entity_name": entity_name,
            "mapping": mapping or {},
            "status": "mapped"
        }
    
    def get_entity_for_concept(self, concept_name: str, 
                              world_id: str = None) -> Optional[Dict]:
        """获取概念对应的实体"""
        if concept_name not in self.grounding_map:
            return None
        
        if world_id:
            mapping = self.grounding_map[concept_name].get(world_id)
        else:
            # 返回第一个有效映射
            mappings = self.grounding_map[concept_name]
            mapping = next(iter(mappings.values()), None)
        
        if mapping and mapping["status"] == "mapped":
            world = self.worlds.get(world_id or list(mappings.keys())[0])
            if world:
                return world.entities.get(mapping["entity_name"])
        
        return None
    
    def save_grounding(self, filepath: str) -> None:
        """保存接地映射"""
        # 只保存可序列化的部分
        serializable = {}
        for concept, mappings in self.grounding_map.items():
            serializable[concept] = {}
            for world_id, mapping in mappings.items():
                serializable[concept][world_id] = {
                    "entity_name": mapping["entity_name"],
                    "mapping": mapping["mapping"],
                    "status": mapping["status"]
                }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    
    def load_grounding(self, filepath: str) -> None:
        """加载接地映射"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.grounding_map = data
        except FileNotFoundError:
            pass


class TheoryValidator:
    """理论验证器 - 验证理论是否可以在世界中实例化"""
    
    def __init__(self, registry: WorldRegistry):
        self.registry = registry
    
    def validate_theory(self, theory: Dict, concepts: Dict) -> Dict:
        """验证理论是否可计算
        
        Args:
            theory: 理论字典，包含 concepts, regularity 等
            concepts: 全局概念字典
            
        Returns:
            {
                "groundable": bool,  # 是否可以接地
                "world_ids": list,   # 可以实例化的世界列表
                "unmapped_concepts": list,  # 未映射的概念
                "predictions": list,  # 可测试的预测
                "confidence_boost": float  # 置信度提升建议
            }
        """
        theory_concepts = theory.get("concepts", [])
        unmapped = []
        grounded_worlds = set()
        
        # 检查理论中的概念是否都有接地映射
        for cid in theory_concepts:
            # 找到概念名
            concept_name = None
            for name, cdata in concepts.items():
                if cdata.get("id") == cid:
                    concept_name = name
                    break
            
            if not concept_name:
                unmapped.append(cid)
                continue
            
            # 检查是否有接地映射
            if concept_name not in self.registry.grounding_map:
                unmapped.append(concept_name)
                continue
            
            # 记录可以接地的世界
            for world_id in self.registry.grounding_map[concept_name]:
                grounded_worlds.add(world_id)
        
        # 判断是否可以接地
        groundable = len(unmapped) == 0 and len(grounded_worlds) > 0
        
        # 计算置信度提升建议
        confidence_boost = 0.0
        if groundable:
            # 可以接地的理论获得置信度加成
            confidence_boost = 0.1 * len(grounded_worlds)
        
        return {
            "groundable": groundable,
            "world_ids": list(grounded_worlds),
            "unmapped_concepts": unmapped,
            "predictions": [],  # TODO: 从理论中提取可测试预测
            "confidence_boost": confidence_boost
        }
    
    def run_simulation(self, theory: Dict, world_id: str, 
                      concepts: Dict, n_steps: int = 10) -> Dict:
        """在世界中运行理论模拟
        
        Returns:
            {
                "success": bool,
                "snapshots": list,
                "observations": dict,
                "consistency": bool
            }
        """
        world = self.registry.get_world(world_id)
        if not world:
            return {"success": False, "error": "世界不存在"}
        
        # 运行模拟
        snapshots = world.step(n_steps)
        
        # 收集观测
        observations = {}
        for obs_name in world.observables:
            observations[obs_name] = []
            for entity_name in world.entities:
                value = world.measure(entity_name, obs_name)
                observations[obs_name].append({
                    "entity": entity_name,
                    "value": value
                })
        
        return {
            "success": True,
            "snapshots": snapshots[-5:],  # 只返回最后5个快照
            "observations": observations,
            "consistency": world.is_consistent()
        }


# ============================================================
# 预定义世界模板
# ============================================================

def create_discrete_world(world_id: str, name: str) -> WorldKernel:
    """创建离散状态世界
    
    适用于：概念系统、信息流、认知网络
    """
    world = WorldKernel(world_id, name, "离散状态世界")
    
    # 默认演化规则：状态值按衰减因子变化
    def default_evolution(entity_name, state):
        new_state = {}
        for key, value in state.items():
            if isinstance(value, (int, float)):
                # 衰减
                new_state[key] = value * 0.95
            else:
                new_state[key] = value
        return new_state
    
    world.set_evolution(default_evolution)
    
    # 默认可观测量：总能量
    def total_energy(state):
        return sum(v for v in state.values() if isinstance(v, (int, float)))
    
    world.add_observable("total_energy", total_energy)
    
    return world


def create_continuous_world(world_id: str, name: str) -> WorldKernel:
    """创建连续动力系统世界
    
    适用于：物理系统、动力系统、微分方程
    """
    world = WorldKernel(world_id, name, "连续动力系统世界")
    
    # 默认演化规则：简单线性演化
    def linear_evolution(entity_name, state):
        new_state = {}
        for key, value in state.items():
            if isinstance(value, (int, float)):
                # 线性增长
                new_state[key] = value + 0.1
            else:
                new_state[key] = value
        return new_state
    
    world.set_evolution(linear_evolution)
    
    return world


def create_agent_world(world_id: str, name: str) -> WorldKernel:
    """创建智能体交互世界
    
    适用于：博弈论、社会系统、多智能体系统
    """
    world = WorldKernel(world_id, name, "智能体交互世界")
    
    # 默认交互规则：合作/背叛
    def prisoner_dilemma(source_state, target_state):
        # 简化版囚徒困境
        source_coop = source_state.get("cooperation", 0.5)
        target_coop = target_state.get("cooperation", 0.5)
        
        # 更新合作度
        new_target = target_state.copy()
        new_target["cooperation"] = (source_coop + target_coop) / 2
        
        return new_target
    
    world.add_rule("interaction", "agent_a", "agent_b", prisoner_dilemma, 
                  "囚徒困境交互")
    
    return world


# ============================================================
# 全局世界管理器
# ============================================================

class WorldManager:
    """世界管理器 - 管理所有世界的生命周期"""
    
    def __init__(self, grounding_file: str = None):
        self.registry = WorldRegistry()
        self.validator = TheoryValidator(self.registry)
        self.grounding_file = grounding_file or "memory/grounding.json"
        
        # 初始化默认世界
        self._init_default_worlds()
        
        # 加载接地映射
        self.registry.load_grounding(self.grounding_file)
    
    def _init_default_worlds(self):
        """初始化默认世界"""
        # 离散世界
        discrete = create_discrete_world("discrete", "离散状态世界")
        self.registry.register_world(discrete)
        
        # 连续世界
        continuous = create_continuous_world("continuous", "连续动力系统世界")
        self.registry.register_world(continuous)
        
        # 智能体世界
        agent = create_agent_world("agent", "智能体交互世界")
        self.registry.register_world(agent)
    
    def validate_theory(self, theory: Dict, concepts: Dict) -> Dict:
        """验证单个理论的可接地性"""
        return self.validator.validate_theory(theory, concepts)
    
    def run_simulation(self, theory: Dict, world_id: str, 
                      concepts: Dict, n_steps: int = 10) -> Dict:
        """在世界中运行理论模拟
        
        Args:
            theory: 理论字典
            world_id: 世界ID
            concepts: 概念字典
            n_steps: 模拟步数
            
        Returns:
            模拟结果
        """
        world = self.registry.get_world(world_id)
        if not world:
            return {"success": False, "error": f"世界{world_id}不存在"}
        
        # 获取理论涉及的概念
        theory_concepts = theory.get("concepts", [])
        concept_names = []
        for cid in theory_concepts:
            for name, cdata in concepts.items():
                if cdata.get("id") == cid:
                    concept_names.append(name)
                    break
        
        if not concept_names:
            return {"success": False, "error": "理论无有效概念"}
        
        # 为世界创建实体（如果还没有）
        for name in concept_names:
            if name not in world.entities:
                # 创建实体，初始状态为概念权重
                concept_data = concepts.get(name, {})
                weight = concept_data.get("weight", 0.5)
                initial_state = {
                    "value": weight,
                    "stability": 1.0,
                    "energy": weight * 10
                }
                world.add_entity(name, initial_state)
        
        # 运行模拟
        try:
            snapshots = world.step(n_steps)
            consistency = world.is_consistent()
            
            # 收集观测
            observations = {}
            for obs_name in world.observables:
                observations[obs_name] = []
                for entity_name in world.entities:
                    if entity_name in concept_names:
                        value = world.measure(entity_name, obs_name)
                        observations[obs_name].append({
                            "entity": entity_name,
                            "value": value
                        })
            
            return {
                "success": True,
                "consistency": consistency,
                "steps": len(snapshots),
                "observations": observations
            }
        except Exception as e:
            logger.warning(f"模拟失败: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_all_theories(self, theories: List[Dict], 
                             concepts: Dict) -> List[Dict]:
        """验证所有理论的可接地性"""
        results = []
        for theory in theories:
            result = self.validator.validate_theory(theory, concepts)
            result["theory_id"] = theory.get("id")
            result["theory_name"] = theory.get("name")
            results.append(result)
        return results
    
    def save_grounding(self) -> None:
        """保存接地映射"""
        self.registry.save_grounding(self.grounding_file)
    
    def get_stats(self) -> Dict:
        """获取世界系统统计"""
        return {
            "total_worlds": len(self.registry.worlds),
            "total_mappings": len(self.registry.grounding_map),
            "worlds": {
                wid: {
                    "name": w.name,
                    "entities": len(w.entities),
                    "rules": len(w.rules)
                }
                for wid, w in self.registry.worlds.items()
            }
        }
