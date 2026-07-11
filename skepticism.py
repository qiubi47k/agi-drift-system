#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skepticism.py - 双向质疑机制模块
设计哲学：理论的置信度不应只靠激活次数堆叠，必须有自我质疑和外部验证能力。
         每一次确认都问"够不够独立？"，每一次反驳都问"够不够可信？"。

核心机制：
  1. 外部验证：记录来自不同source的confirmed/refuted结果
  2. 独立源验证：至少3个独立source_id确认，才允许置信度提升
  3. 质疑分数：综合confirmed/refuted比例 × 独立验证加成
  4. 模拟验证通道：无外部数据时，系统自验证（较低可信度）
  5. 理论间冲突解决：比较evidence_count，优先激活证据链更完整的一方
"""

import os
import json
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger("drift.skepticism")

# ============================================================
# 常量
# ============================================================
MIN_INDEPENDENT_SOURCES = 3       # 至少3个独立源才给满分加成
RELIABILITY_THRESHOLD = 0.8       # 反驳来源可信度阈值
INTERNAL_SOURCE_RELIABILITY = 0.4 # 内部自验证的默认可信度（较低）
VALIDATION_LOG_FILE = "validation_log.json"
SKEPTICISM_DATA_FILE = "skepticism_data.json"

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory")


class ValidationRecord:
    """单条验证记录"""

    __slots__ = [
        "theory_id", "result", "source_reliability",
        "timestamp", "source_id", "note"
    ]

    def __init__(self, theory_id: str, result: str, source_reliability: float,
                 source_id: str, note: str = ""):
        self.theory_id = theory_id
        self.result = result  # "confirmed" | "refuted"
        self.source_reliability = max(0.0, min(1.0, source_reliability))
        self.timestamp = datetime.now().isoformat()
        self.source_id = source_id
        self.note = note

    def to_dict(self) -> dict:
        return {
            "theory_id": self.theory_id,
            "result": self.result,
            "source_reliability": self.source_reliability,
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ValidationRecord":
        rec = cls(
            theory_id=d["theory_id"],
            result=d["result"],
            source_reliability=d["source_reliability"],
            source_id=d["source_id"],
            note=d.get("note", ""),
        )
        rec.timestamp = d.get("timestamp", datetime.now().isoformat())
        return rec


class SkepticismTracker:
    """理论质疑追踪器

    每个 theory_id 维护：
      - confirmed_count: 确认次数
      - refuted_count: 反驳次数（仅 source_reliability > 0.8 时计入）
      - unique_sources: 独立确认源集合
      - disputed: 当前是否存疑
    """

    def __init__(self):
        # theory_id -> {"confirmed_count", "refuted_count", "unique_sources", "disputed"}
        self.trackers: dict = {}
        self.validation_log: list = []
        self._load()

    # ============================================================
    # 持久化
    # ============================================================
    def _load(self):
        """从磁盘加载追踪数据"""
        data_path = os.path.join(MEMORY_DIR, SKEPTICISM_DATA_FILE)
        if os.path.exists(data_path):
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                self.trackers = raw.get("trackers", {})
                # 恢复 unique_sources 为 set
                for tid, data in self.trackers.items():
                    if isinstance(data.get("unique_sources"), list):
                        data["unique_sources"] = set(data["unique_sources"])
                logger.info(f"质疑数据加载完成: {len(self.trackers)}个理论被追踪")
            except Exception as e:
                logger.warning(f"质疑数据加载失败: {e}")
                self.trackers = {}

        log_path = os.path.join(MEMORY_DIR, VALIDATION_LOG_FILE)
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    raw_log = json.load(f)
                self.validation_log = [ValidationRecord.from_dict(r) for r in raw_log]
            except Exception as e:
                logger.warning(f"验证日志加载失败: {e}")
                self.validation_log = []

    def save(self):
        """持久化追踪数据到磁盘"""
        os.makedirs(MEMORY_DIR, exist_ok=True)

        # 序列化 trackers（set -> list）
        serializable = {}
        for tid, data in self.trackers.items():
            entry = dict(data)
            if isinstance(entry.get("unique_sources"), set):
                entry["unique_sources"] = list(entry["unique_sources"])
            serializable[tid] = entry

        data_path = os.path.join(MEMORY_DIR, SKEPTICISM_DATA_FILE)
        tmp = data_path + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({"trackers": serializable}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, data_path)

        # 保存验证日志
        log_path = os.path.join(MEMORY_DIR, VALIDATION_LOG_FILE)
        tmp = log_path + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(
                [r.to_dict() for r in self.validation_log],
                f, ensure_ascii=False, indent=2
            )
        os.replace(tmp, log_path)

    # ============================================================
    # 初始化
    # ============================================================
    def init_theory(self, theory_id: str):
        """为新理论初始化质疑追踪字段

        在 theory 从 pattern 提升时调用。
        """
        if theory_id not in self.trackers:
            self.trackers[theory_id] = {
                "confirmed_count": 0,
                "refuted_count": 0,
                "unique_sources": set(),
                "disputed": False,
            }
            logger.debug(f"质疑追踪初始化: {theory_id}")

    # ============================================================
    # 核心接口：记录验证
    # ============================================================
    def record_validation(self, theory_id: str, result: str,
                          source_reliability: float, source_id: str,
                          note: str = "") -> dict:
        """注入外部验证结果

        Args:
            theory_id: 理论ID
            result: "confirmed" 或 "refuted"
            source_reliability: 来源可信度 [0, 1]
            source_id: 来源标识（用于判断独立性）
            note: 可选备注

        Returns:
            {"status": str, "details": dict} 描述处理结果
        """
        if theory_id not in self.trackers:
            self.init_theory(theory_id)

        tracker = self.trackers[theory_id]
        record = ValidationRecord(theory_id, result, source_reliability, source_id, note)

        if result == "confirmed":
            tracker["confirmed_count"] += 1
            tracker["unique_sources"].add(source_id)
            status = "confirmed"
            details = {
                "confirmed_count": tracker["confirmed_count"],
                "unique_sources": len(tracker["unique_sources"]),
            }
            logger.info(f"理论{theory_id}确认+1 (src={source_id}, "
                        f"reliability={source_reliability:.2f}, "
                        f"total_confirmed={tracker['confirmed_count']})")

        elif result == "refuted":
            if source_reliability > RELIABILITY_THRESHOLD:
                # 高可信度反驳：直接计入
                tracker["refuted_count"] += 1
                status = "refuted_accepted"
                details = {
                    "refuted_count": tracker["refuted_count"],
                    "source_reliability": source_reliability,
                }
                logger.info(f"理论{theory_id}反驳+1 (src={source_id}, "
                            f"reliability={source_reliability:.2f}, "
                            f"total_refuted={tracker['refuted_count']})")
            else:
                # 低可信度反驳：标记存疑，暂不更新计数
                tracker["disputed"] = True
                status = "refuted_pending"
                details = {
                    "refuted_count": tracker["refuted_count"],
                    "source_reliability": source_reliability,
                    "note": "存疑，暂不更新——来源可信度不足",
                }
                logger.info(f"理论{theory_id}反驳存疑 (src={source_id}, "
                            f"reliability={source_reliability:.2f} < {RELIABILITY_THRESHOLD})")
        else:
            status = "unknown_result"
            details = {"error": f"未知验证结果类型: {result}"}
            logger.warning(f"未知验证结果: {result} for {theory_id}")

        self.validation_log.append(record)
        return {"status": status, "details": details}

    # ============================================================
    # 核心接口：质疑分数
    # ============================================================
    def get_skepticism_score(self, theory_id: str) -> float:
        """返回质疑分数 [0, 1]

        分数越高表示理论越可信。
        公式：
          base = confirmed_count / (confirmed_count + refuted_count + 1)
          bonus = min(1.0, unique_sources / MIN_INDEPENDENT_SOURCES)
          score = base * bonus

        如果理论未追踪，返回 0.0
        """
        if theory_id not in self.trackers:
            return 0.0

        tracker = self.trackers[theory_id]
        cc = tracker["confirmed_count"]
        rc = tracker["refuted_count"]
        unique = len(tracker["unique_sources"])

        base = cc / (cc + rc + 1)
        bonus = min(1.0, unique / MIN_INDEPENDENT_SOURCES)
        score = base * bonus

        # 存疑状态下扣分
        if tracker.get("disputed"):
            score *= 0.8

        return round(score, 4)

    def get_independent_verification_bonus(self, theory_id: str) -> float:
        """获取独立验证加成分数 [0, 1]

        bonus = min(1.0, unique_sources / MIN_INDEPENDENT_SOURCES)
        """
        if theory_id not in self.trackers:
            return 0.0
        unique = len(self.trackers[theory_id].get("unique_sources", set()))
        return round(min(1.0, unique / MIN_INDEPENDENT_SOURCES), 4)

    # ============================================================
    # 核心接口：计算置信度
    # ============================================================
    def compute_confidence(self, theory_id: str) -> float:
        """基于质疑分数计算理论的新置信度 [0, 1]

        新置信度 = (confirmed / (confirmed + refuted + 1)) * independent_bonus
        """
        return self.get_skepticism_score(theory_id)

    def get_theory_stats(self, theory_id: str) -> dict:
        """获取单个理论的质疑统计信息"""
        if theory_id not in self.trackers:
            return {
                "theory_id": theory_id,
                "tracked": False,
                "skepticism_score": 0.0,
            }

        tracker = self.trackers[theory_id]
        return {
            "theory_id": theory_id,
            "tracked": True,
            "confirmed_count": tracker["confirmed_count"],
            "refuted_count": tracker["refuted_count"],
            "unique_sources": len(tracker["unique_sources"]),
            "disputed": tracker.get("disputed", False),
            "skepticism_score": self.get_skepticism_score(theory_id),
            "independent_bonus": self.get_independent_verification_bonus(theory_id),
            "computed_confidence": self.compute_confidence(theory_id),
        }

    def get_all_stats(self) -> dict:
        """获取所有被追踪理论的统计概览"""
        result = {}
        for tid in self.trackers:
            result[tid] = self.get_theory_stats(tid)
        return result

    # ============================================================
    # 模拟验证通道
    # ============================================================
    def record_internal_validation(self, theory_id: str, result: str,
                                   round_num: int, note: str = "") -> dict:
        """内部自验证：系统推演中检查新产物与已有理论是否一致

        来源可信度固定为 INTERNAL_SOURCE_RELIABILITY（较低），
        来源ID格式为 "internal_round_{round_num}"。

        Args:
            theory_id: 理论ID
            result: "confirmed" 或 "refuted"
            round_num: 当前轮次
            note: 备注

        Returns:
            同 record_validation 返回格式
        """
        source_id = f"internal_round_{round_num}"
        return self.record_validation(
            theory_id=theory_id,
            result=result,
            source_reliability=INTERNAL_SOURCE_RELIABILITY,
            source_id=source_id,
            note=f"[内部验证 R{round_num}] {note}",
        )

    # ============================================================
    # 理论间冲突解决
    # ============================================================
    def resolve_theory_conflict(self, theory_a_id: str, theory_b_id: str,
                                evidence_count_a: float,
                                evidence_count_b: float) -> dict:
        """理论间不一致时的冲突解决

        比较 evidence_count，优先激活证据链更完整的一方。
        如果证据相当（差异 < 20%），则双方都标记 disputed。

        Args:
            theory_a_id: 理论A的ID
            theory_b_id: 理论B的ID
            evidence_count_a: 理论A的证据数
            evidence_count_b: 理论B的证据数

        Returns:
            {
                "winner": str or None,   # 胜出的theory_id，None表示平局
                "loser": str or None,
                "resolution": str,       # 解决方式描述
                "evidence_ratio": float, # 证据比
            }
        """
        for tid in [theory_a_id, theory_b_id]:
            if tid not in self.trackers:
                self.init_theory(tid)

        max_ev = max(evidence_count_a, evidence_count_b, 1)
        min_ev = min(evidence_count_a, evidence_count_b)
        ratio = min_ev / max_ev if max_ev > 0 else 0

        # 证据相当（差异 < 20%）→ 双方存疑
        if ratio >= 0.8:
            self.trackers[theory_a_id]["disputed"] = True
            self.trackers[theory_b_id]["disputed"] = True
            result = {
                "winner": None,
                "loser": None,
                "resolution": "证据相当，双方标记存疑",
                "evidence_ratio": round(ratio, 4),
            }
            logger.info(f"理论冲突(平局): {theory_a_id}(ev={evidence_count_a}) "
                        f"vs {theory_b_id}(ev={evidence_count_b}), ratio={ratio:.2f}")
        elif evidence_count_a > evidence_count_b:
            result = {
                "winner": theory_a_id,
                "loser": theory_b_id,
                "resolution": f"理论{theory_a_id}证据链更完整({evidence_count_a} vs {evidence_count_b})",
                "evidence_ratio": round(ratio, 4),
            }
            logger.info(f"理论冲突({theory_a_id}胜出): ev={evidence_count_a} vs {evidence_count_b}")
        else:
            result = {
                "winner": theory_b_id,
                "loser": theory_a_id,
                "resolution": f"理论{theory_b_id}证据链更完整({evidence_count_b} vs {evidence_count_a})",
                "evidence_ratio": round(ratio, 4),
            }
            logger.info(f"理论冲突({theory_b_id}胜出): ev={evidence_count_b} vs {evidence_count_a}")

        return result

    # ============================================================
    # 辅助：清除存疑状态
    # ============================================================
    def clear_disputed(self, theory_id: str):
        """清除理论的存疑标记（例如在获得新的高可信验证后）"""
        if theory_id in self.trackers:
            self.trackers[theory_id]["disputed"] = False
            logger.debug(f"理论{theory_id}存疑状态已清除")

    def batch_init_from_theories(self, theories: list):
        """从现有理论列表批量初始化追踪器

        在系统集成时调用，为所有尚无追踪数据的理论创建初始记录。
        """
        count = 0
        for theory in theories:
            tid = theory.get("id", "")
            if tid and tid not in self.trackers:
                self.init_theory(tid)
                count += 1
        if count > 0:
            self.save()
            logger.info(f"批量初始化: {count}个理论已添加质疑追踪")


# ============================================================
# 便捷单例
# ============================================================
_tracker_instance = None


def get_tracker() -> SkepticismTracker:
    """获取全局单例追踪器"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = SkepticismTracker()
    return _tracker_instance
