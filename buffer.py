#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buffer.py - 信息流缓冲池模块
职责：管理原始素材和人类反馈的存取，维护素材索引
"""

import os
import json
import uuid
import logging
from datetime import datetime

logger = logging.getLogger("drift.buffer")

# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "buffer_pool", "raw")
FEEDBACK_DIR = os.path.join(BASE_DIR, "buffer_pool", "feedback")
DIGESTED_DIR = os.path.join(BASE_DIR, "buffer_pool", "digested")
INDEX_FILE = os.path.join(BASE_DIR, "index.json")


class BufferPool:
    """信息流缓冲池

    核心设计原则：
    - raw（原始素材）和 feedback（人类反馈）在消化时权重完全一致，不区分对待
    - 素材永久留存，仅标记消化状态，不删除
    """

    def __init__(self):
        """初始化缓冲池，确保目录和索引文件存在"""
        for d in [RAW_DIR, FEEDBACK_DIR, DIGESTED_DIR]:
            os.makedirs(d, exist_ok=True)
        if not os.path.exists(INDEX_FILE):
            self._save_index({"items": []})
            logger.info("创建了新的素材索引文件")

    def _load_index(self) -> dict:
        """加载素材索引"""
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"索引文件读取失败: {e}，重建空索引")
            return {"items": []}

    def _save_index(self, data: dict):
        """保存素材索引（原子写入）"""
        tmp_path = INDEX_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, INDEX_FILE)

    def write_raw(self, content: str) -> str:
        """写入原始素材

        Args:
            content: 原始文本内容

        Returns:
            素材ID (uuid字符串)
        """
        return self._write_content(content, source="raw", directory=RAW_DIR)

    def write_feedback(self, content: str) -> str:
        """写入人类反馈

        与 write_raw 格式完全一致，仅来源标记不同。
        消化时两者权重完全相同。

        Args:
            content: 反馈文本内容

        Returns:
            素材ID (uuid字符串)
        """
        return self._write_content(content, source="feedback", directory=FEEDBACK_DIR)

    def _write_content(self, content: str, source: str, directory: str) -> str:
        """内部方法：统一写入素材

        Args:
            content: 文本内容
            source: 来源标记 "raw" 或 "feedback"
            directory: 存储目录

        Returns:
            素材ID
        """
        # 生成唯一ID和时间戳文件名
        item_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{item_id[:8]}.md"
        filepath = os.path.join(directory, filename)

        # 写入素材文件（Markdown格式，包含元信息头）
        header = "---\n"
        header += f"id: {item_id}\n"
        header += f"source: {source}\n"
        header += f"created: {datetime.now().isoformat()}\n"
        header += "---\n\n"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + content)

        # 更新索引
        index = self._load_index()
        index["items"].append({
            "id": item_id,
            "file": filepath,
            "source": source,
            "digested": False,
            "digest_round": None
        })
        self._save_index(index)

        logger.info(f"素材入库: id={item_id[:8]}... source={source} file={filename}")
        return item_id

    def read_undigested(self, batch_size: int = 5) -> list:
        """读取未消化的素材批次

        按入库时间顺序（索引中的顺序）读取，raw和feedback混合排列，
        不做任何区分或加权。

        Args:
            batch_size: 每批读取数量

        Returns:
            [(id, text), ...] 列表
        """
        index = self._load_index()
        results = []

        for item in index["items"]:
            if not item["digested"]:
                try:
                    with open(item["file"], "r", encoding="utf-8") as f:
                        text = f.read()
                    results.append((item["id"], text))
                    if len(results) >= batch_size:
                        break
                except FileNotFoundError:
                    logger.warning(f"素材文件丢失: {item['file']}")
                    continue

        logger.debug(f"读取未消化素材: {len(results)}条")
        return results

    def mark_digested(self, id_list: list, round_num: int):
        """标记素材为已消化

        仅修改索引中的状态标记，素材文件本身不变。

        Args:
            id_list: 需要标记的素材ID列表
            round_num: 当前消化轮次编号
        """
        index = self._load_index()
        marked_count = 0

        for item in index["items"]:
            if item["id"] in id_list and not item["digested"]:
                item["digested"] = True
                item["digest_round"] = round_num
                marked_count += 1

        self._save_index(index)
        logger.info(f"标记已消化: {marked_count}条, 轮次={round_num}")

    def get_stats(self) -> dict:
        """获取缓冲池统计信息

        Returns:
            {"total": int, "undigested": int, "digested": int, "last_round": int}
        """
        index = self._load_index()
        total = len(index["items"])
        digested = sum(1 for item in index["items"] if item["digested"])
        undigested = total - digested

        # 找到最大消化轮次
        rounds = [item["digest_round"] for item in index["items"] if item["digest_round"] is not None]
        last_round = max(rounds) if rounds else 0

        return {
            "total": total,
            "undigested": undigested,
            "digested": digested,
            "last_round": last_round
        }

    def get_undigested_count(self) -> int:
        """获取未消化素材数量（轻量查询）"""
        index = self._load_index()
        return sum(1 for item in index["items"] if not item["digested"])
