#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
push.py - 推送模块
职责：将漂移产物保存为markdown文件，并在控制台输出推送信息
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger("drift.push")

# 漂移产物归档目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "drift_output")


def push_drift(drift_content: str, round_num: int) -> str:
    """推送漂移产物
    
    将消化引擎产生的有效涌现内容保存为markdown文件，
    并在控制台打印推送信息。
    
    Args:
        drift_content: 漂移产物的文本内容
        round_num: 当前消化轮次编号
        
    Returns:
        保存的文件路径
    """
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 生成文件名：时间戳 + 轮次
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"drift_{timestamp}_R{round_num}.md"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # 构造完整的markdown文件（含元信息头）
    header = (
        f"---\n"
        f"round: {round_num}\n"
        f"created: {datetime.now().isoformat()}\n"
        f"type: drift_output\n"
        f"---\n\n"
    )
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + drift_content)

    # 控制台推送通知（带时间戳和产物摘要前100字）
    summary = drift_content[:100].replace("\n", " ")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"[{now_str}] 🌊 漂移产物推送 (轮次={round_num})")
    print(f"[{now_str}] 文件: {filename}")
    print(f"[{now_str}] 摘要: {summary}...")
    print(f"{'='*60}\n")

    logger.info(f"漂移产物已推送: {filepath}, 长度={len(drift_content)}字")
    return filepath
