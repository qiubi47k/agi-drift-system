#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - 内生漂移-锚定系统主进程
职责：串联所有模块，驱动消化循环，管理生命周期

系统行为：
- 永不主动退出，仅Ctrl+C终止
- 缓冲池为空时降低轮询频率，不终止
- 全流程日志记录
"""

import os
import sys
import json
import time
import signal
import logging
from datetime import datetime

# 确保模块路径正确
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# 导入自定义模块
from buffer import BufferPool
from engine import DigestEngine
from push import push_drift
from feedback import FeedbackListener

# ============================================================
# 日志配置
# ============================================================
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "runtime.log")


def setup_logging():
    """配置日志系统
    
    同时输出到文件和控制台。
    格式：[时间] [级别] 消息
    """
    # 创建根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 日志格式
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 文件处理器（追加模式）
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger("drift.main")


# ============================================================
# 6条不可违反原则（硬编码检查）
# ============================================================
PRINCIPLES = {
    "1_无指令解析": "所有用户输入全部作为素材，禁止提取任务/问题/指令",
    "2_无终止条件": "程序仅手动kill关闭，缓冲池为空仅降低轮询频率",
    "3_无任务导向": "消化Prompt全程无问答、解题、回答类描述",
    "4_无区分对待": "raw/feedback素材读取消化权重完全一致",
    "5_无权重更新": "全程仅调用模型推理，不做微调训练",
    "6_静默优先": "无有效涌现产物强制输出[SILENT]，禁止编造",
}


def print_banner():
    """打印系统启动横幅"""
    print("\n" + "=" * 60)
    print("  🌀 内生漂移-锚定系统 v3 最小原型")
    print("  Endogenous Drift-Anchoring System - Minimal Prototype")
    print("=" * 60)
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  工作目录: {BASE_DIR}")
    print("=" * 60)
    
    # 打印6条原则确认
    print("\n  [原则确认]")
    for key, desc in PRINCIPLES.items():
        print(f"  ✓ {key}: {desc}")
    print()


def load_config() -> dict:
    """加载全局配置
    
    Returns:
        配置字典
    """
    config_path = os.path.join(BASE_DIR, "config", "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        logging.getLogger("drift.main").info(f"配置加载成功: {config_path}")
        return config
    except Exception as e:
        logging.getLogger("drift.main").error(f"配置加载失败: {e}")
        # 使用默认配置
        return {
            "digest_interval_minutes": 30,
            "batch_size": 5,
            "temperature": 0.95,
            "top_p": 0.95,
            "max_tokens": 2000,
            "silent_threshold": 3,
            "model_name": "qwen-plus",
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "YOUR_API_KEY_HERE"
        }


def main():
    """主循环入口"""
    # 1. 初始化日志
    logger = setup_logging()
    print_banner()
    
    # 2. 加载配置
    config = load_config()
    logger.info(f"配置项: interval={config['digest_interval_minutes']}min, "
                f"batch={config['batch_size']}, model={config['model_name']}")
    
    # 3. 初始化缓冲池
    pool = BufferPool()
    stats = pool.get_stats()
    logger.info(f"缓冲池状态: 总计={stats['total']}条, "
                f"未消化={stats['undigested']}条, "
                f"已消化={stats['digested']}条, "
                f"上次轮次={stats['last_round']}")
    
    # 打印系统状态
    print(f"\n  [系统状态]")
    print(f"  缓冲池素材总数: {stats['total']}")
    print(f"  待消化素材数:   {stats['undigested']}")
    print(f"  已完成消化轮次: {stats['last_round']}")
    print(f"  消化间隔:       {config['digest_interval_minutes']}分钟")
    print(f"  每批处理量:     {config['batch_size']}条")
    print(f"  静默阈值:       {config['silent_threshold']}轮后拉长间隔")
    print()
    
    # 4. 初始化消化引擎
    engine = DigestEngine(config)
    logger.info("消化引擎初始化完成")
    
    # 5. 启动后台反馈监听线程
    feedback_listener = FeedbackListener(pool)
    feedback_listener.start()
    
    # 6. 注册信号处理（优雅关闭）
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        logger.info("收到终止信号，系统正在关闭...")
        print("\n[系统] 收到终止信号，正在安全关闭...")
        running = False
        feedback_listener.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # ============================================================
    # 7. 进入主循环
    # ============================================================
    # 当前消化轮次（从上次最大轮次+1开始）
    current_round = stats["last_round"] + 1
    
    # 静默计数器：连续[SILENT]输出次数
    silent_count = 0
    
    # 当前消化间隔（秒），可能因静默而拉长
    base_interval = config["digest_interval_minutes"] * 60
    current_interval = base_interval
    silent_threshold = config.get("silent_threshold", 3)
    batch_size = config.get("batch_size", 5)
    
    logger.info(f"主循环启动: 起始轮次={current_round}, 间隔={current_interval}秒")
    print(f"[系统] 主循环已启动，消化间隔={config['digest_interval_minutes']}分钟")
    print(f"[系统] 按 Ctrl+C 终止系统\n")
    
    while running:
        try:
            # (a) 读取未消化素材
            batch = pool.read_undigested(batch_size)
            
            if not batch:
                # 缓冲池为空：不终止，仅等待
                logger.info(f"缓冲池为空，等待{current_interval}秒后重试")
                now_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{now_str}] 缓冲池为空，等待下一轮...")
                
                # 分段sleep，以便响应终止信号
                sleep_remaining = current_interval
                while sleep_remaining > 0 and running:
                    time.sleep(min(5, sleep_remaining))
                    sleep_remaining -= 5
                continue
            
            # (b) 调用消化引擎
            logger.info(f"=== 消化轮次 {current_round} 开始 ===")
            result = engine.digest(batch)
            
            # (c) 解析输出
            if result["is_silent"]:
                # 静默输出：计数+1
                silent_count += 1
                logger.info(f"静默计数: {silent_count}/{silent_threshold}")
                now_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{now_str}] 轮次{current_round}: [SILENT] "
                      f"(静默{silent_count}/{silent_threshold})")
                
                # (e) 静默阈值检查：连续静默则拉长间隔
                if silent_count >= silent_threshold:
                    old_interval = current_interval
                    # 间隔翻倍，最大120分钟
                    current_interval = min(current_interval * 2, 120 * 60)
                    logger.info(f"连续静默{silent_count}轮，"
                               f"间隔从{old_interval}秒调整为{current_interval}秒")
                    print(f"[系统] 连续静默，消化间隔拉长至"
                          f"{current_interval//60}分钟")
            else:
                # 有效漂移产物：推送并重置静默计数
                drift_path = push_drift(result["drift_content"], current_round)
                silent_count = 0  # 重置静默计数
                current_interval = base_interval  # 恢复正常间隔
                logger.info(f"漂移产物已保存: {drift_path}")
                logger.info(f"静默计数重置，间隔恢复至{base_interval}秒")
            
            # (d) 标记已消化
            id_list = [item_id for item_id, _ in batch]
            pool.mark_digested(id_list, current_round)
            
            # 推进轮次
            current_round += 1
            
            # (f) 等待下一个消化周期
            logger.info(f"等待{current_interval}秒后进入下一轮")
            sleep_remaining = current_interval
            while sleep_remaining > 0 and running:
                time.sleep(min(5, sleep_remaining))
                sleep_remaining -= 5
            
        except Exception as e:
            # 主循环异常保护：不崩溃，记录日志，继续运行
            logger.error(f"主循环异常: {e}", exc_info=True)
            print(f"[错误] 主循环异常: {e}，10秒后重试")
            time.sleep(10)
    
    # ============================================================
    # 8. 安全关闭
    # ============================================================
    final_stats = pool.get_stats()
    logger.info(f"系统关闭: 最终状态={final_stats}")
    print(f"\n[系统] 已安全关闭。最终状态: {final_stats}")
    print(f"[系统] 日志文件: {LOG_FILE}")


if __name__ == "__main__":
    # 使用 unbuffered 模式的提示
    # 实际运行时请用: python -u main.py
    main()
