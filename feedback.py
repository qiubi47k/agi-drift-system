#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feedback.py - 反馈通道模块
职责：异步接收用户输入，零解析零判断，直接入库
设计原则：用户输入 = 素材，无任何指令提取
"""

import threading
import logging

logger = logging.getLogger("drift.feedback")


class FeedbackListener:
    """异步反馈监听器
    
    在后台线程中运行，持续接收用户的键盘输入。
    所有输入原封不动作为素材写入缓冲池，
    不解析、不判断、不提取任何任务/指令。
    
    这是v3理论的核心机制之一：人类反馈与原始素材在消化时权重完全一致。
    """

    def __init__(self, buffer_pool):
        """初始化反馈监听器
        
        Args:
            buffer_pool: BufferPool实例，用于写入反馈素材
        """
        self.buffer_pool = buffer_pool
        self._running = False
        self._thread = None

    def _input_loop(self):
        """后台输入循环（在独立线程中运行）
        
        使用input()阻塞等待用户输入，每接收到一条非空输入
        就写入缓冲池。
        """
        while self._running:
            try:
                # 提示符使用特殊标记，方便区分系统输出和用户输入
                user_input = input("\n[反馈输入] >>> ")
                
                # 去除首尾空白，空输入忽略
                user_input = user_input.strip()
                if not user_input:
                    continue

                # 零解析零判断：直接作为素材写入
                item_id = self.buffer_pool.write_feedback(user_input)
                logger.info(f"反馈入库成功: {item_id[:8]}...")
                print(f"[系统] 反馈已收录（{len(user_input)}字）")

            except EOFError:
                # 标准输入关闭（如管道输入结束）
                logger.info("反馈通道：输入流结束")
                break
            except KeyboardInterrupt:
                # Ctrl+C由主进程处理
                break
            except Exception as e:
                logger.error(f"反馈输入异常: {e}")

    def start(self):
        """启动后台反馈监听线程
        
        线程设置为daemon模式，主进程退出时自动终止。
        """
        if self._running:
            logger.warning("反馈监听器已在运行中")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._input_loop, 
            daemon=True, 
            name="feedback_listener"
        )
        self._thread.start()
        logger.info("反馈通道已启动（后台线程）")
        print("[系统] 反馈通道已开启，你可以随时输入文本作为素材")

    def stop(self):
        """停止反馈监听"""
        self._running = False
        logger.info("反馈通道已停止")

    @property
    def is_running(self) -> bool:
        """监听器是否正在运行"""
        return self._running
