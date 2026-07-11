#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_loop.py v3 - 自动循环调度器（记忆+搜索+自适应）
核心能力：
1. 记忆系统：推演前检索历史记忆注入上下文，推演后存入记忆库（有积累的推演）
2. 网络搜索：对求知缺口搜索外部信息，注入真实新数据（扩大信息地盘）
3. 自适应调控：根据素材池水位动态调整搜索量，池子少猛搜、池子多收着搜
4. 推演摘要回灌：带"再审视"标签，保证上下文连续性
"""

import os
import sys
import json
import time
import logging
import requests
import re
from urllib.parse import quote_plus, quote
from datetime import datetime
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import DigestEngine
from buffer import BufferPool
from memory import CognitiveMemory
from goal_evolution import GoalEvolver
from grounding import GroundingLayer
from rule_selection import RuleSelector
from forgetting_detector import ForgettingDetector
try:
    from memory import extract_knowledge_from_text
except ImportError:
    extract_knowledge_from_text = None

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "auto_loop.log"),
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger("drift.auto_loop")

# 配置
MAX_ROUNDS = 1000  # 最大循环轮数（v5: 提升到1000，跑一晚上）
COOLDOWN_SECONDS = 15  # 每轮间隔（秒）（v2: 缩短）
MAX_DAILY_COST = 20.0  # 单日成本上限（元）
SEARCH_MAX_GAPS = 3  # 每轮最多搜索几个缺口
SEARCH_MAX_SNIPPETS = 3  # 每个缺口最多取几条搜索结果
SEARCH_TIMEOUT = 12  # 搜索超时（秒）

# 输出目录
DRIFT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "auto_drift_output")
PREDICT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "predict_output")
IMPROVEMENTS_DIR = os.path.join(os.path.dirname(__file__), "improvements")
SEARCH_LOG_DIR = os.path.join(os.path.dirname(__file__), "search_log")
VERIFY_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "verify_output")
STATS_FILE = os.path.join(os.path.dirname(__file__), "auto_loop_stats.json")

# 预演调度参数
RUMINATE_PER_CYCLE = 3   # 每轮周期内反刍次数
PREDICT_PER_CYCLE = 1    # 每轮周期内预演次数

os.makedirs(DRIFT_OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICT_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMPROVEMENTS_DIR, exist_ok=True)
os.makedirs(SEARCH_LOG_DIR, exist_ok=True)
os.makedirs(VERIFY_OUTPUT_DIR, exist_ok=True)


class WebSearcher:
    """轻量网络搜索模块，零外部依赖"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
    def search_ddg_api(self, query, max_results=3):
        """DuckDuckGo Instant Answer API"""
        try:
            url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
            resp = self.session.get(url, timeout=SEARCH_TIMEOUT)
            data = resp.json()
            
            results = []
            # Abstract（最佳结果）
            if data.get("AbstractText"):
                source = data.get("AbstractSource", "")
                results.append({
                    "text": data["AbstractText"][:500],
                    "source": source,
                    "url": data.get("AbstractURL", "")
                })
            
            # Related Topics
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "text": topic["Text"][:500],
                        "source": "DDG Related",
                        "url": topic.get("FirstURL", "")
                    })
                elif isinstance(topic, dict) and "Topics" in topic:
                    # 嵌套主题
                    for sub in topic["Topics"][:2]:
                        if "Text" in sub:
                            results.append({
                                "text": sub["Text"][:500],
                                "source": f"DDG {topic.get('Name', '')}",
                                "url": sub.get("FirstURL", "")
                            })
            
            return results
        except Exception as e:
            logger.debug(f"DDG API搜索失败: {e}")
            return []
    
    def search_ddg_html(self, query, max_results=3):
        """DuckDuckGo HTML Lite搜索（备用）"""
        try:
            url = "https://html.duckduckgo.com/html/"
            resp = self.session.post(url, data={"q": query}, timeout=SEARCH_TIMEOUT)
            
            results = []
            # 解析结果片段
            snippets = re.findall(
                r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                resp.text, re.DOTALL
            )
            # 解析标题+URL
            titles_urls = re.findall(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                resp.text, re.DOTALL
            )
            
            for i, snippet in enumerate(snippets[:max_results]):
                clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                title = ""
                url = ""
                if i < len(titles_urls):
                    url = titles_urls[i][0]
                    title = re.sub(r'<[^>]+>', '', titles_urls[i][1]).strip()
                
                if clean_snippet and len(clean_snippet) > 20:
                    results.append({
                        "text": f"{title}: {clean_snippet}" if title else clean_snippet,
                        "source": "DDG HTML",
                        "url": url
                    })
            
            return results
        except Exception as e:
            logger.debug(f"DDG HTML搜索失败: {e}")
            return []
    
    def search(self, query, max_results=3):
        """统一搜索入口，多策略容错"""
        # 策略1: DDG API
        results = self.search_ddg_api(query, max_results)
        if len(results) >= 2:
            return results
        
        # 策略2: DDG HTML
        html_results = self.search_ddg_html(query, max_results)
        results.extend(html_results)
        
        return results[:max_results]
    
    def search_multiple(self, queries, max_per_query=3):
        """批量搜索"""
        all_results = []
        for query in queries:
            results = self.search(query, max_per_query)
            all_results.extend(results)
            time.sleep(1)  # 避免请求过快
        return all_results


class SelfModifyExecutor:
    """自我修改执行器
    解析系统输出中的[SELF_MODIFY]块，执行修改并记录日志。
    安全约束：不能修改核心prompt，只能操作dynamic_rules和persistent_memory。
    """
    
    def __init__(self):
        self.mod_dir = os.path.join(os.path.dirname(__file__), "self_modifications")
        self.rules_path = os.path.join(self.mod_dir, "dynamic_rules.md")
        self.memory_path = os.path.join(self.mod_dir, "persistent_memory.json")
        self.log_path = os.path.join(self.mod_dir, "modification_log.json")
        os.makedirs(self.mod_dir, exist_ok=True)
    
    def parse_and_execute(self, output: str, round_num: int) -> list:
        """解析输出中的[SELF_MODIFY]块并执行"""
        results = []
        pattern = r'\[SELF_MODIFY\]\s*(.*?)\s*\[/SELF_MODIFY\]'
        matches = re.findall(pattern, output, re.DOTALL)
        
        if not matches:
            return results
        
        # 每轮最多3个修改
        matches = matches[:3]
        logger.info(f"★ 自我修改: 检测到{len(matches)}个修改请求")
        
        for i, block in enumerate(matches):
            try:
                result = self._execute_one(block, round_num)
                results.append(result)
                logger.info(f"  修改{i+1}: {result['action']} [{result['target']}] → {'✓' if result['success'] else '✗'} {result.get('message', '')}")
            except Exception as e:
                results.append({
                    "action": "unknown",
                    "target": "unknown",
                    "success": False,
                    "message": str(e)
                })
                logger.error(f"  修改{i+1}执行失败: {e}")
        
        return results
    
    def _execute_one(self, block: str, round_num: int) -> dict:
        """执行单个修改"""
        # 解析字段
        fields = {}
        current_key = None
        current_value_lines = []
        
        for line in block.strip().split('\n'):
            # 检查是否是新的key: value行
            kv_match = re.match(r'^(\w+):\s*(.*)', line)
            if kv_match:
                # 保存上一个多行字段
                if current_key and current_value_lines:
                    fields[current_key] = '\n'.join(current_value_lines).strip()
                
                current_key = kv_match.group(1)
                value = kv_match.group(2).strip()
                if value == '|':
                    # 多行值，开始收集
                    current_value_lines = []
                else:
                    fields[current_key] = value
                    current_key = None
                    current_value_lines = []
            else:
                # 多行值的续行
                if current_key is not None:
                    current_value_lines.append(line)
        
        # 保存最后一个字段
        if current_key and current_value_lines:
            fields[current_key] = '\n'.join(current_value_lines).strip()
        
        action = fields.get('action', '')
        target = fields.get('target', '')
        content = fields.get('content', '')
        reason = fields.get('reason', '')
        
        if not action or not target:
            return {"action": action, "target": target, "success": False, "message": "缺少action或target"}
        
        # 执行
        if action == 'add_rule':
            return self._add_rule(target, content, reason, round_num)
        elif action == 'remove_rule':
            return self._remove_rule(target, reason, round_num)
        elif action == 'write_memory':
            return self._write_memory(target, content, reason, round_num)
        elif action == 'update_memory':
            return self._update_memory(target, content, reason, round_num)
        else:
            return {"action": action, "target": target, "success": False, "message": f"未知action: {action}"}
    
    def _add_rule(self, target: str, content: str, reason: str, round_num: int) -> dict:
        """添加动态规则"""
        # 检查总长度
        current = ""
        if os.path.exists(self.rules_path):
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                current = f.read()
        
        new_rule = f"\n\n### 规则: {target}\n（R{round_num}添加，原因: {reason}）\n\n{content}\n"
        
        if len(current) + len(new_rule) > 8000:
            return {"action": "add_rule", "target": target, "success": False, "message": "动态规则总长度超限(8000字)"}
        
        with open(self.rules_path, 'w', encoding='utf-8') as f:
            f.write(current + new_rule)
        
        self._log("add_rule", target, content, reason, round_num, True)
        return {"action": "add_rule", "target": target, "success": True, "message": f"已添加规则'{target}'"}
    
    def _remove_rule(self, target: str, reason: str, round_num: int) -> dict:
        """删除动态规则"""
        if not os.path.exists(self.rules_path):
            return {"action": "remove_rule", "target": target, "success": False, "message": "无规则文件"}
        
        with open(self.rules_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按### 规则: target分割
        pattern = rf'\n*### 规则: {re.escape(target)}\n[\s\S]*?(?=\n### 规则:|\Z)'
        new_content = re.sub(pattern, '', content)
        
        if new_content == content:
            return {"action": "remove_rule", "target": target, "success": False, "message": f"未找到规则'{target}'"}
        
        with open(self.rules_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        self._log("remove_rule", target, "", reason, round_num, True)
        return {"action": "remove_rule", "target": target, "success": True, "message": f"已删除规则'{target}'"}
    
    def _write_memory(self, target: str, content: str, reason: str, round_num: int) -> dict:
        """写入持久记忆"""
        mem = {}
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r', encoding='utf-8') as f:
                    mem = json.load(f)
            except:
                mem = {}
        
        mem[target] = {
            "content": content,
            "reason": reason,
            "created_round": round_num,
            "updated_round": round_num
        }
        
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
        
        self._log("write_memory", target, content, reason, round_num, True)
        return {"action": "write_memory", "target": target, "success": True, "message": f"已写入记忆'{target}'"}
    
    def _update_memory(self, target: str, content: str, reason: str, round_num: int) -> dict:
        """更新持久记忆"""
        mem = {}
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r', encoding='utf-8') as f:
                    mem = json.load(f)
            except:
                mem = {}
        
        if target not in mem:
            # 不存在就创建
            mem[target] = {
                "content": content,
                "reason": reason,
                "created_round": round_num,
                "updated_round": round_num
            }
        else:
            mem[target]["content"] = content
            mem[target]["reason"] = reason
            mem[target]["updated_round"] = round_num
        
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
        
        self._log("update_memory", target, content, reason, round_num, True)
        return {"action": "update_memory", "target": target, "success": True, "message": f"已更新记忆'{target}'"}
    
    def _log(self, action: str, target: str, content: str, reason: str, round_num: int, success: bool):
        """记录修改日志"""
        log = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r', encoding='utf-8') as f:
                    log = json.load(f)
            except:
                log = []
        
        log.append({
            "round": round_num,
            "action": action,
            "target": target,
            "content_preview": content[:200] if content else "",
            "reason": reason,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, ensure_ascii=False, indent=2)


class AutoLoop:
    """自动循环调度器 v2"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.engine = DigestEngine(self.config)
        self.buffer = BufferPool()
        self.memory = CognitiveMemory()
        self.searcher = WebSearcher()
        self.goal_evolver = GoalEvolver()
        self.self_modify = SelfModifyExecutor()  # 自我修改执行器
        self.grounding = GroundingLayer()  # 接地层
        self.rule_selector = RuleSelector()  # 规则选择压力器
        self.forgetting_detector = ForgettingDetector()  # 遗忘波动预警算子
        self.consolidate_interval = 10  # 每10轮巩固一次
        
        self.stats = {
            "start_time": datetime.now().isoformat(),
            "rounds_completed": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "curiosity_gaps_extracted": 0,
            "self_maintenance_items": 0,
            "searches_performed": 0,
            "search_results_injected": 0,
            "stopped_reason": None
        }
        
        mem_stats = self.memory.get_stats()
        logger.info(f"自动循环v3初始化完成，最大轮数={MAX_ROUNDS}")
        logger.info(f"记忆网络: {mem_stats['total_concepts']}概念, {mem_stats['total_relations']}关系, {mem_stats['total_patterns']}模式")
    
    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        input_cost = prompt_tokens / 1_000_000 * 1.0
        output_cost = completion_tokens / 1_000_000 * 2.0
        return input_cost + output_cost
    
    def _save_round_output(self, round_num: int, result: dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"round_{round_num:03d}_{timestamp}.md"
        filepath = os.path.join(DRIFT_OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# 自动循环第{round_num}轮输出\n\n")
            f.write(f"生成时间：{datetime.now().isoformat()}\n\n")
            f.write("---\n\n")
            f.write(result["output"])
        
        logger.info(f"第{round_num}轮输出已保存: {filename}")
    
    def _save_self_maintenance(self, round_num: int, self_items: list):
        if not self_items:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"round_{round_num:03d}_{timestamp}.md"
        filepath = os.path.join(IMPROVEMENTS_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# 第{round_num}轮自我维护建议\n\n")
            f.write(f"生成时间：{datetime.now().isoformat()}\n\n---\n\n")
            for item in self_items:
                f.write(f"## [{item['type']}]\n\n{item['content']}\n\n")
        
        logger.info(f"第{round_num}轮自我维护建议已保存")
    
    def _save_code_results(self, round_num: int, code_results: list):
        """保存代码执行结果
        
        Args:
            round_num: 轮次
            code_results: 代码执行结果列表 [{"code": str, "output": str, "success": bool}]
        """
        if not code_results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"round_{round_num:03d}_code_{timestamp}.md"
        filepath = os.path.join(IMPROVEMENTS_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# 第{round_num}轮代码执行结果\n\n")
            f.write(f"生成时间：{datetime.now().isoformat()}\n\n---\n\n")
            for i, item in enumerate(code_results, 1):
                status = "✅ 成功" if item["success"] else "❌ 失败"
                f.write(f"## 代码块 {i} [{status}]\n\n")
                f.write(f"### 代码\n```python\n{item['code']}\n```\n\n")
                f.write(f"### 输出\n```\n{item['output']}\n```\n\n")
        
        logger.info(f"第{round_num}轮代码执行结果已保存")
    
    def _load_prev_code_results(self) -> str:
        """加载上一轮的代码执行结果
        
        Returns:
            上一轮代码执行结果的文本摘要，如果没有则返回空字符串
        """
        try:
            # 查找最新的代码执行结果文件
            code_files = [f for f in os.listdir(IMPROVEMENTS_DIR) if '_code_' in f and f.endswith('.md')]
            if not code_files:
                return ""
            
            # 按文件名排序，取最新的
            latest_file = sorted(code_files)[-1]
            filepath = os.path.join(IMPROVEMENTS_DIR, latest_file)
            
            # 读取文件内容（只取前2000字，避免上下文过长）
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取代码和输出部分
            if len(content) > 2000:
                content = content[:2000] + "\n... (内容过长，已截断)"
            
            return content
        except Exception as e:
            logger.warning(f"加载上一轮代码执行结果失败: {e}")
            return ""
    
    def _load_recent_meta_reactions(self, n: int = 3) -> str:
        """加载最近n条元反应记录
        
        这些元反应会被注入到每轮推演的上下文中，形成持续的自指回路。
        系统在每次推演前都会看到自己过去对自己输出的反应。
        
        Args:
            n: 加载最近几条元反应，默认3条
            
        Returns:
            格式化后的元反应文本，如果没有则返回空字符串
        """
        try:
            meta_dir = "meta_reactions"
            if not os.path.exists(meta_dir):
                return ""
            
            # 获取所有元反应文件，按轮次排序
            meta_files = [f for f in os.listdir(meta_dir) if f.startswith("meta_round_") and f.endswith(".md")]
            if not meta_files:
                return ""
            
            # 按文件名排序（meta_round_010.md < meta_round_020.md），取最近n条
            meta_files_sorted = sorted(meta_files)[-n:]
            
            # 构建上下文文本
            context_parts = []
            for meta_file in meta_files_sorted:
                filepath = os.path.join(meta_dir, meta_file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 提取轮次（从文件名）
                round_num = meta_file.replace("meta_round_", "").replace(".md", "")
                
                # 提取反应内容（---之后的部分）
                if "---\n\n" in content:
                    reaction_text = content.split("---\n\n", 1)[1].strip()
                    # 只取前500字，避免上下文过长
                    if len(reaction_text) > 500:
                        reaction_text = reaction_text[:500] + "..."
                    context_parts.append(f"[第{round_num}轮元反应] {reaction_text}")
            
            if not context_parts:
                return ""
            
            # 组装成上下文块
            result = "\n\n━━━━━━━━━━ [历史元反应：你对自己输出的反应] ━━━━━━━━━━\n"
            result += "\n\n".join(context_parts)
            result += "\n━━━━━━━━━━ 这些是你过去的反应。本轮推演时，考虑它们的影响。 ━━━━━━━━━━\n"
            
            return result
        except Exception as e:
            logger.warning(f"加载历史元反应失败: {e}")
            return ""
    
    def _apply_metacognition_feedback(self, round_num: int):
        """元认知反馈闭环（方案A：半自动）
        
        读取improvements/目录，统计高频建议，生成补丁等待人工确认。
        每20轮触发一次，或手动调用。
        
        Args:
            round_num: 当前轮次
        """
        logger.info(f"[元认知反馈] 第{round_num}轮触发元认知分析...")
        
        try:
            # 1. 读取所有自我维护建议文件
            improvement_files = [f for f in os.listdir(IMPROVEMENTS_DIR) 
                               if f.endswith('.md') and '_code_' not in f]
            
            if not improvement_files:
                logger.info("[元认知反馈] 无自我维护建议文件，跳过")
                return
            
            # 2. 统计高频关键词
            keyword_counts = {}
            keyword_examples = {}  # 保存每个关键词的示例
            
            for filename in improvement_files:
                filepath = os.path.join(IMPROVEMENTS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 提取建议类型（格式：## [类型]）
                    import re
                    section_matches = re.findall(r'##\s*\[([^\]]+)\]', content)
                    
                    # 同时提取具体建议内容中的关键词
                    # 格式：**建议标题**：具体内容
                    content_matches = re.findall(r'\*\*([^*]+)\*\*[：:](.+?)(?:\n\n|\Z)', content, re.DOTALL)
                    
                    # 统计建议类型
                    for section_type in section_matches:
                        keyword = section_type.strip()
                        if not keyword:
                            continue
                        
                        # 标准化关键词
                        keyword_lower = keyword.lower()
                        if 'prompt' in keyword_lower or '缺陷' in keyword_lower:
                            keyword = 'Prompt缺陷'
                        elif '调度' in keyword_lower or '流程' in keyword_lower:
                            keyword = '调度缺陷'
                        elif '能力' in keyword_lower or '边界' in keyword_lower:
                            keyword = '能力边界'
                        else:
                            continue
                            
                        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                    
                    # 从具体内容中提取更细粒度的关键词
                    for title, detail in content_matches:
                        title_lower = title.lower()
                        if '收敛' in title_lower or 'convergence' in title_lower:
                            keyword = '收敛性要求'
                        elif '互斥' in title_lower or '矛盾' in title_lower:
                            keyword = '互斥性检查'
                        elif '数学' in title_lower or '形式化' in title_lower:
                            keyword = '数学化要求'
                        elif '消解' in title_lower or '解决' in title_lower:
                            keyword = '矛盾消解'
                        elif '循环' in title_lower or '重复' in title_lower:
                            keyword = '循环检测'
                        elif '证伪' in title_lower or 'falsif' in title_lower:
                            keyword = '可证伪性要求'
                        elif '边界' in title_lower or '范围' in title_lower or '定义' in title_lower:
                            keyword = '变量边界定义'
                        else:
                            continue
                        
                        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                        
                        # 保存示例
                        if keyword not in keyword_examples:
                            keyword_examples[keyword] = f"**{title}**：{detail.strip()[:200]}"
                except Exception as e:
                    continue
            
            if not keyword_counts:
                logger.info("[元认知反馈] 未提取到有效关键词")
                return
            
            # 3. 排序并生成补丁
            sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            
            logger.info(f"[元认知反馈] 高频建议统计:")
            for kw, count in sorted_keywords:
                logger.info(f"  - {kw}: {count}次")
            
            # 4. 生成补丁文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            patch_filename = f"pending_patch_{timestamp}.md"
            patch_filepath = os.path.join(IMPROVEMENTS_DIR, patch_filename)
            
            with open(patch_filepath, 'w', encoding='utf-8') as f:
                f.write(f"# 元认知补丁（待确认）\n\n")
                f.write(f"**生成时间**：{datetime.now().isoformat()}\n")
                f.write(f"**触发轮次**：第{round_num}轮\n")
                f.write(f"**分析文件数**：{len(improvement_files)}\n\n")
                f.write("---\n\n")
                f.write("## 高频建议统计\n\n")
                
                for kw, count in sorted_keywords:
                    f.write(f"### {kw}（{count}次）\n\n")
                    if kw in keyword_examples:
                        f.write(f"**示例**：{keyword_examples[kw]}\n\n")
                    
                    # 生成具体补丁建议
                    if kw == '收敛性要求':
                        f.write("**补丁建议**：在DIGEST_SYSTEM_PROMPT的'收敛约束'区域添加：\n")
                        f.write("```\n每次进化必须解决至少一个已知矛盾点或坍缩问题，避免纯发散。\n```\n\n")
                    elif kw == '互斥性检查':
                        f.write("**补丁建议**：已应用（路径三：互斥性检查）\n\n")
                    elif kw == '数学化要求':
                        f.write("**补丁建议**：已应用（每轮至少一个产物尝试数学形式化）\n\n")
                    elif kw == '矛盾消解':
                        f.write("**补丁建议**：已应用（每个矛盾点必须附带消解方案）\n\n")
                    elif kw == '循环检测':
                        f.write("**补丁建议**：在DIGEST_SYSTEM_PROMPT添加：\n")
                        f.write("```\n推演前检查是否已解决过该问题，避免重复推演。\n```\n\n")
                    elif kw == '可证伪性要求':
                        f.write("**补丁建议**：在DIGEST_SYSTEM_PROMPT添加：\n")
                        f.write("```\n每个假设必须给出至少一种证伪条件。\n```\n\n")
                    elif kw == '变量边界定义':
                        f.write("**补丁建议**：在DIGEST_SYSTEM_PROMPT添加：\n")
                        f.write("```\n所有数学符号必须定义取值范围和边界条件。\n```\n\n")
                
                f.write("---\n\n")
                f.write("## 操作指南\n\n")
                f.write("1. 审查上述补丁建议\n")
                f.write("2. 如需应用，手动编辑 `engine.py` 中的 `DIGEST_SYSTEM_PROMPT`\n")
                f.write("3. 应用后重启进程：`kill -HUP <pid>` 或重启 `auto_loop.py`\n")
                f.write("4. 如需回滚，删除本文件即可\n")
            
            logger.info(f"[元认知反馈] 补丁已生成: {patch_filename}")
            logger.info(f"[元认知反馈] 请审查后决定是否应用")
            
        except Exception as e:
            logger.error(f"[元认知反馈] 分析失败: {e}")
    
    def _build_search_queries(self, gaps: list) -> list:
        """从求知缺口构建搜索查询
        
        策略：提取核心关键词，而非整句，提高搜索命中率
        """
        queries = []
        # 停用词（搜索时去掉）
        stop_words = {'的', '了', '是', '在', '和', '与', '中', '上', '下', '的', '着', 
                      '到', '被', '把', '让', '从', '对', '等', '而', '及', '其', '这', '那',
                      '如何', '怎样', '什么', '为什么', '是否', '能否', '可以', '需要',
                      '问题', '机制', '现象', '过程', '结果', '方面', '情况', '关系'}
        
        for gap in gaps[:SEARCH_MAX_GAPS]:
            content = gap["content"]
            
            # 去标点
            text = re.sub(r'[，。！？、；：""''（）\[\]【】\-\*\#\·]+', ' ', content)
            
            # 提取中文关键词（2-6字的词组）
            cn_words = re.findall(r'[\u4e00-\u9fff]{2,6}', text)
            # 提取英文关键词
            en_words = re.findall(r'[a-zA-Z][a-zA-Z\s\-]{2,20}', text)
            
            # 过滤停用词，取最核心的
            keywords = []
            for w in cn_words:
                if w not in stop_words and len(w) >= 2:
                    keywords.append(w)
            for w in en_words:
                w = w.strip()
                if len(w) > 2:
                    keywords.append(w)
            
            # 取前3个核心词组成查询
            if keywords:
                query = ' '.join(keywords[:3])
                queries.append(query)
            elif len(content) > 5:
                # 兜底：直接用前30字符
                queries.append(content[:50])
        
        return queries
    
    def _calc_search_budget(self, pool_undigested: int) -> tuple:
        """根据素材池状态动态调整搜索量
        
        核心逻辑：池子少→猛搜扩大地盘，池子多→收着搜避免膨胀
        
        Returns:
            (max_gaps_to_search, max_snippets_per_query)
        """
        if pool_undigested <= 3:
            # 池子快空了，全力搜索
            return SEARCH_MAX_GAPS, SEARCH_MAX_SNIPPETS
        elif pool_undigested <= 10:
            # 池子健康，正常搜索
            return min(SEARCH_MAX_GAPS, 3), SEARCH_MAX_SNIPPETS
        elif pool_undigested <= 25:
            # 池子偏多，减少搜索
            return 2, 2
        elif pool_undigested <= 50:
            # 池子较多，少量搜索
            return 1, 2
        else:
            # 池子很满，不搜索，纯消化存量
            return 0, 0
    
    def _search_and_collect(self, gaps: list, pool_undigested: int) -> list:
        """为求知缺口搜索外部信息（自适应搜索量）
        
        Returns:
            [{"text": str, "source": str, "url": str, "gap_type": str}, ...]
        """
        max_gaps, max_snippets = self._calc_search_budget(pool_undigested)
        
        if max_gaps == 0:
            logger.info(f"素材池充足({pool_undigested}条待消化)，本轮跳过搜索，优先消化存量")
            return []
        
        queries = self._build_search_queries(gaps[:max_gaps])
        if not queries:
            return []
        
        logger.info(f"池子{pool_undigested}条待消化，搜索预算: {len(queries)}个查询×{max_snippets}条/查询")
        
        all_results = []
        gap_types = [g["type"] for g in gaps[:max_gaps]]
        
        for i, query in enumerate(queries):
            results = self.searcher.search(query, max_snippets)
            gap_type = gap_types[i] if i < len(gap_types) else "未知"
            
            for r in results:
                r["gap_type"] = gap_type
                r["query"] = query
                all_results.append(r)
            
            if results:
                logger.info(f"  查询'{query[:40]}...' → {len(results)}条结果")
            else:
                logger.info(f"  查询'{query[:40]}...' → 无结果")
            
            time.sleep(1)  # 搜索间隔
        
        self.stats["searches_performed"] += len(queries)
        return all_results
    
    def _inject_materials(self, gaps: list, search_results: list, drift_summary: str, round_num: int) -> int:
        """统一注入所有新素材到池子
        
        注入策略：
        1. 求知缺口本身（驱动继续探索）
        2. 外部搜索结果（真正的新信息！扩大地盘）
        3. 推演产物摘要（带"再审视"标签，保持连续性）
        
        Returns:
            注入总数
        """
        injected = 0
        
        # 1. 求知缺口
        for gap in gaps:
            text = f"[求知缺口-第{round_num}轮-{gap['type']}]\n{gap['content']}"
            self.buffer.write_raw(text)
            injected += 1
        
        # 2. 外部搜索结果（关键！扩大信息地盘）
        for result in search_results:
            text = f"[外部搜索-{result['gap_type']}]\n"
            text += f"查询: {result.get('query', '')}\n"
            text += f"来源: {result['source']}\n"
            if result.get('url'):
                text += f"链接: {result['url']}\n"
            text += f"\n{result['text']}"
            self.buffer.write_raw(text)
            injected += 1
        
        self.stats["search_results_injected"] += len(search_results)
        
        # 3. 推演产物摘要（带再审视标签）
        if drift_summary and len(drift_summary) > 20:
            text = f"[历史推演再审视-第{round_num}轮]\n"
            text += f"请从新的角度、新的路径重新分析以下发现，不要重复上一轮的思路：\n\n"
            text += drift_summary[:800]
            self.buffer.write_raw(text)
            injected += 1
        
        logger.info(f"第{round_num}轮注入素材: {injected}条 (缺口{len(gaps)} + 搜索{len(search_results)} + 推演摘要{1 if drift_summary else 0})")
        return injected
    
    def _save_math_verification(self, round_num: int, verify_result: dict):
        """保存审稿结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"verify_round_{round_num:03d}_{timestamp}.md"
        filepath = os.path.join(VERIFY_OUTPUT_DIR, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# 审稿报告 第{round_num}轮\n\n")
            f.write(f"生成时间：{datetime.now().isoformat()}\n\n")
            
            # 审稿摘要
            verdict = verify_result.get("verdict", "N/A")
            avg = verify_result.get("avg_score", 0)
            scores = verify_result.get("scores", [])
            f.write(f"**最终裁定：{verdict}**\n\n")
            f.write(f"**平均分：{avg}/5** | 各条评分：{scores}\n\n---\n\n")

            # 逐条审稿（直接用原始输出的逐条部分）
            raw = verify_result.get("raw", "")
            
            # 分类展示
            if verify_result.get("verified"):
                f.write("## 已知换皮（2分以下）\n\n")
                for item in verify_result["verified"]:
                    f.write(f"- {item}\n")
                f.write("\n")

            if verify_result.get("novel_formal"):
                f.write("## 真正新颖可形式化（4-5分）\n\n")
                for item in verify_result["novel_formal"]:
                    f.write(f"- {item}\n")
                f.write("\n")

            if verify_result.get("pre_math"):
                f.write("## 前数学/有直觉但无法形式化（3分）\n\n")
                for item in verify_result["pre_math"]:
                    f.write(f"- {item}\n")
                f.write("\n")

            if verify_result.get("wordplay"):
                f.write("## 废话/术语装饰/重复（1-2分）\n\n")
                for item in verify_result["wordplay"]:
                    f.write(f"- {item}\n")
                f.write("\n")

            # 审稿总结
            if verify_result.get("summary"):
                f.write("## 审稿人总结\n\n")
                f.write(verify_result["summary"])
                f.write("\n\n")

            # 原始完整审稿输出
            if raw:
                f.write("---\n\n## 原始审稿全文\n\n")
                f.write(raw)

        logger.info(f"审稿结果已保存: {filename}")

    def _save_search_log(self, round_num: int, search_results: list):
        """保存搜索日志"""
        if not search_results:
            return
        
        filename = f"search_round_{round_num:03d}.json"
        filepath = os.path.join(SEARCH_LOG_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "round": round_num,
                "timestamp": datetime.now().isoformat(),
                "results_count": len(search_results),
                "results": search_results
            }, f, ensure_ascii=False, indent=2)
    
    def _check_safety(self) -> tuple:
        if self.stats["rounds_completed"] >= MAX_ROUNDS:
            return False, f"达到最大轮数上限({MAX_ROUNDS})"
        if self.stats["estimated_cost"] >= MAX_DAILY_COST:
            return False, f"达到单日成本上限({MAX_DAILY_COST}元)"
        return True, None
    
    def _load_memory_context(self):
        """加载记忆网络上下文（概念+关系），供反刍和预演共用"""
        concepts_file = os.path.join(os.path.dirname(__file__), "memory", "concepts.json")
        relations_file = os.path.join(os.path.dirname(__file__), "memory", "relations.json")
        
        try:
            with open(concepts_file, 'r', encoding='utf-8') as f:
                all_concepts = json.load(f)
            with open(relations_file, 'r', encoding='utf-8') as f:
                all_relations = json.load(f)
            
            sorted_concepts = sorted(
                all_concepts.items(),
                key=lambda x: x[1].get("weight", 0),
                reverse=True
            )[:50]
            
            lines = ["[认知网络-核心概念（按权重排序）]"]
            for name, concept in sorted_concepts:
                weight = concept.get("weight", 0)
                desc = concept.get("description", "")[:250]
                lines.append(f"• {name}(权重{weight:.2f}): {desc}")
            
            sorted_rels = sorted(all_relations, key=lambda x: x.get("weight", 0), reverse=True)[:25]
            lines.append("\n[核心关系]")
            for rel in sorted_rels:
                from_c = rel.get("from_concept", "")
                to_c = rel.get("to_concept", "")
                rel_type = rel.get("type", "")
                weight = rel.get("weight", 0)
                desc = rel.get("description", "")[:300]
                lines.append(f"• {from_c} → {rel_type} → {to_c}(强度{weight:.2f}): {desc}")
            
            return "\n".join(lines), sorted_concepts, sorted_rels
        except Exception as e:
            logger.error(f"读取记忆失败: {e}")
            return "", [], []
    
    def _load_patterns(self):
        """加载已有模式列表"""
        patterns_file = os.path.join(os.path.dirname(__file__), "memory", "patterns.json")
        try:
            with open(patterns_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []

    def _load_theories(self):
        """加载已有理论列表"""
        theories_file = os.path.join(os.path.dirname(__file__), "memory", "theories.json")
        try:
            with open(theories_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    
    def _collect_all_gaps(self):
        """从历史推演产物中收集所有求知缺口"""
        gaps = []
        if not os.path.exists(DRIFT_OUTPUT_DIR):
            return gaps
        for fname in sorted(os.listdir(DRIFT_OUTPUT_DIR)):
            if not fname.endswith('.md'):
                continue
            fpath = os.path.join(DRIFT_OUTPUT_DIR, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 提取求知缺口段落
                if '求知缺口' in content:
                    gap_section = content.split('## 求知缺口')[-1] if '## 求知缺口' in content else ""
                    # 提取每条缺口
                    gap_items = re.findall(r'\d+\.\s*\[([^\]]+)\]\s*(.+?)(?=\n\d+\.|\n##|$)', gap_section, re.DOTALL)
                    for gap_type, gap_text in gap_items:
                        gaps.append({"type": gap_type.strip(), "content": gap_text.strip()[:200]})
            except:
                continue
        return gaps
    
    def _save_predict_output(self, round_num: int, result: dict):
        """保存预演输出到专用目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predict_{round_num:03d}_{timestamp}.md"
        filepath = os.path.join(PREDICT_OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# 预演演算 第{round_num}轮\n\n")
            f.write(f"生成时间：{datetime.now().isoformat()}\n\n---\n\n")
            f.write(result["output"])
        
        logger.info(f"预演输出已保存: {filename}")
        
        # 同时把预演产物注入素材池，作为下一轮反刍的优先素材
        summary = result.get("parsed", {}).get("summary", "") if result.get("parsed") else ""
        drift = result.get("drift_content", "")
        if drift:
            text = f"[预演产物-第{round_num}轮]\n" + drift[:1200]
            self.buffer.write_raw(text)
            logger.info(f"预演产物已注入素材池（{len(text)}字）")
    
    def run(self):
        init_mem = self.memory.get_stats()
        logger.info("=" * 60)
        logger.info("启动自动循环 v5（反刍+预演交替）")
        logger.info(f"调度: {RUMINATE_PER_CYCLE}轮反刍 + {PREDICT_PER_CYCLE}轮预演 循环")
        logger.info(f"最大轮数={MAX_ROUNDS}, 成本上限={MAX_DAILY_COST}元")
        logger.info(f"认知记忆: {init_mem['total_concepts']}概念, {init_mem['total_relations']}关系, {init_mem['total_patterns']}模式")
        logger.info("=" * 60)
        
        round_num = 0
        cycle_position = 0  # 在当前周期内的位置：0,1,2=反刍，3=预演
        prev_output = None  # 上一轮输出文本，用于反重复
        prev_outputs_for_evolution = []  # 用于目标演化模块的历史输出
        
        while True:
            safe, reason = self._check_safety()
            if not safe:
                self.stats["stopped_reason"] = reason
                logger.warning(f"触发安全停止: {reason}")
                break
            
            round_num += 1
            is_predict_round = (cycle_position >= RUMINATE_PER_CYCLE)
            cycle_position += 1
            
            # 完成一个周期后重置
            if cycle_position >= RUMINATE_PER_CYCLE + PREDICT_PER_CYCLE:
                cycle_position = 0
            
            logger.info(f"\n{'='*60}")
            mode_label = "预演演算" if is_predict_round else "反刍推演"
            logger.info(f"开始第{round_num}轮推演 [{mode_label}] (周期位置{cycle_position+1}/{RUMINATE_PER_CYCLE+PREDICT_PER_CYCLE})")
            
            # 检查素材池状态
            pool_stats = self.buffer.get_stats()
            logger.info(f"素材池状态: 总计{pool_stats['total']}条, 未消化{pool_stats['undigested']}条")
            
            # 获取素材批次
            batch = self.buffer.read_undigested(batch_size=5)
            
            # 判断是否进入反刍模式（素材池空时）
            rumination_mode = False
            if not batch:
                mem_stats = self.memory.get_stats()
                if mem_stats['total_concepts'] >= 10:
                    rumination_mode = True
                else:
                    logger.warning("素材池为空且记忆不足，无法继续")
                    self.stats["stopped_reason"] = "素材池为空且记忆不足"
                    break
            else:
                rumination_mode = False
            
            # ===== 预演轮次 =====
            if is_predict_round and rumination_mode:
                # 预演需要记忆上下文+模式+缺口+理论
                logger.info("预演轮次：加载模式+缺口+理论，执行因果外推")
                memory_context, concepts, relations = self._load_memory_context()
                patterns = self._load_patterns()
                theories = self._load_theories()
                all_gaps = self._collect_all_gaps()
                
                if not memory_context:
                    logger.warning("预演轮次：记忆上下文为空，降级为反刍")
                    is_predict_round = False
                else:
                    logger.info(f"预演上下文：{len(concepts)}概念, {len(patterns)}模式, {len(theories)}理论, {len(all_gaps)}缺口")
                    result = self.engine.predict(memory_context, patterns=patterns, theories=theories, gaps=all_gaps)
                    
                    if result["is_silent"]:
                        logger.warning(f"第{round_num}轮预演: SILENT，降级为反刍")
                        is_predict_round = False
                    else:
                        # 保存预演输出（专用目录）
                        self._save_predict_output(round_num, result)
                        
                        # 更新统计
                        self.stats["rounds_completed"] = round_num
                        output_chars = len(result["output"])
                        round_cost = self._estimate_cost(int(output_chars * 0.6), int(output_chars * 0.6))
                        self.stats["estimated_cost"] += round_cost
                        
                        mem_stats = self.memory.get_stats()
                        logger.info(f"第{round_num}轮[预演]完成: {output_chars}字, 成本{round_cost:.4f}元")
                        logger.info(f"  累计: {self.stats['rounds_completed']}轮, 记忆{mem_stats['total_concepts']}概念/{mem_stats['total_relations']}关系/{mem_stats['total_patterns']}模式, {self.stats['estimated_cost']:.4f}元")
                        
                        # ===== 预演轮次也要执行接地层处理和遗忘检测 =====
                        predict_output = result.get("output", "")
                        if predict_output and len(predict_output) > 100:
                            try:
                                predictions = self.grounding.extract_predictions(predict_output, round_num)
                                if predictions:
                                    self.grounding.add_predictions(predictions)
                                    logger.info(f"★ 接地层(预演): 提取{len(predictions)}个可验证预测")
                                
                                if round_num % 10 == 0:
                                    g_stats = self.grounding.get_stats()
                                    logger.info(f"★ 接地层统计(预演): 总预测{g_stats['total_predictions']}, "
                                               f"待验证{g_stats['pending']}, 已证实{g_stats['verified']}, 已证伪{g_stats['refuted']}")
                                    pending = self.grounding.get_pending_predictions()
                                    if pending:
                                        to_verify = pending[-10:]
                                        verdicts = self.grounding.verify_predictions_batch(to_verify, round_num)
                                        if verdicts:
                                            self.grounding.apply_verifications(verdicts)
                            except Exception as e:
                                logger.warning(f"接地层处理(预演)失败: {e}")
                        
                        # 遗忘检测
                        if predict_output and len(predict_output) > 100 and round_num % 3 == 0:
                            try:
                                forgetting_result = self.forgetting_detector.scan_round(round_num)
                                if forgetting_result.get("should_wake"):
                                    logger.info(f"★ 遗忘唤醒触发(预演)！衰退概念组: {[a['group'] for a in forgetting_result['alerts']]}")
                            except Exception as e:
                                logger.warning(f"遗忘检测(预演)失败: {e}")
                        
                        if self.stats["rounds_completed"] < MAX_ROUNDS:
                            logger.info(f"冷却{COOLDOWN_SECONDS}秒...")
                            time.sleep(COOLDOWN_SECONDS)
                        continue
            
            # ===== 执行推演：反刍模式 vs 标准消化 =====
            if rumination_mode:
                mem_stats = self.memory.get_stats()
                logger.info(f"反刍模式：从记忆网络提取核心概念（{mem_stats['total_concepts']}概念/{mem_stats['total_relations']}关系）")
                
                memory_context, concepts, relations = self._load_memory_context()
                
                if not memory_context:
                    logger.warning("反刍模式：记忆上下文为空，跳过")
                    self.stats["stopped_reason"] = f"第{round_num}轮反刍上下文为空"
                    break
                
                # 注入上一轮代码执行结果（如果有）
                prev_code_results = self._load_prev_code_results()
                if prev_code_results:
                    memory_context += f"\n\n[上一轮代码执行结果]\n{prev_code_results}"
                    logger.info(f"反刍模式：注入上一轮代码执行结果")
                
                # 注入历史元反应（持续自指回路）
                meta_reactions_context = self._load_recent_meta_reactions(n=3)
                if meta_reactions_context:
                    memory_context += meta_reactions_context
                    logger.info(f"反刍模式：注入历史元反应（持续自指回路）")
                
                # 获取深度信息用于深度强制
                depth_info = self.memory.get_depth_info()
                logger.info(f"深度追踪: max_depth={depth_info['max_depth']}, 分布={depth_info['depth_distribution']}")
                
                result = self.engine.ruminate(memory_context, depth_info=depth_info, prev_output=prev_output)
                batch = []
            else:
                # 标准消化模式
                batch_ids = [item_id for item_id, _ in batch]
                self.buffer.mark_digested(batch_ids, round_num)
                
                batch_text = " ".join(text for _, text in batch)[:500]
                memory_context = self.memory.get_context(batch_text, max_length=10000)
                if memory_context:
                    batch.append(("cognitive_memory", memory_context))
                    logger.info(f"注入认知记忆上下文")
                
                # 注入历史元反应（持续自指回路）
                meta_reactions_context = self._load_recent_meta_reactions(n=3)
                if meta_reactions_context:
                    batch.append(("meta_reactions", meta_reactions_context))
                    logger.info(f"注入历史元反应（持续自指回路）")
                
                result = self.engine.digest(batch)
            
            if result["is_silent"]:
                logger.warning(f"第{round_num}轮: SILENT")
                self.stats["stopped_reason"] = f"第{round_num}轮SILENT"
                break
            
            # 保存推演输出
            self._save_round_output(round_num, result)
            
            # ===== 处理代码执行结果 =====
            code_results = result.get("code_results", [])
            if code_results:
                logger.info(f"★ 行动力: {len(code_results)}个代码块已执行")
                # 将代码执行结果保存到文件，并在下一轮注入
                self._save_code_results(round_num, code_results)
            
            # 保存当前输出用于下一轮反重复
            prev_output = result.get("output", "")[:2000]  # 只保留前2000字作为反重复参考
            
            # ===== 数学验证（每10轮执行一次，控制成本） =====
            if round_num % 10 == 0 and not result["is_silent"] and result.get("output", ""):
                verify_result = self.engine.verify_mathematically(
                    rumination_output=result.get("output", ""),
                    memory_context=memory_context if rumination_mode else ""
                )
                self._save_math_verification(round_num, verify_result)
                
                # 统计验证结果
                n_known = len(verify_result.get("verified", []))
                n_novel = len(verify_result.get("novel_formal", []))
                n_pre = len(verify_result.get("pre_math", []))
                n_bs = len(verify_result.get("wordplay", []))
                avg = verify_result.get("avg_score", 0)
                verdict = verify_result.get("verdict", "?")
                logger.info(f"★ 审稿: 均分{avg}/5, 裁定{verdict} | 新颖{n_novel}条, 前数学{n_pre}条, 已知换皮{n_known}条, 废话{n_bs}条")
            
            # 自我审视：筛选有用信息进入记忆
            drift_output = result.get("output", "")
            if drift_output and len(drift_output) > 100:
                memory_context_for_review = self.memory.get_context(drift_output[:500], max_length=3000)
                review_result = self.engine.self_review(drift_output, memory_context_for_review)
                
                # 保存审视结果
                keep_count = len(review_result.get("keep", []))
                discard_count = len(review_result.get("discard", []))
                logger.info(f"★ 审视: 保留{keep_count}条, 丢弃{discard_count}条")
                
                # 只保留"有用"的部分进入记忆
                if review_result.get("keep"):
                    filtered_text = "\n\n".join([item["insight"] for item in review_result["keep"]])
                    if extract_knowledge_from_text is not None:
                        extracted = extract_knowledge_from_text(filtered_text, self.config)
                        if extracted.get("concepts") or extracted.get("relations"):
                            self.memory.store_extraction(extracted, round_num)
                            logger.info(f"  存储: {len(extracted.get('concepts', []))}概念, {len(extracted.get('relations', []))}关系")
                else:
                    logger.info(f"  无有用洞察，跳过记忆存储")
            else:
                # 输出太短，直接存储
                if extract_knowledge_from_text is not None:
                    extracted = extract_knowledge_from_text(drift_output, self.config)
                    if extracted.get("concepts") or extracted.get("relations"):
                        self.memory.store_extraction(extracted, round_num)
            
            if round_num % self.consolidate_interval == 0:
                self.memory.consolidate(round_num)
                self.extract_structural_patterns(round_num)
                
                # 验证所有理论的接地状态
                if hasattr(self.memory, 'world_manager') and self.memory.world_manager:
                    grounding_results = self.memory.world_manager.validate_all_theories(
                        self.memory.theories, self.memory.concepts
                    )
                    grounded_count = sum(1 for r in grounding_results if r.get("groundable"))
                    logger.info(f"接地验证: {grounded_count}/{len(grounding_results)}个理论可接地到模拟世界")
                    # 保存接地状态
                    self.memory.world_manager.save_grounding()
            
            # 解析结构化数据
            parsed = result.get("parsed")
            gaps = []
            self_items = []
            drift_summary = ""
            
            if parsed:
                gaps = parsed.get("curiosity_gaps", [])
                self_items = parsed.get("self_maintenance", [])
                drift_summary = parsed.get("summary", "")
                
                self.stats["curiosity_gaps_extracted"] += len(gaps)
                self.stats["self_maintenance_items"] += len(self_items)
                
                logger.info(f"提取: {len(gaps)}条求知缺口, {len(self_items)}条自我维护")
                self._save_self_maintenance(round_num, self_items)
                
                # 自适应搜索
                pool_status = self.buffer.get_stats()
                pool_undigested = pool_status["undigested"]
                
                search_results = []
                if gaps:
                    search_results = self._search_and_collect(gaps, pool_undigested)
                    if search_results:
                        logger.info(f"搜索完成: 获取{len(search_results)}条外部信息")
                        self._save_search_log(round_num, search_results)
                
                self._inject_materials(gaps, search_results, drift_summary, round_num)
            
            # 更新统计
            self.stats["rounds_completed"] = round_num
            
            output_chars = len(result["output"])
            input_chars = sum(len(text) for _, text in batch) if batch else 1000
            estimated_prompt_tokens = int(input_chars * 0.6)
            estimated_completion_tokens = int(output_chars * 0.6)
            round_cost = self._estimate_cost(estimated_prompt_tokens, estimated_completion_tokens)
            
            self.stats["total_tokens"] += estimated_prompt_tokens + estimated_completion_tokens
            self.stats["estimated_cost"] += round_cost
            
            pool_after = self.buffer.get_stats()
            mem_stats = self.memory.get_stats()
            logger.info(f"第{round_num}轮[{mode_label}]完成:")
            logger.info(f"  输出: {output_chars}字, 成本: {round_cost:.4f}元")
            logger.info(f"  累计: {self.stats['rounds_completed']}轮, {self.stats['estimated_cost']:.4f}元, 记忆{mem_stats['total_concepts']}概念/{mem_stats['total_relations']}关系/{mem_stats['total_patterns']}模式({mem_stats['storage_size_mb']}MB)")
            
            # ===== 自我修改：解析输出中的[SELF_MODIFY]块并执行 =====
            current_output_for_modify = result.get("output", "")
            if current_output_for_modify and "[SELF_MODIFY]" in current_output_for_modify:
                modify_results = self.self_modify.parse_and_execute(current_output_for_modify, round_num)
                if modify_results:
                    success_count = sum(1 for r in modify_results if r.get("success"))
                    logger.info(f"★ 自我修改完成: {success_count}/{len(modify_results)}个修改成功")
                    # 保存修改结果到输出文件
                    modify_log_file = os.path.join(
                        os.path.dirname(__file__), 
                        "self_modifications", 
                        f"modify_round_{round_num:03d}.md"
                    )
                    with open(modify_log_file, 'w', encoding='utf-8') as f:
                        f.write(f"# 自我修改记录 - 第{round_num}轮\n\n")
                        f.write(f"执行时间：{datetime.now().isoformat()}\n\n")
                        for i, r in enumerate(modify_results):
                            status = "✓ 成功" if r.get("success") else "✗ 失败"
                            f.write(f"## 修改{i+1}: {status}\n")
                            f.write(f"- 操作: {r.get('action')}\n")
                            f.write(f"- 目标: {r.get('target')}\n")
                            f.write(f"- 结果: {r.get('message')}\n\n")
            
            # ===== 目标演化：评估本轮输出并更新路径权重 =====
            current_output = result.get("output", "")
            if current_output and len(current_output) > 100:
                evolution_result = self.goal_evolver.update_weights(
                    current_output, 
                    prev_outputs_for_evolution[-10:]  # 用最近10轮作为历史
                )
                prev_outputs_for_evolution.append(current_output[:2000])  # 只保留前2000字
                
                # 每10轮输出一次演化统计
                if round_num % 10 == 0:
                    evo_stats = self.goal_evolver.get_stats()
                    logger.info(f"★ 目标演化: 已评估{evo_stats['rounds_evaluated']}轮, "
                               f"平均价值分{evo_stats['avg_value_score']:.3f}, "
                               f"权重分化度{evo_stats['weight_divergence']:.3f}")
                    if evo_stats['weight_divergence'] > 0.3:
                        logger.info(f"  权重已出现明显分化: {evo_stats['current_weights']}")
            
            # ===== 接地层：提取可验证预测 =====
            if current_output and len(current_output) > 100:
                try:
                    predictions = self.grounding.extract_predictions(current_output, round_num)
                    if predictions:
                        self.grounding.add_predictions(predictions)
                        logger.info(f"★ 接地层: 提取{len(predictions)}个可验证预测")
                    
                    # 每10轮输出接地层统计 + 验证待验证预测
                    if round_num % 10 == 0:
                        g_stats = self.grounding.get_stats()
                        logger.info(f"★ 接地层统计: 总预测{g_stats['total_predictions']}, "
                                   f"待验证{g_stats['pending']}, "
                                   f"已证实{g_stats['verified']}, "
                                   f"已证伪{g_stats['refuted']}")
                        
                        # 验证pending预测
                        pending = self.grounding.get_pending_predictions()
                        if pending:
                            # 取最近10条pending预测做验证
                            to_verify = pending[-10:]
                            verdicts = self.grounding.verify_predictions_batch(to_verify, round_num)
                            if verdicts:
                                self.grounding.apply_verifications(verdicts)
                except Exception as e:
                    logger.warning(f"接地层处理失败: {e}")
            
            # ===== 遗忘波动检测：每3轮扫描核心概念密度 =====
            if current_output and len(current_output) > 100:
                try:
                    if round_num % 3 == 0:
                        forgetting_result = self.forgetting_detector.scan_round(round_num)
                        if forgetting_result.get("should_wake"):
                            logger.info(f"★ 遗忘唤醒触发！衰退概念组: {[a['group'] for a in forgetting_result['alerts']]}")
                            # 唤醒内容已通过 forgetting_state.json 传递给 engine 的 _get_system_prompt
                except Exception as e:
                    logger.warning(f"遗忘检测处理失败: {e}")
            
            # ===== 规则选择压力：追踪规则触发，定期清理 =====
            try:
                # 获取当前所有规则名
                rules_path = os.path.join(os.path.dirname(__file__), "self_modifications", "dynamic_rules.md")
                if os.path.exists(rules_path):
                    with open(rules_path, 'r', encoding='utf-8') as f:
                        rules_text = f.read()
                    rule_names = re.findall(r'###\s*规则:\s*(.+?)(?:\n|$)', rules_text)
                    
                    # 检查本轮输出中是否引用了各规则
                    if current_output and rule_names:
                        triggered_any = False
                        for rname in rule_names:
                            if rname.strip()[:10] in current_output:
                                self.rule_selector.track_rule_trigger(rname.strip(), round_num, effective=True)
                                triggered_any = True
                        
                        # 记录未触发的规则
                        if not triggered_any:
                            self.rule_selector.track_round_without_trigger(rule_names, round_num)
                    
                    # 每15轮执行一次清理
                    if round_num % 15 == 0:
                        cleanup_result = self.rule_selector.cleanup(round_num)
                        if cleanup_result["eliminated"]:
                            logger.info(f"★ 规则选择压力: 淘汰{len(cleanup_result['eliminated'])}条规则, "
                                       f"剩余{cleanup_result['remaining_rules']}条")
                            for item in cleanup_result["eliminated"]:
                                logger.info(f"  淘汰: {item['name']} ({item['reason']})")
                            
                            # 检测规则冲突
                            conflicts = self.rule_selector.detect_conflicts(rules_text)
                            if conflicts:
                                logger.info(f"★ 规则冲突检测: 发现{len(conflicts)}对潜在冲突")
                                for c in conflicts:
                                    logger.info(f"  冲突: {c['rule_a'][:20]} ↔ {c['rule_b'][:20]} (相似度{c['similarity']:.0%})")
            except Exception as e:
                logger.warning(f"规则选择压力处理失败: {e}")
            
            # 每20轮触发元认知反馈
            if round_num % 20 == 0:
                self._apply_metacognition_feedback(round_num)
            
            # ===== 元反应：每10轮，把系统自己的输出喂回去 =====
            # v3: 不仅看输出，还看历史元反应，问"你错过了什么"
            # 关键信号：有没有识别到自己输出的逻辑裂缝
            if round_num % 10 == 0 and prev_outputs_for_evolution:
                logger.info(f"★ 触发元反应（第{round_num}轮）：回顾最近{min(len(prev_outputs_for_evolution), 5)}轮输出")
                
                # 加载历史元反应记录
                meta_dir = "meta_reactions"
                os.makedirs(meta_dir, exist_ok=True)
                previous_reactions = []
                for f_name in sorted(os.listdir(meta_dir)):
                    if f_name.startswith("meta_round_") and f_name.endswith(".md"):
                        f_path = os.path.join(meta_dir, f_name)
                        with open(f_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            # 提取反应内容（---之后的部分）
                            if "---\n\n" in content:
                                reaction_text = content.split("---\n\n", 1)[1]
                                previous_reactions.append(reaction_text)
                
                if previous_reactions:
                    logger.info(f"  加载{len(previous_reactions)}条历史元反应")
                
                meta_result = self.engine.meta_react(
                    recent_outputs=prev_outputs_for_evolution[-5:],  # 最近5轮
                    round_num=round_num,
                    previous_reactions=previous_reactions if previous_reactions else None
                )
                
                # 保存元反应结果
                meta_file = os.path.join(meta_dir, f"meta_round_{round_num:03d}.md")
                with open(meta_file, "w", encoding="utf-8") as f:
                    f.write(f"# 元反应 - 第{round_num}轮\n\n")
                    f.write(f"触发时间：{datetime.now().isoformat()}\n")
                    f.write(f"反应层级：{meta_result.get('level', 0)}\n")
                    f.write(f"二阶信号：{'有' if meta_result['has_meta_reaction'] else '无'} ({meta_result.get('signal_count', 0)}个)\n")
                    f.write(f"一阶信号：{meta_result.get('first_order', [])}\n")
                    f.write(f"二阶信号：{meta_result.get('second_order', [])}\n")
                    f.write(f"三阶信号：{meta_result.get('third_order', [])}\n\n")
                    f.write(f"---\n\n{meta_result['reaction']}\n")
                
                logger.info(f"★ 元反应: 层级={meta_result.get('level', 0)}, "
                           f"二阶={'有' if meta_result['has_meta_reaction'] else '无'} → {meta_file}")
            
            if self.stats["rounds_completed"] < MAX_ROUNDS:
                logger.info(f"冷却{COOLDOWN_SECONDS}秒...")
                time.sleep(COOLDOWN_SECONDS)
        
        # 保存统计
        self.stats["end_time"] = datetime.now().isoformat()
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        final_mem = self.memory.get_stats()
        logger.info("\n" + "=" * 60)
        logger.info("自动循环v3结束")
        logger.info(f"完成轮数: {self.stats['rounds_completed']}")
        logger.info(f"求知缺口: {self.stats['curiosity_gaps_extracted']}条")
        logger.info(f"网络搜索: {self.stats['searches_performed']}次, 注入{self.stats['search_results_injected']}条外部信息")
        logger.info(f"自我维护: {self.stats['self_maintenance_items']}条")
        logger.info(f"认知记忆: {final_mem['total_concepts']}概念, {final_mem['total_relations']}关系, {final_mem['total_patterns']}模式, {final_mem['total_theories']}理论 ({final_mem['storage_size_mb']}MB)")
        logger.info(f"估算成本: {self.stats['estimated_cost']:.4f}元")
        logger.info(f"停止原因: {self.stats['stopped_reason']}")
        logger.info("=" * 60)

    def extract_structural_patterns(self, current_round: int):
        """从关系中提取结构特征模式（L2模式层填充）

        调用DeepSeek分析所有关系，找出3条以上共享同一结构特征的关系组，
        提取为模式并存储到patterns.json。
        """
        all_relations = self.memory.relations
        if len(all_relations) < 3:
            logger.info("关系数量不足3条，跳过结构模式提取")
            return

        # 概念名映射
        concept_names = {}
        for name, concept in self.memory.concepts.items():
            concept_names[concept["id"]] = name

        # 取权重最高的200条关系
        sorted_relations = sorted(all_relations, key=lambda r: r.get("weight", 0), reverse=True)[:200]

        relations_text = []
        for rel in sorted_relations:
            from_name = concept_names.get(rel.get("from_concept", ""), rel.get("from_concept", ""))
            to_name = concept_names.get(rel.get("to_concept", ""), rel.get("to_concept", ""))
            rel_type = rel.get("type", "unknown")
            desc = rel.get("description", "")[:150]
            relations_text.append(f"- [{rel.get('id')}] {from_name} --{rel_type}--> {to_name}: {desc}")

        relations_str = "\n".join(relations_text)

        prompt = f"""你是一个认知结构分析师。以下是系统中的{len(sorted_relations)}条关系（按权重排序）：

{relations_str}

请分析这些关系，找出**3条以上共享同一结构特征**的关系组。

结构特征示例：
- "都涉及自指悖论"（描述包含"自身"、"递归"、"悖论"）
- "都涉及极限/边界"（描述包含"无法"、"不可达"、"极限"）
- "都涉及守恒/不变量"（描述包含"守恒"、"不变"、"恒定"）
- "都涉及涌现/突变"（描述包含"涌现"、"相变"、"突变"）

输出JSON格式，不要输出其他内容：
```json
[
  {{
    "structure": "共性结构特征的一句话描述",
    "relations": ["r0001", "r0023", "r0045"],
    "description": "这个模式的整体描述"
  }}
]
```
如果没有满足条件的关系组，输出 []。
"""

        try:
            response = self.engine.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=6000
            )

            content = response.choices[0].message.content.strip()

            # 提取JSON
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            json_str = json_match.group(1) if json_match else content

            new_patterns = json.loads(json_str)
            if not isinstance(new_patterns, list):
                return

            # 过滤：至少3条关系
            valid_patterns = [p for p in new_patterns if len(p.get("relations", [])) >= 3]
            if not valid_patterns:
                logger.info("未找到满足条件的结构模式")
                return

            # 去重：基于结构文本相似度（0.7阈值）+ 关系集合合并
            def text_similarity(a: str, b: str) -> float:
                return SequenceMatcher(None, a.lower(), b.lower()).ratio()

            existing_sa = [
                p for p in self.memory.patterns
                if p.get("extraction_method") == "structural_analysis"
            ]

            added = 0
            merged = 0
            for pattern in valid_patterns:
                rel_set = set(pattern.get("relations", []))
                new_structure = pattern.get("structure", "")

                # 找文本最相似的已有模式
                best_match = None
                best_sim = 0.0
                for ep in existing_sa:
                    sim = text_similarity(new_structure, ep.get("structure", ""))
                    if sim > best_sim:
                        best_sim = sim
                        best_match = ep

                if best_sim > 0.7 and best_match is not None:
                    # 合并：把新关系追加到已有模式
                    existing_rels = set(best_match.get("relations", []))
                    new_rels = rel_set - existing_rels
                    if new_rels:
                        best_match["relations"] = list(existing_rels | rel_set)
                        best_match["evidence_count"] = len(best_match["relations"])
                        best_match["last_merged_round"] = current_round
                        merged += 1
                else:
                    # 真正的新模式
                    pid = self.memory._new_pattern_id()
                    self.memory.patterns.append({
                        "id": pid,
                        "structure": new_structure,
                        "relations": list(rel_set),
                        "description": pattern.get("description", ""),
                        "confidence": 0.6,
                        "evidence_count": len(rel_set),
                        "created_round": current_round,
                        "layer": 2,
                        "extraction_method": "structural_analysis"
                    })
                    existing_sa.append(self.memory.patterns[-1])
                    added += 1

            if added > 0 or merged > 0:
                self.memory._save()
                logger.info(f"★ 结构模式提取完成: 新增{added}个, 合并{merged}个到已有模式, 总计{len(self.memory.patterns)}个")
            else:
                logger.info("未发现新的结构模式")

        except json.JSONDecodeError as e:
            logger.warning(f"模式提取JSON解析失败: {e}")
        except Exception as e:
            logger.error(f"结构模式提取失败: {e}")


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.json")
    
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)
    
    loop = AutoLoop(config_path)
    loop.run()
