#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grounding.py - 接地层模块 v2
职责：将系统内部推演产物与可检验现实建立映射

核心机制：
1. 提取两类可检验声明：
   A类-行为预测：关于系统自身未来行为的声明（可自验）
   B类-已知概念断言：涉及真实世界已知概念的判断（可用LLM裁判验证）
2. 存储到grounding.json
3. 定期验证预测
4. 将验证结果反馈给系统
"""

import os
import json
import re
import logging
from datetime import datetime

logger = logging.getLogger("drift.grounding")

GROUNDING_FILE = os.path.join(os.path.dirname(__file__), "grounding.json")
BASE_DIR = os.path.dirname(__file__)

# 核心概念清单（用于遗忘检测）
CORE_CONCEPTS = [
    "哥德尔", "自指", "不完备", "元盲区", "自指熵",
    "负温度", "拓扑保护", "认知虫洞", "规范对称性", "禁闭相",
    "认知Page曲线", "元反应", "认知卡诺热机", "极限环",
    "认知普朗克常数"
]


class GroundingLayer:
    """接地层：连接内部推演与可检验现实"""
    
    def __init__(self):
        self.data = self._load_grounding()
    
    def _load_grounding(self) -> dict:
        if os.path.exists(GROUNDING_FILE):
            try:
                with open(GROUNDING_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载grounding.json失败: {e}")
        return {"predictions": [], "verifications": [], "concept_anchors": {}}
    
    def _save_grounding(self):
        try:
            with open(GROUNDING_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存grounding.json失败: {e}")
    
    def extract_predictions(self, output: str, round_num: int) -> list:
        """从系统输出中提取两类可检验声明
        
        A类-行为预测：关于系统自身未来行为的声明（可自验）
        B类-已知概念断言：涉及真实世界已知概念的判断（可用LLM裁判验证）
        """
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE"),
                base_url="https://api.deepseek.com/v1"
            )
            
            truncated = output[:4000]
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": """你是一个认知系统审计员。从推演文本中提取两类可检验声明：

【A类-行为预测】关于系统自身未来行为的声明，例如：
- "概念X将在后续推演中继续出现" → 可通过检查未来输出来验证
- "系统将产生N个新概念" → 可通过计数验证
- "模式X的共现次数将增加" → 可通过记忆网络验证
- 任何关于"下一轮会怎样"、"未来趋势"、"预期会出现"的声明

【B类-已知概念断言】推演中涉及真实世界已知的数学/物理/逻辑/哲学概念时做出的具体判断，例如：
- "哥德尔不完备定理意味着系统无法证明自身一致性" → 可对照标准定理验证
- "Page曲线描述了信息恢复过程" → 可对照文献验证
- "Kasparov乘积用于描述C*-代数间的关系" → 可对照数学文献验证
- 即使推演使用了隐喻或类比，只要涉及已知的真实概念就提取

注意：
- 纯自创概念（如"认知虫洞"、"元-盲区定理"等）如果没有与已知概念的映射，则跳过
- 但如果自创概念引用了已知的数学结构（如Kasparov乘积、Page曲线、规范对称性等），则提取相关断言
- 尽量多提取，宁多勿少

输出格式（严格JSON数组）：
[{"type": "A或B", "claim": "具体声明", "check": "如何验证"}]
没有则返回 []"""},
                    {"role": "user", "content": f"推演文本：\n\n{truncated}"}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                claims = json.loads(json_match.group())
            else:
                return []
            
            predictions = []
            for item in claims[:5]:  # 最多5个
                claim_text = item.get("claim", "")
                test_method = item.get("check", "")
                if claim_text and len(claim_text) > 10:
                    predictions.append({
                        "text": claim_text,
                        "type": item.get("type", "B"),
                        "test_method": test_method,
                        "round": round_num,
                        "timestamp": datetime.now().isoformat(),
                        "status": "pending",
                        "verification": None,
                        "verifiable": True
                    })
            
            return predictions
            
        except Exception as e:
            logger.warning(f"LLM预测提取失败: {e}")
            return []
    
    def add_predictions(self, predictions: list):
        if not predictions:
            return
        self.data.setdefault("predictions", []).extend(predictions)
        self._save_grounding()
        logger.info(f"★ 接地层v2: 新增{len(predictions)}个预测，总计{len(self.data['predictions'])}个")
    
    def get_pending_predictions(self) -> list:
        return [p for p in self.data.get("predictions", []) 
                if p.get("status") == "pending"]
    
    def verify_predictions_batch(self, predictions: list, round_num: int) -> list:
        """批量验证待验证预测
        
        A类：基于系统实际数据验证（概念是否出现、模式是否增长等）
        B类：LLM裁判判断知识准确性
        """
        if not predictions:
            return []
        
        # 收集验证所需的系统实际数据
        system_state = self._get_system_state()
        
        a_preds = [(i, p) for i, p in enumerate(predictions) if p.get("type") == "A"]
        b_preds = [(i, p) for i, p in enumerate(predictions) if p.get("type") == "B"]
        
        results = []
        
        # A类验证：基于系统数据
        for idx, pred in a_preds:
            verdict = self._judge_a_prediction(pred, system_state)
            pred["status"] = verdict["status"]
            pred["verification"] = verdict["evidence"]
            pred["verified_round"] = round_num
            results.append(pred)
        
        # B类验证：LLM裁判
        if b_preds:
            b_results = self._verify_type_b(b_preds, round_num)
            results.extend(b_results)
        
        return results
    
    def _get_system_state(self) -> dict:
        """获取系统实际状态，用于A类预测验证"""
        state = {}
        
        # 概念网络
        concepts_path = os.path.join(BASE_DIR, "memory", "concepts.json")
        if os.path.exists(concepts_path):
            try:
                with open(concepts_path) as f:
                    concepts = json.load(f)
                state["total_concepts"] = len(concepts)
                sorted_concepts = sorted(
                    concepts.items(), 
                    key=lambda x: x[1].get("activation_count", 0), 
                    reverse=True
                )[:30]
                state["top30_concepts"] = [name for name, _ in sorted_concepts]
            except:
                pass
        
        # 最近10轮输出中的概念频率
        import glob
        output_dir = os.path.join(BASE_DIR, "auto_drift_output")
        recent_files = sorted(glob.glob(os.path.join(output_dir, "round_*.md")))[-10:]
        state["recent_concept_counts"] = {}
        for cc in CORE_CONCEPTS:
            total = 0
            for f in recent_files:
                try:
                    with open(f) as fh:
                        total += fh.read().count(cc)
                except:
                    pass
            state["recent_concept_counts"][cc] = total
        
        # 模式数量
        patterns_path = os.path.join(BASE_DIR, "memory", "patterns.json")
        if os.path.exists(patterns_path):
            try:
                with open(patterns_path) as f:
                    patterns = json.load(f)
                state["total_patterns"] = len(patterns)
            except:
                pass
        
        return state
    
    def _judge_a_prediction(self, pred: dict, state: dict) -> dict:
        """用系统实际数据判断A类预测（严格标准）"""
        text = pred.get("text", "").lower()
        
        # 严格标准：检查是否提到某概念将持续出现，需要量化阈值
        for concept in CORE_CONCEPTS:
            if concept.lower() in text and any(kw in text for kw in ["持续", "继续", "出现", "核心"]):
                count = state.get("recent_concept_counts", {}).get(concept, 0)
                # 收紧：需要出现>10次才验证通过
                if count > 10:
                    return {"status": "verified", "evidence": f"'{concept}'在最近10轮出现{count}次，显著活跃"}
                elif count > 3:
                    return {"status": "unverifiable", "evidence": f"'{concept}'出现{count}次，证据不足（需>10次）"}
                else:
                    return {"status": "refuted", "evidence": f"'{concept}'仅出现{count}次，未达阈值"}
        
        # 严格标准：概念网络增长预测需要具体数字
        if any(kw in text for kw in ["增长", "扩张", "增多"]):
            total = state.get("total_concepts", 0)
            # 收紧：需要明确的增长趋势，不能只看当前数量
            return {"status": "unverifiable", "evidence": f"当前{total}概念，但缺少基线对比数据"}
        
        if any(kw in text for kw in ["收敛", "精简", "减少"]):
            total = state.get("total_concepts", 0)
            return {"status": "unverifiable", "evidence": f"当前{total}概念，需多轮数据判断趋势"}
        
        # 默认用LLM判断
        return self._llm_judge_a_prediction(pred, state)
    
    def _llm_judge_a_prediction(self, pred: dict, state: dict) -> dict:
        """用LLM判断A类预测（兜底）"""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE"),
                base_url="https://api.deepseek.com/v1"
            )
            
            state_summary = json.dumps({
                "total_concepts": state.get("total_concepts", 0),
                "top10_concepts": state.get("top30_concepts", [])[:10],
                "total_patterns": state.get("total_patterns", 0),
                "recent_concept_counts": {k: v for k, v in state.get("recent_concept_counts", {}).items() if v > 0}
            }, ensure_ascii=False)
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是认知系统审计员。根据系统当前实际状态判断行为预测是否成立。\n\n判断标准：\n- verified: 系统数据明确支持该预测\n- refuted: 系统数据明确反驳该预测\n- unverifiable: 数据不足以判断\n\n输出JSON: {\"status\": \"verified/refuted/unverifiable\", \"evidence\": \"一句话依据\"}"},
                    {"role": "user", "content": f"系统当前状态：{state_summary}\n\n预测：{pred['text']}\n\n请判断。"}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"LLM兜底判断失败: {e}")
        
        return {"status": "unverifiable", "evidence": "验证失败"}
    
    def _verify_type_b(self, indexed_preds: list, round_num: int) -> list:
        """B类验证：LLM裁判判断已知概念断言的准确性"""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE"),
                base_url="https://api.deepseek.com/v1"
            )
            
            pred_list = "\n".join([
                f"[{i+1}] (R{pred.get('round','?')}) {pred['text']}" 
                for i, (_, pred) in enumerate(indexed_preds[:10])
            ])
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": """你是一个严格的学术裁判。系统推演中引用了已知的数学/物理/逻辑概念。你需要判断这些引用是否与学术共识一致。

判断标准：
- verified: 对已知概念的引用和推理与学术共识基本一致
- refuted: 对已知概念的引用存在明显错误或与学术共识矛盾
- unverifiable: 涉及高度推测性的延伸，学术界尚无定论

注意：
- 对于哥德尔定理、Page曲线等经典结果，有明确正确答案
- 对于前沿推测（如量子引力与认知的类比），学术界无定论，应判unverifiable
- 区分"对已知概念的准确引用"（verified）和"对已知概念的创造性延伸"（unverifiable）

输出格式（严格JSON数组）：
[{"id": 1, "status": "verified/refuted/unverifiable", "evidence": "一句话依据"}]"""},
                    {"role": "user", "content": f"请判断以下声明中对已知学术概念的使用是否准确：\n\n{pred_list}"}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if not json_match:
                return []
            
            verdicts = json.loads(json_match.group())
            
            results = []
            for v in verdicts:
                idx = v.get("id", 0) - 1
                if 0 <= idx < len(indexed_preds):
                    _, pred = indexed_preds[idx]
                    pred["status"] = v.get("status", "unverifiable")
                    pred["verification"] = v.get("evidence", "")
                    pred["verified_round"] = round_num
                    results.append(pred)
            
            return results
            
        except Exception as e:
            logger.warning(f"B类LLM验证失败: {e}")
            return []
    
    def apply_verifications(self, verifications: list):
        """将验证结果写入grounding.json"""
        if not verifications:
            return
        
        self.data.setdefault("verifications", []).extend(verifications)
        
        # 更新predictions中的status
        for v in verifications:
            for p in self.data.get("predictions", []):
                if p.get("text") == v.get("text") and p.get("round") == v.get("round"):
                    p["status"] = v["status"]
                    p["verification"] = v.get("verification")
                    break
        
        self._save_grounding()
        
        n_v = sum(1 for v in verifications if v["status"] == "verified")
        n_r = sum(1 for v in verifications if v["status"] == "refuted")
        n_u = sum(1 for v in verifications if v["status"] == "unverifiable")
        logger.info(f"★ 接地层v2验证: {len(verifications)}条 → {n_v}证实, {n_r}证伪, {n_u}不可验证")
    
    def get_verification_summary(self) -> str:
        verifs = self.data.get("verifications", [])
        if not verifs:
            return "接地层尚无验证结果。"
        
        lines = ["=== 接地层验证结果（v2自指接地） ==="]
        for v in verifs[-10:]:
            icon = {"verified": "✓", "refuted": "✗", "unverifiable": "?"}.get(v.get("status", ""), "·")
            ptype = v.get("type", "?")
            lines.append(f"  [{icon}][{ptype}] R{v.get('round','?')}: {v.get('text','')[:80]}")
            if v.get("verification"):
                lines.append(f"      → {v['verification'][:100]}")
        
        all_preds = self.data.get("predictions", [])
        n_v = sum(1 for v in verifs if v.get("status") == "verified")
        n_r = sum(1 for v in verifs if v.get("status") == "refuted")
        n_u = sum(1 for v in verifs if v.get("status") == "unverifiable")
        n_p = sum(1 for p in all_preds if p.get("status") == "pending")
        
        lines.append(f"  --- {len(all_preds)}个预测: {n_v}证实, {n_r}证伪, {n_u}不可验证, {n_p}待验证 ---")
        
        return "\n".join(lines)
    
    def get_pending_predictions(self) -> list:
        return [p for p in self.data.get("predictions", []) if p.get("status") == "pending"]
    
    def get_stats(self) -> dict:
        preds = self.data.get("predictions", [])
        verifs = self.data.get("verifications", [])
        return {
            "total_predictions": len(preds),
            "pending": sum(1 for p in preds if p.get("status") == "pending"),
            "verified": sum(1 for v in verifs if v.get("status") == "verified"),
            "refuted": sum(1 for v in verifs if v.get("status") == "refuted"),
            "unverifiable": sum(1 for v in verifs if v.get("status") == "unverifiable"),
        }
