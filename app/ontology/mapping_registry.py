import json
import logging
import difflib
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class MappingRegistryEngine:
    """
    Mapping Registry Engine (Transitional Phase before Vector DB).
    提供基于 Domain 隔离的模糊语义召回，为 LLM 注入历史映射经验。
    """
    
    def __init__(self, registry_file: str):
        self.registry_file = registry_file
        self._memory_bank: Dict[str, List[Dict[str, Any]]] = {}
        self._load_registry()

    def _load_registry(self):
        """挂载注册表并按业务域 (Domain) 进行物理切片隔离"""
        try:
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            records = raw_data.get("registry", [])
            for record in records:
                domain = record.get("domain", "default")
                if domain not in self._memory_bank:
                    self._memory_bank[domain] = []
                self._memory_bank[domain].append(record)
                
            logger.info(f"Mapping Registry loaded. Found {len(records)} historical records across {len(self._memory_bank)} domains.")
        except FileNotFoundError:
            logger.warning("Mapping registry file not found. Running with empty historical memory.")
        except json.JSONDecodeError as e:
            logger.error(f"Mapping registry JSON format invalid: {e}")

    def recall_candidates(
        self, 
        source_field: str, 
        domain: str = "default", 
        top_k: int = 3,
        similarity_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        核心检索方法：模拟 Vector DB 的 Recall 行为。
        使用 difflib 提供轻量级的文本相似度匹配（基于 Gestalt Pattern Matching）。
        """
        domain_records = self._memory_bank.get(domain, [])
        if not domain_records:
            return []

        # 提取所有历史 known_source_patterns
        known_patterns = [record["source_pattern"] for record in domain_records]
        
        # 执行模糊匹配检索
        matches = difflib.get_close_matches(
            word=source_field, 
            possibilities=known_patterns, 
            n=top_k, 
            cutoff=similarity_threshold
        )
        
        if not matches:
            return []

        # 组装召回结果
        recalled_history = []
        for match in matches:
            # 找回对应的完整记录
            record = next(r for r in domain_records if r["source_pattern"] == match)
            # 计算相似度分值 (0.0 to 1.0)
            similarity_score = difflib.SequenceMatcher(None, source_field, match).ratio()
            
            recalled_history.append({
                "historical_source": match,
                "mapped_target": record.get("target_field"),
                "similarity_score": round(similarity_score, 2),
                "historical_confidence": record.get("confidence", 1.0),
                "hint": f"Historically, '{match}' mapped to '{record.get('target_field')}'."
            })

        return recalled_history