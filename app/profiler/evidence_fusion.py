"""
Evidence Fusion Engine
职责：接收所有检测证据，进行冲突消解和融合，输出最终的语义类型候选
"""
from typing import List, Optional
from app.profiler.detectors import DetectionEvidence, DetectionResult


class EvidenceFusionEngine:
    """证据融合引擎"""
    
    # 互斥类型组：同一列只能属于其中一种
    EXCLUSIVE_GROUPS = [
        ('phone', 'date'),
        ('phone', 'email'),
        ('date', 'email'),
        ('boolean', 'binary_enum'),
        ('uuid', 'identifier'),
        ('currency', 'numeric'),  # 如有 numeric 检测
        ('url', 'identifier'),
        ('url', 'uuid'),
        ('date', 'identifier'),
    ]
    
    # 类型优先级（用于冲突消解时提升某些类型）
    TYPE_PRIORITY = {
        'date': 10,
        'email': 9,
        'phone': 8,
        'uuid': 7,
        'identifier': 6,
        'currency': 5,
        'boolean': 4,
        'binary_enum': 3,
    }
    
    @classmethod
    def fuse(cls, detection_result: DetectionResult) -> DetectionResult:
        """
        融合检测结果，消解冲突
        """
        if not detection_result.candidates:
            return detection_result
        
        # 1. 冲突消解
        resolved = cls._resolve_conflicts(detection_result.candidates)
        
        # 2. 按类型去重（保留置信度最高的）
        type_map = {}
        for evidence in resolved:
            if evidence.type not in type_map or evidence.confidence > type_map[evidence.type].confidence:
                type_map[evidence.type] = evidence
        
        # 3. 排序（先按优先级，再按置信度）
        candidates = sorted(
            type_map.values(),
            key=lambda x: (cls.TYPE_PRIORITY.get(x.type, 0), x.confidence),
            reverse=True
        )
        
        return DetectionResult(candidates=candidates)
    
    @classmethod
    def _resolve_conflicts(cls, candidates: List[DetectionEvidence]) -> List[DetectionEvidence]:
        """消解互斥证据"""
        # 对于每组互斥类型，如果同时存在，降低较低置信度的那个
        for group in cls.EXCLUSIVE_GROUPS:
            group_candidates = [ev for ev in candidates if ev.type in group]
            if len(group_candidates) > 1:
                # 按置信度降序排列
                sorted_candidates = sorted(group_candidates, key=lambda x: x.confidence, reverse=True)
                # 保留最高的，其余乘以 0.6 系数降低
                for ev in sorted_candidates[1:]:
                    ev.confidence *= 0.5
                    # 记录矛盾
                    ev.contradictions.append(f"conflict_with_{sorted_candidates[0].type}")
        return candidates