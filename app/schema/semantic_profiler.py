import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, Any, List, Optional
from app.schema.profile_ir import ColumnProfileIR

logger = logging.getLogger(__name__)

class SemanticProfiler:
    """
    Layer 2: Semantic Profiler (重构版)
    职责：接收经过 TechnicalNormalizer 处理的数据，生成 IR-0 (ColumnProfileIR) 列表。
    不再产出松散的 evidence_pack 字典。
    """
    
    # 业务语义正则（保留原有）
    _CONCEPT_PATTERNS = {
        "email": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
        "date_iso": r"^\d{4}-\d{2}-\d{2}$",
        "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        "phone": r"^\+?\d[\d\s\-()]{7,20}$",
        "currency_code": r"^[A-Z]{3}$"
    }

    @classmethod
    def generate_column_profiles(
        cls, 
        df: pd.DataFrame, 
        dataset_name: str = "default",
        max_sample_rows: int = 10000
    ) -> List[ColumnProfileIR]:
        """
        核心方法：生成 IR-0 列画像列表。
        采样机制防止 OOM，适用于大型数据集。
        """
        total_rows = len(df)
        if total_rows > max_sample_rows:
            logger.info(f"Profiling sampling {max_sample_rows} rows from {total_rows}")
            working_df = df.sample(n=max_sample_rows, random_state=42)
        else:
            working_df = df

        profiles = []
        for col in working_df.columns:
            # ===== 升级版统计计算 =====
            series = working_df[col]
            valid_series = series.dropna()
            valid_count = len(valid_series)
            null_count = len(series) - valid_count
            total_count = len(series)

            # 1. 基础类型判断
            if pd.api.types.is_numeric_dtype(series):
                data_type = "numeric"
                # 百分位数
                percentiles = {}
                if valid_count > 0:
                    p25, p50, p75 = valid_series.quantile([0.25, 0.5, 0.75])
                    percentiles = {"25%": float(p25), "50%": float(p50), "75%": float(p75)}
                else:
                    percentiles = None
                min_val = float(valid_series.min()) if valid_count > 0 else None
                max_val = float(valid_series.max()) if valid_count > 0 else None
                mean_val = float(valid_series.mean()) if valid_count > 0 else None
                std_val = float(valid_series.std()) if valid_count > 0 else None
                pattern = None
                avg_len = None
                max_len = None
            else:
                data_type = "string_or_categorical"
                percentiles = None
                min_val = max_val = mean_val = std_val = None
                # 文本模式推断
                if valid_count > 0:
                    str_series = valid_series.astype(str)
                    avg_len = float(str_series.str.len().mean())
                    max_len = int(str_series.str.len().max())
                    pattern = cls._infer_business_concept(str_series)
                else:
                    avg_len = max_len = None
                    pattern = None

            # 2. 唯一值与重复率
            unique_count = valid_series.nunique() if valid_count > 0 else 0
            unique_ratio = unique_count / valid_count if valid_count > 0 else 0
            duplicate_ratio = round(1 - unique_ratio, 4)  # 新增：重复率

            # 3. 高频值 Top 5（分布统计）
            top_freq = {}
            if valid_count > 0:
                value_counts = valid_series.value_counts()
                for val, cnt in value_counts.head(5).items():
                    # 将值转为字符串，确保可序列化
                    top_freq[str(val) if not isinstance(val, (int, float)) else val] = int(cnt)

            # 4. 候选数据类型推断（新增）
            candidate_types = []
            if data_type == "numeric":
                candidate_types.append("numeric")
                # 如果列名含 price/amount/total，增加 currency 候选
                if any(kw in col.lower() for kw in ['amount', 'price', 'total', 'fee', 'cost']):
                    candidate_types.append("currency")
                if all(v is not None for v in [min_val, max_val]) and min_val >= 0:
                    candidate_types.append("non_negative")
            else:
                candidate_types.append("string")
                if pattern:
                    candidate_types.append(pattern)
                if unique_ratio < 0.05 and valid_count > 10:  # 低基数
                    candidate_types.append("enum")
                if avg_len and avg_len < 10 and unique_ratio > 0.9:
                    candidate_types.append("code")

            # 5. 样本（保留前 5 个非空）
            samples = valid_series.head(5).tolist() if valid_count > 0 else []

            # 6. 构建 Profile 对象（注入所有新字段）
            profile = ColumnProfileIR(
                column_name=col,
                dataset_name=dataset_name,
                data_type=data_type,
                null_ratio=null_count / total_count if total_count > 0 else 1.0,
                unique_ratio=round(unique_ratio, 4),
                duplicate_ratio=duplicate_ratio,      # 新增
                distinct_count=unique_count,
                total_count=total_count,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
                percentiles=percentiles,               # 新增
                pattern=pattern,
                avg_length=avg_len,
                max_length=max_len,
                top_frequencies=top_freq,              # 新增
                samples=samples,
                candidate_types=candidate_types,       # 新增
                _value_set=set(valid_series.astype(str).values) if valid_count > 0 and valid_count < 5000 else None
            )
            profiles.append(profile)
        
        logger.info(f"Generated {len(profiles)} column profiles.")
        return profiles

    @classmethod
    def _infer_business_concept(cls, series: pd.Series) -> Optional[str]:
        """从样本中推断业务概念（保留原有逻辑）"""
        if series.empty:
            return None
        sample_texts = series.head(100).astype(str)
        for concept, pattern in cls._CONCEPT_PATTERNS.items():
            match_count = sample_texts.str.match(pattern).sum()
            if match_count >= len(sample_texts) * 0.9:
                return concept
        return None

    @classmethod
    def build_dependency_matrix(cls, df: pd.DataFrame, sample_rows: int = 10000) -> Dict[str, Any]:
        """
        构建列间依赖矩阵，用于 Chunking 粗分和映射决策。
        返回结构：
        {
            "relationships": [
                {
                    "source_column": "cust_no",
                    "target_column": "id",
                    "source_entity": "Orders",         # 可选，从列名推断
                    "target_entity": "Customer",       # 可选，从列名推断
                    "overlap_ratio": 0.95,
                    "relationship_type": "fk_candidate",
                    "confidence": "HIGH"
                }
            ]
        }
        """
        # 1. 采样（避免 OOM）
        if len(df) > sample_rows:
            working_df = df.sample(n=sample_rows, random_state=42)
        else:
            working_df = df

        # 2. 提取所有列的唯一值集合（只对可哈希类型做）
        col_values = {}
        for col in working_df.columns:
            # 跳过浮点数（浮点数不适合做精确 FK 匹配）
            if pd.api.types.is_float_dtype(working_df[col]):
                continue
            valid_series = working_df[col].dropna()
            if len(valid_series) > 0:
                # 转为字符串集合，保证可哈希
                col_values[col] = set(valid_series.astype(str).values)

        # 3. 计算两两重叠率
        relationships = []
        col_list = list(col_values.keys())

        for i, col_a in enumerate(col_list):
            set_a = col_values[col_a]
            for col_b in col_list[i+1:]:
                set_b = col_values[col_b]
                intersection = len(set_a & set_b)
                if intersection == 0:
                    continue
                ratio = intersection / min(len(set_a), len(set_b))
                rel_type = "fk_candidate" if ratio > 0.8 else "partial_match"
                confidence = "HIGH" if ratio > 0.95 else "MEDIUM" if ratio > 0.8 else "LOW"

                # 避免重复添加
                relationships.append({
                    "source_column": col_a,
                    "target_column": col_b,
                    "overlap_ratio": round(ratio, 4),
                    "relationship_type": rel_type,
                    "confidence": confidence
                })

        return {"relationships": relationships}
    
    @classmethod
    def build_evidence_pack(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        [Deprecated] 保留此方法以兼容旧版 main.py。
        内部转调 generate_column_profiles 并转换为旧字典格式。
        建议尽快切换到 EvidenceGraph。
        """
        logger.warning("build_evidence_pack is deprecated. Use generate_column_profiles + EvidenceGraphBuilder.")
        profiles = cls.generate_column_profiles(df)
        # 构建旧的 evidence_pack 结构（仅为了不报错）
        return {
            "profiles": [p.dict() for p in profiles],
            "columns": [p.column_name for p in profiles],
            "total_rows": len(df)
        }