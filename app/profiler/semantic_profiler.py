import pandas as pd
import numpy as np
import math
import logging
import re
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple, Set
from app.profiler.profile_ir import ColumnProfileIR, PatternFingerprint, SemanticCandidate, EntitySummary
from app.profiler.detectors import DetectorRegistry, DetectionResult
from app.profiler.evidence_fusion import EvidenceFusionEngine

try:
    from pandas_type_detector import TypeDetectionPipeline
    HAS_PANDAS_TYPE_DETECTOR = True
except ImportError:
    HAS_PANDAS_TYPE_DETECTOR = False
    TypeDetectionPipeline = None

HAS_DUCKLING = False
HAS_PRESIDIO = False


logger = logging.getLogger(__name__)


class SemanticProfiler:
    """
    Phase 1: Semantic Profiler - 生成 IR-0 列画像，包含多维度指纹
    """
    
    # 模式检测正则（可扩展）
    _PATTERN_REGEXES = {
        "email": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
        "date_iso": r"^\d{4}-\d{2}-\d{2}$",
        "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        "phone": r"^\+?\d[\d\s\-()]{7,20}$",
        "currency_code": r"^[A-Z]{3}$",
        "url": r"^https?://[^\s]+$",
        "ipv4": r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
        "hex_color": r"^#[0-9a-fA-F]{6}$",
    }

    _type_detector = None
    _duckling = None
    _presidio = None
    _embedding_model = None

    @classmethod
    def _get_embedding_model(cls):
        """懒加载 Sentence Transformer 模型（仅在真正需要时加载）"""
        if cls._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                cls._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence Transformer model loaded: all-MiniLM-L6-v2")
            except ImportError:
                cls._embedding_model = None
                logger.warning("sentence-transformers not installed")
            except Exception as e:
                cls._embedding_model = None
                logger.warning(f"Failed to load Sentence Transformer: {e}")
        return cls._embedding_model

    @classmethod
    def _generate_name_embedding(cls, column_name: str) -> Optional[List[float]]:
        """生成列名的语义向量"""
        model = cls._get_embedding_model()
        if model is None:
            return None
        try:
            embedding = model.encode(column_name, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.debug(f"Failed to generate embedding for '{column_name}': {e}")
            return None

    @classmethod
    def _get_type_detector(cls):
        if cls._type_detector is None and HAS_PANDAS_TYPE_DETECTOR:
            cls._type_detector = TypeDetectionPipeline(locale="en-us")
        return cls._type_detector

    @classmethod
    def _get_duckling(cls):
        if cls._duckling is None:
            try:
                from fb_duckling import Duckling
                cls._duckling = Duckling()
                # 成功导入后设置全局标志（可选）
                global HAS_DUCKLING
                HAS_DUCKLING = True
            except ImportError:
                cls._duckling = None
                logger.warning("fb-duckling not installed")
            except Exception as e:
                cls._duckling = None
                logger.warning(f"Failed to initialize Duckling: {e}")
        return cls._duckling

    @classmethod
    def _get_presidio(cls):
        if cls._presidio is None:
            try:
                from presidio_analyzer import AnalyzerEngine
                cls._presidio = AnalyzerEngine()
                global HAS_PRESIDIO
                HAS_PRESIDIO = True
            except ImportError:
                cls._presidio = None
                logger.warning("presidio-analyzer not installed")
            except Exception as e:
                cls._presidio = None
                logger.warning(f"Failed to initialize Presidio: {e}")
        return cls._presidio

    @classmethod
    def preload_all_detectors(cls) -> None:
        """在应用启动时预加载所有检测器，避免运行时阻塞"""
        logger.info("Preloading all detectors...")
        
        # 1. 预加载 pandas-type-detector
        try:
            cls._get_type_detector()
            logger.info("  - pandas-type-detector loaded")
        except Exception as e:
            logger.warning(f"  - pandas-type-detector failed: {e}")
        
        # 2. 预加载 Duckling
        try:
            cls._get_duckling()
            logger.info("  - Duckling loaded")
        except Exception as e:
            logger.warning(f"  - Duckling failed: {e}")
        
        # 3. 预加载 Presidio（需要显式初始化）
        try:
            analyzer = cls._get_presidio()
            # 触发一次空分析来加载模型
            if analyzer:
                analyzer.analyze(text="test", language='en')
            logger.info("  - Presidio loaded")
        except Exception as e:
            logger.warning(f"  - Presidio failed: {e}")
        
        logger.info("All detectors preloaded.")

    @classmethod
    def generate_column_profiles(
        cls, 
        df: pd.DataFrame, 
        dataset_name: str = "default",
        max_sample_rows: int = 10000
    ) -> List[ColumnProfileIR]:
        """
        生成 IR-0 列画像列表（包含所有新指纹）
        """
        original_total_rows = len(df)
        if original_total_rows > max_sample_rows:
            logger.info(f"Profiling sampling {max_sample_rows} rows from {original_total_rows}")
            working_df = df.sample(n=max_sample_rows, random_state=42)
        else:
            working_df = df

        DEFAULT_MORPH = {
            'numeric_density': 0.0,
            'length_std': 0.0,
            'separator_profile': {},
            'decimal_place_mode': None,
            'value_fingerprint_clusters': {},  # 保留兼容，但不会再使用
            'cluster_coverage': 0.0,
        }

        profiles = []
        for col in working_df.columns:
            series = working_df[col]
            valid_series = series.dropna()
            valid_count = len(valid_series)
            null_count = len(series) - valid_count
            total_count = len(series)

            # ----- 1. 存储类型 -----
            storage_type = cls._infer_storage_type(series)

            # ----- 2. 统计特征 -----
            stats = cls._compute_basic_stats(valid_series, total_count)

            # ----- 3. 结构签名 -----
            structural_signature_simple = cls._compute_structural_signature(valid_series)
            structural_signature_detail = cls._compute_structural_signature_detail(valid_series)

            # ----- 4. 熵 -----
            value_entropy = cls._compute_entropy(valid_series)
            character_entropy = cls._compute_character_entropy(valid_series)
            length_entropy = cls._compute_length_entropy(valid_series)

            # ----- 5. 分布特征 -----
            top_freq, top_10_cov, top_20_cov, singleton_ratio = cls._compute_distribution_features(
                valid_series
            )

            # ----- 6. 形态学特征 -----
            morph_features = {}
            if storage_type == "string" and valid_count > 0:
                morph_features = cls._extract_morphological_features(valid_series)
                for key in DEFAULT_MORPH:
                    if key not in morph_features:
                        morph_features[key] = DEFAULT_MORPH[key]
            else:
                morph_features = DEFAULT_MORPH.copy()

            # 数值列补充 decimal_place_mode
            if storage_type in ["integer", "float"] and valid_count > 0:
                decimals = []
                sample_vals = valid_series.head(200)
                for val in sample_vals:
                    if pd.api.types.is_integer_dtype(valid_series):
                        decimals.append(0)
                    else:
                        s = str(val)
                        if '.' in s:
                            decimals.append(len(s.split('.')[1]))
                        else:
                            decimals.append(0)
                if decimals:
                    morph_features['decimal_place_mode'] = Counter(decimals).most_common(1)[0][0]

            # 值范围画像（针对字符串列）
            value_range_profile = None
            if storage_type == "string" and valid_count > 0:
                value_range_profile = cls._compute_value_range_profile(valid_series)

            precomputed_stats = {
                "pattern_coverage": structural_signature_detail.get("coverage") if structural_signature_detail else 0.0,
                "length_std": morph_features.get("length_std", 0.0),
                "separator_profile": morph_features.get("separator_profile", {}),
            }

            # ---- 7. 模式指纹（供 logical_type 使用, 也调用检测器，传入预计算数据） ----
            pattern_fingerprints = cls._detect_pattern_fingerprints(
                valid_series,
                precomputed_stats=precomputed_stats
            )

            # ----- 8. 逻辑类型（传入 pattern_fingerprints 避免重复正则） -----
            logical_type = cls._infer_logical_type(valid_series, stats, pattern_fingerprints)

            # ----- 9. 值相似度聚类（独立） -----
            value_similarity_clusters, cluster_coverage = cls._compute_value_similarity_clusters(valid_series)

            # ----- 10. 分层采样 -----
            samples = cls._stratified_sample(valid_series)

            # ----- 11. 第三方检测摘要（使用统一采样） -----
            duckling_summary = cls._extract_duckling_summary(valid_series)
            presidio_summary = cls._extract_presidio_summary(valid_series)

            # ----- 12. pandas-type-detector 检测 -----
            detected_type, detection_confidence, detected_format = cls._run_pandas_type_detector(
                valid_series
            )

            # ----- 13. 语义候选 -----
            semantic_candidates = cls._infer_semantic_candidates(
                storage_type, logical_type, pattern_fingerprints, col, stats
            )

            # ----- 14. 新增：生成列名 Embedding -----
            name_embedding = cls._generate_name_embedding(col)


            # ----- 构建 Profile -----
            profile = ColumnProfileIR(
                column_name=col,
                dataset_name=dataset_name,
                # 核心类型
                storage_type=storage_type,
                logical_type=logical_type,
                semantic_candidates=semantic_candidates,
                # 旧字段兼容
                data_type=storage_type,
                # 基础统计
                null_ratio=null_count / total_count if total_count > 0 else 1.0,
                unique_ratio=stats['unique_ratio'],
                duplicate_ratio=stats['duplicate_ratio'],
                distinct_count=stats['distinct_count'],
                total_count=original_total_rows,
                min=stats.get('min'),
                max=stats.get('max'),
                mean=stats.get('mean'),
                std=stats.get('std'),
                percentiles=stats.get('percentiles'),
                # 文本特征
                pattern=pattern_fingerprints[0].pattern_name if pattern_fingerprints else None,
                avg_length=stats.get('avg_length'),
                max_length=stats.get('max_length'),
                # 分布与采样
                top_frequencies=top_freq,
                samples=samples,
                candidate_types=[],  # 逐步废弃
                entropy=value_entropy,
                # 形态
                numeric_density=morph_features['numeric_density'],
                length_std=morph_features['length_std'],
                separator_profile=morph_features['separator_profile'],
                decimal_place_mode=morph_features['decimal_place_mode'],
                length_entropy=length_entropy,
                structural_signature=structural_signature_simple,
                structural_signature_detail=structural_signature_detail,
                value_range_profile=value_range_profile,
                # 值相似度
                value_similarity_clusters=value_similarity_clusters,
                cluster_coverage=cluster_coverage,
                # 新指纹
                singleton_ratio=singleton_ratio,
                character_entropy=character_entropy,
                pattern_fingerprints=pattern_fingerprints,
                top_10_coverage=top_10_cov,
                top_20_coverage=top_20_cov,
                # 第三方检测
                detected_type=detected_type,
                detection_confidence=detection_confidence,
                detected_format=detected_format,
                duckling_summary=duckling_summary,
                presidio_summary=presidio_summary,
                # ===== 新增：列名 Embedding =====
                name_embedding=name_embedding,
                # 内部缓存
                _value_set=set(valid_series.astype(str).values) if valid_count > 0 and valid_count < 5000 else None
            )
            profiles.append(profile)

        logger.info(f"Generated {len(profiles)} column profiles.")
        return profiles

    # ================================================================
    # 辅助方法
    # ================================================================

    @classmethod
    def _infer_storage_type(cls, series: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_integer_dtype(series):
            return "integer"
        if pd.api.types.is_float_dtype(series):
            return "float"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        return "string"

    @classmethod
    def _compute_basic_stats(cls, valid_series: pd.Series, total_count: int) -> Dict[str, Any]:
        valid_count = len(valid_series)
        if valid_count == 0:
            return {
                'unique_ratio': 0.0,
                'duplicate_ratio': 1.0,
                'distinct_count': 0,
                'min': None, 'max': None, 'mean': None, 'std': None,
                'percentiles': None, 'avg_length': None, 'max_length': None
            }

        unique_count = valid_series.nunique()
        unique_ratio = unique_count / valid_count
        duplicate_ratio = round(1 - unique_ratio, 4)

        if pd.api.types.is_numeric_dtype(valid_series) and not pd.api.types.is_bool_dtype(valid_series):
            min_val = float(valid_series.min())
            max_val = float(valid_series.max())
            mean_val = float(valid_series.mean())
            std_val = float(valid_series.std())
            percentiles = {}
            if valid_count > 0:
                p25, p50, p75 = valid_series.quantile([0.25, 0.5, 0.75])
                percentiles = {"25%": float(p25), "50%": float(p50), "75%": float(p75)}
        else:
            min_val = max_val = mean_val = std_val = percentiles = None

        str_series = valid_series.astype(str)
        avg_length = float(str_series.str.len().mean())
        max_length = int(str_series.str.len().max())

        return {
            'unique_ratio': unique_ratio,
            'duplicate_ratio': duplicate_ratio,
            'distinct_count': unique_count,
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'percentiles': percentiles,
            'avg_length': avg_length,
            'max_length': max_length,
        }

    @classmethod
    def _infer_logical_type(
        cls, 
        valid_series: pd.Series, 
        stats: Dict, 
        pattern_fingerprints: List[PatternFingerprint]
    ) -> str:
        """
        推断逻辑类型。
        优先使用检测器结果，如果置信度不足则 fallback 到形态推断。
        """
        if len(valid_series) == 0:
            return "empty"
        
        # 1. 优先使用检测器结果
        high_conf_candidates = [fp for fp in pattern_fingerprints if fp.confidence > 0.5]
        if high_conf_candidates:
            best = max(high_conf_candidates, key=lambda x: x.confidence)
            return f"{best.pattern_name}_like"
        
        # 2. Fallback: 形态推断
        return cls._infer_logical_type_fallback(valid_series, stats)

    @classmethod
    def _infer_logical_type_fallback(cls, valid_series: pd.Series, stats: Dict) -> str:
        """
        基于形态特征的逻辑类型推断（Fallback）
        """
        str_series = valid_series.astype(str)
        lengths = str_series.str.len()
        length_std = lengths.std() if len(lengths) > 1 else 0.0
        digit_ratios = str_series.str.count(r'\d') / str_series.str.len()
        avg_digit_ratio = digit_ratios.mean() if not pd.isna(digit_ratios).all() else 0.0
        unique_ratio = stats['unique_ratio']

        if unique_ratio > 0.9 and length_std < 1.0:
            return "fixed_length_code"
        if unique_ratio > 0.9 and length_std >= 1.0:
            return "variable_length_code"
        if avg_digit_ratio > 0.7 and unique_ratio > 0.8:
            return "numeric_like"
        if unique_ratio < 0.05:
            return "enum_like"
        if avg_digit_ratio < 0.2 and unique_ratio > 0.3:
            return "free_text"
        return "unknown"

    @classmethod
    def _detect_pattern_fingerprints(cls, valid_series: pd.Series, precomputed_stats: Optional[Dict[str, Any]] = None) -> List[PatternFingerprint]:
        """
        检测所有模式（使用新的 DetectorRegistry + EvidenceFusionEngine）
        """
        if len(valid_series) == 0:
            return []

        registry = DetectorRegistry()
        raw_result = registry.detect_all(valid_series, precomputed_stats=precomputed_stats)
        fused_result = EvidenceFusionEngine.fuse(raw_result)
        
        fingerprints = []
        for candidate in fused_result.candidates:
            if candidate.confidence > 0.3:
                fingerprints.append(PatternFingerprint(
                    pattern_name=candidate.type,
                    confidence=candidate.confidence,
                    coverage=candidate.raw_score  # 使用 raw_score 作为 coverage
                ))
        
        # 按置信度降序排序
        fingerprints.sort(key=lambda x: -x.confidence)
        return fingerprints

    @classmethod
    def _compute_structural_signature(cls, valid_series: pd.Series) -> Optional[str]:
        if len(valid_series) == 0:
            return None
        def signature(s: str) -> str:
            result = []
            for ch in s:
                if ch.isalpha():
                    result.append('A')
                elif ch.isdigit():
                    result.append('9')
                else:
                    result.append(ch)
            return ''.join(result)
        str_series = valid_series.astype(str)
        signatures = str_series.apply(signature)
        if signatures.empty:
            return None
        return signatures.value_counts().index[0]

    @classmethod
    def _compute_structural_signature_detail(cls, valid_series: pd.Series) -> Optional[Dict[str, Any]]:
        if len(valid_series) == 0:
            return None

        def signature(s: str) -> str:
            result = []
            # 用 count 表示连续相同类型，如 AAA-999 -> A{3}-9{3}
            last_char = None
            count = 0
            for ch in s:
                if ch.isalpha():
                    type_char = 'A'
                elif ch.isdigit():
                    type_char = '9'
                else:
                    type_char = ch
                if type_char == last_char:
                    count += 1
                else:
                    if last_char is not None:
                        result.append(f"{last_char}{{{count}}}" if count > 1 else last_char)
                    last_char = type_char
                    count = 1
            if last_char is not None:
                result.append(f"{last_char}{{{count}}}" if count > 1 else last_char)
            return ''.join(result)

        def char_class_counts(s: str) -> Dict[str, int]:
            counts = {'digit': 0, 'alpha': 0, 'separator': 0, 'other': 0}
            for ch in s:
                if ch.isdigit():
                    counts['digit'] += 1
                elif ch.isalpha():
                    counts['alpha'] += 1
                elif ch in '-/._ ':
                    counts['separator'] += 1
                else:
                    counts['other'] += 1
            return counts

        str_series = valid_series.astype(str)
        signatures = str_series.apply(signature)
        if signatures.empty:
            return None

        most_common = signatures.value_counts().index[0]
        # 取匹配该签名的第一个样本
        sample_val = str_series[signatures == most_common].iloc[0]
        char_classes = char_class_counts(sample_val)
        total = sum(char_classes.values())
        char_class_ratio = {k: round(v/total, 4) for k, v in char_classes.items()} if total > 0 else {}

        return {
            'signature': most_common,
            'char_class_ratio': char_class_ratio,
            'coverage': round(signatures.value_counts().max() / len(signatures), 4)
        }

    @classmethod
    def _compute_entropy(cls, valid_series: pd.Series) -> Optional[float]:
        if len(valid_series) == 0:
            return None
        probs = valid_series.value_counts(normalize=True)
        if len(probs) == 0:
            return None
        if len(probs) == 1:
            return 0.0
        return round(-sum(p * math.log2(p) for p in probs if p > 0), 4)

    @classmethod
    def _compute_character_entropy(cls, valid_series: pd.Series) -> Optional[float]:
        if len(valid_series) == 0:
            return None
        all_chars = ''.join(valid_series.astype(str))
        if not all_chars:
            return None
        char_counts = Counter(all_chars)
        total_chars = len(all_chars)
        probs = np.array([count / total_chars for count in char_counts.values()])
        return round(-sum(p * math.log2(p) for p in probs if p > 0), 4)

    @classmethod
    def _compute_length_entropy(cls, valid_series: pd.Series) -> Optional[float]:
        if len(valid_series) == 0:
            return None
        lengths = valid_series.astype(str).str.len()
        probs = lengths.value_counts(normalize=True)
        if len(probs) == 0:
            return None
        if len(probs) == 1:
            return 0.0
        return round(-sum(p * math.log2(p) for p in probs if p > 0), 4)

    @classmethod
    def _compute_distribution_features(
        cls, valid_series: pd.Series
    ) -> Tuple[Dict[str, int], Optional[float], Optional[float], Optional[float]]:
        if len(valid_series) == 0:
            return {}, None, None, None

        value_counts = valid_series.value_counts()
        top_freq = {}
        for val, cnt in value_counts.head(5).items():
            if cnt > 1:
                top_freq[str(val)] = int(cnt)

        top_10_coverage = value_counts.head(10).sum() / len(valid_series)
        top_20_coverage = value_counts.head(20).sum() / len(valid_series)
        singleton_count = (value_counts == 1).sum()
        singleton_ratio = singleton_count / len(valid_series)

        return top_freq, round(top_10_coverage, 4), round(top_20_coverage, 4), round(singleton_ratio, 4)

    @classmethod
    def _compute_value_range_profile(cls, valid_series: pd.Series) -> Optional[Dict[str, Any]]:
        """针对字符串列，计算值范围画像（leading_zero_ratio, min_length, max_length 等）"""
        if len(valid_series) == 0:
            return None
        str_series = valid_series.astype(str)
        lengths = str_series.str.len()
        leading_zero_ratio = str_series.str.match(r'^0+').sum() / len(str_series) if len(str_series) > 0 else 0.0
        return {
            'min_length': int(lengths.min()),
            'max_length': int(lengths.max()),
            'leading_zero_ratio': round(leading_zero_ratio, 4)
        }

    @classmethod
    def _compute_value_similarity_clusters(cls, valid_series: pd.Series) -> Tuple[Dict[str, int], float]:
        if len(valid_series) == 0:
            return {}, 0.0

        sample = valid_series.astype(str)
        unique_ratio = sample.nunique() / len(sample) if len(sample) > 0 else 1.0

        if unique_ratio >= 0.15 or len(sample) <= 10:
            return {}, 0.0

        def fingerprint(s):
            s = s.lower()
            s = re.sub(r'[^a-z0-9]', ' ', s)
            tokens = [t for t in s.split() if len(t) > 1]
            tokens.sort()
            return ' '.join(tokens)

        fingerprints = sample.apply(fingerprint)
        cluster_counts = fingerprints.value_counts()
        clusters = {}
        for fp, cnt in cluster_counts.items():
            if cnt > 1:
                clusters[fp] = int(cnt)
        total_clustered = sum(clusters.values())
        coverage = round(total_clustered / len(sample), 4) if len(sample) > 0 else 0.0
        return clusters, coverage

    @classmethod
    def _infer_semantic_candidates(cls, storage_type: str, logical_type: str,
                                   pattern_fingerprints: List[PatternFingerprint],
                                   col_name: str, stats: Dict) -> List[SemanticCandidate]:
        candidates = []
        evidence = []

        # 基于列名的证据
        if any(kw in col_name.lower() for kw in ['id', 'key', 'no', 'code']):
            evidence.append("column_name_suggests_identifier")
        if any(kw in col_name.lower() for kw in ['amount', 'price', 'total', 'fee', 'cost']):
            evidence.append("column_name_suggests_currency")

        # 基于统计的证据
        if stats['unique_ratio'] > 0.95:
            evidence.append("high_uniqueness")
        if stats['unique_ratio'] < 0.05 and stats['distinct_count'] < 20:
            evidence.append("low_cardinality")

        # 基于模式指纹的证据
        for fp in pattern_fingerprints:
            if fp.coverage > 0.7:
                cand_type = fp.pattern_name.replace('_like', '').title()
                cand = SemanticCandidate(
                    type=cand_type,
                    confidence=fp.confidence,
                    evidence=[f"pattern_{fp.pattern_name}_coverage_{fp.coverage}"]
                )
                candidates.append(cand)

        # 基于逻辑类型
        logical_to_semantic = {
            "fixed_length_code": "Code",
            "variable_length_code": "Identifier",
            "enum_like": "Enum",
            "date_like": "Date",
            "numeric_like": "NumericValue",
        }
        if logical_type in logical_to_semantic:
            candidates.append(SemanticCandidate(
                type=logical_to_semantic[logical_type],
                confidence=0.6,
                evidence=[f"logical_type_{logical_type}"]
            ))

        # 去重合并置信度
        merged = {}
        for c in candidates:
            key = c.type
            if key not in merged:
                merged[key] = c
            else:
                merged[key].confidence = max(merged[key].confidence, c.confidence)
                merged[key].evidence.extend(c.evidence)

        # 排序返回
        return sorted(merged.values(), key=lambda x: -x.confidence)[:5]

    @classmethod
    def _stratified_sample(cls, valid_series: pd.Series) -> List[Any]:
        if len(valid_series) == 0:
            return []
        samples = []
        if len(valid_series) >= 2:
            samples.extend(valid_series.sample(2, random_state=42).tolist())
        value_counts = valid_series.value_counts()
        if len(value_counts) >= 2:
            samples.extend(value_counts.head(2).index.tolist())
        if len(value_counts) >= 3:
            samples.extend(value_counts.tail(1).index.tolist())
        unique_samples = []
        seen = set()
        for val in samples:
            if val not in seen:
                seen.add(val)
                unique_samples.append(val)
                if len(unique_samples) >= 5:
                    break
        return unique_samples[:5]

    @classmethod
    def _run_pandas_type_detector(cls, valid_series: pd.Series) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        if not HAS_PANDAS_TYPE_DETECTOR or len(valid_series) < 5:
            return None, None, None
        detector = cls._get_type_detector()
        if detector is None:
            return None, None, None
        try:
            sample_series = valid_series.head(100).astype(str)
            result = detector.detect_column_type(sample_series)
            detected_type = getattr(result, 'data_type', None)
            confidence = getattr(result, 'confidence', None)
            format_info = getattr(result, 'format', None)
            return detected_type, confidence, format_info
        except Exception as e:
            logger.debug(f"pandas-type-detector failed: {e}")
            return None, None, None

    @classmethod
    def _extract_duckling_summary(cls, valid_series: pd.Series) -> Dict[str, EntitySummary]:
        # if not HAS_DUCKLING or len(valid_series) == 0:
        #     return {}

        duckling = cls._get_duckling()
        if duckling is None or len(valid_series) == 0:
            return {}

        # 使用采样数据
        sample_texts = valid_series.astype(str).head(200)
        entity_counts = Counter()
        matched_count = 0
        total_confidence = 0.0

        for val in sample_texts:
            if pd.isna(val) or val == '':
                continue
            try:
                result = duckling.parse(val)
                if result:
                    matched_count += 1
                    for entity in result:
                        dim = entity.get('dim', 'unknown')
                        entity_counts[dim] += 1
                        # 这里无法获取单个实体的置信度，用 1.0 作为近似
                        total_confidence += 1.0
            except Exception:
                pass

        if not entity_counts:
            return {}

        summary = {}
        total = len(sample_texts)
        avg_conf = total_confidence / max(1, sum(entity_counts.values()))
        for entity_type, count in entity_counts.items():
            coverage = count / total if total > 0 else 0.0
            if coverage > 0.05:  # 只保留覆盖率 >5% 的实体
                summary[entity_type] = EntitySummary(
                    coverage=round(coverage, 4),
                    avg_confidence=round(min(avg_conf, 1.0), 4)
                )
        return summary

    @classmethod
    def _extract_presidio_summary(cls, valid_series: pd.Series) -> Dict[str, EntitySummary]:
        # if not HAS_PRESIDIO or len(valid_series) == 0:
        #     return {}

        analyzer = cls._get_presidio()
        if analyzer is None or len(valid_series) == 0:
            return {}

        sample_texts = valid_series.astype(str).head(200)
        entity_scores = {}
        entity_counts = Counter()

        for val in sample_texts:
            if pd.isna(val) or val == '':
                continue
            try:
                result = analyzer.analyze(text=val, language='en')
                if result:
                    for entity in result:
                        entity_type = entity.entity_type
                        score = entity.score
                        if entity_type not in entity_scores:
                            entity_scores[entity_type] = []
                        entity_scores[entity_type].append(score)
                        entity_counts[entity_type] += 1
            except Exception:
                pass

        if not entity_scores:
            return {}

        summary = {}
        total = len(sample_texts)
        for entity_type, scores in entity_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > 0.3:  # 只保留平均置信度 > 0.3 的实体
                coverage = entity_counts[entity_type] / total if total > 0 else 0.0
                summary[entity_type] = EntitySummary(
                    coverage=round(coverage, 4),
                    avg_confidence=round(avg_score, 4)
                )
        return summary

    @classmethod
    def _extract_morphological_features(cls, series: pd.Series) -> Dict[str, Any]:
        sample = series.dropna().astype(str)
        if len(sample) == 0:
            return {
                'numeric_density': 0.0,
                'length_std': 0.0,
                'separator_profile': {},
                'decimal_place_mode': None,
                'value_fingerprint_clusters': {},
                'cluster_coverage': 0.0,
            }

        digit_counts = sample.str.count(r'\d')
        total_chars = sample.str.len()
        valid_mask = total_chars > 0
        avg_digit_ratio = (digit_counts[valid_mask] / total_chars[valid_mask]).mean() if valid_mask.sum() > 0 else 0.0
        numeric_density = round(float(avg_digit_ratio), 4) if not pd.isna(avg_digit_ratio) else 0.0

        lengths = sample.str.len()
        length_std = round(float(lengths.std()), 2) if len(lengths) > 1 else 0.0

        sep_chars = '-/._ '
        sep_counter = Counter()
        for s in sample.head(200):
            for ch in s:
                if ch in sep_chars:
                    sep_counter[ch] += 1
        total_sep = sum(sep_counter.values())
        if total_sep > 0:
            sep_profile = {k: round(v/total_sep, 4) for k, v in sep_counter.items() if v/total_sep > 0.03}
        else:
            sep_profile = {}

        decimal_counts = []
        for s in sample.head(200):
            cleaned = re.sub(r'[^\d.]', '', s.replace(',', ''))
            if '.' in cleaned:
                decimal_counts.append(len(cleaned.split('.')[1]))
            elif cleaned.isdigit():
                decimal_counts.append(0)
        decimal_place_mode = Counter(decimal_counts).most_common(1)[0][0] if decimal_counts else None

        # 保留 value_fingerprint_clusters 但不再使用，已移出
        return {
            'numeric_density': numeric_density,
            'length_std': length_std,
            'separator_profile': sep_profile,
            'decimal_place_mode': decimal_place_mode,
            'value_fingerprint_clusters': {},
            'cluster_coverage': 0.0,
        }

    # ================================================================
    # 公共方法（保持兼容）
    # ================================================================

    @classmethod
    def build_dependency_matrix(cls, df: pd.DataFrame, sample_rows: int = 10000) -> Dict[str, Any]:
        if len(df) > sample_rows:
            working_df = df.sample(n=sample_rows, random_state=42)
        else:
            working_df = df

        col_values = {}
        for col in working_df.columns:
            if pd.api.types.is_float_dtype(working_df[col]):
                continue
            valid_series = working_df[col].dropna()
            if len(valid_series) > 0:
                col_values[col] = set(valid_series.astype(str).values)

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
        logger.warning("build_evidence_pack is deprecated. Use generate_column_profiles + EvidenceGraphBuilder.")
        profiles = cls.generate_column_profiles(df)
        return {
            "profiles": [p.dict() for p in profiles],
            "columns": [p.column_name for p in profiles],
            "total_rows": len(df)
        }