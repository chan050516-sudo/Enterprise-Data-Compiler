import pandas as pd
import numpy as np
import math
import logging
import re
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple
from app.schema.profile_ir import ColumnProfileIR, PatternFingerprint

try:
    from pandas_type_detector import TypeDetectionPipeline
    HAS_PANDAS_TYPE_DETECTOR = True
except ImportError:
    HAS_PANDAS_TYPE_DETECTOR = False
    TypeDetectionPipeline = None

try:
    from fb_duckling import Duckling
    HAS_DUCKLING = True
except ImportError:
    HAS_DUCKLING = False
    Duckling = None

try:
    from presidio_analyzer import AnalyzerEngine
    HAS_PRESIDIO = True
except ImportError:
    HAS_PRESIDIO = False
    AnalyzerEngine = None


logger = logging.getLogger(__name__)

class SemanticProfiler:
    """
    Phase 1: Semantic Profiler - 生成 IR-0 列画像，包含多维度指纹
    
    核心输出:
    - physical_type: Pandas 物理类型 (integer, float, string, boolean, datetime)
    - logical_type: 形态推断的逻辑类型 (fixed_length_code, date_like, enum_like, etc.)
    - pattern_fingerprints: 所有检测到的模式 (带置信度和覆盖率)
    - 分布特征: singleton_ratio, top_10/20_coverage
    - 形态特征: numeric_density, length_std, separator_profile, structural_signature
    - 熵: value_entropy, character_entropy
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

    @classmethod
    def _get_type_detector(cls):
        if cls._type_detector is None and HAS_PANDAS_TYPE_DETECTOR:
            cls._type_detector = TypeDetectionPipeline(locale="en-us")
        return cls._type_detector

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
            'value_fingerprint_clusters': {},
            'cluster_coverage': 0.0,
        }

        profiles = []
        for col in working_df.columns:
            series = working_df[col]
            valid_series = series.dropna()
            valid_count = len(valid_series)
            null_count = len(series) - valid_count
            total_count = len(series)

            # ----- 1. 物理类型 -----
            physical_type = cls._infer_physical_type(series)

            # ----- 2. 统计特征 -----
            stats = cls._compute_basic_stats(valid_series, total_count)

            # ----- 3. 逻辑类型 -----
            logical_type = cls._infer_logical_type(valid_series, stats)

            # ----- 4. 模式指纹 -----
            pattern_fingerprints = cls._detect_pattern_fingerprints(valid_series)

            # ----- 5. 结构签名 -----
            structural_signature = cls._compute_structural_signature(valid_series)

            # ----- 6. 熵 -----
            value_entropy = cls._compute_entropy(valid_series)
            character_entropy = cls._compute_character_entropy(valid_series)

            # ----- 7. 分布特征 -----
            top_freq, top_10_cov, top_20_cov, singleton_ratio = cls._compute_distribution_features(
                valid_series
            )

            # ----- 8. 候选语义类型 -----
            candidate_types = cls._infer_candidate_types(
                physical_type, logical_type, pattern_fingerprints, col
            )

            # ----- 9. 形态学特征 -----
            morph_features = {}
            if physical_type == "string" and valid_count > 0:
                morph_features = cls._extract_morphological_features(valid_series)
                for key in DEFAULT_MORPH:
                    if key not in morph_features:
                        morph_features[key] = DEFAULT_MORPH[key]
            else:
                morph_features = DEFAULT_MORPH.copy()

            # ----- 10. 对数值列补充小数位数模式 -----
            if physical_type in ["integer", "float"] and valid_count > 0:
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

            # ----- 11. 分层采样 -----
            samples = cls._stratified_sample(valid_series)

            # ----- 12. pandas-type-detector 检测 -----
            detected_type, detection_confidence, detected_format = cls._run_pandas_type_detector(
                valid_series
            )

            # ----- 13. Duckling 实体提取（Phase 1C） -----
            duckling_entities, duckling_coverage = cls._extract_duckling_entities(valid_series)

            # ----- 14. Presidio PII 检测（Phase 1C） -----
            presidio_entities, presidio_coverage = cls._extract_presidio_entities(valid_series)

            # ----- 构建 Profile -----
            profile = ColumnProfileIR(
                column_name=col,
                dataset_name=dataset_name,
                # 核心类型系统
                physical_type=physical_type,
                logical_type=logical_type,
                semantic_candidates=[],
                # 旧字段保留（但直接映射到 physical_type）
                data_type=physical_type,
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
                candidate_types=candidate_types,
                entropy=value_entropy,
                # 形态特征
                numeric_density=morph_features['numeric_density'],
                length_std=morph_features['length_std'],
                separator_profile=morph_features['separator_profile'],
                decimal_place_mode=morph_features['decimal_place_mode'],
                value_fingerprint_clusters=morph_features['value_fingerprint_clusters'],
                cluster_coverage=morph_features['cluster_coverage'],
                # 新指纹
                singleton_ratio=singleton_ratio,
                character_entropy=character_entropy,
                structural_signature=structural_signature,
                pattern_fingerprints=pattern_fingerprints,
                top_10_coverage=top_10_cov,
                top_20_coverage=top_20_cov,
                # 第三方检测
                detected_type=detected_type,
                detection_confidence=detection_confidence,
                detected_format=detected_format,
                duckling_entities=duckling_entities,
                duckling_entity_coverage=duckling_coverage,
                presidio_entities=presidio_entities,
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
    def _infer_physical_type(cls, series: pd.Series) -> str:
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

        # 数值型统计（仅对真正的数值列）
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

        # 文本长度（对所有类型都适用）
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
    def _infer_logical_type(cls, valid_series: pd.Series, stats: Dict) -> str:
        if len(valid_series) == 0:
            return "empty"

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
        # 日期检测
        date_pattern = r'^\d{4}[-/.]\d{2}[-/.]\d{2}|\d{2}[-/.]\d{2}[-/.]\d{4}'
        if str_series.str.match(date_pattern).mean() > 0.8:
            return "date_like"
        return "unknown"

    @classmethod
    def _detect_pattern_fingerprints(cls, valid_series: pd.Series) -> List[PatternFingerprint]:
        if len(valid_series) == 0:
            return []
        str_series = valid_series.astype(str)
        total = len(str_series)
        fingerprints = []
        for name, regex in cls._PATTERN_REGEXES.items():
            matches = str_series.str.match(regex).sum()
            coverage = matches / total if total > 0 else 0.0
            if coverage > 0.1:
                confidence = min(0.6 + coverage * 0.4, 0.99)
                fingerprints.append(PatternFingerprint(
                    pattern_name=name,
                    confidence=confidence,
                    coverage=coverage
                ))
        fingerprints.sort(key=lambda x: -x.coverage)
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
    def _infer_candidate_types(cls, physical_type: str, logical_type: str,
                               pattern_fingerprints: List[PatternFingerprint],
                               col_name: str) -> List[str]:
        candidates = []
        if physical_type in ["integer", "float"]:
            candidates.append("numeric")
            if any(kw in col_name.lower() for kw in ['amount', 'price', 'total', 'fee', 'cost']):
                candidates.append("currency")
        else:
            candidates.append("string")
        for fp in pattern_fingerprints:
            if fp.coverage > 0.3:
                candidates.append(fp.pattern_name)
        if logical_type == "enum_like":
            candidates.append("enum")
        if logical_type in ["fixed_length_code", "variable_length_code"]:
            candidates.append("code")
        return list(set(candidates))

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

        unique_ratio = sample.nunique() / len(sample) if len(sample) > 0 else 1.0
        clusters = {}
        cluster_coverage = 0.0
        if unique_ratio < 0.15 and len(sample) > 10:
            def fingerprint(s):
                s = s.lower()
                s = re.sub(r'[^a-z0-9]', ' ', s)
                tokens = [t for t in s.split() if len(t) > 1]
                tokens.sort()
                return ' '.join(tokens)
            fingerprints = sample.apply(fingerprint)
            cluster_counts = fingerprints.value_counts()
            for fp, cnt in cluster_counts.items():
                if cnt > 1:
                    clusters[fp] = int(cnt)
            total_clustered = sum(clusters.values())
            cluster_coverage = round(total_clustered / len(sample), 4) if len(sample) > 0 else 0.0

        return {
            'numeric_density': numeric_density,
            'length_std': length_std,
            'separator_profile': sep_profile,
            'decimal_place_mode': decimal_place_mode,
            'value_fingerprint_clusters': clusters,
            'cluster_coverage': cluster_coverage,
        }

        _duckling = None

    @classmethod
    def _get_duckling(cls):
        if cls._duckling is None and HAS_DUCKLING:
            cls._duckling = Duckling()
        return cls._duckling

    @classmethod
    def _extract_duckling_entities(cls, valid_series: pd.Series) -> Tuple[List[Dict[str, Any]], Optional[float]]:
        """
        使用 Duckling 从文本中提取自然语言实体。
        返回: (entities_list, coverage)
        """
        if not HAS_DUCKLING or len(valid_series) == 0:
            return [], None

        duckling = cls._get_duckling()
        if duckling is None:
            return [], None

        # 采样前 200 行（避免性能问题）
        sample = valid_series.astype(str).head(200)
        all_entities = []
        matched_count = 0

        for val in sample:
            if pd.isna(val) or val == '':
                continue
            try:
                # Duckling 解析
                result = duckling.parse(val)
                if result:
                    matched_count += 1
                    # 提取实体类型和值
                    for entity in result:
                        all_entities.append({
                            'text': val[:100],  # 截断长文本
                            'dimension': entity.get('dim'),
                            'value': entity.get('value'),
                            'start': entity.get('start'),
                            'end': entity.get('end'),
                        })
            except Exception as e:
                logger.debug(f"Duckling parsing failed for '{val[:50]}': {e}")

        coverage = matched_count / len(sample) if len(sample) > 0 else 0.0
        return all_entities[:100], round(coverage, 4)  # 限制返回数量

    _presidio = None

    @classmethod
    def _get_presidio(cls):
        if cls._presidio is None and HAS_PRESIDIO:
            cls._presidio = AnalyzerEngine()
        return cls._presidio

    @classmethod
    def _extract_presidio_entities(cls, valid_series: pd.Series) -> Tuple[List[Dict[str, Any]], Optional[float]]:
        """
        使用 Presidio 从文本中检测 PII 实体。
        返回: (entities_list, coverage)
        """
        if not HAS_PRESIDIO or len(valid_series) == 0:
            return [], None

        analyzer = cls._get_presidio()
        if analyzer is None:
            return [], None

        sample = valid_series.astype(str).head(200)
        all_entities = []
        matched_count = 0

        for val in sample:
            if pd.isna(val) or val == '':
                continue
            try:
                result = analyzer.analyze(text=val, language='en')
                if result:
                    matched_count += 1
                    for entity in result:
                        all_entities.append({
                            'text': val[:100],
                            'entity_type': entity.entity_type,
                            'confidence': entity.score,
                            'start': entity.start,
                            'end': entity.end,
                        })
            except Exception as e:
                logger.debug(f"Presidio analysis failed for '{val[:50]}': {e}")

        coverage = matched_count / len(sample) if len(sample) > 0 else 0.0
        return all_entities[:100], round(coverage, 4)

    # ================================================================
    # 保持兼容的公共方法
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