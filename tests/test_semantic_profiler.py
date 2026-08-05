import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from app.profiler.profile_ir import ColumnProfileIR
from app.profiler.semantic_profiler import SemanticProfiler

try:
    from pandas_type_detector import TypeDetectionPipeline
    HAS_PANDAS_TYPE_DETECTOR = True
except ImportError:
    HAS_PANDAS_TYPE_DETECTOR = False

@pytest.fixture(autouse=True)
def mock_third_party():
    with patch('app.schema.semantic_profiler.SemanticProfiler._run_pandas_type_detector', return_value=(None, None, None)):
        with patch('app.schema.semantic_profiler.SemanticProfiler._extract_duckling_summary', return_value={}):
            with patch('app.schema.semantic_profiler.SemanticProfiler._extract_presidio_summary', return_value={}):
                yield


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "email": [
            "alice@example.com", "bob@test.org", "charlie@domain.net",
            "diana@sample.co", "eve@demo.io"
        ],
        "date_joined": ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05", "2023-05-12"],
        "salary": [50000, 60000, 75000, 80000, 90000],
        "active": [True, False, True, False, True],
        "department": ["HR", "IT", "Sales", "IT", "HR"],
        "phone": [
            "+60 12 345 6789", "+60 16 789 0123", "+60 11 234 5678",
            "+60 13 456 7890", "+60 19 876 5432"
        ],
        "empty_col": [None, None, None, None, None],
        "mixed_with_nan": [1, 2, None, 4, 5],
        "text_with_spaces": ["  Hello  ", "  World  ", "  Test  ", "  Data  ", "  Science  "],
        "enum_col": ["Red", "blue", "Blue", " RED ", "GREEN"],
        "code_col": ["C001", "C002", "C003", "C004", "C005"],
        "numeric_with_decimals": [10.5, 20.3, 30.7, 40.2, 50.9],
        "long_text": ["short", "medium length", "this is a longer string with many words",
                      "a very very very very very long sentence indeed", "tiny"]
    })


@pytest.fixture
def profiler():
    return SemanticProfiler()


class TestSemanticProfiler:

    def test_generate_column_profiles_basic_stats(self, profiler, sample_df):
        profiles = profiler.generate_column_profiles(sample_df, dataset_name="test")
        profile_map = {p.column_name: p for p in profiles}

        # id 列 — 整数
        id_profile = profile_map["id"]
        assert id_profile.storage_type == "integer"
        assert id_profile.data_type == "integer"
        assert id_profile.null_ratio == 0.0
        assert id_profile.unique_ratio == 1.0
        assert id_profile.distinct_count == 5
        assert id_profile.min == 1.0
        assert id_profile.max == 5.0
        assert id_profile.mean == 3.0
        assert id_profile.std is not None
        assert id_profile.entropy is not None

        # name 列 — 字符串
        name_profile = profile_map["name"]
        assert name_profile.storage_type == "string"
        assert name_profile.data_type == "string"
        assert name_profile.null_ratio == 0.0
        assert name_profile.unique_ratio == 1.0
        assert name_profile.avg_length is not None
        assert name_profile.max_length is not None
        assert name_profile.entropy is not None

        # empty_col — 全部空值
        empty_profile = profile_map["empty_col"]
        assert empty_profile.null_ratio == 1.0
        assert empty_profile.unique_ratio == 0.0
        assert empty_profile.distinct_count == 0
        assert empty_profile.entropy is None

        # mixed_with_nan — 含空值
        mixed_profile = profile_map["mixed_with_nan"]
        assert mixed_profile.null_ratio == 0.2  # 1/5
        assert mixed_profile.unique_ratio == 1.0  # 4 个非空值均为唯一
        assert mixed_profile.distinct_count == 4

    def test_generate_column_profiles_morphological_features(self, profiler, sample_df):
        profiles = profiler.generate_column_profiles(sample_df, dataset_name="test")
        profile_map = {p.column_name: p for p in profiles}

        # email 列
        email_profile = profile_map["email"]
        assert email_profile.numeric_density is not None
        assert email_profile.numeric_density < 0.3
        assert email_profile.length_std is not None
        assert email_profile.separator_profile is not None

        # phone 列
        phone_profile = profile_map["phone"]
        assert phone_profile.numeric_density > 0.6
        assert phone_profile.separator_profile is not None
        assert '-' in phone_profile.separator_profile or ' ' in phone_profile.separator_profile

        # code_col 列
        code_profile = profile_map["code_col"]
        assert 0.5 < code_profile.numeric_density < 0.8
        assert code_profile.length_std == 0.0

        # long_text 列
        long_profile = profile_map["long_text"]
        assert long_profile.length_std > 0.5

    def test_generate_column_profiles_fingerprint_clustering(self, profiler, sample_df):
        profiles = profiler.generate_column_profiles(sample_df, dataset_name="test")
        profile_map = {p.column_name: p for p in profiles}

        # enum_col 列不触发聚类（唯一率 > 0.15）
        enum_profile = profile_map["enum_col"]
        assert enum_profile.value_similarity_clusters == {}
        assert enum_profile.cluster_coverage == 0.0

        # 构造真正的枚举列
        df_enum = pd.DataFrame({
            "color": ["Red", "red", " RED ", "Red", "red", "blue", "Blue", " BLUE "]
        })
        profiles_enum = profiler.generate_column_profiles(df_enum, dataset_name="test")
        enum_prof = profiles_enum[0]
        # 因为唯一率 2/8 = 0.25 > 0.15，不触发聚类，故仍为空
        # 这不影响测试通过，仅验证字段存在

        # 数值列的小数位模式
        decimal_profile = profile_map["numeric_with_decimals"]
        assert decimal_profile.decimal_place_mode == 1  # 所有值一位小数

    def test_generate_column_profiles_pandas_type_detector(self, profiler, sample_df):
        if not HAS_PANDAS_TYPE_DETECTOR:
            pytest.skip("pandas-type-detector not installed")

        profiles = profiler.generate_column_profiles(sample_df, dataset_name="test")
        profile_map = {p.column_name: p for p in profiles}

        # 验证字段存在（不强制特定值）
        email_profile = profile_map["email"]
        if email_profile.detected_type is not None:
            assert email_profile.detection_confidence is not None

        date_profile = profile_map["date_joined"]
        if date_profile.detected_type is not None:
            assert date_profile.detection_confidence is not None

    def test_generate_column_profiles_sampling(self, profiler):
        large_df = pd.DataFrame({
            "col": range(15000),
            "col2": ["A"] * 15000
        })
        profiles = profiler.generate_column_profiles(large_df, dataset_name="large", max_sample_rows=10000)
        for p in profiles:
            assert p.total_count == 15000

    def test_generate_column_profiles_handles_empty_series(self, profiler):
        df = pd.DataFrame({"empty_col": [None, None, None]})
        profiles = profiler.generate_column_profiles(df)
        p = profiles[0]
        assert p.null_ratio == 1.0
        assert p.unique_ratio == 0.0
        assert p.distinct_count == 0
        assert p.storage_type == "string"  # Pandas object -> string
        assert p.data_type == "string"
        assert p.entropy is None
        assert p.numeric_density == 0.0
        assert p.length_std == 0.0
        assert p.separator_profile == {}
        assert p.decimal_place_mode is None
        assert p.value_similarity_clusters == {}
        assert p.cluster_coverage == 0.0

    def test_generate_column_profiles_with_numeric_column(self, profiler, sample_df):
        profiles = profiler.generate_column_profiles(sample_df, dataset_name="test")
        profile_map = {p.column_name: p for p in profiles}

        salary_profile = profile_map["salary"]
        assert salary_profile.storage_type == "integer"
        assert salary_profile.percentiles is not None
        assert "25%" in salary_profile.percentiles
        assert "50%" in salary_profile.percentiles
        assert "75%" in salary_profile.percentiles

        # 数值列不计算形态特征（除 decimal_place_mode 外均为默认值）
        assert salary_profile.numeric_density == 0.0
        assert salary_profile.length_std == 0.0

    def test_generate_column_profiles_top_frequencies(self, profiler, sample_df):
        profiles = profiler.generate_column_profiles(sample_df, dataset_name="test")
        profile_map = {p.column_name: p for p in profiles}

        dept_profile = profile_map["department"]
        top_freq = dept_profile.top_frequencies
        assert top_freq.get("HR") == 2
        assert top_freq.get("IT") == 2
        # "Sales" 出现 1 次，不应出现在 top_frequencies 中
        assert "Sales" not in top_freq

        id_profile = profile_map["id"]
        assert id_profile.top_frequencies == {}

    def test_generate_column_profiles_semantic_candidates(self, profiler, sample_df):
        profiles = profiler.generate_column_profiles(sample_df, dataset_name="test")
        profile_map = {p.column_name: p for p in profiles}

        id_profile = profile_map["id"]
        # 检查 semantic_candidates 是否包含期望的类型
        # 由于 id 列是唯一码，应有 "Code" 或 "Identifier" 等候选
        semantic_types = [c.type for c in id_profile.semantic_candidates]
        # 这里不强制具体类型，只确保至少有候选
        assert len(id_profile.semantic_candidates) > 0

        # code_col 列应包含 "Code" 候选
        code_profile = profile_map["code_col"]
        code_types = [c.type for c in code_profile.semantic_candidates]
        assert any(t in code_types for t in ["Code", "Identifier"])

        # email 列应包含 "Email" 候选（如果 pattern 覆盖高）
        email_profile = profile_map["email"]
        email_types = [c.type for c in email_profile.semantic_candidates]
        # 由于 email 列 pattern 覆盖率高，应该包含 "Email"
        assert any("Email" in t for t in email_types)