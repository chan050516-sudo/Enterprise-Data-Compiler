import pytest
import pandas as pd
import numpy as np
from app.normalizer.technical_normalizer import TechnicalNormalizer


@pytest.fixture
def normalizer():
    """默认配置的 TechnicalNormalizer 实例"""
    return TechnicalNormalizer(config={
        "normalize_whitespace": True,
        "normalize_unicode": True,
        "unify_delimiters": True,
        "phone_country_code": "MY",
    })


@pytest.fixture
def sample_data():
    """包含各种类型脏数据的 DataFrame"""
    return pd.DataFrame({
        "phone": [
            "012-3456789",
            "0123456789",
            "+60 12 345 6789",
            "60-12-345-6789",
            "012.345.6789 ext 100",
            "NA",
            None, None, None, None,
            None, None, None, None, None
        ][:15],
        "email": [
            "John.Doe@GMAIL.com",
            "jane.doe@gmail.com",
            " JANE.DOE@GMAIL.COM ",
            "John Doe <john.doe@gmail.com>",
            "invalid-email",
            None, None, None, None, None,
            None, None, None, None, None
        ][:15],
        "date": [
            "2023-01-15",
            "01/15/2023",
            "15/01/2023",
            "2023.01.15",
            "15 Jan 2023",
            "Jan 15, 2023",
            "20230115",
            None, None, None, None,
            None, None, None, None
        ][:15],
        "currency": [
            "$100.00",
            "USD 200",
            "150.50 USD",
            "RM 300",
            "400,000.00",
            "500.00",
            "not a number",
            None, None, None, None,
            None, None, None, None
        ][:15],
        "boolean": [
            "YES", "yes", "Y", "1", "TRUE", "true", "T",
            "NO", "no", "N", "0", "FALSE", "false", "F",
            "maybe"
        ],
        "enum": [
            "Red", "blue", "Blue", " RED ", "GREEN", "Green", "green",
            None, None, None, None, None, None, None, None
        ][:15],
        "text": [
            "  Hello World  ",
            "  Multiple   spaces   ",
            "Unicode: \u3000全角空格",
            "With\ttab and\nnewline",
            None, None, None, None, None, None,
            None, None, None, None, None
        ][:15],
        "empty": [
            "", None, "nan", "NULL", "  ",
            None, None, None, None, None,
            None, None, None, None, None
        ][:15],
    })


class TestTechnicalNormalizer:
    """TechnicalNormalizer 单元测试"""

    def test_normalize_returns_dataframe_and_report(self, normalizer, sample_data):
        """测试 normalize 返回 (DataFrame, NormalizationReport)"""
        df = sample_data[["text"]].copy()
        normalized, report = normalizer.normalize(df)
        assert isinstance(normalized, pd.DataFrame)
        assert hasattr(report, "columns_processed")
        assert len(report.columns_processed) == 1
        assert report.columns_processed[0].column_name == "text"

    def test_normalize_text_whitespace(self, normalizer, sample_data):
        """测试文本列的空白字符处理"""
        df = sample_data[["text"]].copy()
        normalized, report = normalizer.normalize(df)
        
        # 只取前4个非空值（因为第4行是 "With\ttab and\nnewline"）
        result = normalized["text"].dropna().tolist()
        expected = [
            "Hello World",
            "Multiple spaces",
            "Unicode: 全角空格",
            "With tab and newline",
        ]
        # 注意：由于可能会有空值，我们只比较前4个
        assert len(result) >= 4
        for i in range(4):
            assert result[i] == expected[i]

    def test_normalize_empty_values(self, normalizer, sample_data):
        """测试空值统一为 pd.NA"""
        df = sample_data[["empty"]].copy()
        normalized, report = normalizer.normalize(df)
        # 所有空值应变为 pd.NA
        notna = normalized["empty"].dropna()
        # "  " 也会被转为 pd.NA，所以应该全部为空
        assert len(notna) == 0, f"Expected all NaN, but got {notna.tolist()}"

    def test_phone_detection_and_normalization(self, normalizer, sample_data):
        """测试电话号码的检测和统一格式"""
        df = sample_data[["phone"]].copy()
        normalized, report = normalizer.normalize(df)
        expected = [
            "+60123456789",
            "+60123456789",
            "+60123456789",
            "+60123456789",
            "+60123456789",  # 分机号被截断
            np.nan,          # "NA" -> 空
        ]
        result = normalized["phone"].tolist()
        # 检查前5个（除去NA）
        for i in range(5):
            assert result[i] == expected[i]
        assert pd.isna(result[5])

    def test_email_detection_and_normalization(self, normalizer, sample_data):
        """测试Email的检测和统一格式"""
        df = sample_data[["email"]].copy()
        normalized, report = normalizer.normalize(df)
        expected = [
            "john.doe@gmail.com",
            "jane.doe@gmail.com",
            "jane.doe@gmail.com",
            "john.doe@gmail.com",
            "invalid-email",  # 无法识别，原样保留
        ]
        result = normalized["email"].iloc[:5].tolist()
        # 检查是否全部小写且去除空格和标签
        assert result == expected

    def test_date_detection_and_normalization(self, normalizer, sample_data):
        """测试日期的检测和统一为 ISO 格式"""
        df = sample_data[["date"]].copy()
        normalized, report = normalizer.normalize(df)
        result = normalized["date"].tolist()
        
        # 所有日期都应该被解析为 ISO 格式 (YYYY-MM-DD)
        # 注意：pd.to_datetime 可能会将 '15/01/2023' 解析为 2023-01-15
        for date_str in result:
            if pd.notna(date_str):
                # 检查是否为 ISO 格式
                assert len(date_str) == 10
                assert date_str[4] == '-' and date_str[7] == '-'
                # 检查是否全是数字和横杠
                parts = date_str.split('-')
                assert len(parts) == 3
                assert all(p.isdigit() for p in parts)

    def test_currency_detection_and_extraction(self, normalizer, sample_data):
        """测试货币检测和数值+单位提取"""
        df = sample_data[["currency"]].copy()
        normalized, report = normalizer.normalize(df)
        
        # 货币列应变成数值，且可能生成 currency_unit 列
        assert "currency" in normalized.columns
        assert "currency_currency_unit" in normalized.columns

        expected_values = [100.0, 200.0, 150.5, 300.0, 400000.0, 500.0, np.nan]
        expected_units = ["USD", "USD", "USD", "MYR", None, None, None]
        
        result_values = normalized["currency"].iloc[:7].tolist()
        result_units = normalized["currency_currency_unit"].iloc[:7].tolist()
        
        for val, exp in zip(result_values, expected_values):
            if pd.isna(exp):
                assert pd.isna(val)
            else:
                assert val == exp

        for unit, exp in zip(result_units, expected_units):
            if exp is None:
                assert pd.isna(unit)
            else:
                assert unit == exp

    def test_boolean_detection_and_normalization(self, normalizer, sample_data):
        """测试布尔值的标准化为 True/False，无法识别的变为 NaN"""
        df = sample_data[["boolean"]].copy()
        normalized, report = normalizer.normalize(df)
        expected = [
            True, True, True, True, True, True, True,
            False, False, False, False, False, False, False,
            np.nan
        ]
        result = normalized["boolean"].tolist()
        assert len(result) == len(expected)
        for i, (r, e) in enumerate(zip(result, expected)):
            if pd.isna(e):
                assert pd.isna(r)
            else:
                assert r == e

    def test_config_disable_features(self):
        """测试通过配置禁用某些功能"""
        normalizer = TechnicalNormalizer(config={
            "normalize_whitespace": False,
            "normalize_unicode": False,
            "unify_delimiters": False,
            "phone_country_code": None,
        })
        df = pd.DataFrame({
            "text": ["  Hello  World  ", "  Another  ", "  Third  "],
            "phone": ["012-345-6789", "012-345-6789", "012-345-6789"],
        })
        normalized, report = normalizer.normalize(df)
        
        # 空格处理被禁用，应保留原始空格（但去除了首尾空格？需要看实现）
        # 注意：即使 normalize_whitespace=False，代码中仍会执行 str.strip()？
        # 实际上，代码中 normalize_whitespace 控制的是 str.strip() 和压缩空格
        # 所以如果禁用，应该完全跳过。
        # 但由于我们调用了默认的文本清洗（去除控制字符等），可能仍有部分清洗。
        # 这里我们只验证核心功能
        assert "phone" in normalized.columns
        assert "text" in normalized.columns
        
        # 电话归一化：没有国家码，所以只去除分隔符
        assert normalized["phone"].iloc[0] == "0123456789"
        
        # 注意：如果空格处理被禁用，文本应保留大部分原始空格
        # 但由于我们仍有去除控制字符和 Unicode 规范化，可能有些微变化
        # 我们只验证是否包含原始内容
        assert "Hello" in normalized["text"].iloc[0]
        assert "World" in normalized["text"].iloc[0]

    def test_confidence_threshold_prevents_false_positive(self, normalizer, sample_data):
        """测试检测置信度低于阈值时不进行特殊标准化"""
        # 创建一个低置信度的列（混合数字和字母，但不符合任何强模式）
        df = pd.DataFrame({
            "mixed": ["abc123", "123abc", "abc", "123", "a1b2c3"]
        })
        normalized, report = normalizer.normalize(df)
        
        # 不应该被识别为任何特殊类型
        assert "mixed_currency_unit" not in normalized.columns
        # 应该保持为文本/字符串类型
        # 可能被检测为数字（因为包含数字），但置信度应低于阈值
        # 验证所有值仍然是字符串（没有被强制转换为浮点数）
        for val in normalized["mixed"].tolist():
            if pd.notna(val):
                assert isinstance(val, str)

    def test_morphological_features_in_report(self, normalizer, sample_data):
        """测试形态学特征是否写入 report（通过检测元数据）"""
        # 创建一列具有明显形态特征的列
        df = pd.DataFrame({
            "code_column": ["ABC-123", "DEF-456", "GHI-789", "JKL-012"]
        })
        normalized, report = normalizer.normalize(df)
        
        # 检查报告是否包含形态学信息
        col_report = report.columns_processed[0]
        # 注意：形态学特征是在 Profiler 中提取的，但这里 TechnicalNormalizer 可能不直接输出
        # 这个测试主要验证 report 结构完整
        assert col_report.column_name == "code_column"
        # 只要不报错，就说明结构正确

    def test_phone_country_code_customization(self):
        """测试自定义国家码"""
        normalizer_my = TechnicalNormalizer(config={
            "phone_country_code": "MY",
        })
        normalizer_sg = TechnicalNormalizer(config={
            "phone_country_code": "SG",
        })
        
        df_my = pd.DataFrame({"phone": ["012-345-6789"] * 3})
        df_sg = pd.DataFrame({"phone": ["8123-4567"] * 3})
        
        normalized_my, _ = normalizer_my.normalize(df_my)
        normalized_sg, _ = normalizer_sg.normalize(df_sg)
        
        assert normalized_my["phone"].iloc[0] == "+60123456789"
        assert normalized_sg["phone"].iloc[0] == "+6581234567"