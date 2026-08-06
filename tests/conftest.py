import pytest
from app.profiler.semantic_profiler import SemanticProfiler
from unittest.mock import patch

@pytest.fixture(autouse=True)
def disable_preload_for_tests():
    """测试环境下不执行预加载"""
    import builtins
    original_import = builtins.__import__
    # 简单跳过预加载
    original_method = SemanticProfiler.preload_all_detectors
    SemanticProfiler.preload_all_detectors = lambda: None
    yield
    SemanticProfiler.preload_all_detectors = original_method

@pytest.fixture(autouse=True)
def disable_presidio_and_duckling():
    """在测试中禁用 Presidio 和 Duckling（避免下载大型模型）"""
    with patch('app.profiler.semantic_profiler.SemanticProfiler._get_presidio', return_value=None):
        with patch('app.profiler.semantic_profiler.SemanticProfiler._get_duckling', return_value=None):
            with patch('app.profiler.semantic_profiler.SemanticProfiler._get_type_detector', return_value=None):
                with patch('app.profiler.semantic_profiler.SemanticProfiler._generate_name_embedding', return_value=None):
                    yield