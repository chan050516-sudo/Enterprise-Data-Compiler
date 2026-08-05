import pytest
from app.schema.semantic_profiler import SemanticProfiler

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