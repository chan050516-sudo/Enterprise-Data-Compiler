"""
Target Metadata Provider 抽象层

所有具体 Extractor 都继承这个接口。
Phase 5 永远只通过这个接口获取数据，不关心具体来源。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Type
from app.schema.target_knowledge_ir import TargetKnowledgeIR


class TargetMetadataProvider(ABC):
    """
    目标元数据提供者抽象接口
    
    职责：将各种来源（JDBC、DDL、SAP API、文档）的元数据
    统一转换为 TargetKnowledgeIR
    """
    
    @abstractmethod
    def get_metadata(self) -> TargetKnowledgeIR:
        """返回标准化的 TargetKnowledgeIR"""
        pass
    
    @abstractmethod
    def get_source_type(self) -> str:
        """返回来源类型：'jdbc', 'sap_odata', 'ddl', 'manual', 'dbsurveyor'"""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """返回来源名称：'SAP_S4HANA', 'Oracle_19c', 'MiniERP'"""
        pass
    
    def get_table(self, name: str):
        """便捷方法：获取指定表"""
        metadata = self.get_metadata()
        return metadata.get_table(name)


class MetadataProviderRegistry:
    """
    Provider 注册表
    
    支持动态注册和发现 Provider
    """
    
    _providers: dict = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[TargetMetadataProvider]):
        """注册一个 Provider 类"""
        cls._providers[name] = provider_class
        return provider_class
    
    @classmethod
    def get(cls, name: str) -> Type[TargetMetadataProvider]:
        """获取 Provider 类"""
        if name not in cls._providers:
            raise ValueError(f"Provider '{name}' not registered. Available: {cls.list()}")
        return cls._providers[name]
    
    @classmethod
    def list(cls) -> List[str]:
        """列出所有已注册的 Provider"""
        return list(cls._providers.keys())