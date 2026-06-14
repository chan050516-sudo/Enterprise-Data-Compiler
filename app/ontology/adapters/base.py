from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd

class BaseAdapter(ABC):
    """ERP 系统适配器基类"""
    
    @abstractmethod
    def introspect_schema(self, connection_params: Dict[str, Any]) -> Dict[str, Any]:
        """连接目标系统，返回 Vendor Schema Graph（表、列、类型、约束）"""
        pass
    
    @abstractmethod
    def get_mapping_hints(self, vendor_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """返回 vendor 字段到 canonical 字段的映射提示"""
        pass
    
    @staticmethod
    def get_canonical_ontology_path() -> str:
        """返回 canonical ontology 文件路径（静态）"""
        return "app/ontology/canonical_ontology.json"