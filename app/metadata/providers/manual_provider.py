"""
Manual Provider: 从 JSON/YAML 配置加载目标知识

用于：测试、Demo、或无法自动提取的情况
"""

import json
import logging
from pathlib import Path
from typing import Optional
from app.metadata.provider import TargetMetadataProvider
from app.schema.target_knowledge_ir import TargetKnowledgeIR, TargetTable, TargetField

logger = logging.getLogger(__name__)


class ManualProvider(TargetMetadataProvider):
    """从手动配置的 JSON 文件加载元数据"""
    
    def __init__(self, config_path: str, source_name: str = "Manual"):
        self.config_path = Path(config_path)
        self._source_name = source_name
        self._metadata = None
    
    def get_source_type(self) -> str:
        return "manual"
    
    def get_source_name(self) -> str:
        return self._source_name
    
    def get_metadata(self) -> TargetKnowledgeIR:
        """从 JSON 文件加载"""
        if self._metadata is not None:
            return self._metadata
        
        if not self.config_path.exists():
            logger.error(f"Config file not found: {self.config_path}")
            return self._get_empty_metadata()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            tables = []
            for table_data in data.get("tables", []):
                fields = []
                for field_data in table_data.get("fields", []):
                    fields.append(TargetField(**field_data))
                
                tables.append(TargetTable(
                    name=table_data.get("name"),
                    description=table_data.get("description"),
                    module=table_data.get("module"),
                    fields=fields,
                    primary_key=table_data.get("primary_key"),
                    semantic_tags=table_data.get("semantic_tags", [])
                ))
            
            self._metadata = TargetKnowledgeIR(
                source_type="manual",
                source_name=data.get("source_name", self._source_name),
                tables=tables,
                description=data.get("description")
            )
            
            logger.info(f"ManualProvider: Loaded {len(tables)} tables from {self.config_path}")
            return self._metadata
            
        except Exception as e:
            logger.error(f"Failed to load manual config: {e}")
            return self._get_empty_metadata()
    
    def _get_empty_metadata(self) -> TargetKnowledgeIR:
        return TargetKnowledgeIR(
            source_type="manual",
            source_name=self._source_name,
            description="Failed to load manual config"
        )