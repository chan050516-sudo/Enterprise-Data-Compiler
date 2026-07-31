"""
DBSurveyor Provider: 解析 dbsurveyor-collect 输出的 JSON

参考: https://github.com/sjdonado/dbsurveyor
"""

import json
import logging
from pathlib import Path
from typing import Optional
from app.metadata.provider import TargetMetadataProvider
from app.schema.target_knowledge_ir import TargetKnowledgeIR, TargetTable, TargetField

logger = logging.getLogger(__name__)


class DBSurveyorProvider(TargetMetadataProvider):
    """解析 dbsurveyor 导出的 JSON 文件"""
    
    def __init__(self, json_path: str, source_name: str = "DBSurveyor"):
        self.json_path = Path(json_path)
        self._source_name = source_name
        self._metadata = None
    
    def get_source_type(self) -> str:
        return "dbsurveyor"
    
    def get_source_name(self) -> str:
        return self._source_name
    
    def get_metadata(self) -> TargetKnowledgeIR:
        """解析 dbsurveyor JSON"""
        if self._metadata is not None:
            return self._metadata
        
        if not self.json_path.exists():
            logger.error(f"DBSurveyor file not found: {self.json_path}")
            return self._get_empty_metadata()
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            tables = []
            # dbsurveyor 的输出格式可能包含在 'tables' 或 'collections' 字段中
            raw_tables = data.get("tables", data.get("collections", []))
            
            for raw_table in raw_tables:
                fields = []
                raw_columns = raw_table.get("columns", raw_table.get("fields", []))
                
                for raw_col in raw_columns:
                    field = TargetField(
                        name=raw_col.get("name", raw_col.get("column_name")),
                        datatype=raw_col.get("type", raw_col.get("datatype", "string")),
                        nullable=raw_col.get("nullable", True),
                        is_primary_key=raw_col.get("primary_key", False),
                        comment=raw_col.get("comment", raw_col.get("description"))
                    )
                    fields.append(field)
                
                tables.append(TargetTable(
                    name=raw_table.get("name", raw_table.get("table_name")),
                    description=raw_table.get("description"),
                    fields=fields
                ))
            
            self._metadata = TargetKnowledgeIR(
                source_type="dbsurveyor",
                source_name=data.get("database_name", self._source_name),
                tables=tables
            )
            
            logger.info(f"DBSurveyorProvider: Loaded {len(tables)} tables from {self.json_path}")
            return self._metadata
            
        except Exception as e:
            logger.error(f"Failed to parse DBSurveyor JSON: {e}")
            return self._get_empty_metadata()
    
    def _get_empty_metadata(self) -> TargetKnowledgeIR:
        return TargetKnowledgeIR(
            source_type="dbsurveyor",
            source_name=self._source_name,
            description="Failed to parse DBSurveyor JSON"
        )