"""
DDL Provider: 从 SQL DDL 文件解析元数据

支持：PostgreSQL DDL, MySQL DDL, SQL Server DDL
"""

import logging
import re
from pathlib import Path
from typing import Optional
from app.metadata.provider import TargetMetadataProvider
from app.schema.target_knowledge_ir import TargetKnowledgeIR, TargetTable, TargetField

logger = logging.getLogger(__name__)


class DDLFileProvider(TargetMetadataProvider):
    """从 SQL DDL 文件解析元数据"""
    
    def __init__(self, file_path: str, source_name: str = "DDL_Import"):
        self.file_path = Path(file_path)
        self._source_name = source_name
        self._metadata = None
    
    def get_source_type(self) -> str:
        return "ddl"
    
    def get_source_name(self) -> str:
        return self._source_name
    
    def get_metadata(self) -> TargetKnowledgeIR:
        """解析 DDL 文件"""
        if self._metadata is not None:
            return self._metadata
        
        if not self.file_path.exists():
            logger.error(f"DDL file not found: {self.file_path}")
            return self._get_empty_metadata()
        
        try:
            import sqlparse
            from sqlparse.sql import Identifier, IdentifierList, Where
            from sqlparse.tokens import Keyword, DDL, Name
        except ImportError:
            logger.error("sqlparse not installed. Install with: pip install sqlparse")
            return self._get_empty_metadata()
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析 CREATE TABLE 语句
            tables = self._parse_create_tables(content)
            
            self._metadata = TargetKnowledgeIR(
                source_type="ddl",
                source_name=self._source_name,
                tables=tables
            )
            
            logger.info(f"DDLFileProvider: Parsed {len(tables)} tables from {self.file_path}")
            return self._metadata
            
        except Exception as e:
            logger.error(f"Failed to parse DDL: {e}")
            return self._get_empty_metadata()
    
    def _parse_create_tables(self, content: str) -> list:
        """解析 CREATE TABLE 语句"""
        tables = []
        
        # 简化的正则匹配（处理标准 DDL）
        # 格式: CREATE TABLE table_name ( ... );
        pattern = r'CREATE\s+TABLE\s+(\w+)\s*\((.*?)\)\s*;'
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        
        for table_name, columns_def in matches:
            fields = self._parse_columns(columns_def)
            tables.append(TargetTable(
                name=table_name,
                fields=fields,
                primary_key=self._find_primary_key(columns_def)
            ))
        
        return tables
    
    def _parse_columns(self, columns_def: str) -> list:
        """解析列定义"""
        fields = []
        lines = columns_def.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.upper().startswith('CONSTRAINT'):
                continue
            
            # 解析列名和类型
            parts = re.split(r'\s+', line, 2)
            if len(parts) < 2:
                continue
            
            col_name = parts[0].strip('"`[]')
            col_type = parts[1].upper()
            
            # 检查是否是约束（如 PRIMARY KEY）
            if col_type in ['PRIMARY', 'FOREIGN', 'UNIQUE']:
                continue
            
            field = TargetField(
                name=col_name,
                datatype=self._normalize_datatype(col_type)
            )
            
            # 检查是否 NOT NULL
            if 'NOT NULL' in line.upper():
                field.nullable = False
            
            # 检查是否有 DEFAULT
            default_match = re.search(r"DEFAULT\s+([^,\s]+)", line, re.IGNORECASE)
            if default_match:
                field.default = default_match.group(1)
            
            # 检查是否有 COMMENT
            comment_match = re.search(r"COMMENT\s+['\"]([^'\"]+)['\"]", line, re.IGNORECASE)
            if comment_match:
                field.comment = comment_match.group(1)
            
            fields.append(field)
        
        return fields
    
    def _find_primary_key(self, columns_def: str) -> Optional[list]:
        """查找 PRIMARY KEY 约束"""
        pk_match = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", columns_def, re.IGNORECASE)
        if pk_match:
            pk_cols = [c.strip() for c in pk_match.group(1).split(',')]
            return pk_cols
        return None
    
    def _normalize_datatype(self, dtype: str) -> str:
        """规范化数据类型"""
        dtype = dtype.upper()
        if 'VARCHAR' in dtype:
            return 'string'
        elif 'INT' in dtype:
            return 'integer'
        elif 'DECIMAL' in dtype or 'NUMERIC' in dtype:
            return 'float'
        elif 'DATE' in dtype:
            return 'date'
        elif 'TIMESTAMP' in dtype or 'DATETIME' in dtype:
            return 'datetime'
        elif 'BOOLEAN' in dtype or 'BOOL' in dtype:
            return 'boolean'
        else:
            return dtype.lower()
    
    def _get_empty_metadata(self) -> TargetKnowledgeIR:
        return TargetKnowledgeIR(
            source_type="ddl",
            source_name=self._source_name,
            description="Failed to parse DDL"
        )