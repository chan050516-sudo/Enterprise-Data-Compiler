"""
TargetKnowledgeIR: 目标系统知识的标准化表示

这是 Phase 5 的唯一输入来源。
无论数据来自 SAP、Oracle、DDL 还是文档，最终都归一化为这个格式。
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class TargetField(BaseModel):
    """目标系统字段定义"""
    name: str
    datatype: str
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references_table: Optional[str] = None
    references_field: Optional[str] = None
    default: Optional[Any] = None
    comment: Optional[str] = None           # 官方描述
    business_meaning: Optional[str] = None  # 业务含义
    examples: List[str] = Field(default_factory=list)
    semantic_tags: List[str] = Field(default_factory=list)  # ["customer", "identifier"]
    allowed_values: Optional[List[str]] = None


class TargetTable(BaseModel):
    """目标系统表定义"""
    name: str
    description: Optional[str] = None
    module: Optional[str] = None            # 所属模块
    fields: List[TargetField] = Field(default_factory=list)
    primary_key: Optional[List[str]] = None
    foreign_keys: List[Dict[str, str]] = Field(default_factory=list)
    semantic_tags: List[str] = Field(default_factory=list)
    
    def get_field(self, name: str) -> Optional[TargetField]:
        for f in self.fields:
            if f.name.lower() == name.lower():
                return f
        return None


class TargetKnowledgeIR(BaseModel):
    """
    目标系统知识 IR
    
    Phase 5 只消费这个格式，不关心来源。
    """
    source_type: str  # "jdbc", "sap_odata", "ddl", "manual", "dbsurveyor"
    source_name: str  # "SAP_S4HANA", "Oracle_19c", "MiniERP"
    version: Optional[str] = None
    tables: List[TargetTable] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    description: Optional[str] = None
    business_glossary: Dict[str, str] = Field(default_factory=dict)
    
    def get_table(self, name: str) -> Optional[TargetTable]:
        for t in self.tables:
            if t.name.lower() == name.lower():
                return t
        return None
    
    def get_field(self, table_name: str, field_name: str) -> Optional[TargetField]:
        table = self.get_table(table_name)
        if table:
            return table.get_field(field_name)
        return None
    
    def to_prompt_context(self, max_tables: int = 10) -> str:
        """生成 LLM Prompt 可用的上下文摘要"""
        lines = []
        lines.append(f"## Target System: {self.source_name}")
        lines.append(f"Source Type: {self.source_type}")
        lines.append(f"Tables: {len(self.tables)}")
        lines.append("")
        
        for table in self.tables[:max_tables]:
            lines.append(f"### {table.name}")
            if table.description:
                lines.append(f"Description: {table.description}")
            if table.module:
                lines.append(f"Module: {table.module}")
            if table.primary_key:
                lines.append(f"Primary Key: {', '.join(table.primary_key)}")
            lines.append("Fields:")
            for field in table.fields:
                desc = field.comment or field.business_meaning or "No description"
                pk_marker = " [PK]" if field.is_primary_key else ""
                fk_marker = f" [FK->{field.references_table}.{field.references_field}]" if field.is_foreign_key else ""
                lines.append(f"  - {field.name}{pk_marker}{fk_marker}: {desc}")
            lines.append("")
        
        if len(self.tables) > max_tables:
            lines.append(f"... and {len(self.tables) - max_tables} more tables")
        
        return "\n".join(lines)
    
    def to_llm_context(self, entity_columns: List[str], max_tables: int = 5) -> str:
        """
        为 LLM 生成针对特定实体的上下文
        
        只返回与源实体列名相关的表，减少 token 消耗
        """
        if not entity_columns:
            return self.to_prompt_context(max_tables)
        
        # 计算每个表的相关性
        scored_tables = []
        for table in self.tables:
            score = 0
            field_names = [f.name.lower() for f in table.fields]
            for col in entity_columns:
                col_lower = col.lower()
                # 精确匹配
                if col_lower in field_names:
                    score += 3
                # 部分匹配
                for fname in field_names:
                    if col_lower in fname or fname in col_lower:
                        score += 1
                        break
            scored_tables.append((score, table))
        
        # 按相关性排序
        scored_tables.sort(key=lambda x: -x[0])
        relevant_tables = [t for score, t in scored_tables if score > 0][:max_tables]
        
        if not relevant_tables:
            return self.to_prompt_context(3)
        
        lines = []
        lines.append(f"## Target System: {self.source_name}")
        lines.append("Relevant tables (based on column name similarity):")
        lines.append("")
        
        for table in relevant_tables:
            lines.append(f"### {table.name}")
            if table.description:
                lines.append(f"Description: {table.description}")
            lines.append("Fields:")
            for field in table.fields:
                desc = field.comment or field.business_meaning or "No description"
                pk_marker = " [PK]" if field.is_primary_key else ""
                lines.append(f"  - {field.name}{pk_marker}: {desc}")
            lines.append("")
        
        return "\n".join(lines)