"""
JDBC Provider: 通过数据库连接获取元数据

支持：PostgreSQL, MySQL, SQL Server, Oracle, SAP HANA
"""

import logging
from typing import Optional
from app.metadata.provider import TargetMetadataProvider
from app.schema.target_knowledge_ir import TargetKnowledgeIR, TargetTable, TargetField

logger = logging.getLogger(__name__)


class JDBCProvider(TargetMetadataProvider):
    """通过 JDBC/ODBC 连接获取元数据"""
    
    def __init__(
        self,
        connection_string: str,
        driver: str,
        schema: Optional[str] = None,
        source_name: str = "UnknownDB"
    ):
        self.connection_string = connection_string
        self.driver = driver
        self.schema = schema
        self._source_name = source_name
        self._metadata = None
    
    def get_source_type(self) -> str:
        return "jdbc"
    
    def get_source_name(self) -> str:
        return self._source_name
    
    def get_metadata(self) -> TargetKnowledgeIR:
        """从数据库获取元数据"""
        if self._metadata is not None:
            return self._metadata
        
        try:
            import sqlalchemy
            from sqlalchemy import create_engine, inspect
        except ImportError:
            logger.error("sqlalchemy not installed")
            return self._get_empty_metadata()
        
        try:
            engine = create_engine(self.connection_string)
            inspector = inspect(engine)
            
            tables = []
            # 获取所有表
            table_names = inspector.get_table_names(schema=self.schema)
            
            for table_name in table_names:
                # 获取列信息
                columns = inspector.get_columns(table_name, schema=self.schema)
                fields = []
                pk_constraints = inspector.get_pk_constraint(table_name, schema=self.schema)
                pk_columns = pk_constraints.get("constrained_columns", [])
                fk_constraints = inspector.get_foreign_keys(table_name, schema=self.schema)
                
                for col in columns:
                    field = TargetField(
                        name=col["name"],
                        datatype=str(col["type"]),
                        nullable=col.get("nullable", True),
                        is_primary_key=col["name"] in pk_columns,
                        default=col.get("default"),
                        comment=col.get("comment")
                    )
                    fields.append(field)
                
                # 构建外键关系
                foreign_keys = []
                for fk in fk_constraints:
                    for col in fk.get("constrained_columns", []):
                        foreign_keys.append({
                            "column": col,
                            "references_table": fk.get("referred_table"),
                            "references_field": fk.get("referred_columns", [""])[0] if fk.get("referred_columns") else None
                        })
                        # 标记字段为外键
                        for field in fields:
                            if field.name == col:
                                field.is_foreign_key = True
                                field.references_table = fk.get("referred_table")
                                if fk.get("referred_columns"):
                                    field.references_field = fk.get("referred_columns")[0]
                                break
                
                tables.append(TargetTable(
                    name=table_name,
                    fields=fields,
                    primary_key=pk_columns if pk_columns else None,
                    foreign_keys=foreign_keys
                ))
            
            self._metadata = TargetKnowledgeIR(
                source_type="jdbc",
                source_name=self._source_name,
                tables=tables
            )
            
            logger.info(f"JDBCProvider: Loaded {len(tables)} tables from {self._source_name}")
            return self._metadata
            
        except Exception as e:
            logger.error(f"Failed to load metadata from JDBC: {e}")
            return self._get_empty_metadata()
    
    def _get_empty_metadata(self) -> TargetKnowledgeIR:
        return TargetKnowledgeIR(
            source_type="jdbc",
            source_name=self._source_name,
            description="Failed to load metadata"
        )