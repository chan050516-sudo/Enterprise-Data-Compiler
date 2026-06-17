import sqlite3
import json
import logging
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class TargetSchemaIntrospector:
    """
    从目标数据库（SQLite）自动发现表结构，生成 Target Ontology 格式的注册表。
    """
    @staticmethod
    def introspect(db_path: str) -> Dict[str, Any]:
        """
        返回一个字典，key 为表名，value 为 TargetOntology 格式的 dict。
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 获取所有表名（过滤系统表）
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        
        ontology_registry = {}
        
        for table_name in tables:
            # 获取表结构
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            # columns: (cid, name, type, notnull, dflt_value, pk)

            pk_columns = []
            for col in columns:
                if col[5] > 0:          # pk 标志
                    pk_columns.append((col[1], col[5]))  # (列名, pk顺序)
            pk_columns.sort(key=lambda x: x[1])          # 按顺序排列
            pk_cols = [col[0] for col in pk_columns]
            primary_key = pk_cols[0] if len(pk_cols) == 1 else pk_cols if pk_cols else None
            
            fields = {}
            odcs_row_rules = []
            
            for col in columns:
                col_name = col[1]
                col_type = col[2].upper()
                notnull = col[3] == 1
                pk = col[5] > 0
                
                # 映射 SQLite 类型到 ontology 类型
                if "INT" in col_type:
                    ontology_type = "int"
                elif "REAL" in col_type or "FLOAT" in col_type or "DOUB" in col_type:
                    ontology_type = "float"
                elif "DATE" in col_type:
                    ontology_type = "date"
                elif "DATETIME" in col_type:
                    ontology_type = "datetime"
                else:
                    ontology_type = "string"
                
                field_def = {"type": ontology_type}
                # 如果有 NOT NULL 约束，生成 not_null 规则
                if notnull:
                    odcs_row_rules.append({
                        "column": col_name,
                        "assertion": "not_null",
                        "severity": "error"
                    })
                # 如果是主键，添加 unique 规则
                if pk:
                    odcs_row_rules.append({
                        "column": col_name,
                        "assertion": "unique",
                        "severity": "error"
                    })
                
                fields[col_name] = field_def
            
            # 推断外键
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            fks = cursor.fetchall()
            # fk: (id, seq, table, from, to, on_update, on_delete, match)
            for fk in fks:
                from_col = fk[3]
                target_table = fk[2]
                # 添加 foreign_key 规则（假设目标实体为 target_table.id）
                odcs_row_rules.append({
                    "column": from_col,
                    "assertion": "foreign_key",
                    "target_entity": target_table + ".id",  # 简化，假设引用主键
                    "severity": "error"
                })
            
            # 构造 Ontology 字典
            ontology_registry[table_name] = {
                "dataset_name": table_name,
                "description": f"Auto-introspected from SQLite database: {db_path}",
                "fields": fields,
                "odcs_contracts": {
                    "row_level_rules": odcs_row_rules,
                    "dataset_level_rules": [],
                    "global_invariants": []
                },
                "primary_key": primary_key
            }
        
        conn.close()
        return ontology_registry

    @staticmethod
    def save_to_registry(db_path: str, output_path: str):
        """生成并保存为 ontology_registry.json 文件"""
        registry = TargetSchemaIntrospector.introspect(db_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        logger.info(f"Target ontology registry generated and saved to {output_path}")