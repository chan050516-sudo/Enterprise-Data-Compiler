import sqlite3
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SQLiteWriter:
    """
    Layer 8: 目标存储强类型写入器 (SQLite 镜像)
    支持标准原子提交 (commit) 与 Saga 逆向冲销提交 (fallback_commit_reversal)。
    """
    _TYPE_MAP = {
        "string": "TEXT",
        "int": "INTEGER",
        "float": "REAL",
        "boolean": "INTEGER", 
        "date": "TEXT",
        "datetime": "TEXT"
    }

    def __init__(self, db_path: str):
        # [修改 1] 从 classmethod 改为实例，持有数据库连接状态
        self.db_path = db_path

    def commit(self, clean_df: pd.DataFrame, target_ontology: Dict[str, Any]) -> int:
        """
        标准正向物理入库 (被 Orchestrator 调用)
        """
        table_name = target_ontology.get("dataset_name", "compiled_business_table")
        fields_config = target_ontology.get("fields", {})
        
        if clean_df.empty:
            logger.warning(f"No rows to write for table '{table_name}'. Persistence skipped.")
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 1. 动态生成确定性 DDL
            self._ensure_table_exists(cursor, table_name, fields_config)

            # 2. 向量化脱敏清洗 (防 pd.NA 导致 SQLite C 驱动崩溃)
            sanitized_df = self._sanitize_for_sqlite(clean_df)

            # 3. 极速对齐绑定与入库
            rows_inserted = self._execute_bulk_insert(cursor, table_name, sanitized_df, fields_config)
            
            conn.commit()
            logger.info(f"🎉 Successfully persisted {rows_inserted} records to '{table_name}' in SQLite DB.")
            return rows_inserted
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Fatal Write Error. SQLite Transaction rolled back. Error: {str(e)}")
            raise RuntimeError(f"Layer 8 Core Database Commit Panic: {str(e)}")
        finally:
            cursor.close()
            conn.close()

    def fallback_commit_reversal(self, reversal_df: pd.DataFrame, target_ontology: Dict[str, Any]) -> int:
        """
        [新增] Saga 补偿专属接口 (被 SagaManager 调用)
        使用独立短事务强行写入红字冲销凭证，抹平账目。
        """
        table_name = target_ontology.get("dataset_name", "compiled_business_table")
        fields_config = target_ontology.get("fields", {})

        if reversal_df.empty:
            return 0

        logger.critical(f"🚨 SAGA: Initiating emergency write of {len(reversal_df)} reversal records to '{table_name}'...")
        
        # 建立全新的独立应急连接，避开之前可能死锁的事务上下文
        conn = sqlite3.connect(self.db_path, isolation_level="IMMEDIATE")
        cursor = conn.cursor()

        try:
            self._ensure_table_exists(cursor, table_name, fields_config)
            sanitized_df = self._sanitize_for_sqlite(reversal_df)
            rows_inserted = self._execute_bulk_insert(cursor, table_name, sanitized_df, fields_config)
            
            conn.commit()
            logger.critical(f"✅ SAGA: Reversal records successfully hard-committed to target database.")
            return rows_inserted
        except Exception as e:
            conn.rollback()
            logger.critical(f"🔥🔥 SAGA DOUBLE FAULT: Failed to write reversal records! Database requires manual DBA intervention. Error: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

    # --- 内部复用工具方法 ---
    
    def _ensure_table_exists(self, cursor, table_name: str, fields_config: Dict[str, Any]):
        columns_ddl = []
        for col_name, field_attr in fields_config.items():
            ontology_type = field_attr.get("type", "string")
            sql_type = self._TYPE_MAP.get(ontology_type, "TEXT")
            columns_ddl.append(f'"{col_name}" {sql_type}')
        
        columns_ddl.append('"-compiled_at" TEXT DEFAULT CURRENT_TIMESTAMP')
        ddl_query = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(columns_ddl)});'
        cursor.execute(ddl_query)

    def _sanitize_for_sqlite(self, df: pd.DataFrame) -> pd.DataFrame:
        sanitized_df = df.copy()
        for col in sanitized_df.columns:
            if isinstance(sanitized_df[col].dtype, pd.core.dtypes.base.ExtensionDtype):
                sanitized_df[col] = sanitized_df[col].where(sanitized_df[col].notna(), None)
            else:
                sanitized_df[col] = sanitized_df[col].replace({np.nan: None})
        return sanitized_df

    def _execute_bulk_insert(self, cursor, table_name: str, sanitized_df: pd.DataFrame, fields_config: Dict[str, Any]) -> int:
        columns_to_insert = [col for col in fields_config.keys() if col in sanitized_df.columns]
        placeholders = ", ".join(["?"] * len(columns_to_insert))
        insert_query = f'INSERT INTO "{table_name}" ({", ".join([f'"{c}"' for c in columns_to_insert])}) VALUES ({placeholders});'
        
        total_inserted = 0
        
        # [核心改动]: 嗅探是否存在拓扑标识
        if "__tree_depth__" in sanitized_df.columns:
            logger.info(f"Hierarchical topology detected. Executing Top-Down Chunked Commit to prevent FK Deadlocks...")
            max_depth = int(sanitized_df["__tree_depth__"].max())
            
            for depth in range(max_depth + 1):
                chunk = sanitized_df[sanitized_df["__tree_depth__"] == depth]
                if chunk.empty:
                    continue
                
                data_matrix = chunk[columns_to_insert].values.tolist()
                cursor.executemany(insert_query, data_matrix)
                total_inserted += len(data_matrix)
                logger.debug(f" - Inserted Depth {depth} chunk: {len(data_matrix)} records.")
        else:
            # 标准的扁平表写入
            data_matrix = sanitized_df[columns_to_insert].values.tolist()
            cursor.executemany(insert_query, data_matrix)
            total_inserted += len(data_matrix)

        return total_inserted