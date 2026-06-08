import sqlite3
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SQLiteWriter:
    """
    Layer 8: 目标存储强类型写入器 (SQLite 镜像)
    基于 Layer 3 Ontology 自动生成确定性 DDL，并通过向量清洗实现 100% 安全的事务持久化。
    """

    # 映射表：将 Layer 3 的语义类型严格锚定为 SQLite 物理存储类型
    _TYPE_MAP = {
        "string": "TEXT",
        "int": "INTEGER",
        "float": "REAL",
        "boolean": "INTEGER", # SQLite 没有独立的布尔型，0/1 存储
        "date": "TEXT",
        "datetime": "TEXT"
    }

    @classmethod
    def write(cls, clean_df: pd.DataFrame, target_ontology: Dict[str, Any], db_path: str) -> int:
        """
        强类型物理入库。
        :param clean_df: 已经过 Layer 5/6 审判与编译的绝对干净的数据集
        :param target_ontology: 来自 Layer 3 的目标模型契约
        :param db_path: SQLite 数据库文件的物理路径
        :return: 成功写入的记录行数
        """
        table_name = target_ontology.get("dataset_name", "compiled_business_table")
        fields_config = target_ontology.get("fields", {})
        
        if clean_df.empty:
            logger.warning(f"No rows to write for table '{table_name}'. Persistence skipped.")
            return 0

        # 1. 建立独占连接，启动 ACID 事务保护
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # 2. 本体自举：动态生成确定性 DDL
            columns_ddl = []
            for col_name, field_attr in fields_config.items():
                ontology_type = field_attr.get("type", "string")
                sql_type = cls._TYPE_MAP.get(ontology_type, "TEXT")
                columns_ddl.append(f'"{col_name}" {sql_type}')
            
            # 默认追加一条系统级审计字段，用于追踪编译器流水线
            columns_ddl.append('"_compiled_at" TEXT DEFAULT CURRENT_TIMESTAMP')
            
            ddl_query = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(columns_ddl)});'
            logger.info(f"Enforcing explicit DDL Schema for table '{table_name}'...")
            cursor.execute(ddl_query)

            # 3. 终极防御：向量化脱敏清洗 (Sanitize Data for Native SQLite Driver)
            # 彻底擦除 Pandas 特有的 Int64 (pd.NA) 和 numpy.nan，降级为原生 Python 驱动可识别的 None
            sanitized_df = clean_df.copy()
            for col in sanitized_df.columns:
                # 针对 Pandas 可空整数和布尔 ExtensionDtype 的特殊处理
                if isinstance(sanitized_df[col].dtype, pd.core.dtypes.base.ExtensionDtype):
                    sanitized_df[col] = sanitized_df[col].where(sanitized_df[col].notna(), None)
                else:
                    # 针对标准数值列中的 np.nan 转换为 None
                    sanitized_df[col] = sanitized_df[col].replace({np.nan: None})

            # 4. 向量化极速对齐绑定
            columns_to_insert = [col for col in fields_config.keys() if col in sanitized_df.columns]
            placeholders = ", ".join(["?"] * len(columns_to_insert))
            insert_query = f'INSERT INTO "{table_name}" ({", ".join([f'"{c}"' for c in columns_to_insert])}) VALUES ({placeholders});'
            
            # 将 DataFrame 转化为纯 Python 原生对象矩阵
            data_matrix = sanitized_df[columns_to_insert].values.tolist()
            
            # 5. C语言底层批量提交
            logger.info(f"Executing vectorized atomic commit of {len(data_matrix)} records into SQLite...")
            cursor.executemany(insert_query, data_matrix)
            
            # 事务整体提交
            conn.commit()
            logger.info(f"🎉 Successfully persisted {len(data_matrix)} records to '{table_name}' in SQLite DB.")
            return len(data_matrix)
            
        except Exception as e:
            # 任何一个字节发生断层，满盘回滚，确保绝对整洁
            conn.rollback()
            logger.error(f"Fatal Write Error. SQLite Transaction rolled back completely. Error: {str(e)}")
            raise RuntimeError(f"Layer 8 Core Database Commit Panic: {str(e)}")
        finally:
            cursor.close()
            conn.close()