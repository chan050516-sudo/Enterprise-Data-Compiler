import logging
import pandas as pd
from typing import Dict, Any

from app.schema.ir_model import MappingSpec
from app.execution.inverse_compiler import InverseCompiler
from app.output.sqlite_writer import TargetDBWriter
from app.schema.trace_model import ExecutionTrace, SagaTrace

logger = logging.getLogger(__name__)

class SagaManager:
    """
    Saga 分布式事务补偿管理器 (Layer 7 核心护城河)
    当物理写入发生局部失败时，自动生成冲销数据抹平目标 ERP 的状态。
    """
    def __init__(self, db_writer: TargetDBWriter):
        self.inverse_compiler = InverseCompiler()
        self.db_writer = db_writer

    def execute_compensation(self, partially_committed_df: pd.DataFrame, spec: MappingSpec, batch_id: str, target_ontology: Dict[str, Any], trace: Dict[str, Any] = None):
        logger.warning("🛡️ SAGA MANAGER ACTIVATED: Commencing Reversal Transaction...")
        
        try:
            # 1. 生成逆向数据 (红字凭证)
            reversal_df = self.inverse_compiler.generate_reversal(partially_committed_df, spec, target_ontology)
            
            # 在 trace 中记录
            if trace is not None:
                # 将 reversal_df 转为 dict 列表（注意处理 NaN）
                records = reversal_df.where(pd.notna(reversal_df), None).to_dict(orient='records')
                trace.saga = SagaTrace(
                    triggered=True,
                    reversal_records=records[:100],
                    reversal_rows=len(records)
                )
                
            # 2. 将冲销数据打入死信/应急通道写入目标库
            # 这里的 fallback_commit 使用了独立事务或紧急 API 接口
            self.db_writer.fallback_commit_reversal(reversal_df, target_ontology["dataset_name"], batch_id)
            
            logger.info("🛡️ SAGA COMPENSATION COMPLETE: Target system state successfully neutralized (zeroed out).")
        except Exception as e:
            # 极端灾难：冲销也失败了。必须触发最高级别的人工物理干预报警
            logger.critical(f"🔥 SAGA FATAL FAILURE: Unable to apply reversal records. Manual DB intervention required! Error: {e}")
            raise