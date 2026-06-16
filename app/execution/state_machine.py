import logging
from enum import Enum
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BatchState(Enum):
    INIT = "INIT"
    PROFILED = "PROFILED"
    COMPILED = "COMPILED"
    RECONCILED = "RECONCILED"
    COMMITTING = "COMMITTING"
    COMMITTED = "COMMITTED"
    QUARANTINED = "QUARANTINED"
    COMPENSATING = "COMPENSATING" # Saga 冲销状态
    FAILED_CRITICAL = "FAILED_CRITICAL"

class BatchLifecycle:
    """
    数据批次生命周期状态机 (FSM)
    严格控制执行平面的流转，为断点续传和审计提供支持。
    """
    def __init__(self, batch_id: str, spec_id: str):
        self.batch_id = batch_id
        self.spec_id = spec_id
        self.current_state: BatchState = BatchState.INIT
        self.transition_history = []
        self._record_transition("System Initialized")

    def _record_transition(self, reason: str):
        self.transition_history.append({
            "state": self.current_state.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason
        })
        logger.debug(f"[Batch {self.batch_id}] State transitioned to {self.current_state.value}")

    def transition_to(self, new_state: BatchState, reason: str = ""):
        # 工业级防线：防止状态机逆向流转或非法跃迁
        valid_transitions = {
            BatchState.INIT: [BatchState.PROFILED, BatchState.COMPILED],
            BatchState.COMPILED: [BatchState.RECONCILED, BatchState.QUARANTINED],
            BatchState.RECONCILED: [BatchState.COMMITTING, BatchState.QUARANTINED],
            BatchState.COMMITTING: [BatchState.COMMITTED, BatchState.COMPENSATING, BatchState.FAILED_CRITICAL],
            BatchState.COMPENSATING: [BatchState.QUARANTINED, BatchState.FAILED_CRITICAL],
        }

        if new_state not in valid_transitions.get(self.current_state, []):
            logger.error(f"Illegal state transition from {self.current_state} to {new_state}")
            raise StateError(f"Cannot transition from {self.current_state.value} to {new_state.value}")

        self.current_state = new_state
        self._record_transition(reason)
        
class StateError(Exception):
    pass