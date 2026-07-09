import sqlite3
import json
import logging
from typing import Optional, List
from threading import Lock
from app.schema.ir_model import MappingSpec

logger = logging.getLogger(__name__)

class SpecRepository:
    """
    控制平面的物理持久化层 (Repository)
    提供 ACID 事务保证，防止由于并发审批导致的状态错乱。
    """
    def __init__(self, db_path: str = "control_plane.db"):
        self.db_path = db_path
        self._lock = Lock() # 进程内线程安全锁
        self._init_db()

    def _init_db(self):
        """初始化控制平面系统表"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mapping_specs (
                    spec_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_by TEXT,
                    created_at TEXT,
                    approved_by TEXT,
                    approved_at TEXT,
                    parent_spec_id TEXT,
                    ir_payload JSON NOT NULL
                )
            """)
            # 创建业务域与状态的联合索引，加速执行平面查询 LOCKED 契约的速度
            conn.execute("CREATE INDEX IF NOT EXISTS idx_domain_status ON mapping_specs(domain, status)")
            logger.info("Control Plane Database Initialized.")

    def _get_connection(self):
        # 强制开启 SQLite 的外键与事务支持
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, spec: MappingSpec) -> None:
        """Upsert 语义。保存或更新 Spec。"""
        with self._lock, self._get_connection() as conn:
            payload = spec.model_dump_json()
            conn.execute("""
                INSERT INTO mapping_specs 
                (spec_id, domain, version, status, created_by, created_at, approved_by, approved_at, parent_spec_id, ir_payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(spec_id) DO UPDATE SET
                status=excluded.status,
                approved_by=excluded.approved_by,
                approved_at=excluded.approved_at,
                ir_payload=excluded.ir_payload
            """, (
                spec.spec_id, spec.domain, spec.version, spec.status, 
                spec.created_by, spec.created_at, spec.approved_by, 
                spec.approved_at, spec.parent_spec_id, payload
            ))
            logger.debug(f"Saved MappingSpec: {spec.spec_id} [{spec.status}]")

    def get_by_id(self, spec_id: str) -> Optional[MappingSpec]:
        """根据 ID 精确提取"""
        with self._get_connection() as conn:
            row = conn.execute("SELECT ir_payload FROM mapping_specs WHERE spec_id = ?", (spec_id,)).fetchone()
            if row:
                return MappingSpec.model_validate_json(row['ir_payload'])
            return None

    def get_active_locked_spec(self, domain: str) -> Optional[MappingSpec]:
        """执行平面调用的核心方法：拉取指定业务域当前唯一生效的契约"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT ir_payload FROM mapping_specs WHERE domain = ? AND status = 'LOCKED' ORDER BY created_at DESC LIMIT 1", 
                (domain,)
            ).fetchone()
            if row:
                return MappingSpec.model_validate_json(row['ir_payload'])
            return None

    def archive_all_locked_for_domain(self, domain: str, exclude_spec_id: str) -> None:
        """排他性锁：确保同一个业务域下，只有一个 Spec 处于 LOCKED 状态"""
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                UPDATE mapping_specs 
                SET status = 'ARCHIVED' 
                WHERE domain = ? AND status = 'LOCKED' AND spec_id != ?
            """, (domain, exclude_spec_id))

    def promote_to_locked(self, spec_id: str, approver_id: str, approved_at: str) -> None:
        """
        原子地将指定 spec 转为 LOCKED, 并归档同域其他 LOCKED 规格。
        确保同一 domain 只有一个 LOCKED 规格。
        """
        with self._lock, self._get_connection() as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                # 1. 更新当前 spec 为 LOCKED
                conn.execute("""
                    UPDATE mapping_specs 
                    SET status = 'LOCKED', approved_by = ?, approved_at = ?
                    WHERE spec_id = ?
                """, (approver_id, approved_at, spec_id))
                
                # 2. 归档同域其他 LOCKED 规格（排除自身）
                conn.execute("""
                    UPDATE mapping_specs 
                    SET status = 'ARCHIVED' 
                    WHERE domain = (SELECT domain FROM mapping_specs WHERE spec_id = ?)
                    AND status = 'LOCKED' AND spec_id != ?
                """, (spec_id, spec_id))
                
                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK")
                raise RuntimeError(f"Atomic promotion to LOCKED failed: {e}")