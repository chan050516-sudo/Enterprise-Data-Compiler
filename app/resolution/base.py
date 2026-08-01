"""
Base Resolution Operator

所有 Resolution Operator 的抽象基类。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
from app.schema.evidence import Evidence, Candidate
from app.schema.resolution import ResolutionPlan


class BaseResolutionOperator(ABC):
    """
    Resolution Operator 基类
    
    所有 Operator 必须实现 execute 方法。
    输入输出统一为 Evidence 或 Candidate。
    """
    
    @abstractmethod
    def execute(
        self,
        plan: ResolutionPlan,
        df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> List[Evidence]:
        """
        执行 Operator
        
        Args:
            plan: Resolution 执行计划
            df: 原始数据（可选）
            **kwargs: 额外参数
        
        Returns:
            List[Evidence]: 产出的证据列表
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取 Operator 名称"""
        pass
    
    @abstractmethod
    def get_category(self) -> str:
        """获取 Operator 分类"""
        pass