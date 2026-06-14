import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta
from faker import Faker
from typing import Dict, Any, Optional, List

fake = Faker()

class OntologyDataGenerator:
    """
    基于 Target Ontology（JSON 格式）生成干净的测试数据。
    支持 canonical_ontology.json 以及任意符合 TargetOntology 结构的字典。
    """

    @staticmethod
    def generate(
        ontology: Dict[str, Any],
        num_rows: int = 100,
        random_seed: int = 42,
        enforce_global_invariants: bool = False
    ) -> pd.DataFrame:
        """
        根据 ontology 定义生成 DataFrame。

        Args:
            ontology: TargetOntology 字典，包含 fields 和可选的 odcs_contracts。
            num_rows: 生成行数。
            random_seed: 随机种子。
            enforce_global_invariants: 是否尝试满足全局不变量（暂不实现复杂逻辑）。

        Returns:
            生成的干净 DataFrame。
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        fake.seed_instance(random_seed)

        fields_config = ontology.get("fields", {})
        data = {}

        # 先记录需要唯一值的列
        unique_columns = []
        for field_name, field_attrs in fields_config.items():
            # 目前 ontology 中没有直接标记 unique，但我们可以从 odcs_contracts 中解析
            # 简化：先不处理唯一约束，后续可以扩展
            pass

        # 为每个字段生成数据
        for field_name, field_attrs in fields_config.items():
            field_type = field_attrs.get("type", "string")
            nullable = field_attrs.get("nullable", True)  # 默认可空
            # 获取描述或其他信息（用于生成更真实的数据）
            description = field_attrs.get("description", "")

            if field_type == "string":
                # 根据字段名或描述生成不同类型字符串
                if "id" in field_name or "id" in description.lower():
                    data[field_name] = [OntologyDataGenerator._gen_id(field_name) for _ in range(num_rows)]
                elif "name" in field_name or "name" in description.lower():
                    data[field_name] = [fake.name() for _ in range(num_rows)]
                elif "email" in field_name:
                    data[field_name] = [fake.email() for _ in range(num_rows)]
                elif "address" in field_name:
                    data[field_name] = [fake.address() for _ in range(num_rows)]
                elif "phone" in field_name:
                    data[field_name] = [fake.phone_number() for _ in range(num_rows)]
                else:
                    data[field_name] = [fake.word() for _ in range(num_rows)]

            elif field_type == "int":
                # 整数，默认范围 1-10000
                low, high = 1, 10000
                data[field_name] = np.random.randint(low, high, size=num_rows)

            elif field_type == "float":
                # 浮点数，默认范围 0.0-10000.0
                data[field_name] = np.random.uniform(0.0, 10000.0, size=num_rows).round(2)

            elif field_type == "date":
                # 日期，范围 2020-01-01 到 2025-12-31
                start = datetime(2020, 1, 1)
                end = datetime(2025, 12, 31)
                delta = (end - start).days
                dates = [start + timedelta(days=random.randint(0, delta)) for _ in range(num_rows)]
                data[field_name] = pd.Series(dates).dt.date

            elif field_type == "boolean":
                data[field_name] = np.random.choice([True, False], size=num_rows)

            elif field_type == "datetime":
                start = datetime(2020, 1, 1, 0, 0, 0)
                end = datetime(2025, 12, 31, 23, 59, 59)
                delta = (end - start).total_seconds()
                datetimes = [start + timedelta(seconds=random.randint(0, int(delta))) for _ in range(num_rows)]
                data[field_name] = pd.Series(datetimes)

            else:
                # 未知类型，默认字符串
                data[field_name] = [f"unknown_{i}" for i in range(num_rows)]

            # 处理非空约束：如果 nullable=False，则将生成的 None 替换为默认值
            # 当前生成逻辑不会产生 None，但若后期引入缺失值，可在此处理
            if not nullable:
                # 例如将空字符串替换为 "N/A"
                if field_type == "string":
                    data[field_name] = [v if v and v != "" else "N/A" for v in data[field_name]]
                elif field_type in ["int", "float"]:
                    # 确保没有 NaN
                    data[field_name] = pd.Series(data[field_name]).fillna(0)

        df = pd.DataFrame(data)

        # 可选：处理字段间派生关系（如 total_amount = line_amount_sum）
        # 由于 canonical_ontology.json 中的 global_invariants 是跨行/跨实体检查，不是字段级派生，
        # 这里不实现复杂逻辑。但可以留出扩展接口。
        if enforce_global_invariants:
            # 简单示例：若有 total_amount 和 line_amount 列，使 line_amount 之和等于 total_amount
            if 'total_amount' in df.columns and 'line_amount' in df.columns:
                # 随机分配行金额，使每行总金额等于 total_amount 的对应值
                for idx in range(num_rows):
                    total = df.loc[idx, 'total_amount']
                    if total > 0:
                        # 将 total 拆分到 line_amount（假设每行只有一行明细，直接相等）
                        df.loc[idx, 'line_amount'] = total
                    else:
                        df.loc[idx, 'line_amount'] = 0.0

        return df

    @staticmethod
    def _gen_id(prefix: str = "id") -> str:
        """生成唯一标识符"""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"