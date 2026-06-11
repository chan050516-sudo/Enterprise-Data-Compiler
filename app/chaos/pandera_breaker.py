import pandas as pd
import pandera as pa
import numpy as np
import json
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ODCSContractBreaker:
    """
    Layer 0 - Chaos Engine: Contract Breaker (Driven by Pandera + Hypothesis)
    职责：
    1. 解析 Layer 3 的 ODCS 契约。
    2. 翻译为 Pandera Schema，生成 100% 合规的基准数据。
    3. 实施反向契约破坏（穿透极值、打破数学守恒、破坏外键/时效）。
    """
    def __init__(self, ontology_path: str = "app/ontology/ontology_registry.json"):
        self.ontology_path = ontology_path
        self.ontologies = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        try:
            with open(self.ontology_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Ontology registry not found at {self.ontology_path}")

    def _translate_to_pandera_schema(self, domain: str) -> pa.DataFrameSchema:
        """【核心魔法】：将 JSON ODCS 契约动态编译为 Pandera Schema"""
        ontology = self.ontologies.get(domain)
        if not ontology:
            raise ValueError(f"Domain '{domain}' not found in registry.")

        fields_config = ontology.get("fields", {})
        rules = ontology.get("odcs_contracts", {}).get("row_level_rules", [])

        # 1. 初始化列属性收集器
        col_defs = {}
        for col, config in fields_config.items():
            dtype_map = {
                "string": str,
                "float": float,
                "int": int,
                "date": "datetime64[ns]"
            }
            pa_type = dtype_map.get(config["type"], str)
            col_defs[col] = {
                "type": pa_type,
                "nullable": True, # 默认允许空值
                "checks": [],
                "unique": False
            }

        # 2. 挂载 ODCS 规则断言到收集器
        for rule in rules:
            col_name = rule.get("column")
            assertion = rule.get("assertion")
            
            if not col_name or col_name not in col_defs:
                continue

            if assertion == "not_null":
                col_defs[col_name]["nullable"] = False
            elif assertion == "non_negative":
                col_defs[col_name]["checks"].append(pa.Check.ge(0.0))
            elif assertion == "pattern":
                col_defs[col_name]["checks"].append(pa.Check.str_matches(rule.get("regex")))
            elif assertion == "unique":
                col_defs[col_name]["unique"] = True

        # 3. 一次性实例化
        pa_columns = {}
        for col, props in col_defs.items():
            pa_columns[col] = pa.Column(
                props["type"],
                nullable=props["nullable"],
                checks=props["checks"] if props["checks"] else None,
                allow_duplicates=not props["unique"]
            )

        return pa.DataFrameSchema(pa_columns)

    def generate_adversarial_payload(self, domain: str, size: int = 50, poison_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        生成对抗数据负载。
        返回: (金牌干净数据, 被契约破坏的毒化数据)
        """
        logger.info(f"🛡️ Compiling Pandera Schema for domain: {domain}")
        schema = self._translate_to_pandera_schema(domain)
        
        # 步骤 1：利用 Hypothesis 引擎生成数学完美的 Baseline 数据
        logger.info(f"✨ Generating {size} rows of mathematically perfect Golden Data...")
        try:
            golden_df = schema.example(size=size)
        except Exception as e:
            logger.warning(f"Hypothesis strict generation failed: {str(e)}. Fallback to basic dummy generation.")
            # 【修复】：强大的降级生成器，保证流水线不崩溃
            dummy_data = {}
            for col_name, col_obj in schema.columns.items():
                if col_obj.dtype.type == np.float64: 
                    dummy_data[col_name] = np.random.uniform(10.0, 1000.0, size)
                elif col_obj.dtype.type == np.int64: 
                    dummy_data[col_name] = np.random.randint(1, 100, size)
                elif "datetime" in str(col_obj.dtype): 
                    dummy_data[col_name] = pd.date_range("2026-01-01", periods=size)
                else: 
                    dummy_data[col_name] = [f"DUMMY_VAL_{i}" for i in range(size)]
            golden_df = pd.DataFrame(dummy_data)

        toxic_df = golden_df.copy()
        ontology = self.ontologies.get(domain)
        rules = ontology.get("odcs_contracts", {}).get("row_level_rules", [])

        # 步骤 2：基于 ODCS 实施精准的反向物理规则破坏
        logger.warning(f"☣️ Injecting Contract Breaches (Poison Ratio: {poison_ratio:.0%})...")
        num_poison = max(1, int(size * poison_ratio))

        for rule in rules:
            assertion = rule.get("assertion")
            col = rule.get("column")
            
            # 攻击 1：非负防线击穿 -> 注入负数异常值
            if assertion == "non_negative":
                idx = toxic_df.sample(n=num_poison).index
                # 强行乘以 -1.5 穿透 0 的下限
                toxic_df.loc[idx, col] = toxic_df.loc[idx, col] * -1.5 - 10.0
                logger.debug(f"  [Breach] Injected Negative Values into '{col}'")

            # 攻击 2：物理守恒定律打破 (如: 销售额 != 单价 * 数量)
            elif assertion == "expression":
                formula = rule.get("formula", "")
                # 【修复】：动态从公式中提取依赖列，杜绝硬编码
                import re
                # 提取公式中所有可能属于 DataFrame 列名的变量
                potential_cols = [c for c in re.findall(r'[a-zA-Z_]\w*', formula) if c in toxic_df.columns]
                
                if potential_cols:
                    # 随机挑一个公式里的列进行破坏
                    target_col = potential_cols[0] 
                    idx = toxic_df.sample(n=num_poison).index
                    
                    # 强行给该列注入极大偏移量，撕裂数学守恒
                    if pd.api.types.is_numeric_dtype(toxic_df[target_col]):
                        toxic_df.loc[idx, target_col] += 999.99
                    else:
                        toxic_df.loc[idx, target_col] = "BROKEN_REF"
                        
                    logger.debug(f"  [Breach] Broken mathematical expression on dynamically extracted column '{target_col}'")

            # 攻击 3：合规时效击穿 (如: 质检日期与到货日期相差超过 5 天)
            elif assertion == "date_tolerance":
                target_col = rule.get("target_column")
                if col in toxic_df.columns and target_col in toxic_df.columns:
                    idx = toxic_df.sample(n=num_poison).index
                    window = rule.get("tolerance_window_days", 5)
                    # 强行推迟 N+10 天，直接触发 Layer 5 的合规警报
                    toxic_df.loc[idx, col] = toxic_df.loc[idx, target_col] + pd.Timedelta(days=window + 10)
                    logger.debug(f"  [Breach] Exceeded date tolerance between '{col}' and '{target_col}'")

        return golden_df, toxic_df

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
#     breaker = ODCSContractBreaker("app/ontology/ontology_registry.json")
    
#     # 直接攻击你的采购与物流仓储模型
#     golden, toxic = breaker.generate_adversarial_payload(domain="procurement_logistics", size=10, poison_ratio=0.3)
    
#     print("\n✅ 【Golden Data (Layer 5 会 100% 放行)】:")
#     print(golden[['order_quantity', 'unit_net_price', 'total_net_amount']].head(3))
    
#     print("\n❌ 【Toxic Data (带有财务与逻辑裂痕，将考验 Layer 5)】:")
#     print(toxic[['order_quantity', 'unit_net_price', 'total_net_amount']].head(3))