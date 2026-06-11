import pandas as pd
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

class HumanNoiseInjector:
    """
    Layer 0 - Chaos Engine: Human Noise Injector
    专门针对 Layer 6 的清理算子 (PARSE_DATE, CLEAN_CURRENCY) 进行极限施压
    """
    
    @staticmethod
    def inject_noise(df: pd.DataFrame, poison_ratio: float = 0.2) -> pd.DataFrame:
        toxic_df = df.copy()
        num_poison = max(1, int(len(toxic_df) * poison_ratio))
        
        logger.info(f"☣️ Injecting formatting noise & typos (Poison Ratio: {poison_ratio:.0%})...")

        for col in toxic_df.columns:
            dtype = toxic_df[col].dtype

            # 1. 货币与数值格式变异 (100.5 -> "RM 100.50", "100,5")
            if pd.api.types.is_numeric_dtype(toxic_df[col]):
                idx = toxic_df.sample(n=num_poison).index
                toxic_df[col] = toxic_df[col].astype(str)  # 强行退化为字符串
                
                for i in idx:
                    val = toxic_df.loc[i, col]
                    if val == "nan": continue
                    # 模拟财务手工乱填
                    noise_variants = [f"RM {val}", f"{val} MYR", val.replace(".", ",")]
                    toxic_df.loc[i, col] = random.choice(noise_variants)

            # 2. 日期格式变异 (2026-06-01 -> "01/06/2026", "Jun 1st 26")
            elif pd.api.types.is_datetime64_any_dtype(toxic_df[col]):
                idx = toxic_df.sample(n=num_poison).index
                # 预转为 datetime 对象以便重格式化
                temp_dates = pd.to_datetime(toxic_df.loc[idx, col], errors='coerce')
                
                for i, dt in temp_dates.items():
                    if pd.isnull(dt): continue
                    formats = ["%d/%m/%Y", "%b %d %Y", "%Y%m%d"]
                    toxic_df.loc[i, col] = dt.strftime(random.choice(formats))

            # 3. 字符串拼写退化与两端空格 (Pending -> "Pnding", "  Pending \n")
            elif pd.api.types.is_string_dtype(toxic_df[col]) or pd.api.types.is_object_dtype(toxic_df[col]):
                idx = toxic_df.sample(n=num_poison).index
                for i in idx:
                    val = str(toxic_df.loc[i, col])
                    if val in ["nan", "None", ""]: continue
                    
                    if random.random() > 0.5:
                        # 删除随机元音模拟错别字
                        val = "".join([c for c in val if c.lower() not in "aeiou" or random.random() > 0.5])
                    else:
                        # 注入隐形不可见字符
                        val = f"  {val} \t\n"
                    toxic_df.loc[i, col] = val

            # 4. 强制 Null 注入 (穿透防线)
            null_idx = toxic_df.sample(n=max(1, int(num_poison * 0.5))).index
            toxic_df.loc[null_idx, col] = np.nan

        return toxic_df