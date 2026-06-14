import pandas as pd
import numpy as np
import random
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SpreadsheetCorruptor:
    """模拟电子表格典型问题：日期序列号、自动日期解析、科学计数法、前导零丢失、布尔值污染"""

    @staticmethod
    def corrupt(df: pd.DataFrame, poison_ratio: float, log: List[Dict]) -> pd.DataFrame:
        df = df.copy()
        num_poison = max(1, int(len(df) * poison_ratio))

        for col in df.columns:
            # 日期列污染
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(object)
                for i in idx:
                    val = df.at[i, col]
                    if isinstance(val, pd.Series): val = val.iloc[0] if not val.empty else np.nan
                    if pd.isna(val):
                        continue
                    old_val = val
                    # 随机选择污染类型
                    noise = random.choices(
                        ['serial', 'auto_date'],
                        weights=[0.5, 0.5]
                    )[0]
                    if noise == 'serial':
                        # 转为 Excel 序列号（约值）
                        try:
                            excel_serial = (val - pd.Timestamp('1899-12-30')).days
                            new_val = str(excel_serial)
                        except:
                            new_val = str(val)
                    else:  # auto_date
                        new_val = SpreadsheetCorruptor._excel_auto_date(val)
                    df.at[i, col] = new_val
                    log.append({
                        "row": i,
                        "column": col,
                        "corruption_type": f"SPREADSHEET_{noise.upper()}",
                        "original_value": old_val,
                        "new_value": new_val
                    })

            # 数值列污染
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(object)
                # idx = df.sample(n=num_poison).index

                for i in idx:
                    val = df.at[i, col]
                    if isinstance(val, pd.Series): val = val.iloc[0] if not val.empty else np.nan
                    if pd.isna(val):
                        continue
                    old_val = val
                    if random.random() < 0.5:
                        new_val = f"{val:.2e}"
                        df.at[i, col] = new_val
                        log.append({
                            "row": i,
                            "column": col,
                            "corruption_type": "SPREADSHEET_SCIENTIFIC",
                            "original_value": old_val,
                            "new_value": new_val
                        })

            # 字符串列可能前导零丢失（仅当全数字时）
            elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                idx = df.sample(n=num_poison).index
                for i in idx:
                    val = str(df.at[i, col])
                    if isinstance(val, pd.Series): val = val.iloc[0] if not val.empty else np.nan
                    val_str = str(val).strip()
                    if val_str.isdigit() and val_str.startswith('0'):
                        new_val = str(int(val_str))
                        df.at[i, col] = new_val
                        log.append({
                            "row": i,
                            "column": col,
                            "corruption_type": "SPREADSHEET_LEADING_ZERO",
                            "original_value": val,
                            "new_value": new_val
                        })
                    # 布尔值污染：YES/NO/TRUE/FALSE/Y/N -> 1/0 或大小写变体
                    elif val.upper() in ['YES', 'NO', 'TRUE', 'FALSE', 'Y', 'N']:
                        if random.random() < 0.3:
                            new_val = '1' if val.upper() in ['YES', 'TRUE', 'Y'] else '0'
                            df.at[i, col] = new_val
                            log.append({
                                "row": i,
                                "column": col,
                                "corruption_type": "SPREADSHEET_BOOLEAN",
                                "original_value": val,
                                "new_value": new_val
                            })

        return df

    @staticmethod
    def _excel_auto_date(val):
        """模拟 Excel 自动将日期解析为短格式字符串"""
        try:
            dt = pd.to_datetime(val)
            fmt = random.choice(['%b %d', '%d-%b', '%b-%d'])
            return dt.strftime(fmt)
        except:
            return str(val)