import pandas as pd
import random
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# 键盘邻近映射（QWERTY 布局）
KEYBOARD_NEIGHBORS = {
    'q': ['w','a','s'], 'w': ['q','e','a','s','d'], 'e': ['w','r','s','d','f'],
    'r': ['e','t','d','f','g'], 't': ['r','y','f','g','h'], 'y': ['t','u','g','h','j'],
    'u': ['y','i','h','j','k'], 'i': ['u','o','j','k','l'], 'o': ['i','p','k','l'],
    'p': ['o','l'],
    'a': ['q','w','s','z'], 's': ['q','w','e','a','d','z','x'],
    'd': ['w','e','r','s','f','x','c'], 'f': ['e','r','t','d','g','c','v'],
    'g': ['r','t','y','f','h','v','b'], 'h': ['t','y','u','g','j','b','n'],
    'j': ['y','u','i','h','k','n','m'], 'k': ['u','i','o','j','l','m'],
    'l': ['i','o','p','k'],
    'z': ['a','s','x'], 'x': ['z','s','d','c'], 'c': ['x','d','f','v'],
    'v': ['c','f','g','b'], 'b': ['v','g','h','n'], 'n': ['b','h','j','m'],
    'm': ['n','j','k']
}

class HumanCorruptor:
    """模拟人工录入错误：拼写、大小写、空格、缩写（带权重）"""

    @staticmethod
    def corrupt(df: pd.DataFrame, poison_ratio: float, log: List[Dict]) -> pd.DataFrame:
        """
        对字符串列进行人类风格污染。
        """
        df = df.copy()
        num_poison = max(1, int(len(df) * poison_ratio))

        for col in df.columns:
            # 仅处理字符串/对象列
            if not (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
                continue
            idx = df.sample(n=num_poison).index
            for i in idx:
                val = str(df.at[i, col])
                if val in ['nan', 'None', '']:
                    continue
                old_val = val

                # 加权选择错误类型
                error_type = random.choices(
                    ['typo', 'case', 'space', 'abbrev'],
                    weights=[0.45, 0.25, 0.20, 0.10]
                )[0]

                if error_type == 'typo':
                    # 80% 键盘邻近错误，20% 随机字母
                    if random.random() < 0.8:
                        val = HumanCorruptor._keyboard_typo(val)
                    else:
                        val = HumanCorruptor._random_typo(val)
                elif error_type == 'case':
                    val = val.upper() if random.random() > 0.5 else val.lower()
                elif error_type == 'space':
                    if random.random() > 0.5:
                        val = f" {val} "  # 两边加空格
                    else:
                        val = val.strip().replace(" ", "")  # 删除空格
                elif error_type == 'abbrev':
                    if len(val) > 5:
                        val = val[:3] if random.random() > 0.5 else val[0]
                    else:
                        val = val[0] if random.random() > 0.5 else val

                df.at[i, col] = val
                log.append({
                    "row": i,
                    "column": col,
                    "corruption_type": f"HUMAN_{error_type.upper()}",
                    "original_value": old_val,
                    "new_value": val
                })
        return df

    @staticmethod
    def _keyboard_typo(word: str) -> str:
        """根据键盘邻近替换一个字符"""
        if len(word) < 2:
            return word
        idx = random.randint(0, len(word)-1)
        c = word[idx].lower()
        if c in KEYBOARD_NEIGHBORS:
            new_c = random.choice(KEYBOARD_NEIGHBORS[c])
            if word[idx].isupper():
                new_c = new_c.upper()
            return word[:idx] + new_c + word[idx+1:]
        else:
            return HumanCorruptor._random_typo(word)

    @staticmethod
    def _random_typo(word: str) -> str:
        """随机替换为一个字母"""
        if len(word) < 2:
            return word
        idx = random.randint(0, len(word)-1)
        new_c = random.choice('abcdefghijklmnopqrstuvwxyz')
        if word[idx].isupper():
            new_c = new_c.upper()
        return word[:idx] + new_c + word[idx+1:]