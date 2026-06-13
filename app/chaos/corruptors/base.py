import random
import string

def random_string(length: int = 5) -> str:
    """生成随机小写字符串"""
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def random_choice_weighted(choices: list, weights: list):
    """加权随机选择"""
    return random.choices(choices, weights=weights)[0]