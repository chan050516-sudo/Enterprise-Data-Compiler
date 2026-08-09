import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from app.profiler.detectors import DateDetector

detector = DateDetector()

# tax_id 的样本值
test_values = ["T001", "T065", "T043", "T100", "T002"]

for val in test_values:
    is_date, fmt, conf = detector._classify_value(val)
    print(f"{val:10} -> is_date={is_date}, fmt={fmt}, conf={conf:.2f}")