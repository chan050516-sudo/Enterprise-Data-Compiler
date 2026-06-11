import pandas as pd
import logging
from typing import Dict, Any, Tuple
from app.chaos.pandera_breaker import ODCSContractBreaker
from app.chaos.noise_injector import HumanNoiseInjector
from app.chaos.semantic_drifter import SemanticDrifter

logger = logging.getLogger(__name__)

class ChaosEngine:
    """
    Layer 0: Unified Chaos Orchestrator
    生成用于验证大模型和编译器的“企业级剧毒数据集”及“判卷基准答案”
    """
    def __init__(self, ontology_path: str = "app/ontology/ontology_registry.json"):
        self.breaker = ODCSContractBreaker(ontology_path)

    def generate_benchmark_payload(self, domain: str, size: int = 50, intensity: str = "Filthy") -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        intensity 级别控制破坏率：
        - Dirty: 5% 破坏率
        - Filthy: 20% 破坏率
        - Disgusting: 50% 破坏率
        """
        poison_ratio = 0.05 if intensity == "Dirty" else (0.20 if intensity == "Filthy" else 0.50)
        
        logger.info(f"🚀 Initializing Layer 0 Chaos Engine for Domain: [{domain}] | Intensity: [{intensity}]")

        # 阶段 1：Pandera / Hypothesis 自举生成金牌数据，并实施数学契约破坏
        golden_df, toxic_df = self.breaker.generate_adversarial_payload(domain, size=size, poison_ratio=poison_ratio)
        
        # 阶段 2：Faker 注入人工录入格式噪声 (此时必须依据标准 Ontology 列名注入，不可乱序)
        noisy_df = HumanNoiseInjector.inject_noise(toxic_df, poison_ratio=poison_ratio)
        
        # 阶段 3：语义变异 (篡改表头 Schema) 并产出 Ground Truth
        # Semantic Drift 必须最后执行，以此锁死送入 Layer 1 的最终物理表头
        messy_payload, ground_truth = SemanticDrifter.apply_drift(noisy_df, drift_ratio=0.8)
        
        logger.info("✅ Chaos Payload successfully constructed. Ready for Compiler execution.")
        
        return messy_payload, ground_truth

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    engine = ChaosEngine()
    # 一键生成 10 行 Filthy 级别的仓储剧毒数据
    payload_df, truth_matrix = engine.generate_benchmark_payload("procurement_logistics", size=10, intensity="Filthy")
    
    print("\n💀 【Messy Payload (Ready for Layer 1)】:")
    print(payload_df.head())
    
    print("\n🎯 【Ground Truth Matrix (For Benchmark Arena Evaluation)】:")
    for messy_col, standard_col in truth_matrix.items():
        print(f"   {messy_col.ljust(20)} ➔ {standard_col}")