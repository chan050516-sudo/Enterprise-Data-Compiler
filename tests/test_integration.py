"""
集成测试：端到端验证 Phase 0.5 -> Phase 1 -> Phase 2
输出原始数据、IR-0（完整列画像）和 IR-1（证据图节点+边）
无断言，纯人工检查
"""
import pytest
import pandas as pd
import numpy as np
from pprint import pprint
from unittest.mock import patch
from app.legacy.technical_normalizer import TechnicalNormalizer
from app.profiler.semantic_profiler import SemanticProfiler
from app.evidence.builder import EvidenceGraphBuilder


@pytest.fixture(scope="module")
def realistic_df():
    """生成 100 行模拟企业数据（含重复值、缺失值、多种格式）"""
    np.random.seed(42)
    n = 100

    customer_ids = [f"C{str(i).zfill(4)}" for i in range(1, n + 1)]

    first_names = ["John", "Jane", "Bob", "Alice", "Tom", "Mary", "David", "Sarah", "Michael", "Emma"]
    last_names = ["Smith", "Johnson", "Lee", "Chen", "Wong", "Tan", "Ng", "Lim", "Goh", "Chan"]
    full_names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n)]

    emails = [f"user{i}@company.com" for i in range(1, n + 1)]

    phones = []
    for _ in range(n):
        prefix = np.random.choice(["+60 12", "+6012", "012", "60-12"])
        number = f"{np.random.randint(1000000, 9999999)}"
        phones.append(f"{prefix} {number}")

    amounts = np.random.uniform(10, 1000, n).round(2)
    for i in range(n):
        if np.random.rand() < 0.3:
            amounts[i] = amounts[np.random.randint(0, n)]
    amounts[np.random.choice(n, 5, replace=False)] = np.nan

    dates = pd.date_range('2024-01-01', periods=n)
    date_strs = []
    for d in dates:
        if np.random.rand() > 0.3:
            date_strs.append(d.strftime('%Y-%m-%d'))
        else:
            fmt = np.random.choice(['%d/%m/%Y', '%m/%d/%Y', '%Y.%m.%d'])
            date_strs.append(d.strftime(fmt))

    countries = np.random.choice(['MY', 'SG', 'TH', 'ID', 'PH', 'MY', 'MY', 'SG'], n)
    countries[:20] = 'MY'
    countries[20:35] = 'SG'

    tax_ids = [f"T{str(i).zfill(3)}" for i in range(1, n + 1)]
    tax_ids = [None if i in np.random.choice(n, 3, replace=False) else tax_ids[i] for i in range(n)]

    statuses = np.random.choice(['pending', 'shipped', 'delivered', 'cancelled'], n, p=[0.1, 0.3, 0.5, 0.1])

    df = pd.DataFrame({
        "customer_id": pd.Series(customer_ids, dtype=str),
        "full_name": full_names,
        "email": emails,
        "phone": phones,
        "amount": amounts,
        "order_date": date_strs,
        "country_code": countries,
        "tax_id": tax_ids,
        "status": statuses,
    })
    df["customer_id"] = df["customer_id"].astype(str)
    return df


def print_ir0(profile):
    """打印单个 ColumnProfileIR 的所有非 None 字段"""
    data = profile.model_dump(exclude_none=True, exclude={'_value_set'})
    # 重命名某些字段使输出更清晰
    print(f"  Column: {data.get('column_name')}")
    for key, value in data.items():
        if key == 'column_name':
            continue
        if isinstance(value, float):
            value = round(value, 4)
        print(f"    {key}: {value}")
    print()


def print_ir1_node(node):
    """打印 GraphNode 的关键属性"""
    print(f"  Node: {node.column_name}")
    props = node.properties
    # 只显示感兴趣的特征
    key_fields = [
        'storage_type', 'logical_type', 'unique_ratio', 'null_ratio',
        'entropy', 'numeric_density', 'length_std', 'separator_profile',
        'pattern', 'candidate_types', 'pk_score', 'has_id_pattern'
    ]
    for k in key_fields:
        if k in props:
            v = props[k]
            if isinstance(v, float):
                v = round(v, 4)
            print(f"    {k}: {v}")
    print()


def print_ir1_edge(edge):
    """打印单条边及其证据详情"""
    ev = edge.evidence
    print(f"  {edge.source_column} --[{edge.edge_type.value}]--> {edge.target_column}  (权重: {edge.weight:.4f})")
    # 打印所有非 None 证据
    ev_dict = ev.model_dump(exclude_none=True)
    if ev_dict:
        for k, v in ev_dict.items():
            if isinstance(v, float):
                v = round(v, 4)
            print(f"      {k}: {v}")
    print()


def test_end_to_end_integration(realistic_df, capsys):
    """完整端到端测试，打印原始数据、IR-0、IR-1（无断言）"""
    with capsys.disabled():
        print("\n" + "=" * 80)
        print("  EDC 端到端集成测试 - 详细输出")
        print("=" * 80)

        # ---- 原始数据 ----
        print("\n[原始数据]")
        print(f"  形状: {realistic_df.shape}")
        print("  前5行:")
        print(realistic_df.head().to_string())
        print("\n  列信息:")
        for col in realistic_df.columns:
            print(f"    {col}: {realistic_df[col].dtype}, 非空 {realistic_df[col].count()}/{len(realistic_df)}")

        # ---- Phase 0.5: Normalization ----
        normalizer = TechnicalNormalizer(config={
            "phone_country_code": "MY",
            "normalize_dates": True,
            "normalize_currency": True,
            "normalize_whitespace": True,
            "normalize_unicode": True,
        })
        df_clean, norm_report = normalizer.normalize(realistic_df)

        # ---- Phase 1: Profiling（禁用 embedding 避免下载模型） ----
        with patch('app.profiler.semantic_profiler.SemanticProfiler._generate_name_embedding', return_value=None):
            profiler = SemanticProfiler()
            profiles = profiler.generate_column_profiles(df_clean, dataset_name="integration_test")

            # ---- Phase 2: Evidence Graph ----
            builder = EvidenceGraphBuilder()
            graph = builder.build(
                profiles=profiles,
                df=df_clean,
                enable_blocking=False,
            )

        # ============================================================
        # 输出 IR-0：完整列画像
        # ============================================================
        print("\n" + "=" * 80)
        print("  IR-0: 列画像 (ColumnProfileIR)")
        print("=" * 80)
        for p in profiles:
            print_ir0(p)

        # ============================================================
        # 输出 IR-1：证据图
        # ============================================================
        print("\n" + "=" * 80)
        print("  IR-1: 证据图 (EvidenceGraph)")
        print("=" * 80)
        print(f"  节点数: {len(graph.nodes)}")
        print(f"  边数: {len(graph.edges)}")

        print("\n--- 节点 (GraphNode) ---")
        for node in graph.nodes:
            print_ir1_node(node)

        print("\n--- 边 (GraphEdge) ---")
        if graph.edges:
            # 按权重降序排列
            sorted_edges = sorted(graph.edges, key=lambda e: e.weight, reverse=True)
            for edge in sorted_edges:
                print_ir1_edge(edge)
        else:
            print("  没有边")

        print("=" * 80)
        print("测试完成（无断言，仅输出）")
        print("=" * 80 + "\n")