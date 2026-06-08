import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
import logging
from app.schema.ir_model import AdvancedTransformationIR, IRNode, IRArgument

logger = logging.getLogger(__name__)

class CompilationError(Exception):
    pass

class IRCompiler:
    """
    Layer 6: Determinsitic IR Compiler (Upgraded with Relational Algebra & AST Eval)
    Based on defined operator lists, parse IR as Pandas into vectorization operation
    """
    def __init__(self):
        pass

    def _resolve_arg(self, arg: IRArgument, source_df: pd.DataFrame, context: Dict[str, Union[pd.Series, pd.DataFrame]], extra_tables: Dict[str, pd.DataFrame] = None) -> Union[pd.Series, pd.DataFrame, Any]:
        if arg.type == "COLUMN_REF":
            if arg.value not in source_df.columns:
                raise ValueError(f"Compile Error: Source column '{arg.value}' missing.")
            return source_df[arg.value]
        elif arg.type == "STEP_REF":
            if arg.value not in context:
                raise ValueError(f"Compile Error: Intermediate step '{arg.value}' was not computed yet.")
            return context[arg.value]
        elif arg.type == "LITERAL":
            # Convert into Pandas Series to ease vectors computations
            return pd.Series(arg.value, index=source_df.index)
        elif arg.type == "TABLE_REF":
            if arg.value == "source":
                return source_df
            elif extra_tables and arg.value in extra_tables:
                return extra_tables[arg.value]
            raise ValueError(f"Compile Error: External table '{arg.value}' not found in runtime.")
        raise ValueError(f"Unsupported argument type: {arg.type}")

    def _execute_node(self, node: IRNode, source_df: pd.DataFrame, context: Dict[str, Union[pd.Series, pd.DataFrame]], extra_tables: Dict[str, pd.DataFrame] = None) -> Union[pd.Series, pd.DataFrame]:
        # Dynamically parse arguments list
        resolved_args = [self._resolve_arg(arg, source_df, context, extra_tables) for arg in node.inputs]
        op = node.operation

        # ----------------------------------------------------
        # 1. 高阶算子 (Relational Ops & Expressions) - [本次重点新增]
        # ----------------------------------------------------
        if op == "COMPUTE_EXPR":
            # 极速、安全的表达式计算（防注入），例如: "gross_amount - discount_amount * tax_rate"
            formula = node.options.get("formula")
            if not formula:
                raise ValueError("COMPUTE_EXPR requires a 'formula' in options.")
            try:
                # 建立安全沙盒：将 source_df 和之前的所有中间步骤(Series)合并作为求值环境
                eval_env = source_df.copy()
                for k, v in context.items():
                    if isinstance(v, pd.Series):
                        eval_env[k] = v
                # df.eval 底层调用 C++ NumExpr，绝对安全且速度极快
                res = eval_env.eval(formula)
            except Exception as e:
                raise CompilationError(f"Expression evaluation failed for '{formula}': {str(e)}")
            
        elif op == "UNION":
            if len(resolved_args) < 2:
                raise ValueError("UNION requires at least two TABLE_REF or DataFrame inputs.")
            # 垂直拼接，去重（或不去重，取决于业务）
            res = pd.concat(resolved_args, ignore_index=True).drop_duplicates()

        elif op == "JOIN":
            # 确定性表关联
            if len(resolved_args) < 2:
                raise ValueError("JOIN requires at least two TABLE_REF inputs.")
            left_df = resolved_args[0]
            right_df = resolved_args[1]
            how = node.options.get("how", "left")
            on = node.options.get("on")
            left_on = node.options.get("left_on")
            right_on = node.options.get("right_on")
            # 执行底层 merge
            res = pd.merge(left_df, right_df, how=how, on=on, left_on=left_on, right_on=right_on)

        elif op == "GROUP_BY":
            df_to_group = resolved_args[0]
            by_cols = node.options.get("by")
            agg_dict = node.options.get("agg")  # e.g., {"revenue": "sum"}
            res = df_to_group.groupby(by_cols).agg(agg_dict).reset_index()

        # ----------------------------------------------------
        # 2. 基础与规范化算子 (保持原样，增加对 DataFrame 抽列的支持)
        # ----------------------------------------------------
        elif op == "COPY":
            # [升级] 允许从一个 DataFrame(比如 JOIN 后的表) 中 COPY 出特定列
            if isinstance(resolved_args[0], pd.DataFrame) and node.options and "column" in node.options:
                series = resolved_args[0][node.options["column"]]
            else:
                series = resolved_args[0]
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                res = series.astype(str).str.strip().replace(['nan', 'None', 'N/A', ''], np.nan)
            else:
                res = series
        elif op == "CONCAT":
            # Force into str type before concatenation to avoid NaN
            res = resolved_args[0].astype(str) + resolved_args[1].astype(str)
        elif op == "ADD":
            res = pd.to_numeric(resolved_args[0], errors='coerce').fillna(0) + \
                  pd.to_numeric(resolved_args[1], errors='coerce').fillna(0)
        elif op == "SUBTRACT":
            res = pd.to_numeric(resolved_args[0], errors='coerce').fillna(0) - \
                  pd.to_numeric(resolved_args[1], errors='coerce').fillna(0)
        elif op == "MULTIPLY":
            res = pd.to_numeric(resolved_args[0], errors='coerce').fillna(0) * \
                  pd.to_numeric(resolved_args[1], errors='coerce').fillna(0)
        elif op == "DIVIDE":
            # Handle zero-denominator or non-numeric impurities
            numerator = pd.to_numeric(resolved_args[0], errors='coerce').fillna(0)
            denominator = pd.to_numeric(resolved_args[1], errors='coerce').fillna(1) # Impurities default to convert to 1，avoid system collapse
            res = numerator / denominator.replace(0, 1)   # Fallback
        elif op == "TO_FLOAT":
            res = pd.to_numeric(resolved_args[0], errors='coerce')
        elif op == "TO_INT":
            res = pd.to_numeric(resolved_args[0], errors='coerce').fillna(0).astype(int)
        
        # ----------------------------------------------------
        # 3. 规范化算子 (Canonicalization - 解决格式噪声的核心)
        # ----------------------------------------------------
        elif op == "PARSE_DATE":
            # 将类似 "2026/06/08", "Jun 8 2026" 强制转换为标准 ISO8601 格式
            res = pd.to_datetime(resolved_args[0], errors='coerce').dt.strftime('%Y-%m-%d')
        elif op == "CLEAN_CURRENCY":
            # 暴力清洗：剥离所有非数字字符 (保留小数点和负号)，"RM 1,000.50" -> 1000.50
            cleaned_str = resolved_args[0].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
            res = pd.to_numeric(cleaned_str, errors='coerce')
        elif op == "FUZZY_MAP":
            # 基础规范化：去两端空格 + 转大写 (例如 " pNding " -> "PNDING")
            res = resolved_args[0].astype(str).str.strip().str.upper()
            # 如果 node 里带了字典，可以执行确定的 Enum 映射
            if node.options and "mapping_dict" in node.options:
                mapping = node.options["mapping_dict"]
                # 匹配不到的保留原样，交给 Layer 5 去抓
                res = res.map(mapping).fillna(res)
        
        else:
            raise NotImplementedError(f"Operator {op} is not supported by runtime compiler.")

        return res
    
    def _topological_sort(self, steps: Dict[str, IRNode]) -> List[str]:
        """Kahn's Algorithm Topological Sorting, decide the sequence of operations of the compiler"""
        in_degree = {name: 0 for name in steps.keys()}
        adj_list = {name: [] for name in steps.keys()}

        # Generate adjacent list and in degree list
        for step_name, node in steps.items():
            for arg in node.inputs:
                if arg.type == "STEP_REF":
                    if arg.value in steps:
                        adj_list[arg.value].append(step_name)
                        in_degree[step_name] += 1

        # Find out starting node with degree=0
        queue = [name for name, deg in in_degree.items() if deg == 0]
        sorted_steps = []

        while queue:
            current = queue.pop(0)
            sorted_steps.append(current)
            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_steps) != len(steps):
            raise CompilationError("Topological sort failed. Hidden cycle detected during compilation.")

        return sorted_steps

    def compile(self, source_df: pd.DataFrame, ir: AdvancedTransformationIR, target_ontology: Dict[str, Any] = None, extra_tables: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        logger.info("Initializing Native Compilation & Relational Context...")
        runtime_context: Dict[str, Union[pd.Series, pd.DataFrame]] = {}
        output_df = pd.DataFrame()

        execution_order = self._topological_sort(ir.intermediate_steps)

        # 1. Compile and execute calculation steps
        for step_name in execution_order:
            node = ir.intermediate_steps[step_name]
            logger.debug(f"Compiling intermediate virtual register: {step_name}")
            runtime_context[step_name] = self._execute_node(node, source_df, runtime_context, extra_tables)

        ontology_fields = target_ontology.get("fields", {}) if target_ontology else {}

        # 2. Compile and generate final target Ontology column
        for target_field, node in ir.output_mappings.items():
            logger.info(f"Compiling target business field: {target_field}")
            final_series = self._execute_node(node, source_df, runtime_context, extra_tables)
            
            # --- 业务语义注入 (Policy Imputation) ---
            field_config = ontology_fields.get(target_field, {})
            fallback_strategy = field_config.get("fallback_strategy", "KEEP_NULL")
            default_val = field_config.get("default_value")

            if fallback_strategy == "USE_ZERO":
                final_series = pd.to_numeric(final_series, errors='coerce').fillna(0)
            elif fallback_strategy == "DEFAULT_STRING":
                final_series = final_series.fillna(default_val if default_val else "UNKNOWN")
            elif fallback_strategy == "HALT":
                # 对于强审计列（HALT），拒绝填充。
                # 保留 NaN 状态，这样稍后在 Layer 5 Trust Engine 扫描 `not_null` 时会精准地把这行抓进隔离区。
                pass

            # --- ERP 级类型锚定 (Type Anchoring) ---
            if node.target_type == "float":
                output_df[target_field] = pd.to_numeric(final_series, errors='coerce')
            elif node.target_type == "int":
                # 【终极防御】：使用 Pandas 的 'Int64' (可空整数) 替代原生 'int'。
                # 防止由于某行触发了 HALT 策略保留了 NaN，导致整列发生浮点数逃逸 (例如 "单据号 100" 变成 "100.0")
                output_df[target_field] = pd.to_numeric(final_series, errors='coerce').astype('Int64')
            else:
                output_df[target_field] = final_series

        return output_df