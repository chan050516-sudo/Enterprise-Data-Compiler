import pandas as pd
import numpy as np
import difflib
from typing import Dict, Any, List, Union
import logging
from app.schema.ir_model import AdvancedTransformationIR, IRNode, IRArgument

logger = logging.getLogger(__name__)

class CompilationError(Exception):
    pass

class IRCompiler:
    """
    Layer 6: Determinsitic IR Compiler (V3: Operator Registry Architecture)
    Based on defined operator lists, parse IR as Pandas into vectorization operation.
    采用高扩展性的算子注册表模式，修复所有索引对齐与排序 Bug，并完整保留业务兜底逻辑。
    """
    def __init__(self):
        # ==========================================
        # 核心架构跃迁：算子注册表 (Operator Registry)
        # ==========================================
        self._operator_registry = {
            # 基础数学与类型
            "COPY": self._op_copy,
            "CONCAT": self._op_concat,
            "ADD": self._op_add,
            "SUBTRACT": self._op_subtract,
            "MULTIPLY": self._op_multiply,
            "DIVIDE": self._op_divide,
            "TO_FLOAT": self._op_to_float,
            "TO_INT": self._op_to_int,
            
            # 清洗与规范化
            "PARSE_DATE": self._op_parse_date,
            "CLEAN_CURRENCY": self._op_clean_currency,
            "FUZZY_MAP": self._op_fuzzy_map,
            "RESOLVE_ENTITIES": self._op_resolve_entities,
            "REGEX_EXTRACT": self._op_regex_extract,  # [新增]
            "REPLACE": self._op_replace,              # [新增]
            "FILLNA": self._op_fillna,                # [新增]
            
            # 关系代数与结构重塑
            "JOIN": self._op_join,
            "UNION": self._op_union,
            "GROUP_BY": self._op_group_by,
            "FILTER": self._op_filter,
            "EXPLODE": self._op_explode,
            
            # 图灵完备级 SQL 操作
            "WINDOW_APPLY": self._op_window_apply,
            "CASE_WHEN": self._op_case_when,
            "PIVOT": self._op_pivot,
            "UNPIVOT": self._op_unpivot,
            "ORDER_BY": self._op_order_by,
            "LIMIT": self._op_limit
        }

    # ==========================================
    # 参数解析器 (完全保留原版逻辑)
    # ==========================================
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

    # ==========================================
    # 算子实现区 1: 基础操作 (100% 继承原版)
    # ==========================================
    def _op_copy(self, args, opts):
        if isinstance(args[0], pd.DataFrame) and opts and "column" in opts:
            series = args[0][opts["column"]]
        else:
            series = args[0]
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            return series.astype(str).str.strip().replace(['nan', 'None', 'N/A', ''], np.nan)
        return series

    def _op_concat(self, args, opts):
        # Force into str type before concatenation to avoid NaN
        return args[0].astype(str) + args[1].astype(str)

    def _op_add(self, args, opts): 
        return pd.to_numeric(args[0], errors='coerce').fillna(0) + pd.to_numeric(args[1], errors='coerce').fillna(0)
        
    def _op_subtract(self, args, opts): 
        return pd.to_numeric(args[0], errors='coerce').fillna(0) - pd.to_numeric(args[1], errors='coerce').fillna(0)
        
    def _op_multiply(self, args, opts): 
        return pd.to_numeric(args[0], errors='coerce').fillna(0) * pd.to_numeric(args[1], errors='coerce').fillna(0)
        
    def _op_divide(self, args, opts):
        # Handle zero-denominator or non-numeric impurities
        numerator = pd.to_numeric(args[0], errors='coerce').fillna(0)
        denominator = pd.to_numeric(args[1], errors='coerce').fillna(1) # Impurities default to convert to 1，avoid system collapse
        return numerator / denominator.replace(0, 1)   # Fallback

    def _op_to_float(self, args, opts): 
        return pd.to_numeric(args[0], errors='coerce')
        
    def _op_to_int(self, args, opts): 
        return pd.to_numeric(args[0], errors='coerce').fillna(0).astype(int)

    # ==========================================
    # 算子实现区 2: 清洗与空值处理
    # ==========================================
    def _op_parse_date(self, args, opts): 
        return pd.to_datetime(args[0], errors='coerce', utc=True).dt.strftime('%Y-%m-%d')
        
    def _op_clean_currency(self, args, opts): 
        cleaned_str = args[0].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
        return pd.to_numeric(cleaned_str, errors='coerce')
        
    def _op_fuzzy_map(self, args, opts):
        res = args[0].astype(str).str.strip().str.upper()
        if opts and "mapping_dict" in opts:
            res = res.map(opts["mapping_dict"]).fillna(res)
        return res
        
    def _op_resolve_entities(self, args, opts):
        series = args[0]
        threshold = opts.get("similarity_threshold", 0.85)
        valid_series = series.dropna().astype(str)
        unique_vals = valid_series.value_counts().index.tolist()
        canonical_mapping, processed = {}, set()
        for val in unique_vals:
            if val in processed: continue
            matches = difflib.get_close_matches(val, unique_vals, n=len(unique_vals), cutoff=threshold)
            for match in matches:
                if match not in processed:
                    canonical_mapping[match] = val
                    processed.add(match)
        return series.map(canonical_mapping).fillna(series)

    def _op_regex_extract(self, args, opts):
        pattern = opts.get("pattern")
        if not pattern: raise ValueError("REGEX_EXTRACT requires 'pattern'.")
        return args[0].astype(str).str.extract(pattern, expand=False)

    def _op_replace(self, args, opts):
        pattern = opts.get("pattern")
        repl = opts.get("replacement", "")
        is_regex = opts.get("is_regex", True)
        if not pattern: raise ValueError("REPLACE requires 'pattern'.")
        return args[0].astype(str).str.replace(pattern, repl, regex=is_regex)

    def _op_fillna(self, args, opts):
        method = opts.get("method")
        value = opts.get("value")
        if method in ['ffill', 'bfill']:
            return args[0].fillna(method=method)
        return args[0].fillna(value)

    # ==========================================
    # 算子实现区 3: 关系代数与结构重塑
    # ==========================================
    def _op_join(self, args, opts):
        if len(args) < 2: raise ValueError("JOIN requires at least two TABLE_REF inputs.")
        return pd.merge(args[0], args[1], how=opts.get("how", "left"), on=opts.get("on"), left_on=opts.get("left_on"), right_on=opts.get("right_on"))

    def _op_union(self, args, opts):
        if len(args) < 2: raise ValueError("UNION requires at least two inputs.")
        # [修复] 增加 distinct 选项以支持 UNION ALL
        res = pd.concat(args, ignore_index=True)
        if opts.get("distinct", True):
            res = res.drop_duplicates()
        return res

    def _op_group_by(self, args, opts):
        return args[0].groupby(opts.get("by")).agg(opts.get("agg")).reset_index()

    def _op_filter(self, args, opts):
        condition = opts.get("condition")
        if not condition: raise ValueError("FILTER requires a 'condition'.")
        try:
            return args[0].query(condition).copy()
        except Exception as e:
            raise CompilationError(f"FILTER evaluation failed for '{condition}': {str(e)}")

    def _op_explode(self, args, opts):
        col, delimiter = opts.get("column"), opts.get("delimiter", ",")
        if not col or col not in args[0].columns:
            raise ValueError(f"EXPLODE requires a valid 'column'. Got: {col}")
        res = args[0].copy()
        res[col] = res[col].astype(str).str.split(delimiter)
        res = res.explode(col)
        if pd.api.types.is_string_dtype(res[col]) or pd.api.types.is_object_dtype(res[col]):
            res[col] = res[col].str.strip().replace(['nan', 'None', ''], np.nan)
        return res

    # ==========================================
    # 算子实现区 4: 高阶 SQL 操作
    # ==========================================
    def _op_window_apply(self, args, opts):
        df_in = args[0].copy()
        partition_by, order_by, ascending = opts.get("partition_by", []), opts.get("order_by", []), opts.get("ascending", True)
        func, target_col = opts.get("function"), opts.get("target_column")
        if not func or not target_col: raise ValueError("WINDOW_APPLY requires 'function' and 'target_column'.")
        
        # [修复] 保证组内严格排序
        if order_by:
            df_in = df_in.sort_values(by=partition_by + order_by, ascending=ascending)
            
        grouped = df_in.groupby(partition_by)[target_col] if partition_by else df_in[target_col]
        
        if func == "row_number":
            res_series = df_in.groupby(partition_by).cumcount() + 1 if partition_by else pd.Series(range(1, len(df_in) + 1), index=df_in.index)
        elif func == "rank": res_series = grouped.rank(method='min', ascending=ascending)
        elif func == "dense_rank": res_series = grouped.rank(method='dense', ascending=ascending)
        elif func in ["sum", "mean", "max", "min"]: res_series = grouped.transform(func)
        elif func == "cumulative_sum": res_series = grouped.cumsum()
        else: raise NotImplementedError(f"Unsupported window function: {func}")
            
        # [修复] 通过原索引强制对齐恢复时序
        return res_series.sort_index()

    def _op_case_when(self, args, opts):
        df_or_series = args[0]
        # 如果输入是 Series，转换为单列 DataFrame
        if isinstance(df_or_series, pd.Series):
            df_in = df_or_series.to_frame(name='_temp')
        else:
            df_in = df_or_series.copy()
        
        cases = opts.get("cases", [])
        default_val = opts.get("default", np.nan)
        conditions = []
        choices = []
        
        try:
            for case in cases:
                # 使用 df.eval 评估条件
                cond_mask = df_in.eval(case["condition"])
                conditions.append(cond_mask)
                val = case["value"]
                # 如果 value 是字符串且在 DataFrame 列中，取该列；否则作为字面量
                if isinstance(val, str) and val in df_in.columns:
                    choices.append(df_in[val])
                else:
                    choices.append(val)
            # 使用 np.select 进行多路选择
            result = pd.Series(np.select(conditions, choices, default=default_val), index=df_in.index)
            return result
        except Exception as e:
            raise CompilationError(f"CASE_WHEN evaluation failed: {str(e)}")

    def _op_pivot(self, args, opts):
        res = pd.pivot_table(args[0], index=opts.get("index"), columns=opts.get("columns"), values=opts.get("values"), aggfunc=opts.get("aggfunc", "sum")).reset_index()
        res.columns = [str(c) for c in res.columns]
        return res

    def _op_unpivot(self, args, opts):
        return pd.melt(args[0], id_vars=opts.get("id_vars"), value_vars=opts.get("value_vars"), var_name=opts.get("var_name", "variable"), value_name=opts.get("value_name", "value"))

    def _op_order_by(self, args, opts):
        res = args[0].sort_values(by=opts.get("by"), ascending=opts.get("ascending", True))
        # [修复] 重置底层索引，彻底防止下游隔离区因索引错乱导致定位失败
        if opts.get("reset_index", True):
            res = res.reset_index(drop=True)
        return res

    def _op_limit(self, args, opts):
        return args[0].head(opts.get("n", 100))

    # ==========================================
    # 执行路由与主控 (完全保留原版的防御机制与策略注入)
    # ==========================================
    def _execute_node(self, node: IRNode, source_df: pd.DataFrame, context: Dict[str, Union[pd.Series, pd.DataFrame]], extra_tables: Dict[str, pd.DataFrame] = None) -> Union[pd.Series, pd.DataFrame]:
        # Dynamically parse arguments list
        resolved_args = [self._resolve_arg(arg, source_df, context, extra_tables) for arg in node.inputs]
        op = node.operation

        # ----------------------------------------------------
        # 截获并特殊处理全局环境沙盒算子 COMPUTE_EXPR (100% 继承原版安全机制)
        # ----------------------------------------------------
        if op == "COMPUTE_EXPR":
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
                return eval_env.eval(formula)
            except Exception as e:
                raise CompilationError(f"Expression evaluation failed for '{formula}': {str(e)}")

        # ----------------------------------------------------
        # 标准注册表路由分发
        # ----------------------------------------------------
        if op not in self._operator_registry:
            raise NotImplementedError(f"Operator {op} is not supported by runtime compiler.")
            
        try:
            return self._operator_registry[op](resolved_args, node.options or {})
        except Exception as e:
            raise CompilationError(f"Execution failed at Operator [{op}]: {str(e)}")

    def _topological_sort(self, steps: Dict[str, IRNode]) -> List[str]:
        """Kahn's Algorithm Topological Sorting, decide the sequence of operations of the compiler"""
        in_degree = {name: 0 for name in steps.keys()}
        adj_list = {name: [] for name in steps.keys()}

        for step_name, node in steps.items():
            for arg in node.inputs:
                if arg.type == "STEP_REF":
                    if arg.value in steps:
                        adj_list[arg.value].append(step_name)
                        in_degree[step_name] += 1

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
            
            # --- 业务语义注入 (Policy Imputation) (100% 继承原版) ---
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

            # --- ERP 级类型锚定 (Type Anchoring) (100% 继承原版) ---
            if node.target_type == "float":
                output_df[target_field] = pd.to_numeric(final_series, errors='coerce')
            elif node.target_type == "int":
                # 【终极防御】：使用 Pandas 的 'Int64' (可空整数) 替代原生 'int'。
                # 防止由于某行触发了 HALT 策略保留了 NaN，导致整列发生浮点数逃逸 (例如 "单据号 100" 变成 "100.0")
                output_df[target_field] = pd.to_numeric(final_series, errors='coerce').astype('Int64')
            else:
                output_df[target_field] = final_series

        return output_df