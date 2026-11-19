"""The strategy lint's allowlists (ADR-001 A4, A14 f G5). finalizes them.

No listed name takes a path, a URL, SQL text, a code string or a foreign pointer, and none draws random
numbers. A module attribute (np.X, pl.X, math.X under any alias) must be on that module's list; any other
attribute name must be on the union of the numpy and polars lists; a bare name that the file does not define
must be on the builtins list.
"""

from __future__ import annotations

NUMPY_NAMES: frozenset[str] = frozenset(
    {
        # dtypes and constants
        "bool_", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "intp",
        "float32", "float64", "ndarray", "iinfo", "nan", "inf", "pi", "newaxis",
        # (ndarray: in annotations only; every other load is refused by the lint)
        # construction
        # (no empty / empty_like: uninitialized memory may hold an earlier call's arrays)
        "array", "asarray", "arange", "zeros", "zeros_like", "ones", "ones_like", "full", "full_like",
        "linspace", "concatenate", "stack", "hstack", "vstack", "append", "repeat", "tile", "broadcast_to",
        # element-wise and ufuncs
        "abs", "absolute", "add", "subtract", "multiply", "divide", "floor_divide", "mod", "negative",
        "power", "sign", "sqrt", "square", "exp", "log", "log1p", "floor", "ceil", "rint", "round", "around",
        "trunc", "clip", "maximum", "minimum", "equal", "not_equal", "greater", "greater_equal", "less",
        "less_equal", "logical_and", "logical_or", "logical_not", "logical_xor", "isfinite", "isnan", "where",
        # ufunc methods
        "accumulate", "reduce", "reduceat",
        # reductions and searches
        "all", "any", "sum", "cumsum", "cumprod", "prod", "mean", "median", "std", "var", "max", "min",
        "nanmax", "nanmin", "nanmean", "argmax", "argmin", "argsort", "sort", "searchsorted", "unique",
        "isin", "nonzero", "flatnonzero", "count_nonzero", "diff", "percentile", "quantile", "array_equal",
        "take", "flip", "convolve",
        # ndarray attributes
        "astype", "copy", "item", "tolist", "shape", "size", "ndim", "dtype", "reshape", "ravel", "flatten",
        "squeeze", "transpose", "T",
    }
)  # fmt: skip

POLARS_NAMES: frozenset[str] = frozenset(
    {
        # frames, expressions and dtypes
        "DataFrame", "Series", "LazyFrame", "col", "lit", "when", "then", "otherwise", "concat", "int_range",
        "Boolean", "Int8", "Int16", "Int32", "Int64", "UInt32", "UInt64", "Float32", "Float64", "String",
        "Date", "Datetime", "Duration",
        # frame verbs
        "select", "with_columns", "with_row_index", "filter", "group_by", "agg", "sort", "join", "head",
        "tail", "slice", "unique", "lazy", "collect", "get_column", "to_numpy", "to_series", "height",
        "width", "columns", "schema", "dtypes",
        # expression verbs
        "alias", "cast", "over", "shift", "diff", "len", "count", "n_unique", "sum", "mean", "max", "min",
        "std", "var", "median", "quantile", "first", "last", "arg_max", "arg_min", "rank", "abs", "round",
        "floor", "ceil", "sqrt", "log", "exp", "clip", "sign", "pct_change", "cum_sum", "cum_max", "cum_min",
        "cum_count", "cum_prod", "rolling_mean", "rolling_sum", "rolling_max", "rolling_min", "rolling_std",
        "ewm_mean", "fill_null", "fill_nan", "forward_fill", "is_null", "is_not_null", "is_in", "is_between",
        "and_", "or_", "not_", "all_horizontal", "any_horizontal", "max_horizontal", "min_horizontal",
        "sum_horizontal", "dt", "hour", "minute", "weekday",
    }
)  # fmt: skip

MATH_NAMES: frozenset[str] = frozenset(
    {
        "ceil", "floor", "trunc", "sqrt", "exp", "log", "log10", "log2", "fabs", "copysign", "isfinite",
        "isnan", "isinf", "isclose", "inf", "nan", "pi", "e", "gcd", "hypot", "fsum", "prod", "comb",
    }
)  # fmt: skip

BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "frozenset", "int",
        "isinstance", "len", "list", "map", "max", "min", "range", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "zip", "ArithmeticError", "IndexError", "KeyError", "RuntimeError",
        "TypeError", "ValueError", "ZeroDivisionError",
    }
)  # fmt: skip

MODULE_LISTS: dict[str, frozenset[str]] = {"numpy": NUMPY_NAMES, "polars": POLARS_NAMES, "math": MATH_NAMES}
OTHER_ATTRIBUTE_NAMES: frozenset[str] = NUMPY_NAMES | POLARS_NAMES

__all__ = [
    "BUILTIN_NAMES",
    "MATH_NAMES",
    "MODULE_LISTS",
    "NUMPY_NAMES",
    "OTHER_ATTRIBUTE_NAMES",
    "POLARS_NAMES",
]
