"""The strategy lint (ADR-001 A4, A14 f G5): one refused snippet per escape; the reference strategy passes."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pfb.engine.lint import lint_source, lint_strategy
from pfb.engine.lint_allow import BUILTIN_NAMES, MATH_NAMES, NUMPY_NAMES, POLARS_NAMES
from pfb.engine.strategy import LintFailed, load_strategy

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "strategies" / "reference_breakout.py"
NP = "import numpy as np\n"
PL = "import polars as pl\ndf = pl.DataFrame()\n"


def _rules(source: str) -> set[str]:
    return {v.rule for v in lint_source(source)}


def _polars_io_names() -> list[tuple[str, str]]:
    """Every read_/scan_ of the polars module and every write_/sink_ of its frames and series."""
    names = [("pl", n) for n in dir(pl) if n.startswith(("read_", "scan_"))]
    for owner in (pl.DataFrame, pl.LazyFrame, pl.Series):
        names += [("df", n) for n in dir(owner) if n.startswith(("write_", "sink_"))]
    return sorted(set(names))


def test_the_reference_strategy_passes() -> None:
    assert lint_strategy(REFERENCE) == []


@pytest.mark.parametrize(
    "snippet",
    [
        PL + "x = pl.sql('select 1')",
        PL + "ctx = pl.SQLContext()",
        PL + "x = df.sql('select 1')",
        PL + "x = pl.DataFrame.deserialize('f')",
        PL + "x = df.deserialize('f')",
        PL + "pl.Config.set_tbl_rows(1)",
        NP + "x = np.load('f')",
        NP + "np.save('f', 1)",
        NP + "x = np.loadtxt('f')",
        NP + "x = np.genfromtxt('f')",
        NP + "x = np.fromregex('f', 'r', 'i8')",
        NP + "x = np.fromfile('f')",
        NP + "np.zeros(1).tofile('f')",
        NP + "x = np.memmap('f')",
        NP + "x = np.DataSource()",
        NP + "x = np.lib",
        NP + "x = np.ctypeslib",
        NP + "x = np.zeros(1).ctypes",
        "import ctypes",
        NP + "x = np.f2py",
        NP + "x = np.random.default_rng(1)",
        "x = open('f')",
        "x = eval('1')",
        "exec('x = 1')",
        "x = getattr(1, 'real')",
        "x = (1).__class__",
        "x = __import__('os')",
    ],
)
def test_each_named_escape_is_refused(snippet: str) -> None:
    assert lint_source(snippet), snippet


@pytest.mark.parametrize(("owner", "name"), _polars_io_names())
def test_every_polars_reader_scanner_writer_and_sink_is_refused(owner: str, name: str) -> None:
    assert lint_source(f"{PL}x = {owner}.{name}('f')")


def test_g5_a_module_attribute_must_be_on_that_modules_list() -> None:
    # Arrange: each name is on SOME list, but not on the list of the module it is read from
    cases = [
        "import math\nx = math.where",
        NP + "x = np.DataFrame",
        "import polars as pl\nx = pl.cumsum",
        "import math\nx = math.zeros(1)",
    ]
    # Act / Assert
    for source in cases:
        assert _rules(source) == {"module_attribute_not_allowed"}, source


def test_g5_scoping_follows_the_alias_not_the_name() -> None:
    assert _rules("import numpy as m\nx = m.col('a')") == {"module_attribute_not_allowed"}
    assert _rules("import math as np\nx = np.zeros(1)") == {"module_attribute_not_allowed"}
    assert _rules("import math as np\nx = np.floor(1.5)") == set()


def test_g5_any_other_attribute_must_be_on_the_numpy_or_polars_list() -> None:
    assert _rules(NP + "a = np.zeros(3)\nb = a.cumsum()") == set()
    assert _rules(PL + "b = df.with_columns(pl.col('a').cum_sum())") == set()
    assert _rules(NP + "a = np.zeros(3)\nb = a.isclose") == {"attribute_not_allowed"}  # math only
    assert _rules("x = '{0}'.format(1)") == {"attribute_not_allowed"}


@pytest.mark.parametrize(
    ("snippet", "rule"),
    [
        ("import os", "import_not_allowed"),
        ("import numpy.random", "submodule_import"),
        ("from numpy.random import default_rng", "submodule_import"),
        ("from . import helper", "submodule_import"),
        ("from numpy import *", "star_import"),
        ("from numpy import load", "from_import_not_allowed"),
        ("from math import where", "from_import_not_allowed"),
        ("from __future__ import division", "import_not_allowed"),
        ("def f():\n    global x\n    x = 1", "global_statement"),
        ("def f():\n    x = 1\n    def g():\n        nonlocal x\n        x = 2", "nonlocal_statement"),
        ("g = open", "builtin_not_allowed"),
        ("def print():\n    return 1", "builtin_not_allowed"),
        ("x = undefined_name", "name_not_allowed"),
        (NP + "np = 1", "alias_rebound"),
        ("def __init__():\n    return 1", "dunder"),
        ("x = 1 +", "syntax_error"),
    ],
)
def test_each_structural_rule_has_a_refused_case(snippet: str, rule: str) -> None:
    assert rule in _rules(snippet)


def test_allowed_imports_and_local_names_pass() -> None:
    source = (
        "from __future__ import annotations\nimport math\nimport numpy as np\nimport polars as pl\n"
        "from numpy import where\n\n"
        "def helper(a, b=2):\n    return where(a > b, a, b)\n\n"
        "def generate(bars, params):\n    c = bars['close']\n    n = len(c)\n"
        "    z = np.zeros(n, dtype=np.int64)\n    f = np.zeros(n, dtype=np.bool_)\n"
        "    y = helper(c, int(math.floor(params['k'])))\n"
        "    return {'entry_long': y > 0, 'entry_short': f, 'exit': f, 'stop_ticks': z, 'target_ticks': z}\n"
    )
    assert lint_source(source) == []


@pytest.mark.parametrize(
    ("snippet", "rule"),
    [
        ("def f(x, seen=[]):\n    return x", "mutable_default"),
        ("def f(x, *, seen={}):\n    return x", "mutable_default"),
        (NP + "def f(x, z=np.zeros(3)):\n    return x", "mutable_default"),
        ("g = lambda x, s=set(): x", "mutable_default"),
        (NP + "np.pi = 3", "attribute_store"),
        (NP + "def f(x):\n    s = np.sort\n    s.max = x\n    return x", "attribute_store"),
        ("class C:\n    T = 1\n\ndef f(x):\n    C.T = x\n    return x", "attribute_store"),
        ("S = {}\n\ndef f(x):\n    S['c'] = x\n    return x", "item_store_not_local"),
        ("S = [0]\n\ndef f(x):\n    S[0] += x\n    return x", "item_store_not_local"),
        ("S = {}\n\ndef f(x):\n    del S['c']\n    return x", "item_store_not_local"),
        (NP + "def f(x):\n    np.zeros(2)[0] = x\n    return x", "item_store_not_local"),
        ("from math import pi\npi[0] = 1", "item_store_not_local"),
        ("def f():\n    S = {}\n    def g(x):\n        S['c'] = x\n    return g", "item_store_not_local"),
        (NP + "x = np.ndarray((3,))", "uninitialized_array"),
        ("from numpy import ndarray\nx = ndarray((3,))", "uninitialized_array"),
        (NP + "x = np.empty(3)", "module_attribute_not_allowed"),
        (NP + "x = np.zeros(3)\ny = x.empty_like", "attribute_not_allowed"),
    ],
)
def test_e1_each_cheap_state_channel_is_refused(snippet: str, rule: str) -> None:
    assert rule in _rules(snippet), snippet


def test_e1_local_stores_and_constant_defaults_pass() -> None:
    # (isinstance(x, np.ndarray) used to pass here; it is now refused, see below)
    source = NP + (
        "T = np.zeros(3)\nT[0] = 1\n\n"
        "def f(x, k=2, s='a', t=(1, -2), z=None):\n"
        "    out = np.zeros(len(x))\n    out[0] = k\n    d = {}\n    d['a'] = 1\n"
        "    out[1:] += 1\n"
        "    for i, row in enumerate([out]):\n        row[i] = 0\n    return out, d\n"
    )
    assert lint_source(source) == []


# ------------------------------------------- B1: ndarray only inside annotations ----
NDARRAY_RULE = "ndarray_outside_annotation"
LEAK_ALIAS = NP + (  # the alias trick of an uninitialized-memory leak probe, cut down
    "RAW = np.ndarray\n\ndef _scan(n):\n    buf = RAW((n,), dtype=np.int64)\n    return buf\n"
)


@pytest.mark.parametrize(
    "snippet",
    [
        LEAK_ALIAS,
        NP + "RAW = np.ndarray",
        "from numpy import ndarray as A\nx = A((3,))",
        "from numpy import ndarray as A\nB = A",
        "from numpy import ndarray\nx = ndarray((3,))",
        NP + "x = np.ndarray((3,))",
        "import numpy\nx = numpy.ndarray((3,))",
        NP + "x = np.zeros(3)\ny = x.ndarray",
        NP + "def f(x):\n    return isinstance(x, np.ndarray)",
        NP + "def f(x, k=np.ndarray):\n    return x",
        NP + "def f(x) -> np.ndarray:\n    return np.ndarray((3,))",
        NP + "type RAW = np.ndarray",
        NP + "fs = [np.ndarray]",
    ],
)
def test_b1_ndarray_outside_an_annotation_is_refused(snippet: str) -> None:
    assert NDARRAY_RULE in _rules(snippet), snippet


@pytest.mark.parametrize(
    "snippet",
    [
        NP + "def generate(bars, params) -> np.ndarray:\n    x: np.ndarray = np.zeros(3)\n    return x\n",
        NP
        + (
            "def f(a: np.ndarray, *b: np.ndarray, c: dict[str, np.ndarray] | None = None)"
            " -> tuple[np.ndarray, np.ndarray]:\n    return a, a\n"
        ),
        NP + "y: np.ndarray\nz: list[np.ndarray] = []\n",
        "import numpy as np\nfrom numpy import ndarray\n\ndef f(a: ndarray) -> ndarray:\n    return a\n",
        "import numpy as np\nfrom numpy import ndarray as A\n\ndef f(a: A) -> list[A]:\n    return [a]\n",
        NP + "def f(a: 'np.ndarray') -> 'np.ndarray':\n    return a\n",
    ],
)
def test_b1_ndarray_in_annotations_stays_clean(snippet: str) -> None:
    assert lint_source(snippet) == [], snippet


def test_b1_the_alias_violation_points_at_the_load() -> None:
    hits = [v for v in lint_source(LEAK_ALIAS) if v.rule == NDARRAY_RULE]
    assert [(v.line, v.name) for v in hits] == [(2, "ndarray")]


def test_no_listed_name_reads_writes_or_draws_random_numbers() -> None:
    listed = NUMPY_NAMES | POLARS_NAMES | MATH_NAMES | BUILTIN_NAMES
    banned = (
        "read",
        "scan",
        "write",
        "sink",
        "sql",
        "load",
        "save",
        "file",
        "memmap",
        "serial",
        "random",
        "ctypes",
        "config",
        "pandas",
        "arrow",
        "buffer",
        "path",
        "url",
        "open",
        "eval",
        "exec",
        "import",
        "plugin",
        "map_",
        "pipe",
        "format",
        "lib",
    )
    offending = sorted(n for n in listed if any(b in n.lower() for b in banned))
    assert offending == []


def test_lint_never_executes_and_load_strategy_refuses_a_failing_file(tmp_path: Path) -> None:
    # Arrange: a file that would create a marker if it were executed
    marker = tmp_path / "executed.txt"
    path = tmp_path / "evil.py"
    path.write_text(f"open({str(marker)!r}, 'w').write('x')\nSPEC = {{}}\n", encoding="utf-8")
    # Act
    violations = lint_strategy(path)
    with pytest.raises(LintFailed):
        load_strategy(path)
    # Assert
    assert violations
    assert not marker.exists()
