"""Static allowlist lint of a strategy file (ADR-001 A4, A14 f G5). Nothing is executed."""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from pathlib import Path

from pfb.engine.lint_allow import BUILTIN_NAMES, MODULE_LISTS, OTHER_ATTRIBUTE_NAMES

ALLOWED_MODULES = frozenset(MODULE_LISTS)
FUTURE_MODULE = "__future__"
FUTURE_NAMES = frozenset({"annotations"})
PYTHON_BUILTINS = frozenset(dir(builtins))
FORBIDDEN_BUILTINS = PYTHON_BUILTINS - BUILTIN_NAMES


@dataclass(frozen=True)
class LintViolation:
    rule: str
    line: int
    col: int
    name: str
    message: str


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _at(node: ast.AST, rule: str, name: str, message: str) -> LintViolation:
    return LintViolation(rule, getattr(node, "lineno", 0), getattr(node, "col_offset", 0), name, message)


class _Imports:
    """Module aliases (alias -> module) and names bound by allowed `from` imports."""

    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}
        self.imported: set[str] = set()
        self.from_targets: dict[str, str] = {}  # bound name -> "module.name"
        self.violations: list[LintViolation] = []

    def visit(self, tree: ast.Module) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self._import(node)
            elif isinstance(node, ast.ImportFrom):
                self._import_from(node)

    def _import(self, node: ast.Import) -> None:
        for alias in node.names:
            if "." in alias.name:
                self.violations.append(_at(node, "submodule_import", alias.name, "a submodule import"))
            elif alias.name not in ALLOWED_MODULES:
                self.violations.append(
                    _at(node, "import_not_allowed", alias.name, "only numpy, polars, math")
                )
            else:
                self.aliases[alias.asname or alias.name] = alias.name

    def _import_from(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level or "." in module:
            self.violations.append(_at(node, "submodule_import", module, "a relative or submodule import"))
            return
        for alias in node.names:
            self._from_name(node, module, alias)

    def _from_name(self, node: ast.ImportFrom, module: str, alias: ast.alias) -> None:
        if alias.name == "*":
            self.violations.append(_at(node, "star_import", module, "a star import"))
        elif module == FUTURE_MODULE and alias.name in FUTURE_NAMES and alias.asname is None:
            return
        elif module not in ALLOWED_MODULES:
            self.violations.append(_at(node, "import_not_allowed", module, "only numpy, polars, math"))
        elif alias.name not in MODULE_LISTS[module]:
            self.violations.append(
                _at(
                    node,
                    "from_import_not_allowed",
                    f"{module}.{alias.name}",
                    "a name not on that module's list",
                )
            )
        else:
            self.imported.add(alias.asname or alias.name)
            self.from_targets[alias.asname or alias.name] = f"{module}.{alias.name}"


# ------------------------------------------------- state rules ----
FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
UNINITIALIZED = "ndarray"  # np.ndarray(shape) hands out uninitialized memory, possibly an earlier call's


def _is_constant(node: ast.expr) -> bool:
    """A default that cannot carry state: a literal constant, a signed one, or a tuple of those."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        return _is_constant(node.operand)
    return isinstance(node, ast.Tuple) and all(_is_constant(e) for e in node.elts)


def _annotation_nodes(tree: ast.Module) -> set[ast.AST]:
    """Every node inside an argument or return annotation, or the annotation of an annotated assignment."""
    roots: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg | ast.AnnAssign) and node.annotation is not None:
            roots.append(node.annotation)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.returns is not None:
            roots.append(node.returns)
    return {inner for root in roots for inner in ast.walk(root)}


def _root_name(node: ast.expr) -> str | None:
    while isinstance(node, ast.Attribute | ast.Subscript | ast.Starred):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _function_locals(fn: FunctionNode) -> set[str]:
    """Parameters and names bound in the function's own body (nested functions and classes not entered)."""
    a = fn.args
    names = {x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    names |= {x.arg for x in (a.vararg, a.kwarg) if x is not None}
    stack: list[ast.AST] = [] if isinstance(fn, ast.Lambda) else list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        stack.extend(ast.iter_child_nodes(node))
    return names


class _StateRules:
    """Refuses cheaply what lets a value outlive a generate call: a default that is not a constant, every
    attribute store (a local alias of a shared numpy or polars object would otherwise do), an item store
    whose root is not a name local to its function (or is an import, a builtin or no name at all), a
    direct call of ndarray, and every load of the name ndarray (any `<x>.ndarray`, or a name imported as
    numpy.ndarray) outside an annotation, so no alias can reach its uninitialized memory.
    A mutation through a method call is NOT refused here: the fresh module per call
    (pfb.engine.strategy.IsolatedGenerate) is what stops it."""

    def __init__(self, tree: ast.Module, imports: _Imports) -> None:
        self.imports = imports
        self.foreign = set(imports.aliases) | imports.imported | BUILTIN_NAMES
        self.parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        self.locals: dict[ast.AST, set[str]] = {}
        self.annotations = _annotation_nodes(tree)
        self.violations: list[LintViolation] = []
        for node in ast.walk(tree):
            self._node(node)

    def _node(self, node: ast.AST) -> None:
        self._ndarray_load(node)
        if isinstance(node, FunctionNode):
            self._defaults(node)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store | ast.Del):
            self.violations.append(_at(node, "attribute_store", node.attr, "an attribute store or delete"))
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store | ast.Del):
            self._item_store(node)
        elif isinstance(node, ast.Call) and self._is_uninitialized(node.func):
            self.violations.append(_at(node, "uninitialized_array", UNINITIALIZED, "uninitialized memory"))

    def _defaults(self, fn: FunctionNode) -> None:
        for default in (*fn.args.defaults, *(d for d in fn.args.kw_defaults if d is not None)):
            if not _is_constant(default):
                self.violations.append(_at(default, "mutable_default", "default", "a non-constant default"))

    def _enclosing(self, node: ast.AST) -> FunctionNode | None:
        parent = self.parents.get(node)
        while parent is not None and not isinstance(parent, FunctionNode):
            parent = self.parents.get(parent)
        return parent

    def _item_store(self, node: ast.Subscript) -> None:
        root = _root_name(node.value)
        fn = self._enclosing(node)
        if fn is not None and fn not in self.locals:
            self.locals[fn] = _function_locals(fn)
        local = root is not None and root not in self.foreign and (fn is None or root in self.locals[fn])
        if not local:
            self.violations.append(
                _at(node, "item_store_not_local", root or "<expression>", "an item store on a non-local name")
            )

    def _ndarray_load(self, node: ast.AST) -> None:
        if node in self.annotations:
            return
        if not isinstance(node, ast.Attribute | ast.Name) or not isinstance(node.ctx, ast.Load):
            return
        attr = isinstance(node, ast.Attribute) and node.attr == UNINITIALIZED
        name = isinstance(node, ast.Name) and self._is_imported_ndarray(node.id)
        if attr or name:
            self.violations.append(
                _at(node, "ndarray_outside_annotation", UNINITIALIZED, "ndarray loaded outside an annotation")
            )

    def _is_imported_ndarray(self, name: str) -> bool:
        return self.imports.from_targets.get(name) == "numpy.ndarray"

    def _is_uninitialized(self, func: ast.expr) -> bool:
        if isinstance(func, ast.Attribute):
            return func.attr == UNINITIALIZED
        return isinstance(func, ast.Name) and self._is_imported_ndarray(func.id)


def _bound_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    """(every name the file binds, the names bound by a statement other than an import)."""
    bound: set[str] = set()
    plain: set[str] = set()
    for node in ast.walk(tree):
        name: str | None = None
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            name = node.name
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            name = node.id
        elif isinstance(node, ast.arg):
            name = node.arg
        elif isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar):
            name = node.name
        elif isinstance(node, ast.MatchMapping):
            name = node.rest
        if name is not None:
            bound.add(name)
            plain.add(name)
        if isinstance(node, ast.Import | ast.ImportFrom):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
    return bound, plain


class _Checker(ast.NodeVisitor):
    def __init__(self, imports: _Imports, bound: set[str]) -> None:
        self.imports = imports
        self.bound = bound
        self.violations: list[LintViolation] = []

    def _add(self, node: ast.AST, rule: str, name: str, message: str) -> None:
        self.violations.append(_at(node, rule, name, message))

    def _check_binding_name(self, node: ast.AST, name: str | None) -> None:
        if name is None:
            return
        if _is_dunder(name):
            self._add(node, "dunder", name, "a dunder name")
        elif name in FORBIDDEN_BUILTINS:
            self._add(node, "builtin_not_allowed", name, "a builtin that is not on the list")

    def visit_Global(self, node: ast.Global) -> None:
        self._add(node, "global_statement", ",".join(node.names), "global is not allowed")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._add(node, "nonlocal_statement", ",".join(node.names), "nonlocal is not allowed")

    def visit_Name(self, node: ast.Name) -> None:
        self._check_binding_name(node, node.id)
        is_load = isinstance(node.ctx, ast.Load)
        checked = is_load and not _is_dunder(node.id) and node.id not in FORBIDDEN_BUILTINS
        if checked and node.id not in self.bound and node.id not in BUILTIN_NAMES:
            self._add(node, "name_not_allowed", node.id, "neither defined in the file nor on the list")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = node.attr
        module = self._module_of(node.value)
        if _is_dunder(name):
            self._add(node, "dunder", name, "a dunder attribute")
        elif module is not None and name not in MODULE_LISTS[module]:
            self._add(node, "module_attribute_not_allowed", f"{module}.{name}", "not on that module's list")
        elif module is None and name not in OTHER_ATTRIBUTE_NAMES:
            self._add(node, "attribute_not_allowed", name, "not on the numpy or polars list")
        self.generic_visit(node)

    def _module_of(self, value: ast.expr) -> str | None:
        if isinstance(value, ast.Name):
            return self.imports.aliases.get(value.id)
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_binding_name(node, node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_binding_name(node, node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._check_binding_name(node, node.name)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self._check_binding_name(node, node.arg)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg is not None and _is_dunder(node.arg):
            self._add(node, "dunder", node.arg, "a dunder keyword")
        self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> None:
        for part in (node.name, node.asname):
            if part is not None and any(_is_dunder(p) for p in part.split(".")) and part != FUTURE_MODULE:
                self._add(node, "dunder", part, "a dunder import name")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is not None and node.module != FUTURE_MODULE and _is_dunder(node.module):
            self._add(node, "dunder", node.module, "a dunder module")
        self.generic_visit(node)


def lint_source(source: str, filename: str = "<strategy>") -> list[LintViolation]:
    """Every violation of the allowlist in `source`, in line order; an empty list means the file passes."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [LintViolation("syntax_error", exc.lineno or 0, exc.offset or 0, filename, str(exc.msg))]
    imports = _Imports()
    imports.visit(tree)
    bound, plain = _bound_names(tree)
    checker = _Checker(imports, bound | imports.imported)
    checker.visit(tree)
    rebound = [
        LintViolation("alias_rebound", 0, 0, alias, "a module alias is also bound by another statement")
        for alias in sorted(imports.aliases)
        if alias in plain
    ]
    found = imports.violations + checker.violations + rebound + _StateRules(tree, imports).violations
    return sorted(found, key=lambda v: (v.line, v.col, v.rule, v.name))


def lint_strategy(path: Path | str) -> list[LintViolation]:
    """Lint a strategy file by path (read as UTF-8 text; nothing is imported or executed)."""
    source = Path(path).read_text(encoding="utf-8")
    return lint_source(source, filename=str(path))


__all__ = ["LintViolation", "lint_source", "lint_strategy"]
