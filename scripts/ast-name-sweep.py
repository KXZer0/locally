#!/usr/bin/env python3
"""Report names which Python modules load without binding or importing.

This is deliberately mechanical.  It is not a style checker; it catches the
easy-to-miss failure mode in a file split where a function moves but one of its
module globals does not.  ``global name`` is treated as a reference to a real
module binding, not as a binding by itself.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPECIALS = {
    "__builtins__", "__cached__", "__file__", "__loader__", "__name__",
    "__package__", "__spec__",
}


def bound_names(target: ast.AST) -> set[str]:
    """Return every name bound by an assignment-like target."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(bound_names(item) for item in target.elts))
    if isinstance(target, ast.Starred):
        return bound_names(target.value)
    return set()


class BindingCollector(ast.NodeVisitor):
    """Collect bindings in one lexical scope, stopping at nested scopes."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Import(self, node: ast.Import | ast.ImportFrom) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".", 1)[0])

    visit_ImportFrom = visit_Import

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self.names.add(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest:
            self.names.add(node.rest)
        self.generic_visit(node)


def direct_bindings(body: list[ast.stmt]) -> set[str]:
    collector = BindingCollector()
    for stmt in body:
        collector.visit(stmt)
    return collector.names


def argument_names(args: ast.arguments) -> set[str]:
    values = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if args.vararg:
        values.append(args.vararg)
    if args.kwarg:
        values.append(args.kwarg)
    return {arg.arg for arg in values}


class Sweep(ast.NodeVisitor):
    def __init__(self, tree: ast.Module):
        self.module_bindings = direct_bindings(tree.body)
        self.scopes: list[set[str]] = [set(self.module_bindings)]
        self.globals: list[set[str]] = [set()]
        self.unbound: list[tuple[int, int, str]] = []

    def _known(self, name: str) -> bool:
        if name in SPECIALS or hasattr(builtins, name):
            return True
        if name in self.globals[-1]:
            return name in self.module_bindings
        return any(name in scope for scope in reversed(self.scopes))

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and not self._known(node.id):
            self.unbound.append((node.lineno, node.col_offset, node.id))

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Defaults and decorators execute in the enclosing scope.
        for item in [*node.decorator_list, *node.args.defaults,
                     *(x for x in node.args.kw_defaults if x),
                     *getattr(node, "type_params", ())]:
            self.visit(item)
        if node.returns:
            self.visit(node.returns)
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
                    node.args.vararg, node.args.kwarg]:
            if arg and arg.annotation:
                self.visit(arg.annotation)
        local = direct_bindings(node.body) | argument_names(node.args)
        declared_global = {
            name for child in ast.walk(node) if isinstance(child, ast.Global)
            for name in child.names
        }
        local -= declared_global
        self.scopes.append(local)
        self.globals.append(declared_global)
        for stmt in node.body:
            self.visit(stmt)
        self.globals.pop()
        self.scopes.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Lambda(self, node: ast.Lambda) -> None:
        local = argument_names(node.args)
        self.scopes.append(local)
        self.globals.append(set())
        self.visit(node.body)
        self.globals.pop()
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for item in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(item)
        self.scopes.append(direct_bindings(node.body))
        self.globals.append(set())
        for stmt in node.body:
            self.visit(stmt)
        self.globals.pop()
        self.scopes.pop()


def python_files() -> list[Path]:
    return [ROOT / "locally.py", *sorted((ROOT / "core").rglob("*.py"))]


def main() -> int:
    total = 0
    for path in python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            print(f"{path.relative_to(ROOT)}: {exc}")
            total += 1
            continue
        sweep = Sweep(tree)
        sweep.visit(tree)
        findings = sorted(set(sweep.unbound))
        for line, column, name in findings:
            print(f"{path.relative_to(ROOT)}:{line}:{column + 1}: unbound name {name}")
        total += len(findings)
    print(f"AST name sweep: {total} unbound name{'s' if total != 1 else ''}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
