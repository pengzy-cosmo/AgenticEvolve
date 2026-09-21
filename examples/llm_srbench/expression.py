"""A candidate equation as a compact expression: trace, count, check, compile.

trace runs equation() on symbolic inputs and symbolic params[0..9] and returns
the expression the program computes, as written: no algebraic simplification,
so 0.5*(x+y) stays 5 nodes and a reused intermediate is counted at every use.
Programs that branch on the data, convert values to Python numbers or use
other numpy functions than the operators below cannot be traced and fail.

size counts every operator, input, params entry and numeric constant as one
node, an n-ary sum or product as n - 1 binary operators (PySR's maxsize
convention); a - b, a / b and sqrt(a) are 3, 3 and 2 nodes as in PySR, a bare
sign -a costs nothing here and 3 in PySR.

check rejects operators outside + - * / pow, exp log sqrt sin cos, abs max min
and piecewise selection (np.where), and expressions above MAX_SIZE nodes.

compile turns the expression back into a numpy equation(*columns, params).

load checks imports and prohibited names before executing a candidate and
returning equation. All Python candidate loaders share these protocol checks;
filesystem and process isolation require a separate execution sandbox.
"""

import ast
import builtins
from functools import reduce

import numpy as np
import sympy as sp
from sympy.printing.numpy import NumPyPrinter

N_PARAMS = 10
MAX_SIZE = 30
PARAMS = sp.symbols(f"params0:{N_PARAMS}")
RULE = (
    "The evaluation traces equation into a single expression by running it on symbolic "
    "inputs and params, so the body must be a plain formula: + - * / and powers, np.exp, "
    "np.log, np.sqrt, np.sin, np.cos, np.abs, np.maximum, np.minimum and np.where; other "
    f"functions and data-dependent branches fail. At most {MAX_SIZE} nodes, every operator, "
    "input, params entry and numeric constant counting one, as written: an intermediate "
    "variable is counted at every use, and a literal constant costs the same node as a "
    "params entry. Larger expressions fail the evaluation. The program may import numpy "
    "and math only."
)


def _add(a, b):
    return sp.Add(a, b, evaluate=False)


def _mul(a, b):
    return sp.Mul(a, b, evaluate=False)


def _neg(a):
    return sp.Mul(-1, a, evaluate=False)


def _sub(a, b):
    return _add(a, _neg(b))


def _pow(a, b):
    return sp.Pow(a, b, evaluate=False)


def _div(a, b):
    return _mul(a, _pow(b, -1))


def _root(n):
    return lambda a: _pow(a, sp.Rational(1, n))


def _function(f):
    return lambda *a: f(*a, evaluate=False)


_UFUNCS = {
    np.add: _add,
    np.subtract: _sub,
    np.multiply: _mul,
    np.divide: _div,
    np.true_divide: _div,
    np.power: _pow,
    np.float_power: _pow,
    np.negative: _neg,
    np.positive: lambda a: a,
    np.square: lambda a: _pow(a, 2),
    np.sqrt: _root(2),
    np.cbrt: _root(3),
    np.exp: _function(sp.exp),
    np.log: _function(sp.log),
    np.sin: _function(sp.sin),
    np.cos: _function(sp.cos),
    np.absolute: _function(sp.Abs),
    np.fabs: _function(sp.Abs),
    np.maximum: _function(sp.Max),
    np.minimum: _function(sp.Min),
    np.greater: sp.Gt,
    np.greater_equal: sp.Ge,
    np.less: sp.Lt,
    np.less_equal: sp.Le,
    np.equal: sp.Eq,
    np.not_equal: sp.Ne,
}
_ALLOWED = (
    sp.Symbol,
    sp.Number,
    sp.NumberSymbol,
    sp.Add,
    sp.Mul,
    sp.Pow,
    sp.exp,
    sp.log,
    sp.sin,
    sp.cos,
    sp.Abs,
    sp.Max,
    sp.Min,
    sp.Piecewise,
    sp.functions.elementary.piecewise.ExprCondPair,
    sp.core.relational.Relational,
    sp.logic.boolalg.BooleanAtom,
)


def _expr(value):
    return value.expr if isinstance(value, _Trace) else sp.sympify(value)


class _Trace:
    """Scalar stand-in that records the operations applied to it as a sympy expression."""

    __array_priority__ = 1000

    def __init__(self, expr):
        self.expr = expr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs or ufunc not in _UFUNCS:
            return NotImplemented
        fn = _UFUNCS[ufunc]
        if any(isinstance(i, np.ndarray) for i in inputs):
            return np.frompyfunc(lambda *a: _Trace(fn(*map(_expr, a))), ufunc.nin, 1)(*inputs)
        return _Trace(fn(*map(_expr, inputs)))

    def __array_function__(self, func, types, args, kwargs):
        if kwargs:
            return NotImplemented
        if func is np.where:
            cond, a, b = map(_expr, args)
            return _Trace(sp.Piecewise((a, cond), (b, True), evaluate=False))
        if func is np.clip:
            x, low, high = map(_expr, args)
            return _Trace(_UFUNCS[np.minimum](_UFUNCS[np.maximum](x, low), high))
        return NotImplemented

    def __bool__(self):
        raise TypeError("equation must not branch on its inputs")

    def __float__(self):
        raise TypeError("equation must not convert its inputs to numbers")

    __hash__ = object.__hash__

    def __neg__(self):
        return _Trace(_neg(self.expr))

    def __pos__(self):
        return self

    def __abs__(self):
        return _Trace(_UFUNCS[np.absolute](self.expr))


def _binary(op):
    def forward(self, other):
        return _Trace(op(self.expr, _expr(other)))

    def reverse(self, other):
        return _Trace(op(_expr(other), self.expr))

    return forward, reverse


for _name, _op in {"add": _add, "sub": _sub, "mul": _mul, "truediv": _div, "pow": _pow}.items():
    setattr(_Trace, f"__{_name}__", _binary(_op)[0])
    setattr(_Trace, f"__r{_name}__", _binary(_op)[1])
for _name, _op in {
    "lt": sp.Lt,
    "le": sp.Le,
    "gt": sp.Gt,
    "ge": sp.Ge,
    "eq": sp.Eq,
    "ne": sp.Ne,
}.items():
    setattr(_Trace, f"__{_name}__", _binary(_op)[0])
for _name in ("exp", "log", "sqrt", "sin", "cos"):  # numpy falls back to these on object arrays
    _fn = _UFUNCS[getattr(np, _name)]
    setattr(_Trace, _name, (lambda fn: lambda self: _Trace(fn(self.expr)))(_fn))


def trace(equation, inputs):
    """The expression equation(*inputs, params) computes, inputs being column names."""
    params = np.array([_Trace(p) for p in PARAMS], dtype=object)
    result = equation(*[_Trace(sp.Symbol(name)) for name in inputs], params)
    if isinstance(result, _Trace):
        return result.expr
    if isinstance(result, (int, float, np.number)):
        return sp.Float(result)
    raise TypeError(f"equation must compute a single expression, got {type(result).__name__}")


def size(expr):
    """Nodes of the expression tree; see the module docstring for the convention."""
    if isinstance(expr, sp.Piecewise):
        args = [part for piece in expr.args for part in piece.args]
    elif isinstance(expr, sp.Pow) and expr.exp in (sp.Rational(1, 2), sp.Rational(1, 3)):
        return 1 + size(expr.base)  # a root
    elif isinstance(expr, sp.Mul):
        args = [arg for arg in expr.args if arg != -1]  # a sign
        args = [arg.base if isinstance(arg, sp.Pow) and arg.exp == -1 else arg for arg in args]
        return max(len(args) - 1, 0) + sum(size(arg) for arg in args)  # b**(-1): a division
    else:
        args = expr.args
    if not args:
        return 1
    operators = len(args) - 1 if isinstance(expr, (sp.Add, sp.Max, sp.Min)) else 1
    return operators + sum(size(arg) for arg in args)


MODULES = {"numpy", "math"}
FORBIDDEN = {
    "open", "exec", "eval", "compile", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "input", "__import__",
}  # fmt: skip


def load(source, filename="program.py"):
    """Check imports and names, execute the source, and return equation.

    Imports are limited to numpy and math. Prohibited names and dunder access
    raise ValueError before execution.
    """
    tree = ast.parse(source, filename)
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=aliases):
                modules = [alias.name for alias in aliases]
            case ast.ImportFrom(module=module):
                modules = [module or ""]
            case ast.Name(id=name) | ast.Attribute(attr=name) if (
                name in FORBIDDEN or name.startswith("__")
            ):
                raise ValueError(f"{name} is not allowed in a program")
            case _:
                continue
        for module in modules:
            if module.partition(".")[0] not in MODULES:
                raise ValueError(f"import of {module}: a program may import numpy and math only")
    namespace = {}
    exec(builtins.compile(tree, filename, "exec"), namespace)
    if "equation" not in namespace:
        raise ValueError("program must define equation(..., params)")
    return namespace["equation"]


def check(expr):
    """The expression if it is within the operator set and MAX_SIZE, else ValueError."""
    for node in sp.preorder_traversal(expr):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"operator not allowed: {type(node).__name__}")
    if (n := size(expr)) > MAX_SIZE:
        raise ValueError(f"expression has {n} nodes, at most {MAX_SIZE} allowed")
    return expr


class _Printer(NumPyPrinter):
    def _print_Max(self, expr):
        return reduce(lambda a, b: f"numpy.maximum({a}, {b})", map(self._print, expr.args))

    def _print_Min(self, expr):
        return reduce(lambda a, b: f"numpy.minimum({a}, {b})", map(self._print, expr.args))


def compile(expr, inputs):
    """Numeric equation(*columns, params) of the expression."""
    f = sp.lambdify([*(sp.Symbol(name) for name in inputs), *PARAMS], expr, printer=_Printer)

    def equation(*columns, params):
        return np.broadcast_to(np.asarray(f(*columns, *params), dtype=float), columns[0].shape)

    return equation
