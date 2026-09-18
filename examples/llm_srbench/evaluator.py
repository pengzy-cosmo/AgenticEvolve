"""Evaluator: a candidate is a compact expression whose parameters are fitted on the
training observations. Shared by every method: Full's evaluation, the LLM-SR
baseline's sandbox (run_llmsr.py) and the final scoring (score.py).

1. trace: equation runs on symbolic inputs and symbolic params[0..9]; the
   result is the expression the program computes. Programs that branch on
   the data, convert values to Python numbers or use other numpy functions
   than the operators below cannot be traced and fail.
2. check: operators limited to + - * / pow, exp log sqrt sin cos, abs max min
   and piecewise selection (np.where); size at most MAX_SIZE, counting every
   operator, input, params entry and numeric constant as one node, n-ary sums
   and products as n - 1 binary operators (PySR's maxsize convention).
3. fit: the traced expression is compiled to numpy and its 10 params fitted
   by least squares from all ones. LLM-SR minimizes the mean squared error
   with BFGS instead; on CRK21 that fit stops at train NMSE 7e-7 for the
   generating equation itself, where least squares reaches 1e-15.

Metrics: mse; nmse = mse / var(target); combined_score = -nmse, the fitness;
complexity = size / MAX_SIZE. A non-finite mse fails the evaluation.
data/train.csv next to this file holds the observations, target column first.
"""

import importlib.util
import operator
from functools import reduce
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import least_squares
from sympy.printing.numpy import NumPyPrinter

N_PARAMS = 10
MAX_SIZE = 30
PARAMS = sp.symbols(f"params0:{N_PARAMS}")
RULE = (
    "The evaluation traces equation into a single expression by running it on symbolic "
    "inputs and params, so the body must be a plain formula: + - * / and powers, np.exp, "
    "np.log, np.sqrt, np.sin, np.cos, np.abs, np.maximum, np.minimum and np.where; other "
    f"functions and data-dependent branches fail. At most {MAX_SIZE} nodes, every operator, "
    "input, params entry and numeric constant counting one."
)

_UFUNCS = {
    np.add: operator.add,
    np.subtract: operator.sub,
    np.multiply: operator.mul,
    np.divide: operator.truediv,
    np.true_divide: operator.truediv,
    np.power: operator.pow,
    np.float_power: operator.pow,
    np.negative: operator.neg,
    np.positive: operator.pos,
    np.square: lambda a: a**2,
    np.sqrt: sp.sqrt,
    np.cbrt: sp.cbrt,
    np.exp: sp.exp,
    np.log: sp.log,
    np.sin: sp.sin,
    np.cos: sp.cos,
    np.absolute: sp.Abs,
    np.fabs: sp.Abs,
    np.maximum: sp.Max,
    np.minimum: sp.Min,
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
            return _Trace(sp.Piecewise((a, cond), (b, True)))
        if func is np.clip:
            x, low, high = map(_expr, args)
            return _Trace(sp.Min(sp.Max(x, low), high))
        return NotImplemented

    def __bool__(self):
        raise TypeError("equation must not branch on its inputs")

    def __float__(self):
        raise TypeError("equation must not convert its inputs to numbers")

    def __neg__(self):
        return _Trace(-self.expr)

    def __pos__(self):
        return self

    def __abs__(self):
        return _Trace(sp.Abs(self.expr))


def _binary(op):
    def forward(self, other):
        return _Trace(op(self.expr, _expr(other)))

    def reverse(self, other):
        return _Trace(op(_expr(other), self.expr))

    return forward, reverse


for _name, _op in {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "truediv": operator.truediv,
    "pow": operator.pow,
}.items():
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
_Trace.__hash__ = object.__hash__
for _name in ("exp", "log", "sqrt", "sin", "cos"):  # numpy's fallback on object arrays
    setattr(_Trace, _name, (lambda fn: lambda self: _Trace(fn(self.expr)))(getattr(sp, _name)))


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
    """Nodes of the expression tree: every operator, symbol and constant one node.

    Sympy writes a - b as a + (-1)*b, a / b as a * b**(-1) and sqrt(b) as
    b**(1/2); the -1 factor counts as a sign, the b**(-1) factor as a division
    and the 1/2 or 1/3 exponent as a unary root, so all three are PySR's count
    (3, 3, 2). A bare sign, -x, costs nothing here and 3 in PySR.
    """
    if isinstance(expr, sp.Piecewise):
        args = [part for piece in expr.args for part in piece.args]
    elif isinstance(expr, sp.Pow) and expr.exp in (sp.Rational(1, 2), sp.Rational(1, 3)):
        return 1 + size(expr.base)
    elif isinstance(expr, sp.Mul):
        args = [arg for arg in expr.args if arg != -1]
        args = [arg.base if isinstance(arg, sp.Pow) and arg.exp == -1 else arg for arg in args]
        return max(len(args) - 1, 0) + sum(size(arg) for arg in args)
    else:
        args = expr.args
    if not args:
        return 1
    operators = len(args) - 1 if isinstance(expr, (sp.Add, sp.Max, sp.Min)) else 1
    return operators + sum(size(arg) for arg in args)


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


def fit(equation, x, y):
    """Least-squares fit of params from all ones. x has one column per input; returns (params, mse)."""

    def residual(params):
        return equation(*x.T, params=params) - y

    with np.errstate(all="ignore"):
        result = least_squares(residual, np.ones(N_PARAMS))
    return result.x, float(np.mean(result.fun**2))


def load(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    if not hasattr(program, "equation"):
        raise ValueError("program must define equation(..., params)")
    return program.equation


def evaluate(program_path):
    data = Path(__file__).parent / "data/train.csv"
    inputs = data.read_text().split("\n", 1)[0].split(",")[1:]
    rows = np.loadtxt(data, delimiter=",", skiprows=1)
    y, x = rows[:, 0], rows[:, 1:]

    expr = check(trace(load(program_path), inputs))
    _, mse = fit(compile(expr, inputs), x, y)
    if not np.isfinite(mse):
        raise ValueError(f"fit produced a non-finite mse: {mse}")
    nmse = mse / float(np.var(y))
    return {
        "combined_score": -nmse,
        "mse": mse,
        "nmse": nmse,
        "complexity": size(expr) / MAX_SIZE,
    }
