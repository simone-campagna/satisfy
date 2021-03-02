import abc
import functools
import itertools
import logging


__all__ = [
    'expression_globals',
    'Expression',
    'Const',
    'Variable',
    'Parameter',
    'InputReader',
    'InputConst',
    'GlobalVariable',
    'FunctionCall',
    'CompiledExpression',
]


LOG = logging.getLogger(__name__)


def prod(values, start=1):
    for value in values:
        start *= value
    return start


EXPRESSION_GLOBALS = {
    'sum': sum,
    'prod': prod,
}


def expression_globals():
    return EXPRESSION_GLOBALS


class ExpressionError(RuntimeError):
    pass


class EvaluationError(ExpressionError):
    pass


class ExpressionBase(abc.ABC):
    @abc.abstractmethod
    def free_vars(self, substitution):
        raise NotImplementedError()

    def vars(self):
        yield from self.free_vars({})

    def is_free(self, substitution=None):
        if substitution is None:
            substitution = {}
        for _ in self.free_vars(substitution):
            return False
        return True

    @abc.abstractmethod
    def is_externally_updated(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate(self, substitution):
        raise NotImplementedError()


class Expression(ExpressionBase):
    @abc.abstractmethod
    def equals(self, other):
        raise NotImplementedError()

    def _sort_key(self):
        return str(self)

    @classmethod
    def coerce(cls, val):
        if isinstance(val, Expression):
            return val
        else:
            return Const(val)

    def compile_py_expr(self):
        return compile(self.py_expr(), '<stdin>', 'eval')

    # def py_source(self):
    #     alist = list(self.vars())
    #     alist.append("**__dummy_kwargs")
    #     return "lambda {}: ({})".format(", ".join(alist), self.py_expr())

    # def compile_py_function(self):
    #     py_source = self.py_source()
    #     return eval(py_source, EXPRESSION_GLOBALS)

    # def as_function(self):
    #     try:
    #         return self.compile_py_function()
    #     except:
    #         LOG.exception("compilation of py function failed:")
    #     return self.evaluate

    @abc.abstractmethod
    def py_expr(self):
        raise NotImplementedError()

    # unops:
    def __pos__(self):
        return self

    def __neg__(self):
        return Neg(self)

    def __abs__(self):
        return Abs(self)

    # binops:
    def __add__(self, other):
        if isinstance(other, Const) and other.value == 0:
            return self
        return _make_sum(self, other)

    def __sub__(self, other):
        if isinstance(other, Const) and other.value == 0:
            return self
        return Sub(self, other)

    def __mul__(self, other):
        if isinstance(other, Const):
            if other.value == 0:
                return other
            elif other.value == 1:
                return self
        return _make_prod(self, other)

    def __div__(self, other):
        if isinstance(other, Const):
            if other.value == 1:
                return self
        return Div(self, other)

    def __mod__(self, other):
        return Mod(self, other)

    def __pow__(self, other):
        if isinstance(other, Const):
            if other.value == 0:
                return Const(1)
            elif other.value == 1:
                return self
        return Pow(self, other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __floordiv__(self, other):
        return self.__div__(other)

    def __eq__(self, other):
        return Eq(self, other)

    def __ne__(self, other):
        return Ne(self, other)

    def __lt__(self, other):
        return Lt(self, other)

    def __le__(self, other):
        return Le(self, other)

    def __gt__(self, other):
        return Gt(self, other)

    def __ge__(self, other):
        return Ge(self, other)

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __not__(self):
        return Not(self)

    # r-binops:
    def __radd__(self, other):
        return self.coerce(other) + self

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __rmul__(self, other):
        return self.coerce(other) * self

    def __rdiv__(self, other):
        return self.coerce(other) // self

    def __rtruediv__(self, other):
        return self.coerce(other) // self

    def __rmod__(self, other):
        return self.coerce(other) % self

    def __rpow__(self, other):
        return self.coerce(other) ** self

    def __req__(self, other):
        return self.coerce(other) == self

    def __rne__(self, other):
        return self.coerce(other) != self

    def __rlt__(self, other):
        return self.coerce(other) < self

    def __rle__(self, other):
        return self.coerce(other) <= self

    def __rgt__(self, other):
        return self.coerce(other) > self

    def __rge__(self, other):
        return self.coerce(other) >= self

    def __rand__(self, other):
        return And(self.coerce(other), self)

    def __ror__(self, other):
        return Or(self.coerce(other), self)


class Const(Expression):
    def __init__(self, value):
        self._value = value

    def is_externally_updated(self):
        return False

    def free_vars(self, substitution):
        yield from ()

    def equals(self, other):
        return type(self) == type(other) and self._get_value() == other.value

    def _get_value(self):
        return self._value

    @property
    def value(self):
        return self._get_value()

    def evaluate(self, substitution):
        return self._get_value()

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self._get_value())

    def __str__(self):
        return str(self._get_value())

    def py_expr(self):
        return str(self)

    def __add__(self, other):
        value = self._get_value()
        if value == 0:
            return self.coerce(other)
        elif isinstance(other, Const):
            return Const(value + other.value)
        else:
            return _make_sum(self, other)

    def __sub__(self, other):
        value = self._get_value()
        if value == 0:
            return Neg(other)
        elif isinstance(other, Const):
            return Const(value - other.value)
        else:
            return Sub(self, other)

    def __mul__(self, other):
        value = self._get_value()
        if value == 0:
            return self
        elif value == 1:
            return self.coerce(other)
        elif isinstance(other, Const):
            return Const(value * other.value)
        return _make_prod(self, other)

    def __div__(self, other):
        value = self._get_value()
        if value == 0:
            return self
        elif isinstance(other, Const):
            return Const(value // other.value)
        else:
            return Div(self, other)

    def __pow__(self, other):
        value = self._get_value()
        if value == 1:
            return self
        elif isinstance(other, Const):
            return Const(value ** other.value)
        return Pow(self, other)

    def __mod__(self, other):
        if isinstance(other, Const):
            return Const(self.value % other.value)
        return Mod(self, other)

    def __eq__(self, other):
        if isinstance(other, Const):
            return int(self.value == other.value)
        return Eq(self, other)

    def __ne__(self, other):
        if isinstance(other, Const):
            return int(self.value != other.value)
        return Ne(self, other)

    def __lt__(self, other):
        if isinstance(other, Const):
            return int(self.value < other.value)
        return Lt(self, other)

    def __le__(self, other):
        if isinstance(other, Const):
            return int(self.value <= other.value)
        return Le(self, other)

    def __gt__(self, other):
        if isinstance(other, Const):
            return int(self.value > other.value)
        return Gt(self, other)

    def __ge__(self, other):
        if isinstance(other, Const):
            return int(self.value >= other.value)
        return Ge(self, other)

    def __and__(self, other):
        if isinstance(other, Const):
            return int(self.value and other.value)
        return And(self, other)

    def __or__(self, other):
        if isinstance(other, Const):
            return int(self.value or other.value)
        return Or(self, other)


class InputReader:
    def __init__(self, input_file=None, output_file=None, prompt=None, input_type=int):
        if input_file is None:
            input_file = sys.stdin
        self.input_file = input_file
        if output_file is None:
            output_file = sys.stdout
        self.output_file = output_file
        self.prompt = prompt
        self.input_type = input_type

    def __call__(self):
        if self.prompt:
            print(self.prompt, end='', file=self.output_file, flush=True)
        return self.input_type(self.input_file.readline())


class InputConst(Const):
    def __init__(self, reader=input):
        self.reader = reader
        super().__init__(None)

    def has_value(self):
        return self._value is not None

    def is_externally_updated(self):
        return False

    def _get_value(self):
        if self._value is None:
            self._value = self.reader()
        return self._value

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self.reader)


class Collection(Expression):
    def __init__(self, expressions):
        if False: #self.is_commutative():
            elist = [self.coerce(expr) for expr in expressions]
            elist.sort(key=lambda x: x._sort_key())
            self._expressions = tuple(elist)
        else:
            self._expressions = tuple(self.coerce(expr) for expr in expressions)
        vlist = []
        for expr in self._expressions:
            for var in expr.vars():
                if var not in vlist:
                    vlist.append(var)
        self._vars = tuple(vlist)

    def is_externally_updated(self):
        return any(expression.is_externally_updated() for expression in self._expressions)

    @classmethod
    def is_commutative(cls):
        return False

    def equals(self, other):
        return type(self) == type(other) and len(self) == len(other) and all(l_e == r_e for l_e, r_e in zip(self, other))

    def __iter__(self):
        yield from self._expressions

    def __len__(self):
        return len(self._expressions)

    def vars(self):
        yield from self._vars

    def free_vars(self, substitution):
        for var in self._vars:
            if var not in substitution:
                yield var

    def __repr__(self):
        return "{}({})".format(
            type(self).__name__,
            ", ".join(repr(expr) for expr in self._expressions))

    def __str__(self):
        return "{}({})".format(
            type(self).__name__,
            ", ".join(str(expr) for expr in self._expressions))


def _make_accumulate(expressions, neutral_element, absorbing_element, binop_class, accumulate_class):
    const = neutral_element
    expanded_expressions = []
    for expression in expressions:
        if isinstance(expression, binop_class):
           expanded_expressions.extend(expression)
        else:
           expanded_expressions.append(Expression.coerce(expression))
    elist = []
    for expression in expanded_expressions:
        if isinstance(expression, Const):
            if expression.value == neutral_element:
                continue
            elif expression.value == absorbing_element:
                const = absorbing_element
                elist.clear()
                break
            else:
                const = accumulate_class.op(const, [expression.value])
                if const == absorbing_element:
                    elist.clear()
                    break
        elif isinstance(expression, accumulate_class) and len(expression) == 0:
            const = accumulate_class.op(const, [expression.const])
        elif isinstance(expression, accumulate_class):
            if expression.const != neutral_element:
                const = accumulate_class.op(const, [expression.const])
                if const == absorbing_element:
                    elist.clear()
                    break
            elist.extend(expression)
        else:
            elist.append(expression)
    if elist:
        if const is neutral_element:
            if len(elist) == 1:
                return elist[0]
            elif len(elist) == 2:
                return binop_class(elist[0], elist[1])
            else:
                return accumulate_class(elist)
        else:
            if len(elist) == 1:
                return binop_class(const, elist[0])
            else:
                return accumulate_class(elist, const=const)
    else:
        return Const(const)


def _make_prod(l_expr, r_expr):
    # return Mul(l_expr, r_expr)
    return _make_accumulate([l_expr, r_expr], 1, 0, Mul, Prod)


def _make_sum(l_expr, r_expr):
    # return Add(l_expr, r_expr)
    return _make_accumulate([l_expr, r_expr], 0, None, Add, Sum)


class UnOp(Collection):
    def __init__(self, operand):
        super().__init__([operand])

    def free_vars(self, substitution):
        yield from self._expressions[0].free_vars(substitution)

    def evaluate(self, substitution):
        return self._op(self._expressions[0].evaluate(substitution))

    @property
    def operand(self):
        return self._expressions[0]


class Neg(UnOp):
    def evaluate(self, substitution):
        return -self._expressions[0].evaluate(substitution)

    def __str__(self):
        return "-({})".format(self.operand)

    def py_expr(self):
        return str(self)


class Abs(UnOp):
    def evaluate(self, substitution):
        return abs(self._expressions[0].evaluate(substitution))

    def __str__(self):
        return "abs({})".format(self.operand)

    def py_expr(self):
        return str(self)


class BinOp(Collection):
    __symbol__ = "?"
    __py_symbol__ = "?"

    def __init__(self, left_operand, right_operand):
        super().__init__([left_operand, right_operand])

    @property
    def left_operand(self):
        return self._expressions[0]

    @property
    def right_operand(self):
        return self._expressions[1]

    @abc.abstractmethod
    def _op(self, left_value, right_value):
        raise NotImplementedError()

    def evaluate(self, substitution):
        return self._op(self._expressions[0].evaluate(substitution), self._expressions[1].evaluate(substitution))

    def __str__(self):
        return "({} {} {})".format(self.left_operand, self.__symbol__, self.right_operand)

    def py_expr(self):
        return "({} {} {})".format(self.left_operand.py_expr(), self.__py_symbol__, self.right_operand.py_expr())


class Add(BinOp):
    __symbol__ = '+'
    __py_symbol__ = '+'

    def _op(self, left_value, right_value):
        return left_value + right_value


class Mul(BinOp):
    __symbol__ = '*'
    __py_symbol__ = '*'

    def _op(self, left_value, right_value):
        return left_value * right_value


class Sub(BinOp):
    __symbol__ = '-'
    __py_symbol__ = '-'

    def _op(self, left_value, right_value):
        return left_value - right_value


class Div(BinOp):
    __symbol__ = '/'
    __py_symbol__ = '//'

    def _op(self, left_value, right_value):
        return left_value // right_value


class Mod(BinOp):
    __symbol__ = '%'
    __py_symbol__ = '%'

    def _op(self, left_value, right_value):
        return left_value % right_value


class Pow(BinOp):
    __symbol__ = '**'
    __py_symbol__ = '**'

    def _op(self, left_value, right_value):
        return left_value ** right_value


class Eq(BinOp):
    __symbol__ = '=='
    __py_symbol__ = '=='

    @classmethod
    def is_commutative(cls):
        return True

    def _op(self, left_value, right_value):
        print("==", left_value, right_value)
        return left_value == right_value


class Ne(BinOp):
    __symbol__ = '!='
    __py_symbol__ = '!='

    @classmethod
    def is_commutative(cls):
        return True

    def _op(self, left_value, right_value):
        return left_value != right_value


class Lt(BinOp):
    __symbol__ = '<'
    __py_symbol__ = '<'

    def _op(self, left_value, right_value):
        return left_value < right_value


class Le(BinOp):
    __symbol__ = '<='
    __py_symbol__ = '<='

    def _op(self, left_value, right_value):
        return left_value <= right_value


class Gt(BinOp):
    __symbol__ = '>'
    __py_symbol__ = '>'

    def _op(self, left_value, right_value):
        return left_value > right_value


class Ge(BinOp):
    __symbol__ = '>='
    __py_symbol__ = '>='

    def _op(self, left_value, right_value):
        return left_value >= right_value


class And(BinOp):
    __symbol__ = '&'
    __py_symbol__ = 'and'

    def _op(self, left_value, right_value):
        return left_value and right_value


class Or(BinOp):
    __symbol__ = '|'
    __py_symbol__ = 'or'

    def _op(self, left_value, right_value):
        return left_value or right_value


class Not(UnOp):
    def evaluate(self, substitution):
        return not self._expressions[0].evaluate(substitution)

    def __str__(self):
        return "^{}".format(self._expressions[0])

    def py_expr(self):
        return "not {}".format(self._expressions[0].py_expr())


class Accumulation(Collection):
    __neutral_element__ = None
    __absorbing_element__ = None

    def __init__(self, expressions, const=None):
        cls = type(self)
        neutral_element = cls.__neutral_element__
        absorbing_element = cls.__absorbing_element__
        if const is None:
            const = neutral_element
        elist = []
        for expression in expressions:
            expression = self.coerce(expression)
            if isinstance(expression, Const):
                if expression.value == neutral_element:
                    continue
                elif expression.value == absorbing_element:
                    const = absorbing_element
                    elist.clear()
                    break
                else:
                    const = self.op(const, [expression.value])
            elif isinstance(expression, Accumulation) and len(expression) == 0:
                const = self.op(const, [expression.const])
            elif self._has_same_type(expression):
                if expression.const == absorbing_element:
                    const = absorbing_element
                    elist.clear()
                    break
                elif expression.const != neutral_element:
                    const = self.op(const, [expression.const])
                elist.extend(expression)
            else:
                elist.append(expression)
        self._const = const
        super().__init__(elist)

    @property
    def const(self):
        return self._const

    @classmethod
    @abc.abstractmethod
    def op(cls, start, values):
        raise NotImplementedError()

    def evaluate(self, substitution):
        return self.op(self._const, (e.evaluate(substitution) for e in self._expressions))

    def __repr__(self):
        return "{}({}, const={!r})".format(
            type(self).__name__,
            ", ".join(repr(e) for e in self),
            self._const)

    def _as_str(self, op, fun, convert=str):
        values = []
        if self._const != self.__neutral_element__:
            values.append(str(self._const))
        values.extend(convert(e) for e in self)
        if len(values) == 0:
            return str(self.__neutral_element__)
        elif len(values) == 1:
            return str(values[0])
        elif len(values) == 2:
            return "{} {} {}".format(values[0], op, values[1])
        else:
            return "{}([{}])".format(fun, ", ".join(values))


class Sum(Accumulation):
    __neutral_element__ = 0

    @classmethod
    def is_commutative(cls):
        return True

    def _has_same_type(self, other):
        return isinstance(other, Sum)

    @classmethod
    def op(self, start, values):
        return sum(values, start)

    def __str__(self):
        return self._as_str(op='+', fun='sum')

    def py_expr(self):
        return self._as_str(op='+', fun='sum', convert=lambda x: x.py_expr())


class Prod(Accumulation):
    __neutral_element__ = 1
    __absorbing_element__ = 0

    @classmethod
    def is_commutative(cls):
        return True

    def _has_same_type(self, other):
        return isinstance(other, Prod)

    @classmethod
    def op(self, start, values):
        return prod(values, start)

    def __str__(self):
        return self._as_str(op='*', fun='prod')

    def py_expr(self):
        return self._as_str(op='*', fun='prod', convert=lambda x: x.py_expr())


class Variable(Expression):
    def __init__(self, name):
        if not isinstance(name, str):
            raise TypeError("bad name {}".format(name))
        self._name = name

    def equals(self, other):
        return type(self) == type(other) and self._name == other.name

    @property
    def name(self):
        return self._name

    def is_externally_updated(self):
        return False

    def free_vars(self, substitution):
        if self._name not in substitution:
            yield self._name

    def evaluate(self, substitution):
        if self._name in substitution:
            return substitution[self._name]
        else:
            raise EvaluationError("undefined variable {}".format(self._name))

    def __str__(self):
        return self._name

    def __repr__(self):
        return "{}({!r})".format(
            type(self).__name__,
            self._name)

    def py_expr(self):
        return self._name


class Parameter(Variable):
    def __init__(self, name, value):
        super().__init__(name)
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v

    def free_vars(self):
        yield from ()

    def vars(self):
        yield from ()

    def evaluate(self, substitution):
        return self._value

    def py_expr(self):
        return str(self._value)


def build_expression(value):
    if isinstance(value, Expression):
        return value
    else:
        return Const(value)


class GlobalVariable(Expression):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def is_externally_updated(self):
        return True

    def is_free(self, substitution=None):
        return True

    def free_vars(self, substitution):
        yield from ()

    def equals(self, other):
        return type(self) == type(other) and self.name == other.name

    def py_expr(self):
        return self.name

    def evaluate(self, substitution):
        raise NotImplementedError()


class FunctionCall(Expression):
    def __init__(self, name, args, kwargs):
        super().__init__()
        self.name = name
        self.args = [build_expression(arg) for arg in args]
        self.kwargs = {key: build_expression(value) for key, value in kwargs.items()}

    @property
    def function(self):
        return self.globals_dict[self.name]

    def is_externally_updated(self):
        return True

    def is_free(self, substitution=None):
        for arg in itertools.chain(self.args, self.kwargs.values()):
            if not arg.is_free(substitution):
                return False
        return True

    def equals(self, other):
        if type(self) != type(other) or self.name != other.name:
            return False
        if len(self.args) != len(other.args):
            return False
        if len(self.kwargs) != len(other.kwargs) or set(self.kwargs) != set(other.kwargs):
            return False
        if not all(a0.equals(a1) for a0, a1 in zip(self.args, other.args)):
            return False
        if not all(v0.equals(v1) for v0, v1 in zip(self.kwargs.values(), other.kwargs.values())):
            return False
        return True

    def free_vars(self, substitution):
        seen = set()
        for arg in itertools.chain(self.args, self.kwargs.values()):
            for var in arg.free_vars(substitution):
                if var not in seen:
                    yield var
                    seen.add(var)

    def py_expr(self):
        alist = [arg.py_expr() for arg in self.args]
        for key, value in self.kwargs.items():
            alist.append('{}={}'.format(key, value.py_expr()))
        return '{}({})'.format(self.name, ', '.join(alist))

    def evaluate(self, substitution):
        raise NotImplementedError()
        # args = [arg.evaluate(substitution) for arg in self.args]
        # kwargs = {key: value.evaluate(substitution) for key, value in self.kwargs.items()}
        # return self.function(*args, **kwargs)


class BoundExpression(ExpressionBase):
    def __init__(self, expression, globals_dict=None):
        if not isinstance(expression, Expression):
            raise TypeError("{} is not an Expression".format(expression))
        if globals_dict is None:
            globals_dict = expression_globals()
        self._globals_dict = globals_dict
        self._expression = expression
        self._compiled_function = None
        self._var_names = set(self._expression.vars())

    @property
    def globals(self):
        if self._globals is None:
            return expression_globals()
        return self._globals

    @globals.setter
    def globals(self, globals_d):
        self._globals = globals_d

    @property
    def expression(self):
        return self._expression

    def is_externally_updated(self):
        return self._expression.is_externally_updated()

    def is_compiled(self):
        return self._compiled_function is not None

    def _compile_function(self):
        ce = self._expression.compile_py_expr()
        # def cfun(subs):
        #     return  eval(ce, self._globals, subs)
        # return cfun
        return lambda subs: eval(ce, self._globals, subs)

    def compile(self):
        self._compiled_function = self._compile_function()

    @property
    def compiled_function(self):
        if self._compiled_function is None:
            self._compiled_function = self.compile_function()
        return self._compiled_function

    def evaluate(self, substitution):
        efun = self._compiled_function
        if efun is None:
            gdict = dict(self._globals_dict)
            gdict.update(substitution)
            return self._expression.evaluate(gdict)
        else:
            return efun(substitution)

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self._expression)

    def __str__(self):
        return str(self._expression)

    def free_vars(self, substitution):
        return self._var_names.difference(substitution)

    def vars(self):
        return self._var_names
