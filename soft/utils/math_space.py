import random

from soft.utils.operator import Operator
from soft.utils.syntax_space import SyntaxSpace
import numpy as np


def and_op(a, b, o):
    return np.fmin(np.array(a), np.array(b)) * 2 * o


def and3_op(a, b, c, o):
    return np.fmin(np.array(a), np.fmin(np.array(b), np.array(c))) * 2 * o


def or_op(a, b, o):
    return np.fmax(np.array(a), np.array(b)) * 2 * o


def or3_op(a, b, c, o):
    return np.fmax(np.array(a), np.fmax(np.array(b), np.array(c))) * 2 * o


def neg(a, o):
    r = -o * np.array(a)
    return r


def add(a, b, o):
    r = o * (np.array(a) + np.array(b)) + (1 - o) * (np.array(a) - np.array(b))
    return r


def mul(a, b, o):
    r = np.array(a) * np.power(np.abs(np.array(b)), o) * np.sign(b)
    return r


def sigmoid(z):
    x = 1. / (1 + np.exp(-np.array(z)))
    return x


def greater_than(a, b, o):
    result = np.greater(a, b)
    return 2 * o * result


def lower_than(a, b, o):
    result = np.greater(b, a)
    return 2 * o * result


def create_param_list(params = 0):
    param_list = []
    for i in range(params):
        param_list.append(f'P{i+1}_')
    return param_list


def create_trivial_math_space(params = 0):
    return SyntaxSpace(
        operators = math_operators(params)
    )


def create_nonlinear_math_space(params = 0):
    return SyntaxSpace(
        operators = math_extended_operators(params)
    )


def math_extended_operators(params = 0):
    operators = math_operators(params)
    operators.append(
        Operator(
            o_name = 'AND',
            o_type = 'logic',
            func = lambda a, b, o: and_op(a, b, o),
            input_types = ['bool', 'bool', 'operator'],
            return_type = 'bool'
        )
    )
    operators.append(
        Operator(
            o_name = 'LOGIC_MEAN',
            o_type = 'logic',
            func = lambda a, b, o: o * (np.array(a) + np.array(b)) / 2,
            input_types = ['bool', 'bool', 'operator'],
            return_type = 'bool'
        )
    )
    operators.append(
        Operator(
            o_name = 'PARAM',
            o_type = 'real',
            func = lambda: random.choice(create_param_list(params)),
            input_types = [],
            return_type = 'param',
            mutation_rule = lambda x: random.choice(create_param_list(params))
        )
    )
    operators.append(
        Operator(
            o_name = 'SIG',
            o_type = 'func',
            func = lambda a, o: sigmoid(a) * o,
            input_types = ['real', 'operator'],
            return_type = 'real'
        )
    )
    operators.append(
        Operator(
            o_name = 'MAX',
            o_type = 'func',
            func = lambda a, b, o: np.fmax(np.array(a), np.array(b)) * o,
            input_types = ['real', 'real', 'operator'],
            return_type = 'real'
        )
    )
    operators.append(
        Operator(
            o_name = 'MIN',
            o_type = 'func',
            func = lambda a, b, o: np.fmin(np.array(a), np.array(b)) * o,
            input_types = ['real', 'real', 'operator'],
            return_type = 'real'
        )
    )
    operators.append(
        Operator(
            o_name = 'MEAN',
            o_type = 'func',
            func = lambda a, b, o: o * (np.array(a) + np.array(b)) / 2,
            input_types = ['real', 'real', 'operator'],
            return_type = 'real'
        )
    )
    operators.append(
        Operator(
            o_name = 'LIN3',
            o_type = 'func',
            func = lambda a, b, c, o1, o2, o3: o1 * np.array(a) + o2 * np.array(b) + o3 * np.array(c),
            input_types = ['param', 'param', 'param', 'operator', 'operator', 'operator'],
            return_type = 'real'
        )
    )
    operators.append(
        Operator(
            o_name = 'LIN4',
            o_type = 'func',
            func = lambda a, b, c, d, o1, o2, o3, o4: o1 * np.array(a) + o2 * np.array(b) + o3 * np.array(
                c) + o4 * np.array(d),
            input_types = ['param', 'param', 'param', 'param', 'operator', 'operator', 'operator', 'operator'],
            return_type = 'real'
        )
    )

    return operators


def math_operators(params = 0):
    return [
        Operator(
            o_name = 'OR',
            o_type = 'logic',
            func = lambda a, b, o: or_op(a, b, o),
            input_types = ['bool', 'bool', 'operator'],
            return_type = 'bool'
        ),
        Operator(
            o_name = 'OR3',
            o_type = 'logic',
            func = lambda a, b, c, o: or3_op(a, b, c, o),
            input_types = ['bool', 'bool', 'bool', 'operator'],
            return_type = 'bool'
        ),
        Operator(
            o_name = 'NOT',
            o_type = 'logic',
            func = lambda a, o: np.fmin(np.fmax(1 - o * np.array(a), 0), 1),
            input_types = ['bool', 'operator'],
            return_type = 'bool'
        ),
        Operator(
            o_name = 'LT',
            o_type = 'compare',
            func = lambda a, b, o: lower_than(a, b, o),
            input_types = ['real', 'real', 'operator'],
            return_type = 'bool'
        ),
        Operator(
            o_name = 'GT',
            o_type = 'compare',
            func = lambda a, b, o: greater_than(a, b, o),
            input_types = ['real', 'real', 'operator'],
            return_type = 'bool'
        ),
        Operator(
            o_name = 'ADD',
            o_type = 'func',
            func = lambda a, b, o: add(a, b, o),
            input_types = ['real', 'real', 'operator'],
            return_type = 'real'
        ),
        Operator(
            o_name = 'NEG',
            o_type = 'func',
            func = lambda x, o: neg(x, o),
            input_types = ['real', 'operator'],
            return_type = 'real'
        ),
        Operator(
            o_name = 'MUL',
            o_type = 'func',
            func = lambda a, b, o: mul(a, b, o),
            input_types = ['real', 'real', 'operator'],
            return_type = 'real'
        ),
        Operator(
            o_name = 'SYMBOLIC',
            o_type = 'real',
            func = lambda: random.choice(create_param_list(params)),
            input_types = [],
            return_type = 'real',
            mutation_rule = lambda x: random.choice(create_param_list(params))
        ),
        Operator(
            o_name = 'CONSTANT',
            o_type = 'real',
            func = lambda: round(random.random(), 2),
            input_types = [],
            return_type = 'real',
            mutation_rule = lambda i: round(min(max(i + random.gauss(0, 2), 0), 10), 2)
        ),
        Operator(
            o_name = 'OPERATOR',
            o_type = 'operator',
            func = lambda: round(random.random(), 2),
            input_types = [],
            return_type = 'operator',
            mutation_rule = lambda i: round(min(max(i + random.gauss(0, .5), 0), 10), 2)
        ),
    ]


def create_hard_math_space(params = 0):
    return SyntaxSpace(
        operators = [
            Operator(
                o_name = 'AND',
                o_type = 'logic',
                func = lambda a, b: np.logical_and(np.array(a), np.array(b)),
                input_types = ['bool', 'bool'],
                return_type = 'bool'
            ),
            Operator(
                o_name = 'OR',
                o_type = 'logic',
                func = lambda a, b: np.logical_or(np.array(a), np.array(b)),
                input_types = ['bool', 'bool'],
                return_type = 'bool'
            ),
            Operator(
                o_name = 'NOT',
                o_type = 'logic',
                func = lambda a: np.logical_not(np.array(a)),
                input_types = ['bool'],
                return_type = 'bool'
            ),

            Operator(
                o_name = 'LT',
                o_type = 'compare',
                func = lambda a, b: np.greater(np.array(b), np.array(a)),
                input_types = ['real', 'real'],
                return_type = 'bool'
            ),
            Operator(
                o_name = 'GT',
                o_type = 'compare',
                func = lambda a, b: np.greater(np.array(a), np.array(b)),
                input_types = ['real', 'real'],
                return_type = 'bool'
            ),

            Operator(
                o_name = 'ADD',
                o_type = 'func',
                func = lambda a, b: np.array(a) + np.array(b),
                input_types = ['real', 'real'],
                return_type = 'real'
            ),
            Operator(
                o_name = 'NEG',
                o_type = 'func',
                func = lambda x: -np.array(x),
                input_types = ['real'],
                return_type = 'real'
            ),
            Operator(
                o_name = 'MUL',
                o_type = 'func',
                func = lambda a, b: np.array(a) * np.array(b),
                input_types = ['real', 'real'],
                return_type = 'real'
            ),
            Operator(
                o_name = 'SYMBOLIC',
                o_type = 'real',
                func = lambda: random.choice(create_param_list(params)),
                input_types = [],
                return_type = 'real',
                mutation_rule = lambda x: random.choice(create_param_list(params))
            ),
            Operator(
                o_name = 'CONSTANT',
                o_type = 'real',
                func = lambda: round(random.random(), 2),
                input_types = [],
                return_type = 'real',
                mutation_rule = lambda i: round(min(max(i + random.gauss(0, 2), 0), 10), 2)
            ),
        ],
    )
