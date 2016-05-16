#!/usr/bin/env python
from py_expression_eval import Parser

class Function(object):
    def __init__(self, exp):
        p = Parser()
        self.expression = p.parse(exp)
        self.var = self.expression.variables()
        self.var.sort()

    def __call__(self, x):
        values = dict( (self.var[i], float(x[i])) for i in range(self.dim) )
        return self.expression.evaluate(values)

    @property
    def dim(self):
        return len(self.var)



if __name__ == '__main__':
    from test_functions import *

    fun = Function(ros)
    x = (1, 2)

    from fmindfp import *
    x0 = [0.4, -0.6]
    x = fmindfp(fun, x0, maxiter=10000, disp=True)
    print(x, x0)

