#!/usr/bin/env python
from py_expression_eval import Parser

class Function(object):
    def __init__(self, exp):
        p = Parser()
        self.expression = p.parse(exp)
        self.var = self.expression.variables()
        self.var.sort()

    def __call__(self, *x):
        values = dict((self.var[i], x[i]) for i in range (len(self.var)))
        return self.expression.evaluate(values)

    @property
    def dim(self):
        return len(self.var)



if __name__ == '__main__':
    from test_functions import *

    fun = Function(ros)
    print fun.dim


#from function import Function
#from test_functions import *
#fun = Function(ros)
