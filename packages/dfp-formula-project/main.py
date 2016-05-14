#!/usr/bin/env python
from function import Function

#start GUI, get data from GUI;
from test_functions import *

given_function = ros
fun = Function(given_function)


def extend_range(val, scale=1):
    return np.absolute(val)/val * scale + val

def build_function(exp):
    parser = Parser()
    expression = parser.parse(exp)
    var = expression.variables()
    var.sort()
    def function_wrapper(x):
        values = dict((var[i], x[i]) for i in range (len(var)))
        return expression.evaluate(values)
    return function_wrapper


# rysowanie wykresow 3d w myjavi, plus warstwice i wektor
# olicoznych punktow kolejnych osiagnietych wattosci funckji.
import numpy as np
from mayavi import mlab

#tworzenie zakresu wzgledem x0 oraz x znalezionego przez fmindfp():

if __name__ == '__main__':
    from fmindfp import *
    x0 = [0.4, -0.6]
    x = fmindfp(fun, x0, maxiter=10000, disp=True)




    N = 8
    dn = 0.05
    X = np.arange(-N, N, dn)
    Y = np.arange(-N, N, dn)
    X, Y = np.mgrid[-N:N:dn, -N:N:dn]

    z = [fun((x, y)) for x, y in np.nditer([X, Y])]

    Z = np.asanyarray(z)
    Z = Z.reshape(X.shape)


    mlab.surf(Z, warp_scale='auto')
    mlab.show()
