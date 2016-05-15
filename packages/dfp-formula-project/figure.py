#!/usr/bin/env python
import numpy as np
from mayavi import mlab

class Figure():
    def __init__(self, fun, x, x0, x_vec=None, lr=100):
        self.fdim = fun.dim
        if self.fdim <= 2:
            self.fun = fun
            self.x = x
            self.x0 = x0

            self.lin_range = lr

            self.x_range = self.create_range(x[0], x0[0])
            self.y_range = self.create_range(x[1], x0[1])



    def plot_surf(self):
        if self.fdim <= 2:
            X, Y = np.meshgrid(self.x_range, self.y_range)
            Z = np.asanyarray([self.fun((x, y)) for x, y in np.nditer([X, Y])])
            Z = Z.reshape(X.shape)

            mlab.surf(Z, warp_scale='auto')
            mlab.show()


    def plot_contour(self):
        if self.fdim <= 2:
            pass

    def show(self):
        pass

    def add_abs_value(self, val, scale=1):
        return np.absolute(val)/val * scale + val

    def create_range(self, x, x0):
        x0 = self.add_abs_value(x0)
        xd = x - x0
        xF = x + xd
        return np.linspace(start=x0, stop=xF, num=self.lin_range)


if __name__ == '__main__':
    from function import Function
    from test_functions import *

    given_function = ros
    fun = Function(given_function)


    from fmindfp import fmindfp
    x0 = [0.4, -0.6]
    x = fmindfp(fun, x0, maxiter=10000, disp=True)

    x_vec = x[1]
    x = x[0]

    fig = Figure(fun, x, x0)

    fig.plot_surf()
