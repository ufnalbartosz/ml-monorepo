#!/usr/bin/env python
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt


class Figure():
    def __init__(self, fun, x, x0, x_vec=None, lr=500):
        self.fdim = fun.dim
        if self.fdim <= 2:
            self.fun = fun
            self.x = x
            self.x0 = x0

            self.lin_range = lr

            self.x_range = self.create_range(x[0], x0[0])
            self.y_range = self.create_range(x[1], x0[1])

            self.eval()

            self.vec = x_vec



    def plot_surf(self):
        if self.fdim <= 2:
            mlab.surf(self.x_range, self.y_range, self.values, warp_scale='auto')
            mlab.show()


    def plot_contour(self):
        if self.fdim <= 2:
            #mlab.contour_surf(self.val_vec)
            #fig = mlab.gcf()
            #mlab.plot3d(self.vec[1], self.vec[2], self.vec[0], figure=fig)
            #mlab.contour_surf(self.x_range, self.y_range, self.values, contours=20, figure=fig)
            #mlab.show()

            plt.contour(self.x_range, self.y_range, self.values)
            print('---'*4)
            x_ = []
            y_ = []
            for i in self.vec:
                x_.append(i[0])
                y_.append(i[1])

            plt.plot(x_, y_)
            plt.show()


    def eval(self):
        X, Y = np.meshgrid(self.x_range, self.y_range)
        Z = np.asanyarray([self.fun((x, y)) for x, y in np.nditer([X, Y])])
        self.values = Z.reshape(X.shape)

    def show(self):
        pass

    def add_abs_value(self, val, scale=1):
        return np.absolute(val)/val * scale + val

    def create_range(self, x, x0):
        x0 = self.add_abs_value(x0, 2)
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

    #normalizacja otrzymanych danych wektora x
    x_vec = np.asarray(x[1:])
    x_ = x_vec.tolist()
    x_ = x_[0]

    x = x[0]


    fig = Figure(fun, x, x0, x_)

    #fig.plot_surf()
    fig.plot_contour()
