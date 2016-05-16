#!/usr/bin/env python
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt


class Figure():
    def __init__(self, fun, x, x0, x_vec, lr=500):
        self.fdim = x_vec.shape[1]
        if self.fdim == 2:
            self.fun = fun

            #dodac warunek min max i podawc je do funckji crt_range
            const = 5
            maxx = np.max(x_vec[:, 0]) + const
            minx = np.min(x_vec[:, 0]) - const

            maxy = np.max(x_vec[:, 1]) + const
            miny = np.min(x_vec[:, 1]) - const

            self.x_range = np.linspace(start=minx, stop=maxx, num=lr)
            self.y_range = np.linspace(start=miny, stop=maxy, num=lr)

            self.calculate_function_values()

            self.vec = x_vec

        else:
            print('Dimension not equal 2')


    def plot_surf(self):
        if self.fdim == 2:
            mlab.surf(self.x_range, self.y_range, self.values, warp_scale='auto')
            mlab.show()
        else:
            print('Dimension not equal 2')


    def plot_contour(self):
        if self.fdim == 2:
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
        else:
            print('Dimension not equal 2')


    def calculate_function_values(self):
        X, Y = np.meshgrid(self.x_range, self.y_range)
        Z = np.asanyarray([self.fun((x, y)) for x, y in np.nditer([X, Y])])
        self.values = Z.reshape(X.shape)

    def show(self):
        pass

if __name__ == '__main__':
    from function import Function
    from test_functions import *

    given_function = f10
    fun = Function(given_function)


    from fmindfp import fmindfp
    x0 = [0.1, 0.1]
    x = fmindfp(fun, x0, maxiter=10000, disp=True)

    #normalizacja otrzymanych danych wektora x
    x_vec = x[1]
    x_vec = np.asanyarray(x_vec)
    for line in x[2]:
        print(line)



    x = x[0]
    x = x.tolist()

    fig = Figure(fun, x, x0, x_vec)

    #fig.plot_surf()
    fig.plot_contour()
