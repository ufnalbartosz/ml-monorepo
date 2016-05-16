#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
#matplotlib3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Figure():
    def __init__(self, fun, x_vec, lr=500):
        self.fdim = x_vec.shape[1]
        if self.fdim == 2:
            self.fun = fun

            #dodac warunek min max i podawc je do funckji crt_range
            const = 10
            maxx = np.max(x_vec[:, 0]) + const
            minx = np.min(x_vec[:, 0]) - maxx

            maxy = np.max(x_vec[:, 1]) + const
            miny = np.min(x_vec[:, 1]) - maxy

            self.x_range = np.linspace(start=minx, stop=maxx, num=lr)
            self.y_range = np.linspace(start=miny, stop=maxy, num=lr)

            self.calculate_function_values()

            self.vec = x_vec

        else:
            print('Dimension not equal 2')


    def plot_surf_mayavi(self):
        if self.fdim == 2:
            mlab.surf(self.x_range, self.y_range, self.Zval, warp_scale='auto')
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

            CS = self.ax.contour(self.x_range, self.y_range, self.Zval)
            self.ax.clabel(CS, inline=1, fontsize=10)
            print('---'*4)
            x_ = []
            y_ = []
            for i in self.vec:
                x_.append(i[0])
                y_.append(i[1])

            self.ax.plot(x_, y_)
#            plt.show()
        else:
            print('Dimension not equal 2')


    def calculate_function_values(self):
        X, Y = np.meshgrid(self.x_range, self.y_range)
        Z = np.asanyarray([self.fun((x, y)) for x, y in np.nditer([X, Y])])
        self.Zval = Z.reshape(X.shape)
        self.Xmesh = X
        self.Ymesh = Y


    def plot_surf(self):
        self.ax.plot_surface(self.Xmesh, self.Ymesh, self.Zval,
                             rstride=1, cstride=1, cmap=cm.coolwarm,
                             linewidth=0, antialiased=False)



    def show(self):
        fig = plt.figure(figsize=plt.figaspect(0.5)) #twice as wide as it's tall
        self.ax = fig.add_subplot(1, 2, 1, projection='3d')

        self.plot_surf()

        self.ax = fig.add_subplot(1, 2, 2)
        self.plot_contour()


        plt.show()

if __name__ == '__main__':
    from function import Function
    from test_functions import *

    given_function = f10
    fun = Function(given_function)


    from fmindfp import fmindfp
    x0 = [0.1, 0.1]
    x = fmindfp(fun, x0, maxiter=10000, disp=True)

    #normalizacja otrzymanych danych wektora x
    vec = x[1]
    vec = np.asanyarray(vec)


    fig = Figure(fun, vec)

    fig.show()

    #fig.plot_surf()
    #fig.plot_contour()
