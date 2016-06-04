#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
#matplotlib3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Figure():
    def __init__(self, fun, vec, lr=100):
        self.fdim = vec.ndim
        if self.fdim == 2:
            self.fun = fun

            #dodac warunek min max i podawc je do funckji crt_range
            scale = 1.3;
            x = vec[:, 0]
            y = vec[:, 1]

            x_distance = scale * np.abs(x.max() - x.min())
            y_distance = scale * np.abs(y.max() - y.min())

            const = 0.3
            x_max = x.max() + const * x_distance
            x_min = x.min() - const * x_distance

            y_max = y.max() + const * y_distance
            y_min = y.min() - const * y_distance

            if x_max < x_min:
                x_min, x_max = x_max, x_min

            if y_max < y_min:
                y_min, y_max = y_max, y_min

            self.x_range = np.linspace(start=x_min, stop=x_max, num=lr)
            self.y_range = np.linspace(start=y_min, stop=y_max, num=lr)

            self.calculate_function_values()

            self.vec = vec

        else:
            print('Dimension not equal 2')


    def plot_surf_mayavi(self):
        if self.fdim == 2:
            mlab.surf(self.x_range, self.y_range, self.z_val, warp_scale='auto')
            mlab.show()
        else:
            print('Dimension not equal 2')


    def plot_contour(self):
        if self.fdim == 2:

            p = [self.x_range.min(), self.x_range.max(),
                               self.y_range.max(), self.y_range.min()]
            p2 = [self.x_range.min(), self.x_range.max(),
                               self.y_range.min(), self.y_range.max()]
            plt.imshow(self.z_val, vmin=self.z_val.min(), vmax=self.z_val.max(),
                       extent=p)

            CS = plt.contour(self.x_range, self.y_range, self.z_val, 15,  colors='k')
            plt.clabel(CS, inline=1, fontsize=10)
            x_ = []
            y_ = []
            for i in self.vec:
                x_.append(i[0])
                y_.append(i[1])

            plt.plot(x_, y_, color='k')
            plt.ylabel("x2")
            plt.xlabel("x1")

            if self.y_range.max() > 0:
                plt.ylim(self.y_range.min(), self.y_range.max())
            else:
                plt.ylim(self.y_range.max(), self.y_range.min())

            if self.x_range.max() > 0:
                plt.ylim(self.x_range.min(), self.x_range.max())
            else:
                plt.ylim(self.x_range.max(), self.x_range.min())

            plt.show()
        else:
            print('Dimension not equal 2')


    def calculate_function_values(self):
        X, Y = np.meshgrid(self.x_range, self.y_range)
        Z = np.asanyarray([self.fun((x, y)) for x, y in np.nditer([X, Y])])
        self.z_val = Z.reshape(X.shape)
        self.x_mesh = X
        self.y_mesh = Y


    def plot_surf(self):
        self.ax.plot_surface(self.x_mesh, self.y_mesh, self.z_val,
                             rstride=1, cstride=1, cmap=cm.coolwarm,
                             linewidth=0, antialiased=False)



    def show(self):
#        fig = plt.figure(figsize=plt.figaspect(0.5)) #twice as wide as it's tall
#        self.ax = fig.add_subplot(1, 2, 1, projection='3d')

#        self.plot_surf()

#        self.ax = fig.add_subplot(1, 2, 2)
        self.plot_contour()
        plt.show()

if __name__ == '__main__':
    from function import Function
    from test_functions import *

    given_function = f10
    fun = Function(given_function)


    from fmindfp import fmindfp
    x0 = [-0.1, 0.1]
    x = fmindfp(fun, x0, maxiter=10000, disp=True)

    #normalizacja otrzymanych danych wektora x
    vec = x[1]
    vec = np.asanyarray(vec)


    fig = Figure(fun, vec)

    #fig.plot_surf_mayavi()
    fig.plot_contour()
    #fig.show()
