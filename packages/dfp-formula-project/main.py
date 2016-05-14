#!/usr/bin/env python


#start GUI, get data from GUI;





# rysowanie wykresow 3d w myjavi, plus warstwice i wektor
# olicoznych punktow kolejnych osiagnietych wattosci funckji.

#tworzenie zakresu wzgledem x0 oraz x znalezionego przez fmindfp():

if __name__ == '__main__':
    from function import Function
    from test_functions import *
    given_function = goldstein
    fun = Function(given_function)


    from fmindfp import fmindfp
    x0 = [0.4, -0.6]
    x = fmindfp(fun, x0, maxiter=10000, disp=True)

    from figure import Figure

    fig = Figure(fun, x, x0)

    fig.plot_surf()
