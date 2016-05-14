#!/usr/bin/env python

if __name__ == '__main__':
    #start GUI, get data from GUI;
    from function import Function
    from test_functions import *
    given_function = goldstein
    fun = Function(given_function)


    #TODO: fmin powinno zwracac wektor osiagnietych
    # wartosci x podczas kazdej iteracji algorytmu

    #TODO: dodac warunki stopu, oraz wypisywac
    # ich wartosci w kazdej iteracji algorytmu

    #TODO: zmienic algorytm bfgs na dfp

    from fmindfp import fmindfp
    x0 = [0.4, -0.6]
    x = fmindfp(fun, x0, maxiter=10000, disp=True)


    # rysowanie wykresow 3d w myjavi, plus warstwice i wektor
    # olicoznych punktow kolejnych osiagnietych wattosci funckji.
    from figure import Figure

    x_vec = None
    fig = Figure(fun, x, x0, x_vec)

    fig.plot_surf()
