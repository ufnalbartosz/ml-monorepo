#!/usr/bin/env python

if __name__ == '__main__':
    #start GUI, get data from GUI;
    from dfp_formula_project.function import Function
    from dfp_formula_project.test_functions import *
    given_function = f4
    fun = Function(given_function)


    #TODO: fmin powinno zwracac wektor osiagnietych
    # wartosci x podczas kazdej iteracji algorytmu

    #TODO: dodac warunki stopu, oraz wypisywac
    # ich wartosci w kazdej iteracji algorytmu

    #TODO: zmienic algorytm bfgs na dfp

    from dfp_formula_project.fmindfp import fmindfp
    x0 = [0.4, -0.6]
    x = fmindfp(fun, x0, maxiter=10000, disp=False)


    # rysowanie wykresow 3d w myjavi, plus warstwice i wektor
    # olicoznych punktow kolejnych osiagnietych wattosci funckji.
    from dfp_formula_project.figure import Figure
    import numpy as np

    vec = np.asanyarray(x[1])
    fig = Figure(fun, vec)

    fig.show()
