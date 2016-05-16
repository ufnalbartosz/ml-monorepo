#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import warnings
import sys
import numpy
from scipy._lib.six import callable
from numpy import (atleast_1d, eye, mgrid, argmin, zeros, shape, squeeze,
                   vectorize, asarray, sqrt, Inf, asfarray, isinf)
import numpy as np

from scipy.optimize.linesearch import (line_search_wolfe1, line_search_wolfe2,
                         line_search_wolfe2 as line_search,
                         LineSearchWarning)
from scipy._lib._util import getargspec_no_self as _getargspec


#from scipy.optimize.optimize import (_epsilon, rosen, rosen_der, _check_unknown_options,
#                                     wrap_function, vecnorm, _LineSearchError, _line_search_wolfe12,
#                                     OptimizeResult, approx_fprime, _status_message)

from scipy.optimize.optimize import (_epsilon, rosen, approx_fprime, _check_unknown_options,
                                     wrap_function, vecnorm, _LineSearchError, _line_search_wolfe12,
                                     OptimizeResult, _status_message)

def fmindfp(f, x0, fprime=None, args=(), gtol=1e-5, xtol=1e-09, fxtol=1e-09, norm=Inf,
              epsilon=_epsilon, maxiter=None, full_output=False, disp=False,
              retall=True, callback=None):
    """
    Minimize a function using the DFP algorithm.
    Parameters
    ----------
    f : callable f(x,*args)
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    fprime : callable f'(x,*args), optional
        Gradient of f.
    args : tuple, optional
        Extra arguments passed to f and fprime.
    gtol : float, optional
        Gradient norm must be less than gtol before successful termination.
    norm : float, optional
        Order of norm (Inf is max, -Inf is min)
    epsilon : int or ndarray, optional
        If fprime is approximated, use this value for the step size.
    callback : callable, optional
        An optional user-supplied function to call after each
        iteration.  Called as callback(xk), where xk is the
        current parameter vector.
    maxiter : int, optional
        Maximum number of iterations to perform.
    full_output : bool, optional
        If True,return fopt, func_calls, grad_calls, and warnflag
        in addition to xopt.
    disp : bool, optional
        Print convergence message if True.
    retall : bool, optional
        Return a list of results at each iteration if True.
    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e. f(xopt) == fopt.
    fopt : float
        Minimum value.
    gopt : ndarray
        Value of gradient at minimum, f'(xopt), which should be near 0.
    Bopt : ndarray
        Value of 1/f''(xopt), i.e. the inverse hessian matrix.
    func_calls : int
        Number of function_calls made.
    grad_calls : int
        Number of gradient calls made.
    warnflag : integer
        1 : Maximum number of iterations exceeded.
        2 : Gradient and/or function calls not changing.
    allvecs  :  list
        `OptimizeResult` at each iteration.  Only returned if retall is True.
    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'DFP' `method` in particular.
    Notes
    -----
    Optimize the function, f, whose gradient is given by fprime
    using the quasi-Newton method of Davidon- Fletcher-Powell (DFP)
    """
    opts = {'gtol': gtol,
            'xtol': xtol,
            'fxtol': fxtol,
            'norm': norm,
            'eps': epsilon,
            'disp': disp,
            'maxiter': maxiter,
            'return_all': retall}

    res = _minimize(f, x0, args, fprime, callback=callback, **opts)

    if full_output:
        retlist = (res['x'], res['fun'], res['jac'], res['hess_inv'],
                   res['nfev'], res['njev'], res['status'], res['lst'])
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs'], res['lst']
        else:
            return res['x'], res['lst']


def _minimize(fun, x0, args=(), jac=None, callback=None,
                   gtol=1e-5, fxtol=1e-09, xtol=1e-09, norm=Inf,
                   eps=_epsilon, maxiter=None, disp=False,
                   return_all=False, **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    DFP algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.
    """
    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200
    func_calls, f = wrap_function(f, args)

    grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))


    gfk = myfprime(x0)
    k = 0
    N = len(x0)
    I = numpy.eye(N, dtype=int)
    Hk = I
    old_fval = f(x0)
    old_old_fval = None
    xk = x0
    if retall:
        allvecs = [x0]
    sk = [2 * gtol]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    xnorm = np.Inf
    fx = np.Inf
    print_lst = []
    while (gnorm > gtol) and (xnorm > xtol) and (fx > fxtol) and (k < maxiter):
        pk = -numpy.dot(Hk, gfk)
        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, myfprime, xk, pk, gfk,
                                          old_fval, old_old_fval)
        except _LineSearchError:
            # Line search failed to find a better solution.
            print("Line search failed to find a better solution.")
            warnflag = 2
            break

        xkp1 = xk + alpha_k * pk

        fx = np.absolute(old_old_fval - old_fval)
        xnorm = vecnorm(xkp1 - xk)
        if retall:
            allvecs.append(xkp1)

        sk = xkp1 - xk
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)

        yk = gfkp1 - gfk
        gfk = gfkp1
        if callback is not None:
            callback(xk)
        k += 1

        if disp:
            print_ = ('Iter: ' + str(k) + '\n')
            print_ += ('x: ' + str(xk) + '\n')
            print_ += ('f(x): ' + str(f(xk)) + '\n') #zmiana na fx
            print_ +=('gtol: ' + str(gnorm) + '\n')
            print_ +=('xtol: ' + str(xnorm) + '\n')
            print_ +=('fxtol: ' + str(fx) + '\n')
            print_lst.append(print_)

        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break

        if not numpy.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            print("We correctly found +-Inf as optimal value, or something went wrong.")
            warnflag = 2
            break

        try:  # this was handled in numeric, let it remaines for more safety
            rhok = 1.0 / (numpy.dot(yk, sk))
        except ZeroDivisionError:
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        if isinf(rhok):  # this is patch for numpy
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        A1 = I - sk[:, numpy.newaxis] * yk[numpy.newaxis, :] * rhok
        A2 = I - yk[:, numpy.newaxis] * sk[numpy.newaxis, :] * rhok
        Hk = numpy.dot(A1, numpy.dot(Hk, A2)) + (rhok * sk[:, numpy.newaxis] *
                                                 sk[numpy.newaxis, :])

    fval = old_fval
    if np.isnan(fval):
        # This can happen if the first call to f returned NaN;
        # the loop is then never entered.
        print("This can happen if the first call to f returned NaN; the loop is then never entered.")
        warnflag = 2

    if warnflag == 2:
        msg = _status_message['pr_loss']
        if disp:
            print("Warning: " + msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])

    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
        if disp:
            print("Warning: " + msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])

    result = OptimizeResult(fun=fval,lst=print_lst, jac=gfk, hess_inv=Hk, nfev=func_calls[0],
                            njev=grad_calls[0], status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result


if __name__ == "__main__":
    import time

    times = []
    algor = []
    x0 = [0.4, -0.6]

    from function import Function
    from test_functions import *
    fun = Function(ros)

    start = time.time()
    x = fmindfp(fun, x0, maxiter=80, disp=True)

    times.append(time.time() - start)
    algor.append('DFP Quasi-Newton\t')
