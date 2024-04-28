'''
Scaled Conjugate Gradient - "Second order gradient" descent - no
hyperparams or optimizer grid search need apply.
'''
import numpy as np
EPSILON = 1e-15
from typing import Callable
from numpy.typign improt NDArray
import time
import copy

def scg(w:NDArray, error_f:Callable,
        fargs: list=[],
        n_iterations: int=100,
        error_gradient_f: Callable=None,
        eval_f: Callable=lambda x: x,
        save_wtrace: bool = False,
        verbose: bool =False):
    """
    org Written by Prof Charles Anderson - adapted here
    Parameters
    ----------
    w :
    error_f :
    fargs :
    n_iterations :
    error_gradient_f :
    eval_f :
    save_wtrace :
    verbose :

    Returns
    -------

    """
    start_time = time.time()
    start_time_last_verbose = start_time

    w = w.copy()
    wtrace = [w.copy()] if save_wtrace else None

    sigma0 = 1.0e-6
    error_old = error_f(w, *fargs)
    error_now = error_old
    gradnew = error_gradient_f(w, *fargs)
    ftrace = [eval_f(error_old)]

    gradold = copy.deepcopy(gradnew)
    d = -gradnew      # Initial search direction.
    success = True    # Force calculation of directional derivs.
    nsuccess = 0      # nsuccess counts number of successes.
    beta = 1.0e-6     # Initial scale parameter. Lambda in Moeller.
    betamin = 1.0e-15 # Lower bound on scale.
    betamax = 1.0e20  # Upper bound on scale.
    nvars = len(w)
    iteration = 1     # count of number of iterations


    thisIteration = 1

    while thisIteration <= n_iterations:

        if success:
            mu = d.T @ gradnew
            if mu >= 0:
                d = -gradnew
                mu = d.T @ gradnew
            kappa = d.T @ d

            if np.isnan(kappa):
                print('kappa', kappa)

            if kappa < EPSIOLON:
                return {'w': w,
                        'f': error_now,
                        'n_iterations': iteration,
                        'wtrace': np.array(wtrace)[:iteration + 1, :] if save_wtrace else None,
                        'ftrace': np.array(ftrace)[:iteration + 1],
                        'reason': 'limit on machine precision',
                        'time': time.time() - start_time}

            sigma = sigma0 / np.sqrt(kappa)

            w_smallstep = w + sigma * d
            error_f(w_smallstep, *fargs)
            g_smallstep = error_gradient_f(w_smallstep, *fargs)

            theta = d.T @ (g_smallstep - gradnew) / sigma
            if np.isnan(theta):
                print(f'theta {theta} sigma {sigma} d[0] {d[0]} g_smallstep[0] {g_smallstep[0]} gradnew[0] {gradnew[0]}')

        # Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        if np.isnan(delta):
            print(f'delta is NaN theta {theta} beta {beta} kappa {kappa}')
        elif delta <= 0:
            delta = beta * kappa
            beta = beta - theta / kappa

        if delta == 0:
            success = False
            error_now = error_old
        else:
            alpha = -mu / delta

            # Calculate the comparison ratio Delta
            wnew = w + alpha * d
            error_new = error_f(wnew, *fargs)
            Delta = 2 * (error_new - error_old) / (alpha * mu)

            if not np.isnan(Delta) and Delta  >= 0:
                success = True
                nsuccess += 1
                w[:] = wnew
                error_now = error_new
            else:
                success = False
                error_now = error_old

        iterations_per_print = np.ceil(n_iterations/10)
        if verbose and thisIteration % np.max(1, iterations_per_print) == 0:
            seconds = time.time() - start_time_last_verbose
            print(f'SCG: Iteration {iteration:d} ObjectiveF={eval_f(error_now):.5f} Scale={beta:.3e} Seconds={seconds:.3f}')
            start_time_last_verbose = time.time()
        if save_wtrace:
            wtrace.append(w.copy())
        ftrace.append(eval_f(error_now))


        if success:

            error_old = error_new
            gradold[:] = gradnew
            gradnew[:] = error_gradient_f(w, *fargs)

            # If the gradient is zero then we are done.
            gg = gradnew.T @ gradnew
            if gg == 0:
                return {'w': w,
                        'f': error_now,
                        'n_iterations': iteration,
                        'wtrace': np.array(wtrace)[:iteration + 1, :] if save_wtrace else None,
                        'ftrace': np.array(ftrace)[:iteration + 1],
                        'reason': 'zero gradient',
                        'time': time.time() - start_time}

        if np.isnan(Delta) or Delta < 0.25:
            beta = min(4.0 * beta, betamax)
        elif Delta > 0.75:
            beta = max(0.5 * beta, betamin)

        # Update search direction using Polak-Ribiere formula, or re-start
        # in direction of negative gradient after nparams steps.
        if nsuccess == nvars:
            d[:] = -gradnew
            nsuccess = 0
        elif success:
            gamma = (gradold - gradnew).T @ (gradnew / mu)
            d[:] = gamma * d - gradnew

        thisIteration += 1
        iteration += 1

        # If we get here, then we haven't terminated in the given number of iterations.

    return {'w': w,
            'f': error_now,
            'n_iterations': iteration,
            'wtrace': np.array(wtrace)[:iteration + 1,:] if save_wtrace else None,
            'ftrace': np.array(ftrace)[:iteration + 1],
            'reason': 'did not converge',
            'time': time.time() - start_time}
