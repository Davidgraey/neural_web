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
#
# def scg(w:NDArray, error_f:Callable,
#         fargs: list=[],
#         n_iterations: int=100,
#         error_gradient_f: Callable=None,
#         eval_f: Callable=lambda x: x,
#         save_wtrace: bool = False,
#         verbose: bool =False):
#     """
#     org Written by Prof Charles Anderson - adapted here
#     Parameters
#     ----------
#     w :
#     error_f :
#     fargs :
#     n_iterations :
#     error_gradient_f :
#     eval_f :
#     save_wtrace :
#     verbose :
#
#     Returns
#     -------
#
#     """
#     start_time = time.time()
#     start_time_last_verbose = start_time
#
#     w = w.copy()
#     wtrace = [w.copy()] if save_wtrace else None
#
#     sigma0 = 1.0e-6
#     error_old = error_f(w, *fargs)
#     error_now = error_old
#     gradnew = error_gradient_f(w, *fargs)
#     ftrace = [eval_f(error_old)]
#
#     gradold = copy.deepcopy(gradnew)
#     d = -gradnew      # Initial search direction.
#     success = True    # Force calculation of directional derivs.
#     nsuccess = 0      # nsuccess counts number of successes.
#     beta = 1.0e-6     # Initial scale parameter. Lambda in Moeller.
#     betamin = 1.0e-15 # Lower bound on scale.
#     betamax = 1.0e20  # Upper bound on scale.
#     nvars = len(w)
#     iteration = 1     # count of number of iterations
#
#
#     thisIteration = 1
#
#     while thisIteration <= n_iterations:
#
#         if success:
#             mu = d.T @ gradnew
#             if mu >= 0:
#                 d = -gradnew
#                 mu = d.T @ gradnew
#             kappa = d.T @ d
#
#             if np.isnan(kappa):
#                 print('kappa', kappa)
#
#             if kappa < EPSIOLON:
#                 return {'w': w,
#                         'f': error_now,
#                         'n_iterations': iteration,
#                         'wtrace': np.array(wtrace)[:iteration + 1, :] if save_wtrace else None,
#                         'ftrace': np.array(ftrace)[:iteration + 1],
#                         'reason': 'limit on machine precision',
#                         'time': time.time() - start_time}
#
#             sigma = sigma0 / np.sqrt(kappa)
#
#             w_smallstep = w + sigma * d
#             error_f(w_smallstep, *fargs)
#             g_smallstep = error_gradient_f(w_smallstep, *fargs)
#
#             theta = d.T @ (g_smallstep - gradnew) / sigma
#             if np.isnan(theta):
#                 print(f'theta {theta} sigma {sigma} d[0] {d[0]} g_smallstep[0] {g_smallstep[0]} gradnew[0] {gradnew[0]}')
#
#         # Increase effective curvature and evaluate step size alpha.
#         delta = theta + beta * kappa
#         if np.isnan(delta):
#             print(f'delta is NaN theta {theta} beta {beta} kappa {kappa}')
#         elif delta <= 0:
#             delta = beta * kappa
#             beta = beta - theta / kappa
#
#         if delta == 0:
#             success = False
#             error_now = error_old
#         else:
#             alpha = -mu / delta
#
#             # Calculate the comparison ratio Delta
#             wnew = w + alpha * d
#             error_new = error_f(wnew, *fargs)
#             Delta = 2 * (error_new - error_old) / (alpha * mu)
#
#             if not np.isnan(Delta) and Delta  >= 0:
#                 success = True
#                 nsuccess += 1
#                 w[:] = wnew
#                 error_now = error_new
#             else:
#                 success = False
#                 error_now = error_old
#
#         iterations_per_print = np.ceil(n_iterations/10)
#         if verbose and thisIteration % np.max(1, iterations_per_print) == 0:
#             seconds = time.time() - start_time_last_verbose
#             print(f'SCG: Iteration {iteration:d} ObjectiveF={eval_f(error_now):.5f} Scale={beta:.3e} Seconds={seconds:.3f}')
#             start_time_last_verbose = time.time()
#         if save_wtrace:
#             wtrace.append(w.copy())
#         ftrace.append(eval_f(error_now))
#
#
#         if success:
#
#             error_old = error_new
#             gradold[:] = gradnew
#             gradnew[:] = error_gradient_f(w, *fargs)
#
#             # If the gradient is zero then we are done.
#             gg = gradnew.T @ gradnew
#             if gg == 0:
#                 return {'w': w,
#                         'f': error_now,
#                         'n_iterations': iteration,
#                         'wtrace': np.array(wtrace)[:iteration + 1, :] if save_wtrace else None,
#                         'ftrace': np.array(ftrace)[:iteration + 1],
#                         'reason': 'zero gradient',
#                         'time': time.time() - start_time}
#
#         if np.isnan(Delta) or Delta < 0.25:
#             beta = min(4.0 * beta, betamax)
#         elif Delta > 0.75:
#             beta = max(0.5 * beta, betamin)
#
#         # Update search direction using Polak-Ribiere formula, or re-start
#         # in direction of negative gradient after nparams steps.
#         if nsuccess == nvars:
#             d[:] = -gradnew
#             nsuccess = 0
#         elif success:
#             gamma = (gradold - gradnew).T @ (gradnew / mu)
#             d[:] = gamma * d - gradnew
#
#         thisIteration += 1
#         iteration += 1
#
#         # If we get here, then we haven't terminated in the given number of iterations.
#
#     return {'w': w,
#             'f': error_now,
#             'n_iterations': iteration,
#             'wtrace': np.array(wtrace)[:iteration + 1,:] if save_wtrace else None,
#             'ftrace': np.array(ftrace)[:iteration + 1],
#             'reason': 'did not converge',
#             'time': time.time() - start_time}

LAMBDA_MAX = 1e24
LAMBDA_MIN =  1e-24
def scaled_conjugate_gradient(model: BasalModel,
                              x_data: NDArray,
                              y_data: NDArray,
                              iterations: int) -> NDArray:
    """
    This implementation is based on Charles Anderson's (Colorado State
    University) SCG method

    Use with caution  -- weights are adjusted as we take steps down the
    gradient; if you don't account for those object updates, you might have a
    nasty surprise

    Requres a model object with specific methods implemented, hence the
    BasalModel class --

    Parameters
    ----------
    model : BasalModel -- relies on the model object to be of BasalModel
        class- we need methods for
                model.get_weights(),
                model.calculate_gradients()
                model.calculate_loss()
        and for the model.weights to be accessible / settable

    x_data : input data
    y_data : labels
    iterations : maximum number of SCG steps to take down this gradient

    Returns
    -------
    final optimized weights resultant from scaled cojugate descent.
        These values can be directly used, or combined with a learning rate
        to slow convergence
    """
    sigma_zero = 1e-6
    lamb = 1e-6
    lamb_ = 0

    vector = model.get_weights().reshape(-1, 1)
    grad_new, _ = model.calculate_gradients(x_data, y_data)
    grad_new = -1 * grad_new.reshape(-1, 1)
    r_new = grad_new.copy()
    success = True

    for _i in range(iterations):
        r = r_new.copy()
        grad = grad_new.copy()
        mu = grad.T @ grad

        if success:
            success = False
            sigma = sigma_zero / np.sqrt(mu)

            grad_old, _ = model.calculate_gradients(x_data, y_data)
            grad_old = grad_old.reshape(-1, 1)

            # update our model's weights -- (take a step down the gradient)
            model.weights = (vector + (sigma * grad)).reshape(model._weight_shape)
            grad_step, _ = model.calculate_gradients(x_data, y_data)

            step = (grad_old - grad_step.reshape(-1, 1)) / sigma
            delta = grad.T @ step

        # increase the curvature - "reach deeper"
        zeta = lamb - lamb_
        step += zeta * grad
        delta += zeta * mu

        if delta <= 0:
            step += (lamb - 2 * delta / mu) * grad
            lamb_ = 2 * (lamb - delta / mu)
            delta -= lamb * mu
            delta *= -1
            lamb = lamb_

        phi = grad.T @ r
        alpha = phi / delta
        vector_new = vector + alpha * grad
        loss_old = model.calculate_loss(x_data, y_data)

        # update our model's weights -- (take a step down the gradient)
        model.weights = vector_new.copy().reshape(model._weight_shape)
        loss_new = model.calculate_loss(x_data, y_data)

        comparison = 2 * delta * (loss_old - loss_new) / (phi ** 2)

        if comparison >= 0:
            # break condition?
            vector = vector_new.copy()
            loss_old = loss_new

            # update our model's weights -- (take a step down the gradient)
            model.weights = vector_new.copy().reshape(model._weight_shape)
            r_new, _ = model.calculate_gradients(x_data, y_data)
            r_new = -1 * r_new.reshape(-1, 1)
            success = True
            lamb_ = 0

            if _i % model.weight_shape[0] == 0:
                grad_new = r_new
            else:
                beta = ((r_new.T @ r_new) - (r_new.T @ r)) / phi
                # update our model's weights -- (take a step down the gradient)
                grad_new = r_new + beta * grad

            if comparison > 0.75:
                lamb = max(0.5 * lamb, LAMBDA_MIN)
        else:
            lamb_ = lamb

        if comparison < 0.25:
            lamb = min(4 * lamb, LAMBDA_MAX)

    return vector_new.reshape(model._weight_shape)
