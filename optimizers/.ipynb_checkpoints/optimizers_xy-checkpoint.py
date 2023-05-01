import numpy as np

def gradient_descent_xy(initial_x, initial_y, max_iters, gamma):
    """Gradient descent algorithm for an optimization problem of the form minmax f(x,y)=xy.
    In this algorithm the updates are independent from each others and for the update in y
    it is considered that maximizing f(x,y) is equal to minimizing -f(x,y), hence
    the update has the plus sign."""
    # Define parameters to store x and objective func. values
    xs = [initial_x]
    ys = [initial_y]
    objectives = []
    x = initial_x
    y = initial_y
    for n_iter in range(max_iters):
        # compute the gradients wrt to x and y
        gradf_x = y
        gradf_y = x
        gradg_x = -y
        gradg_y = -x
        obj = x*y # for this algorithm, the function is f(x,y)=xy
        # update x and y
        x = x - gamma * gradf_x 
        y = y - gamma * gradg_y
        # store x and objective function value
        xs.append(x)
        ys.append(y)
        objectives.append(obj)
        print("Gradient Descent({bi}/{ti}): objective={l}".format(
              bi=n_iter, ti=max_iters - 1, l=obj))
    return objectives, xs, ys


def LCGD_xy(initial_x, initial_y, max_iters, gamma, eta):
    """LCGD algorithm for an optimization problem of the form minmax f(x,y).
    It is considered that maximizing f(x,y) is equal to minimizing -f(x,y), hence
    the update has the plus sign."""
    # Define parameters to store x and objective func. values
    xs = [initial_x]
    ys = [initial_y]
    objectives = []
    x = initial_x
    y = initial_y
    for n_iter in range(max_iters):
        # compute the gradients wrt to x and y
        gradf_x = y
        gradf_y = x
        gradg_x = -y
        gradg_y = -x
        df_dxdy = 1
        dg_dxdy = -1
        obj = x*y # for this algorithm, the function is f(x,y)=xy
        # update x and y
        x = x - gamma * gradf_x - eta*df_dxdy*gradf_y
        y = y - gamma * gradg_y - eta*dg_dxdy*gradg_x
        # store x and objective function value
        xs.append(x)
        ys.append(y)
        objectives.append(obj)
        print("LCGD({bi}/{ti}): objective={l}".format(
              bi=n_iter, ti=max_iters - 1, l=obj))
    return objectives, xs, ys


def ConOpt_xy(initial_x, initial_y, max_iters, gamma, eta):
    """ConOpt algorithm for an optimization problem of the form minmax f(x,y).
    It is considered that maximizing f(x,y) is equal to minimizing -f(x,y), hence
    the update has the plus sign."""
    # Define parameters to store x and objective func. values
    xs = [initial_x]
    ys = [initial_y]
    objectives = []
    x = initial_x
    y = initial_y
    for n_iter in range(max_iters):
        # compute the gradients wrt to x and y
        gradf_x = y
        gradf_y = x
        gradg_x = -y
        gradg_y = -x
        df_dxdy = 1
        dg_dxdy = -1
        df_dxdx = 0
        dg_dydy = 0
        obj = x*y # for this algorithm, the function is f(x,y)=xy
        # update x and y
        x = x - gamma * gradf_x - eta*df_dxdy*gradf_y - eta*df_dxdx*gradf_x
        y = y - gamma * gradg_y - eta*dg_dxdy*gradg_x - eta*dg_dydy*gradg_y
        # store x and objective function value
        xs.append(x)
        ys.append(y)
        objectives.append(obj)
        print("ConOpt({bi}/{ti}): objective={l}".format(
              bi=n_iter, ti=max_iters - 1, l=obj))
    return objectives, xs, ys


def CGD_xy(initial_x, initial_y, max_iters, gamma, eta):
    """Conpetitive Gradient Descent algorithm for an optimization problem of the form minmax f(x,y).
    It is considered that maximizing f(x,y) is equal to minimizing -f(x,y), hence
    the update has the plus sign."""
    # Define parameters to store x and objective func. values
    xs = [initial_x]
    ys = [initial_y]
    objectives = []
    x = initial_x
    y = initial_y
    for n_iter in range(max_iters):
        # compute the gradients wrt to x and y
        gradf_x = y
        gradf_y = x
        gradg_x = -y
        gradg_y = -x
        df_dxdy = 1
        dg_dxdy = -1
        obj = x*y # for this algorithm, the function is f(x,y)=xy
        # update x and y
        x = x + 1/(1 + (eta*df_dxdy**2))*(- gamma*gradf_x - eta*df_dxdy*gradf_y)
        y = y + 1/(1 + (eta*dg_dxdy**2))*(- gamma*gradg_y - eta*dg_dxdy*gradg_x)
        # store x and objective function value
        xs.append(x)
        ys.append(y)
        objectives.append(obj)
        print("Competitive Gradient Descent({bi}/{ti}): objective={l}".format(
              bi=n_iter, ti=max_iters - 1, l=obj))
    return objectives, xs, ys
