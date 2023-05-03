import numpy as np

def gradient_descent_xy(initial_x, initial_y, alpha, max_iters, eta):
    """Gradient descent algorithm for an optimization problem of the form minmax f(x,y)=alpha*xy.
    In this algorithm the updates are independent from each others and the only term present is 
    the gradient term, with learning rate eta"""
    # Define parameters to store x and objective func. values
    xs = [initial_x]
    ys = [initial_y]
    objectives = []
    x = initial_x
    y = initial_y
    for n_iter in range(max_iters):
        # compute the gradients wrt to x and y
        gradf_x = alpha*y
        gradf_y = alpha*x
        gradg_x = -alpha*y
        gradg_y = -alpha*x
        obj = alpha*x*y # for this algorithm, the function is f(x,y)=alpha*xy
        # update x and y
        x = x - eta * gradf_x 
        y = y - eta * gradg_y
        # store x and objective function value
        xs.append(x)
        ys.append(y)
        objectives.append(obj)
        print("Gradient Descent({bi}/{ti}): objective={l}".format(
              bi=n_iter, ti=max_iters - 1, l=obj))
    return objectives, xs, ys


def SG_xy(initial_x, initial_y, alpha, max_iters, eta, gamma):
    """SG algorithm for an optimization problem of the form minmax f(x,y)=alpha*xy.
    In this algorithm it is present the gradient term and the competitive term. 
    eta is the hyper-parameter of the gradient term and gamma is the hyper-parameter of the comeptitive term.
    Please notice that SG with gamma equal to eta leads into LCGD."""
    # Define parameters to store x and objective func. values
    xs = [initial_x]
    ys = [initial_y]
    objectives = []
    x = initial_x
    y = initial_y
    for n_iter in range(max_iters):
        # compute the gradients wrt to x and y
        gradf_x = alpha*y
        gradf_y = alpha*x
        gradg_x = -alpha*y
        gradg_y = -alpha*x
        df_dxdy = alpha
        dg_dxdy = -alpha
        obj = alpha*x*y # for this algorithm, the function is f(x,y)=alpha*xy
        # update x and y
        x = x - eta * gradf_x - gamma*df_dxdy*gradf_y
        y = y - eta * gradg_y - gamma*dg_dxdy*gradg_x
        # store x and objective function value
        xs.append(x)
        ys.append(y)
        objectives.append(obj)
        print("SG({bi}/{ti}): objective={l}".format(
              bi=n_iter, ti=max_iters - 1, l=obj))
    return objectives, xs, ys


def ConOpt_xy(initial_x, initial_y, alpha, max_iters, eta, gamma):
    """ConOpt algorithm for an optimization problem of the form minmax f(x,y)=alpha*xy.
    In this algorithm it is present the gradient term, the competitive term and also
    the so called consensus term. 
    eta is the hyper-parameter of the gradient term and 
    gamma is the hyper-parameter of the consensus and comeptitive term"""
    # Define parameters to store x and objective func. values
    xs = [initial_x]
    ys = [initial_y]
    objectives = []
    x = initial_x
    y = initial_y
    for n_iter in range(max_iters):
        # compute the gradients wrt to x and y
        gradf_x = alpha*y
        gradf_y = alpha*x
        gradg_x = -alpha*y
        gradg_y = -alpha*x
        df_dxdy = alpha
        dg_dxdy = -alpha
        df_dxdx = 0
        dg_dydy = 0
        obj = x*y # for this algorithm, the function is f(x,y)=alpha*xy
        # update x and y
        x = x - eta * gradf_x - gamma*df_dxdy*gradf_y - gamma*df_dxdx*gradf_x
        y = y - eta * gradg_y - gamma*dg_dxdy*gradg_x - gamma*dg_dydy*gradg_y
        # store x and objective function value
        xs.append(x)
        ys.append(y)
        objectives.append(obj)
        print("ConOpt({bi}/{ti}): objective={l}".format(
              bi=n_iter, ti=max_iters - 1, l=obj))
    return objectives, xs, ys


def CGD_xy(initial_x, initial_y, alpha, max_iters, eta):
    """Conpetitive Gradient Descent algorithm for an optimization problem of the form minmax f(x,y)=alpha*xy.
    The in this algorithm are the equilibrium term that is inverted and multiplied by the sum of the gradient
    term with the competitive term. eta is the only hyperparameter involved."""
    # Define parameters to store x and objective func. values
    xs = [initial_x]
    ys = [initial_y]
    objectives = []
    x = initial_x
    y = initial_y
    for n_iter in range(max_iters):
        # compute the gradients wrt to x and y
        gradf_x = alpha*y
        gradf_y = alpha*x
        gradg_x = -alpha*y
        gradg_y = -alpha*x
        df_dxdy = alpha
        dg_dxdy = -alpha
        obj = x*y # for this algorithm, the function is f(x,y)=alpha*xy
        # update x and y
        x = x - eta/(1 - (eta**2)*df_dxdy*dg_dxdy)*(- gradf_x - eta*df_dxdy*gradg_y)
        y = y - eta/(1 - (eta**2)*df_dxdy*dg_dxdy)*(- gradg_y - eta*dg_dxdy*gradf_x)
        # store x and objective function value
        xs.append(x)
        ys.append(y)
        objectives.append(obj)
        print("Competitive Gradient Descent({bi}/{ti}): objective={l}".format(
              bi=n_iter, ti=max_iters - 1, l=obj))
    return objectives, xs, ys