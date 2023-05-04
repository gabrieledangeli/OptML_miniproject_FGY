import numpy as np

def GDA_alphaxy(initial_x, initial_y, alpha, max_iters, eta):
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
        print("Gradient Descent Ascent (GDA) ({bi}/{ti}): objective={l}".format(
              bi=n_iter, ti=max_iters - 1, l=obj))
    return objectives, xs, ys

def LCGD_alphaxy(initial_x, initial_y, alpha, max_iters, eta):
    xs = [initial_x]
    ys = [initial_y]
    objectives = [initial_x*initial_y]
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
        x = x - eta*(gradf_x - eta*df_dxdy*gradg_y)
        y = y - eta*(gradg_y - eta*dg_dxdy*gradf_x)
        # store x and objective function value
        xs.append(x)
        ys.append(y)
        objectives.append(obj)
        print("Linearized Competitive Gradient Descent (LCDG) ({bi}/{ti}): objective={l}".format(
              bi=n_iter, ti=max_iters - 1, l=obj))

    return objectives, xs, ys 

def SGA_alphaxy(initial_x, initial_y, alpha, max_iters, eta, gamma):
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
        obj = x*y # for this algorithm, the function is f(x,y)=alpha*xy
        # update x and y
        x = x - eta*(gradf_x - gamma*df_dxdy*gradg_y)
        y = y - eta*(gradg_y - gamma*dg_dxdy*gradf_x)
        # store x and objective function value
        xs.append(x)
        ys.append(y)
        objectives.append(obj)
        print("Symplectic Gradient Adjustment (SGA) ({bi}/{ti}): objective={l}".format(
              bi=n_iter, ti=max_iters - 1, l=obj))
    return objectives, xs, ys


def CGD_alphaxy(initial_x, initial_y, alpha, max_iters, eta):
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
        x = x - eta*(((1 - (eta**2)*df_dxdy*dg_dxdy))**-1)*(gradf_x - eta*df_dxdy*gradg_y)
        y = y - eta*(((1 - (eta**2)*dg_dxdy*df_dxdy))**-1)*(gradg_y - eta*dg_dxdy*gradf_x)
        # store x and objective function value
        xs.append(x)
        ys.append(y)
        objectives.append(obj)
        print("Competitive Gradient Descent (CGD) ({bi}/{ti}): objective={l}".format(
              bi=n_iter, ti=max_iters - 1, l=obj))
    return objectives, xs, ys

def OGDA_alphaxy(initial_x, initial_y, alpha, max_iters, eta):
    """Optimistic Gradient descent algorithm for an optimization problem of the form minmax f(x,y)=alpha*xy.
    """
    # Define parameters to store x and objective func. values
    xs = [initial_x]
    ys = [initial_y]
    objectives = []
    x_prec = 0
    y_prec = 0
    x = initial_x
    y = initial_y
    for n_iter in range(max_iters):
        # compute the gradients wrt to x and y
        gradf_x = alpha*y
        gradf_y = alpha*x
        gradg_x = -alpha*y
        gradg_y = -alpha*x
        gradf_x_prec = alpha*y_prec
        gradg_y_prec = -alpha*x_prec
        df_dxdy = alpha
        dg_dxdy = -alpha
        obj = x*y # for this algorithm, the function is f(x,y)=alpha*xy
        # update x and y
        x_prec = x
        y_prec = y
        x = x - eta*(gradf_x + (gradf_x - gradf_x_prec))
        y = y - eta*(gradg_y + (gradg_y - gradg_y_prec))
        # store x and objective function value
        xs.append(x)
        ys.append(y)
        objectives.append(obj)
        print("Optimistic Gradient Descent Ascent (OGDA) ({bi}/{ti}): objective={l}".format(
              bi=n_iter, ti=max_iters - 1, l=obj))
    return objectives, xs, ys