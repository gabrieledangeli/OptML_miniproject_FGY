import math

def Gradx_f(alpha,x,y):
    """
    Gradient of f(x,y)=alpha*(-x^2 + y^2) wrt x
    """
    return -2*alpha*x

def Grady_f(alpha,x,y):
    """
    Gradient of f(x,y)=alpha*(-x^2 + y^2) wrt y
    """
    return 2*alpha*y

def Dxy_f(alpha,x,y):
    """
    partial derivative in dx and dy of f(x,y)=alpha*(-x^2 + y^2)
    """
    return 0

def Dyx_f(alpha,x,y):
    """
    partial derivative in dy and dx of f(x,y)=alpha*(-x^2 + y^2)
    """
    return 0

def Dxx_f(alpha,x,y):
    """
    second partial derivative in dx of f(x,y)=alpha*(-x^2 + y^2)
    """    
    return -2*alpha

def Dyy_f(alpha,x,y):
    """
    second partial derivative in dy of f(x,y)=alpha*(-x^2 + y^2)
    """    
    return 2*alpha

def Gradx_g(alpha,x,y):
    """
    Gradient of g(x,y) = -alpha*(-x^2 + y^2) wrt x
    """
    return 2*alpha*x

def Grady_g(alpha,x,y):
    """
    Gradient of g(x,y) = -alpha*(-x^2 + y^2) wrt y
    """
    return -2*alpha*y

def Dxy_g(alpha,x,y):
    """
    partial derivative of g(x,y) = -alpha*(-x^2 + y^2) in dx and then in dy
    """
    return 0

def Dyx_g(alpha,x,y):
    """
    partial derivative of g(x,y) = -alpha*(-x^2 + y^2) in dy and then in dx
    """
    return 0

def Dxx_g(alpha,x,y):
    """
    second partial derivative of g(x,y) = -alpha*(-x^2 + y^2) in dx 
    """
    return 2*alpha

def Dyy_g(alpha,x,y):
    """
    second partial derivative of g(x,y) = -alpha*(-x^2 + y^2) in dy 
    """
    return -2*alpha

def GDA_step(alpha,x,y,eta):
    """
    GDA step with parameters alpha(in the loss functions) and eta(learning rate).
    """
    #update of x
    deltaX=-Gradx_f(alpha,x,y)
    x+=eta*deltaX
    #update of y
    deltaY=-Grady_g(alpha,x,y)
    y+=eta*deltaY
    return x,y

#try eta inside or outside
def LCGD_step(alpha,x,y,eta):
    """
    LCGD step with parameter alpha(in the loss functions).
    eta is the learning rate.
    """
    #update of x
    deltaX=-Gradx_f(alpha,x,y)-eta*Dxy_f(alpha,x,y)*Grady_f(alpha,x,y)
    x+=eta*deltaX
    #update of y
    deltaY=-Grady_g(alpha,x,y)-eta*Dyx_g(alpha,x,y)*Gradx_g(alpha,x,y)
    y+=eta*deltaY
    return x,y

def SGA_step(alpha,x,y,eta,gamma):
    """
    SGA step with parameters alpha(in the loss functions) 
    eta is the hyper-parameter of the gradient term and gamma is the hyper-parameter of the comeptitive term.
    """
    #update of x
    deltaX=-Gradx_f(alpha,x,y)-gamma*Dxy_f(alpha,x,y)*Grady_f(alpha,x,y)
    x+=eta*deltaX
    #update of y
    deltaY=-Grady_g(alpha,x,y)-gamma*Dyx_g(alpha,x,y)*Gradx_g(alpha,x,y)
    y+=eta*deltaY
    return x,y

def ConOpt_step(alpha,x,y,eta,gamma):
    """
    ConOpt step with parameters alpha(in the loss functions).
    eta is the hyper-parameter of the gradient term and gamma is the hyper-parameter of the comeptitive term.
    """
    #update of x
    deltaX=-Gradx_f(alpha,x,y)-gamma*Dxy_f(alpha,x,y)*Grady_f(alpha,x,y)-gamma*Dxx_f(alpha,x,y)*Gradx_f(alpha,x,y)
    x+=eta*deltaX
    #update of y
    deltaY=-Grady_g(alpha,x,y)-gamma*Dyx_g(alpha,x,y)*Gradx_g(alpha,x,y)-gamma*Dyy_g(alpha,x,y)*Grady_g(alpha,x,y)
    y+=eta*deltaY
    return x,y

def OGDA_step(alpha,x,y,eta):
    """
    OGDA step with parameter alpha(in the loss functions).
    eta is the learning rate
    """
    #update of x
    deltaX=-Gradx_f(alpha,x,y)-eta*Dxy_f(alpha,x,y)*Grady_f(alpha,x,y)+eta*Dxx_f(alpha,x,y)*Gradx_f(alpha,x,y)
    x+=eta*deltaX
    #update of y
    deltaY=-Grady_g(alpha,x,y)-eta*Dyx_g(alpha,x,y)*Gradx_g(alpha,x,y)+eta*Dyy_g(alpha,x,y)*Grady_g(alpha,x,y)
    y+=eta*deltaY
    return x,y

def CGD_step(alpha,x,y,eta):
    """
    CGD step with parameter alpha(parameter of the loss functions).
    """
    #update of x
    deltaX=(1/(1+(eta**2)*Dxy_f(alpha,x,y)*Dyx_f(alpha,x,y)))*(-Gradx_f(alpha,x,y)-eta*Dxy_f(alpha,x,y)*Grady_f(alpha,x,y))
    x+=eta*deltaX
    #update of y
    deltaY=(1/(1+(eta**2)*Dyx_g(alpha,x,y)*Dxy_g(alpha,x,y)))*(-Grady_g(alpha,x,y)-eta*Dyx_g(alpha,x,y)*Gradx_g(alpha,x,y))
    y+=eta*deltaY
    return x,y

def L1norm(x,y):
    """
    This function calculate the L1 norm of the vector with origin in (0,0) pointing in (x,y) and returns the log10 of it
    """
    norm=abs(x)+abs(y)
    log10norm=math.log10(norm)
    return log10norm

def run(alpha,eta,gamma,x,y,iter=100,optimizer='GDA'):
    """
    The function executes a specified optimization algorithm for a given number of iterations. 
    It takes several parameters needed for the different optimizers (alpha, eta, gamma). "optimizer" specifies the optimization algorithm
    to use and x and y define the starting point of the algorithm.
    The function maintains a "record" list to store the L1 norm of "x" and "y" at regular intervals (every 25 iterations). The L1 norm is
    calculated using the "L1norm" function.
    """
    record=[]
    for i in range(iter):
        log10norm=L1norm(x,y)
        if(i%25==0):
            record.append(log10norm)
        if(optimizer=='GDA'):
            x,y=GDA_step(alpha,x,y,eta)
        elif(optimizer=='LCGD'):
            x,y=LCGD_step(alpha,x,y,eta)
        elif(optimizer=='SGA'):
            x,y=SGA_step(alpha,x,y,eta,gamma)
        elif(optimizer=='ConOpt'):
            x,y=ConOpt_step(alpha,x,y,eta,gamma)
        elif(optimizer=='OGDA'):
            x,y=OGDA_step(alpha,x,y,eta)
        elif(optimizer=='CGD'):
            x,y=CGD_step(alpha,x,y,eta)
    record.append(L1norm(x,y))
    return record






    

    