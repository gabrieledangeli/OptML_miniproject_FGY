import math

def Gradx_f(alpha,x,y):
    return 2*alpha*x

def Grady_f(alpha,x,y):
    return -2*alpha*y

def Dxy_f(alpha,x,y):
    return 0

def Dyx_f(alpha,x,y):
    return 0

def Dxx_f(alpha,x,y):
    return 2*alpha

def Dyy_f(alpha,x,y):
    return -2*alpha

def Gradx_g(alpha,x,y):
    return -2*alpha*x

def Grady_g(alpha,x,y):
    return 2*alpha*y

def Dxy_g(alpha,x,y):
    return 0

def Dyx_g(alpha,x,y):
    return 0

def Dxx_g(alpha,x,y):
    return -2*alpha

def Dyy_g(alpha,x,y):
    return 2*alpha

def GDA_step(alpha,x,y,eta):
    deltaX=-Gradx_f(alpha,x,y)
    x+=eta*deltaX
    deltaY=-Grady_g(alpha,x,y)
    y+=eta*deltaY
    return x,y

#try eta inside or outside
def LCGD_step(alpha,x,y,eta):
    deltaX=-Gradx_f(alpha,x,y)-eta*Dxy_f(alpha,x,y)*Grady_f(alpha,x,y)
    x+=eta*deltaX
    deltaY=-Grady_g(alpha,x,y)-eta*Dyx_g(alpha,x,y)*Gradx_g(alpha,x,y)
    y+=eta*deltaY
    return x,y

def SGA_step(alpha,x,y,eta,gamma):
    deltaX=-Gradx_f(alpha,x,y)-gamma*Dxy_f(alpha,x,y)*Grady_f(alpha,x,y)
    x+=eta*deltaX
    deltaY=-Grady_g(alpha,x,y)-gamma*Dyx_g(alpha,x,y)*Gradx_g(alpha,x,y)
    y+=eta*deltaY
    return x,y

def ConOpt_step(alpha,x,y,eta,gamma):
    deltaX=-Gradx_f(alpha,x,y)-gamma*Dxy_f(alpha,x,y)*Grady_f(alpha,x,y)-gamma*Dxx_f(alpha,x,y)*Gradx_f(alpha,x,y)
    x+=eta*deltaX
    deltaY=-Grady_g(alpha,x,y)-gamma*Dyx_g(alpha,x,y)*Gradx_g(alpha,x,y)-gamma*Dyy_g(alpha,x,y)*Grady_g(alpha,x,y)
    y+=eta*deltaY
    return x,y

def OGDA_step(alpha,x,y,eta):
    deltaX=-Gradx_f(alpha,x,y)-eta*Dxy_f(alpha,x,y)*Grady_f(alpha,x,y)+eta*Dxx_f(alpha,x,y)*Gradx_f(alpha,x,y)
    x+=eta*deltaX
    deltaY=-Grady_g(alpha,x,y)-eta*Dyx_g(alpha,x,y)*Gradx_g(alpha,x,y)+eta*Dyy_g(alpha,x,y)*Grady_g(alpha,x,y)
    y+=eta*deltaY
    return x,y

def CGD_step(alpha,x,y,eta):
    deltaX=(1/(1+(eta**2)*Dxy_f(alpha,x,y)*Dyx_f(alpha,x,y)))*(-Gradx_f(alpha,x,y)-eta*Dxy_f(alpha,x,y)*Grady_f(alpha,x,y))
    x+=eta*deltaX
    deltaY=(1/(1+(eta**2)*Dyx_g(alpha,x,y)*Dxy_g(alpha,x,y)))*(-Grady_g(alpha,x,y)-eta*Dyx_g(alpha,x,y)*Gradx_g(alpha,x,y))
    y+=eta*deltaY
    return x,y

def L1norm(x,y):
    norm=abs(x)+abs(y)
    log10norm=math.log10(norm)
    return log10norm

def run(alpha,eta,gamma,x,y,iter=100,optimizer='GDA'):
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

"""Xinit=0.5
Yinit=0.5
alpha=1
eta=0.2
gamma=1

x=Xinit
y=Yinit
#optimizers=['GDA','LCGD','SGA','ConOpt','OGDA','CGD']
optimizers=['ConOpt']

for optimizer in optimizers:
    record=[]
    record=run(x,y,optimizer=optimizer)
    print(optimizer,':',record)

#step=[i*25 for i in range(5)]
#import matplotlib.pyplot as plt

#plt.plot(step,record)
#plt.show()"""




    

    