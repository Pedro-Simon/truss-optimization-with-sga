import pysga
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functions import function

problem = 4

def my_fobj(x):
    #STANDARD FUNCTION
    # print(x)
    if problem == 1:
        y = np.zeros((x.shape[0], 1))
        x1 = x[:, 0]
        x2 = x[:, 1]
        y[:, 0] = function(problem, x1, x2)
        return y
    #TEST FUNCTION (1 VARIABLE)
    elif problem ==2:
        y=function(problem, x1)
        return y
    #GOLDSTEIN PRICE FUNCTION (2 VARIABLES)
    elif problem ==3:
        y = np.zeros((x.shape[0], 1))
        x1 = x[:, 0]
        x2 = x[:, 1]
        y[:, 0] = function(problem, x1, x2)
        return y
    #BRANIN FUNCTION (2 VARIABLES)
    elif problem ==4:
        y = np.zeros((x.shape[0],1))
        x1 = x[:, 0]
        x2 = x[:, 1]
        y[:, 0] = function(problem, x1, x2)
        return y
    #SHEKEL (4 VARIABLES)
    elif problem ==5:
        y= np.zeros((x.shape[0],1))
        y[:, 0] = function(problem, x)
        return y


if problem == 1:
    lim_inf = [-10., -10.]
    lim_sup = [10., 10.]
elif problem == 2:
    lim_inf = [0.]
    lim_sup = [6.]
elif problem ==3:
    lim_inf = [-2., -2.]
    lim_sup = [2., 2.]
elif problem ==4:
    lim_inf = [-5., 0]
    lim_sup = [10, 15]
elif problem ==5:
    lim_inf = [0, 0, 0, 0]
    lim_sup = [10, 10, 10, 10]

f = pysga.ObjectiveFunc(inf=lim_inf, sup=lim_sup)
f.evaluate = my_fobj

sga_params = pysga.ParamsSGA()

def Graphs(inf, sup):
    x1 = np.arange(inf[0], sup[0]+0.05, 0.05)
    x2 = np.arange(inf[1], sup[1]+0.05, 0.05)

    X1, X2 = np.meshgrid(x1, x2)
    Y = function(problem, X1, X2)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X1, X2, Y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    # fig = plt.figure()
    plt.figure(figsize=(12, 8))
    plt.contourf(X1, X2, Y)
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')

try:
    Graphs(lim_inf, lim_sup)
except:
    pass

x, y = pysga.run(fobj=f)
print(x,y)

try:
    plt.show()
except:
    pass
