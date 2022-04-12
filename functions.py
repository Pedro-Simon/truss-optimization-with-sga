import numpy as np

def function(problem, x1, x2=None, x3=None, x4=None):
    #STANDARD FUNCTION
    if problem == 1:
        s1 = np.zeros(x1.shape[0])
        s2 = np.zeros(x1.shape[0])
        for i in range(5):
            n = i + 1
            s1 = s1 + n * np.cos((n + 1) * x1 + n)
            s2 = s2 + n * np.cos((n + 1) * x2 + n)
        y = s1 * s2
        return y
    #TEST FUNCTION (1 VARIABLE)
    if problem ==2:
        y=0.1*x1*np.cos(2*x1)
        return y
    #GOLDSTEIN PRICE FUNCTION (2 VARIABLES)
    if problem ==3:
        y = (1+((x1+x2+1)**2)*(19-14*x1+3*(x1**2)-14*x2+6*x1*x2+(3*x2**2)))*(30+((2*x1-3*x2)**2)*(18-32*x1+(12*x1**2)+48*x2-36*x1*x2+(27*x2**2)))
        return y
    #BRANIN FUNCTION (2 VARIABLES)
    if problem ==4:
        a = 1
        b = 5.1/(4*np.pi**2)
        c = 5/np.pi
        r = 6
        s = 10
        t = 1/(8*np.pi)
        y = a*(x2 - b*(x1**2)+c*x1-r)**2+s*(1-t)*np.cos(x1)+s
        return y
    elif problem == 5:
        var = x1
        beta = 0.1*np.transpose(np.array([1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 3.0, 7.0, 5.0, 5.0]))
        C = np.array([[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]])
        outer_sum = 0

        for i in range(10):
            inner_sum = 0
            for j in range(4):
              inner_sum += (var[:, j]-C[j][i])**2
            outer_sum += 1/((inner_sum)+beta[i])
        return -outer_sum
