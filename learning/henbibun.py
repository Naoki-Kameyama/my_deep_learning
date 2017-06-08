import numpy as np
import matplotlib.pylab as plt

#def function_2(x):
#    return x[0]**2 + x[1]**2

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0 

def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2*h)

x0 = 3
x1 = 4
x0_diff = numerical_diff(function_tmp1, 3.0)
x1_diff = numerical_diff(function_tmp2, 4.0)
print(x0_diff)
print(x1_diff)

