import numpy as np

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.exp(a)
    y = exp_a / sum_exp_a

    return y