"""
Activation function
"""

import numpy as np

def step_function(x):
    y = x>0
    return y.astype(np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c) #for avoid overflow
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y
