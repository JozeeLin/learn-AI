import numpy as np

def numerical_gradient(f, x):
    """x can be vector or matrix"""
    h = 1e-4
    grad = np.zeros_like(x) #generate one array shape=x.shape
    count = 0

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    #for idx in range(x.size):
    while not it.finished:
        count += 1
        idx = it.multi_index

        tmp_val = x[idx]
        x[idx] = tmp_val+h
        fxh1 = f(x)

        x[idx] = tmp_val-h
        fxh2 = f(x)

        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] = tmp_val

        it.iternext()

    return grad

def numerical_diff(f, x):
    """numerical differentiation"""
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


if __name__ == '__main__':
    def function_2(x):
        return x[0]**2 + x[1]**2

    init_x = np.array([-3.0,4.0])
    result = gradient_descent(function_2, init_x, 0.1,100)
    print result

