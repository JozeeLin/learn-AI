import numpy as np

def mean_squared_error(y,t):
    """mean squared error"""
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y,t):
    """cross entropy error"""
    delta = 1e-7 #to avoid log0 overflow
    if y.ndim == 1:
        #return -np.sum(t*np.log(y+delta))
        y = y.reshape(1,y.size)

    batch_size = y.shape[0]
    if type(t) is int:
        return -np.sum(np.log(y[np.arange(batch_size), t]+delta))/batch_size

    if y.ndim == 1:
        t = t.reshape(1,t.size)
    return -np.sum(t*np.log(y+delta))/batch_size #t is a vector

#def cross_entropy_error(y,t):
#    print y.size, t.size
#    if y.ndim == 1:
#        t = t.reshape(1, t.size)
#        y = y.reshape(1, y.size)
#
#    if t.size == y.size:
#        t = t.argmax(axis=1)
#
#    batch_size = y.shape[0]
#    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


if __name__ == '__main__':
    print cross_entropy_error(np.array([0.2,0.8]),[0,1])
