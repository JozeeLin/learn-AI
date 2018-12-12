#coding:utf8
"""
1.mini-batch
从训练数据中随机选出一部分数据，这部分数据成为mini-batch。目标是减小mini-batch的损失函数的值

2.计算梯度

3.更新参数

4.重复步骤1，2，3
"""
import sys,os
sys.path.append(os.pardir)
from collections import OrderedDict
import numpy as np
from activate import sigmoid
from activate import softmax
from loss import cross_entropy_error
from gradient import numerical_gradient
from dataset.mnist import load_mnist
from layer import *

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #生成层
        self.layers = OrderedDict()
        #hidden layer
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        #output layer
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        #softmax layer
        self.lastLayer = SoftmaxWithLoss()



    def predict(self, x):

        # version 1.0 implement
        #W1,W2 = self.params['W1'], self.params['W2']
        #b1,b2 = self.params['b1'], self.params['b2']

        #a1 = np.dot(x,W1)+b1
        #z1 = sigmoid(a1)
        #a2 = np.dot(z1, W2)+b2
        #y = softmax(a2)

        #return y

        # version 2.0 implement
        for l in self.layers.values():
            x = l.forward(x)

        return x

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    def loss(self, x, t):
        y = self.predict(x)

        #version 1.0 implement
        #return cross_entropy_error(y,t)
        #version 2.0 implement
        return self.lastLayer.forward(y,t)

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        self.loss(x,t)

        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for l in layers:
            dout = l.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads



if __name__=='__main__':
    #net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    #x = np.random.rand(100,784)
    #t = np.random.rand(100,10)
    #y = net.predict(x)
    #print("begin compute grads")
    #net.numerical_gradient(x,t)

    #implement mini-batch
    (x_train, t_train),(x_test, t_test) = load_mnist(normalize=True,one_hot_label=True)
    train_loss_list = []

    #超参数
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    count = 0
    for i in range(iters_num):
        #generate mini-batch datasets
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        #compute gradient
        #version 1.0
        #grad = net.numerical_gradient(x_batch, t_batch)
        #version 2.0
        grad = net.gradient(x_batch, t_batch)

        #update weights
        for key in ('W1','b1','W2','b2'):
            net.params[key] -= learning_rate * grad[key]

        #record the loss
        loss = net.loss(x_batch, t_batch)
        #print loss
        train_loss_list.append(loss)
        count += 1

    print train_loss_list
    print count

