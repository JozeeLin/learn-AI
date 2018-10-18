#!/usr/bin/env python
# coding=utf-8
import numpy as np
import scipy.special

class neuralNetwork(object):
    '''
    neural network class definition
    '''

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        '''
        initialise the neural network
        设置输入层节点,隐藏层节点和输出层节点的数量
        '''
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # 初始化权重矩阵,第一个矩阵为hiddennodes x inputnodes
        # 第二个矩阵为outputnodes x hiddennodes
        # 减去0.5是为了使得权重区间设定为[-0.5, 0.5]
        self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)

#        # 使用正态概率分布采样权重
#        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
#        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 定义激活函数
        self.activation_function = lambda x: scipy.special.expit(x)



    def train(self, inputs_list, targets_list):
        '''
        train the neural network
        '''

        #convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into hidden layer
        final_inputs = np.dot(self.who, hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weight for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors*final_outputs*(1.0-final_outputs)), np.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr*np.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)), np.transpose(inputs))

    def query(self, inputs_list):
        '''
        query the neural network
        接受神经网络的输入,返回网络的输出
        '''
        # 把输入数组变成二维矩阵,并获取它的转置矩阵
        inputs = np.array(inputs_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


if __name__ == '__main__':
    inputnodes = 3
    hiddennodes = 3
    outputnodes = 3

    learningrate = 0.5

    n = neuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)

