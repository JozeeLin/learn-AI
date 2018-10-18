#!/usr/bin/env python
# coding=utf-8
import numpy as np
import numpy
import scipy.special

class neuralNetwork(object):
    '''
    neural network class definition
    神经网络改进
    1. 使用更多的数据集
    2. 学习率调整(使用更小的学习率,但是训练时间会更长)
    3. 多次迭代训练设置epoch>1
    4. 改变网络形状
    5. 数据集增强(使用现有的数据,通过对现有数据进行旋转角度生成新的样本)
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
        you can call it 'predict' method
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


#class neuralNetwork:
#
#    # initialise the neural network
#    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
#        # set number of nodes in each input, hidden, output layer
#        self.inodes = inputnodes
#        self.hnodes = hiddennodes
#        self.onodes = outputnodes
#
#        # link weight matrices, wih and who
#        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
#        # w11 w21
#        # w12 w22 etc
#        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
#        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
#
#        # learning rate
#        self.lr = learningrate
#
#        # activation function is the sigmoid function
#        self.activation_function = lambda x: scipy.special.expit(x)
#
#        pass
#
#
#    # train the neural network
#    def train(self, inputs_list, targets_list):
#        # convert inputs list to 2d array
#        inputs = numpy.array(inputs_list, ndmin=2).T
#        targets = numpy.array(targets_list, ndmin=2).T
#
#        # calculate signals into hidden layer
#        hidden_inputs = numpy.dot(self.wih, inputs)
#        # calculate the signals emerging from hidden layer
#        hidden_outputs = self.activation_function(hidden_inputs)
#
#        # calculate signals into final output layer
#        final_inputs = numpy.dot(self.who, hidden_outputs)
#        # calculate the signals emerging from final output layer
#        final_outputs = self.activation_function(final_inputs)
#
#        # output layer error is the (target - actual)
#        output_errors = targets - final_outputs
#        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
#        hidden_errors = numpy.dot(self.who.T, output_errors)
#
#        # update the weights for the links between the hidden and output layers
#        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
#
#        # update the weights for the links between the input and hidden layers
#        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
#
#        pass
#
#
#    # query the neural network
#    def query(self, inputs_list):
#        # convert inputs list to 2d array
#        inputs = numpy.array(inputs_list, ndmin=2).T
#
#        # calculate signals into hidden layer
#        hidden_inputs = numpy.dot(self.wih, inputs)
#        # calculate the signals emerging from hidden layer
#        hidden_outputs = self.activation_function(hidden_inputs)
#
#        # calculate signals into final output layer
#        final_inputs = numpy.dot(self.who, hidden_outputs)
#        # calculate the signals emerging from final output layer
#        final_outputs = self.activation_function(final_inputs)
#
#        return final_outputs



if __name__ == '__main__':
#    inputnodes = 3
#    hiddennodes = 3
#    outputnodes = 3
#
#    learningrate = 0.5
#
#    n = neuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)
#

    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10

    learningrate = 0.1

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learningrate)

    #load the mnist traning data CSV file into a list
    with open("data/mnist_train_100.csv") as fp_r:
        training_data_list = fp_r.readlines()


    # 训练网络
    epochs = 5

    #train the neural network
    for _ in range(epochs):
        # go through all records in the training data set
        for record in training_data_list:
            #split the record by the ',' commas
            all_values = record.split(',')
            #scale and shift the inputs
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99)+0.01
            #create the target output values (all 0.01, except the desired label which is 0.99
            targets = np.zeros(output_nodes)+0.01
            targets[int(all_values[0])] = 0.99 # 目标向量为[0.01,0.01,...,0.99,...,0.01]

            n.train(inputs, targets)


    # 测试网络
    with open('data/mnist_test_10.csv') as fp_r:
        test_data_list = fp_r.readlines()


    scorecard = []
    #go through all the records in the test data set
    for record in test_data_list:
        #split the record by the ',' commas
        all_values = record.split(',')
        #correct answer is first value
        correct_label = int(all_values[0])
        print(correct_label, "correct label")

        #scale and shift the inputs
        inputs = (np.asfarray(all_values[1:])/255.0+0.99)+0.01
        #query the network
        outputs = n.query(inputs)
        #the index of the highest value corresponds to the label
        label = np.argmax(outputs)
        print(label, "network's answer")
        #append correct or incorrect to list
        if (label==correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)

    print(scorecard)
    print(round(float(sum(scorecard))/len(scorecard),3))
