#coding:utf-8
"""
fork from https://github.com/SkalskiP/ILearnDeepLearning.py
"""
import numpy as np

NN_ARCHITECTURE = [
        {"input_dim":2,"output_dim":25,"activation":"relu"},
        {"input_dim":25,"output_dim":50,"activation":"relu"},
        {"input_dim":50,"output_dim":50,"activation":"relu"},
        {"input_dim":50,"output_dim":25,"activation":"relu"},
        {"input_dim":25,"output_dim":1,"activation":"sigmoid"},
        ]

def init_layers(nn_architecture, seed=99):
    #random seed initiation
    np.random.seed(seed)
    #number of layers in our neural network
    number_of_layers = len(nn_architecture)
    #parameters storage initiation
    params_values = {}

    #iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        #we number network layers from 1
        layer_idx = idx+1

        #extracting the number of units in layers
        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_dim']

        #权重矩阵初始化
        params_values['W'+str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size)*0.1
        #偏置向量初始化
        params_values['b'+str(layer_idx)] = np.random.randn(layer_output_size, 1)*0.1

    return params_values

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def sigmoid_backward(dA, Z):
    '''sigmoid函数导数'''
    sig = sigmoid(Z)
    return dA*sig*(1-sig)

def relu_backward(dA,Z):
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0
    return dZ

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation='relu'):
    Z_curr = np.dot(W_curr, A_prev)+b_curr

    if activation is 'relu':
        activation_func = relu
    elif activation is 'sigmoid':
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')

    #return of calculated activation A and the intermediate Z matrix
    return activation_func(Z_curr), Z_curr

def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}

    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx =idx+1
        A_prev = A_curr

        activ_function_curr = layer['activation']
        W_curr = params_values["W"+str(layer_idx)]
        b_curr = params_values["b"+str(layer_idx)]

        A_curr, z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

        memory["A"+str(idx)] = A_prev
        memory['Z'+str(layer_idx)] = Z_curr

    #return of prediction vector and a dictionary containing intermediate values
    return A_curr, memory

def get_cost_value(Y_hat, Y):
    #number of examples
    m = Y_hat.shape[1]
    #calculation of the cost according to the binary crossentropy formula
    cost = -1/m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1-Y, np.log(1-Y_hat).T))

    return np.squeeze(cost)

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation='relu'):
    #number of examples
    m = A_prev.shape[1]

    #selection of activation function
    if activation is 'relu':
        backward_activation_func = relu_backward
    elif activation is 'sigmoid':
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('None-supported activation function')

    #calculation of the activation function derivative
    dZ_curr = backward_activation_func(dA_curr, Z_curr)

    #derivative of the matrix W
    dW_curr = np.dot(dZ_curr, A_prev.T)/m
    #derivative of the vector b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True)/m
    #derivative of the matrix A_prev
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}

    #number of examples
    m = Y.shape[1]
    #a hack ensuring the same shape of the prediction vectior and labels vector
    Y = Y.reshape(Y_hat.shape)

    #initiation of gradient descent algorithm
    dA_prev = -(np.divide(Y, Y_hat)-np.divide(1-Y, 1-Y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev+1

        activ_function_curr = layer['activation']

        dA_curr=dA_prev

        A_prev = memory["A"+str(layer_idx_prev)]
        Z_curr = memory["Z"+str(layer_idx_curr)]

        W_curr = params_values["W"+str(layer_idx_curr)]
        b_curr = params_values["b"+str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

        grads_values["dW"+str(layer_idx_curr)] = dW_curr
        grads_values["db"+str(layer_idx_curr)] = db_curr

    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W"+str(layer_idx)] -= learning_rate*grads_values["dW"+str(layer_idx)]

        params_values["b"+str(layer_idx)] -= learning_rate*grads_values["db"+
str(layer_idx)]

    return params_values


def train(X,Y,nn_architecture, epochs, learning_rate, verbose=False, callback=None):
    params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        Y_hat, cashe = full_forward_propagation(X,params_values, nn_architecture)

        #calculating metrics and saving them in history
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)

        #step backward - calculating gradient
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        #updating model state
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)

        if(i%50 == 0):
            if verbose:
                print("Iteration: {:05}-cost:{:.5f}-accuracy: {:.5f}".format(i, cost, accuracy))

            if(callback is not None):
                callback(i, params_values)

    return params_values


