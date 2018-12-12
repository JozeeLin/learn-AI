from sys import stdout
import copy
import numpy as np
np.random.seed(0)

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):
    return output*(1-output)

#training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1) #[256x8]
for i in range(largest_number):
    int2binary[i] = binary[i]

#input variables
alpha = 0.1 #learning rate
input_dim = 2
hidden_dim = 16
output_dim = 1

#initialize neural network weights
synapse_0 = 2*np.random.random((input_dim, hidden_dim)) -1
synapse_1 = 2*np.random.random((hidden_dim, output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim, hidden_dim))-1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

#dataset 's size is 10000
for j in range(10000):
    a_int = np.random.randint(largest_number/2) #int version
    a = int2binary[a_int] #binary encoding

    b_int = np.random.randint(largest_number/2)
    b = int2binary[b_int]

    #true answer[label]
    c_int = a_int+b_int
    c = int2binary[c_int]

    #predict answer
    d = np.zeros_like(c)

    overallError = 0
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))

    #moving along the positions in the binary encoding
    #sequence length = binary_dim
    for position in range(binary_dim):
        #generate input and output
        X = np.array([[a[binary_dim-position-1], b[binary_dim-position-1]]]) #X[t]= concat[sequence-a[t];sequence-b[t]]
        y = np.array([[c[binary_dim-position-1]]]).T

        #hidden layer(input ~+ prev_hidden) prev_hidden=layer_1_values[-1]
        layer_1 = sigmoid(np.dot(X, synapse_0)+np.dot(layer_1_values[-1], synapse_h))
        #store hidden state so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))

        #output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        layer_2_error = y - layer_2

        #back propagation
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))

        overallError += np.abs(layer_2_error[0])

        #[predict]decode estimate so we can print it out
        d[binary_dim-position-1] = np.round(layer_2[0][0])


    #[!!!!!!!!!!!!!!]this is very importen(this error is come from next timestep,because current hidden state as input to next timestep)
    future_layer_1_delta = np.zeros(hidden_dim)

    # it is a stack structure
    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])

        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]

        #error at output layer
        layer_2_delta = layer_2_deltas[-position-1]

        #[!!!!!!!!!!!]error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T)+layer_2_delta.dot(synapse_1.T))* \
                            sigmoid_output_to_derivative(layer_1)

        #let's update all our weights
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    #reset
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    #print out progress
    if(j%1000==0):
        #print("\r Error:"+str(overallError)+" Pred:"+str(d)),
        print("Error:"+str(overallError)+" Pred:"+str(d)," True:"+str(c))
        #stdout.flush()
