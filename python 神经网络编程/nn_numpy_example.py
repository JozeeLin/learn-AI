import nn_numpy

import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

N_SAMPLES = 1000
TEST_SIZE = 0.1

X, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=TEST_SIZE, random_state=42)

params_values = nn_numpy.train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0],1))), nn_numpy.NN_ARCHITECTURE, 10000, 0.01)

#predict
Y_test_hat, _  = nn_numpy.full_forward_propagation(np.transpose(X_test), params_values, nn_numpy.NN_ARCHITECTURE)

acc_test = nn_numpy.get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0],1))))
print("Test set accuracy:{:.2f}-David".format(acc_test))
