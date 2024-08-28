#### Libraries
# Standard library
import _pickle as cPickle
import gzip

# Third-party libraries
import numpy as np

def load_data():

    f = gzip.open('D:\Charm\Code\MLP\FCN_Cost\MNIST\mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='bytes')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():

    tr_d, va_d, te_d = load_data()
    # cause the network's input is a vector 784X1, so we reshape the image so as to more convenient
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e