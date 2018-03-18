import sys
import numpy as np


def terminate():
    print('Usage: \n - net.py train POLYNOMIAL DEGREE PATH_TO_CSV \n'
          ' - net.py estimate X')
    sys.exit()

# constants
TRAIN = True
INFER = False
PATH_TO_CSV = '/home/kuba/Development/Quantum/tensorflow/ai-task-samples.csv'
POLYNOMIAL_DEGREE = 10


def compute_gradient(w, x, y):
    y_estimate = x.dot(w).flatten()
    error = (y.flatten() - y_estimate)
    mse = (1.0 / len(x)) * np.sum(np.power(error, 2))
    gradient = -(1.0 / len(x)) * error.dot(x)
    return gradient, mse


def train():
    # load .csv data
    my_data = np.genfromtxt(PATH_TO_CSV, delimiter=',')
    
    data_x_raw = np.zeros((len(my_data), 1))
    data_y = np.zeros((len(my_data), 1))
    
    for i in range(len(my_data)):
        data_x_raw[i] = my_data[i][0]
        data_y[i] = my_data[i][1]

    degree_data = []
    for model_degree in range(1, POLYNOMIAL_DEGREE):
        # turn input X into matrix of Xs to nth power
        data_x = np.power(data_x_raw, range(model_degree))
        # normalize
        data_x /= np.max(data_x, axis=0)
        # generate random initial values for weights
        w = np.random.randn(model_degree)

        # SGD params
        learning_rate = 0.5
        error_tolerance = 1e-3
        epochs = 1
        decay = 0.99
        batch_size = 10
        iterations = 0
        max_epochs = 1000

        while True:
            # shuffle
            degree = np.random.permutation(len(data_x))
            data_x = data_x[degree]
            data_y = data_y[degree]

            used_data = 0
            while used_data < len(data_x):
                tx = data_x[used_data: used_data + batch_size]
                ty = data_y[used_data: used_data + batch_size]
                gradient = compute_gradient(w, tx, ty)[0]
                error = compute_gradient(w, data_x, data_y)[1]
                w -= learning_rate * gradient
                iterations += 1
                used_data += batch_size

            # Keep track of our performance
            if epochs % 100 == 0:
                new_error = compute_gradient(w, data_x, data_y)[1]
                print "Epoch: %d - Error: %.4f" % (epochs, new_error)

                # stopping conditions
                if abs(new_error - error) < error_tolerance:
                    print "Converged, stopping"
                    break

            if epochs > max_epochs:
                print 'Early stopping'
                break

            learning_rate = learning_rate * (decay ** int(epochs / 1000))
            epochs += 1

        print "weights (from 0th to nth) =", w
        degree_data.append([model_degree, new_error, w, iterations])

    min_iter = degree_data[0][3]
    winner_row = degree_data[0]
    for elem in degree_data:
        if elem[3] < min_iter:
            min_iter = elem[3]
            winner_row = elem
    print winner_row

if TRAIN:
    train()

