#! /usr/bin/python 
import sys
import numpy as np

# set seed for deterministic output
np.random.seed(5)

TRAIN = False
INFER = False
PATH_TO_CSV = './ai-task-samples.csv'
SAVE_PATH = './weights.npy'
POLYNOMIAL_DEGREE = 2
INPUT = 0


def terminate():
    ''' Show prompt and exit '''
    print('Usage: \n - ./polynomial train POLYNOMIAL_'
          'DEGREE PATH_TO_CSV \n - ./polynomial estimate X')
    sys.exit()


if len(sys.argv) > 2:
    if sys.argv[1] == 'train':
        try:
            POLYNOMIAL_DEGREE += int(sys.argv[2])
            PATH_TO_CSV = str(sys.argv[3])
            TRAIN = True
        except Exception:
            terminate()
    elif sys.argv[1] == 'estimate':
        try:
            INPUT = float(sys.argv[2])
            INFER = True
        except Exception:
            terminate()
    else:
        terminate()
else:
    terminate()


def compute_gradient(w, x, y):
    ''' Compute gradient and MSE '''
    y_estimate = x.dot(w).flatten()
    error = (y.flatten() - y_estimate)
    mse = (1.0 / len(x)) * np.sum(np.power(error, 2))
    gradient = -(1.0 / len(x)) * error.dot(x)
    return gradient, mse


def infer():
    ''' Run forward pass through the network '''
    weights = np.load(SAVE_PATH)[::-1]
    output = 0
    for i, coef in enumerate(weights):
        output += coef * INPUT ** i
    return output


def train_nth_degree(data_x_raw, data_y, model_degree):
    ''' Find optimal coefficients and error for giben degree'''
    # turn input X into matrix of Xs to nth power
    data_x = np.power(data_x_raw, range(model_degree))
    # normalize inputs
    scaling = np.max(data_x, axis=0)
    data_x /= scaling
    # generate random initial values for weights
    w = np.random.randn(model_degree)

    # SGD parameters
    learning_rate = 0.5
    error_tolerance = 1e-6
    epochs = 1
    decay = 0.99
    batch_size = 100
    iterations = 0
    max_epochs = 1000

    while True:
        # shuffle training data
        degree = np.random.permutation(len(data_x))
        data_x = data_x[degree]
        data_y = data_y[degree]

        used_data = 0
        while used_data < len(data_x):  # one iteration per batch
            tx = data_x[used_data: used_data + batch_size]
            ty = data_y[used_data: used_data + batch_size]
            # compute gradient
            gradient = compute_gradient(w, tx, ty)[0]
            error = compute_gradient(w, data_x, data_y)[1]
            # correct weights
            w -= learning_rate * gradient
            iterations += 1
            used_data += batch_size

        if epochs % 100 == 0:
            new_error = compute_gradient(w, data_x, data_y)[1]
            # check if the model converges
            if abs(new_error - error) < error_tolerance:
                break

        if epochs > max_epochs:
            # early stopping
            new_error = compute_gradient(w, data_x, data_y)[1]
            break

        # decay the LR
        learning_rate = learning_rate * (decay ** int(epochs / 1000))
        epochs += 1

    return model_degree, new_error, w, iterations, scaling


def train():
    # load .csv data
    my_data = np.genfromtxt(PATH_TO_CSV, delimiter=',')
    
    data_x_raw = np.zeros((len(my_data), 1))
    data_y = np.zeros((len(my_data), 1))
    
    for i in range(len(my_data)):
        data_x_raw[i] = my_data[i][0]
        data_y[i] = my_data[i][1]

    degree_data = []
    # one iteration per model degree
    for model_degree in range(1, POLYNOMIAL_DEGREE):
        single_output = train_nth_degree(data_x_raw, data_y, model_degree)
        degree_data.append(single_output)

    min_error = degree_data[0][1]
    winner_row = degree_data[0]
    # find model with least error
    for elem in degree_data:
        if elem[1] < min_error:
            min_error = elem[1]
            winner_row = elem
    
    weights = winner_row[2]
    scaling = winner_row[4]
    # scale the polynomial coefficients
    scaled_weights = [b / a for a, b in zip(scaling, weights)]
    output = scaled_weights[::-1]
    np.save(SAVE_PATH, output)
    return output


if TRAIN:
    print train()
if INFER:
    print infer()
