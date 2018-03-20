#! /home/kuba/Development/Quantum/quantum/bin/python
import sys
import numpy as np


#np.random.seed(5)

# constants
TRAIN = False
INFER = False
PATH_TO_CSV = './ai-task-samples.csv'
SAVE_PATH = './weights.npy'
SAVE_PATH_SCALE = './scaling.npy'
POLYNOMIAL_DEGREE = 2
INPUT = 0


def terminate():
    print('Usage: \n - net.py train POLYNOMIAL DEGREE PATH_TO_CSV \n'
          ' - net.py estimate X')
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
    y_estimate = x.dot(w).flatten()
    error = (y.flatten() - y_estimate)
    mse = (1.0 / len(x)) * np.sum(np.power(error, 2))
    gradient = -(1.0 / len(x)) * error.dot(x)
    return gradient, mse


def infer():
    weights = np.load(SAVE_PATH)
    output = 0
    for i, coef in enumerate(weights[::-1]):
        output += coef * INPUT ** i
    return 200*  output


# def save_scaling(data_x, data_y):
#     scaling_coef = np.array([np.max(np.abs(data_x), axis=0),
#                              np.max(np.abs(data_y), axis=0)])
#     np.save(SAVE_PATH_SCALE, scaling_coef)
#     return scaling_coef[0], scaling_coef[1]

def train_nth_degree(data_x_raw, data_y, model_degree):
    # turn input X into matrix of Xs to nth power
    data_x = np.power(data_x_raw, range(model_degree))
    print np.max(data_x, axis=0)
    data_x /= np.max(data_x, axis=0)
    # generate random initial values for weights
    w = np.random.randn(model_degree)

    # SGD params
    learning_rate = 0.5
    error_tolerance = 1e-6
    epochs = 1
    decay = 0.99
    batch_size = 100
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

        if epochs % 100 == 0:
            new_error = compute_gradient(w, data_x, data_y)[1]
            print "Epoch: %d - Error: %.4f" % (epochs, new_error)
            # stopping conditions
            if abs(new_error - error) < error_tolerance:
                break

        if epochs > max_epochs:
            new_error = compute_gradient(w, data_x, data_y)[1]
            break

        learning_rate = learning_rate * (decay ** int(epochs / 1000))
        epochs += 1
    return model_degree, new_error, w, iterations


def train():
    # load .csv data
    my_data = np.genfromtxt(PATH_TO_CSV, delimiter=',')
    
    data_x_raw = np.zeros((len(my_data), 1))
    data_y = np.zeros((len(my_data), 1))
    
    for i in range(len(my_data)):
        data_x_raw[i] = my_data[i][0]
        data_y[i] = my_data[i][1]

    # scale_x, scale_y = save_scaling(data_x_raw, data_y)
    # print scale_x, scale_y
    #
    # data_x_raw /= scale_y
    # data_y /= scale_y

    degree_data = []
    for model_degree in range(1, POLYNOMIAL_DEGREE):
        single_output = train_nth_degree(data_x_raw, data_y, model_degree)
        print single_output
        degree_data.append(single_output)

    min_error = degree_data[0][1]
    winner_row = degree_data[0]
    for elem in degree_data:
        if elem[1] < min_error:
            min_error = elem[1]
            winner_row = elem
    
    output = winner_row[2][::-1]
    np.save(SAVE_PATH, output)
    return output

if TRAIN:
    print(train())
if INFER:
    print(infer())

