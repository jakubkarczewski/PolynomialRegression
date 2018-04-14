#! /home/kuba/Development/Quantum/quantum/bin/python
import sys
import numpy as np

# set seed for deterministic output
np.random.seed(5)

PATH_TO_CSV = None
SAVE_PATH = None
POLYNOMIAL_DEGREE = 2


def terminate():
    ''' Show prompt and exit '''
    print('Usage: \n - ./polynomial train POLYNOMIAL_'
          'DEGREE PATH_TO_CSV [SAVE_PATH] \n - ./polynomial estimate X [WEIGHTS_PATH]')
    return


if len(sys.argv) > 2:
    if sys.argv[1] == 'train':
        try:
            POLYNOMIAL_DEGREE += int(sys.argv[2])
            PATH_TO_CSV = str(sys.argv[3])
            if len(sys.argv) >= 5:
                SAVE_PATH = str(sys.argv[4])
        except Exception:
            terminate()
    elif sys.argv[1] == 'estimate':
        try:
            _ = float(sys.argv[2])
            if len(sys.argv) >= 5:
                SAVE_PATH = str(sys.argv[4])
        except Exception:
            terminate()
    else:
        terminate()
else:
    terminate()

if not SAVE_PATH:
    SAVE_PATH = './weights.npy' # todo: add directory where everyone can write


class ShallowNet:

    def __init__(self, learning_rate=0.5, error_tolerance=1e-6, decay=0.99,
                 batch_size=100, max_epochs=100, polynomial_degree=4):
        # SGD parameters
        self.learning_rate = learning_rate
        self.error_tolerance = error_tolerance
        self.decay = decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.polynomial_degree = polynomial_degree

    def compute_gradient(self, w, x, y):
        ''' Compute gradient and MSE '''
        y_estimate = x.dot(w).flatten()
        error = (y.flatten() - y_estimate)
        mse = (1.0 / len(x)) * np.sum(np.power(error, 2))
        gradient = -(1.0 / len(x)) * error.dot(x)
        return gradient, mse

    def infer(self, input):
        ''' Run forward pass through the network '''
        weights = np.load(SAVE_PATH)[::-1]
        output = 0
        for i, coef in enumerate(weights):
            output += coef * input ** i
        return output

    def train_nth_degree(self, data_x_raw, data_y, model_degree):
        ''' Find optimal coefficients and error for given degree'''
        # turn input X into matrix of Xs to nth power
        data_x = np.power(data_x_raw, range(model_degree))
        # normalize inputs
        scaling = np.max(data_x, axis=0)
        data_x /= scaling
        # generate random initial values for weights
        w = np.random.randn(model_degree)

        epochs = 1
        iterations = 0
        new_error = 10
        error = 1

        while (abs(new_error - error) > self.error_tolerance
               and epochs < self.max_epochs):
            # shuffle training data
            degree = np.random.permutation(len(data_x))
            data_x = data_x[degree]
            data_y = data_y[degree]

            used_data = 0
            while used_data < len(data_x):  # one iteration per batch
                tx = data_x[used_data: used_data + self.batch_size]
                ty = data_y[used_data: used_data + self.batch_size]
                # compute gradient
                gradient = self.compute_gradient(w, tx, ty)[0]
                error = self.compute_gradient(w, data_x, data_y)[1]
                # correct weights
                w -= self.learning_rate * gradient
                iterations += 1
                used_data += self.batch_size

            new_error = self.compute_gradient(w, data_x, data_y)[1]
            # decay the LR
            self.learning_rate = (self.learning_rate * (self.decay **
                                                        int(epochs / 1000)))
            epochs += 1

        return model_degree, new_error, w, iterations, scaling

    def train(self):
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
            single_output = self.train_nth_degree(data_x_raw, data_y,
                                                 model_degree)
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

if __name__ == '__main__':
    net = ShallowNet(polynomial_degree=POLYNOMIAL_DEGREE)
    if sys.argv[1] == 'train':
        print(net.train())
    if sys.argv[1] == 'estimate':
        print(net.infer(float(sys.argv[2])))
