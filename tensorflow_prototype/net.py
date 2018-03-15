import numpy as np
import pandas as pd
import tensorflow as tf

MAX_DEGREE = 20     # TODO: PARAMETRIZE
path = '/path/to/csv/file/ai-task-samples.csv'
dataset = pd.read_csv(path, names=['x', 'y'], header=None)
xs = dataset['x'].as_matrix()
ys = dataset['y'].as_matrix()

xs /= np.max(np.abs(xs), axis=0)
ys /= np.max(np.abs(ys), axis=0)

n_observations = ys.shape[0]

degree_data = []
for degree in range(2, MAX_DEGREE):

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    b = tf.Variable(tf.random_normal([1]), name='bias')

    weights = []
    for i in range(1, degree):
        weights.append(tf.Variable(tf.random_normal([1]),
                                   name='weight_%d' % i))

    for pow_i, weight in zip(range(1, degree), weights):
        weight = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
        Y_pred = tf.add(tf.multiply(tf.pow(X, pow_i), weight), b)

    cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)

    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    saver = tf.train.Saver()

    n_epochs = 100
    sess = tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())

    prev_training_cost = 0.0
    for epoch_i in range(n_epochs):
        for (x, y) in zip(xs, ys):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        training_cost = sess.run(
            cost, feed_dict={X: xs, Y: ys})

        # print('Cost is:', training_cost)

        if np.abs(prev_training_cost - training_cost) < 0.0001:
            break
        prev_training_cost = training_cost

    weights_values = [b.eval(session=sess)]
    for i, W in enumerate(weights):
        value = W.eval(session=sess)
        weights_values.append(value)
        # print(value, " of power: ", i + 1)

    nth_degree_data = [degree, training_cost, weights_values]

    degree_data.append(nth_degree_data)

# choose best
min_cost = degree_data[0][1]
chosen_degree = degree_data[0]
for deg in degree_data:
    if min_cost > deg[1]:
        min_cost = deg[1]
        chosen_degree = deg

# print('Optimal degree of polynomial for the data after', n_epochs,
#       'of training is:', chosen_degree[0], 'with cost:', chosen_degree[1],
#       'and following parameters (from 0th to nth)', chosen_degree[2])
#
# print('All outputs:')
# print(degree_data)

output = [x for x in chosen_degree[2]]
output = output[::-1]
print(output)



