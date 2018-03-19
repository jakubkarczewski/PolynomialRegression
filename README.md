# Polynomial Regression

This repo contains an implementation of simple neural network for polynomial regression.


## Depenedencies

The net was implemented using Python 2.7 with Numpy. To get Numpy on Ubuntu run:

```
pip install numpy
```

## Training the network

To train the network you will ned .csv training data file with 2 floats per line separated by comma. The floats stand for x and y coordinate accordingly. You will also have to specify the highest order of polynomial that you think will model the data best. Note that high maximal degree will result in longer training.

To train run:
```
./net.py train POLYNOMIAL_DEGREE PATH_TO_CSV
```
where POLYNOMIAL_DEGREE is an integer number of highest order of polynomial and PATH_TO_CSV is the path to .csv data training file

For example for a job of finding the optimal polynomial of order no higher than 6 you want to run:
```
./net.py train 6 ./data_dir/my_csv_data.py
```

## Forward pass through the network

To run a forward pass through the trained network you need to provide it with an input X for which it will calculate the value of polynomial. The coefficients for this polynomial estimation were found during training the network.

To estimate run:
```
./net.py estimate X
```
where X is the input.
For example, to find the value of polynomial at point x=3 you should run
```
./net.py estimate 3
```

### Networks structure

The implemented network is a very simple 2 layer network with N inputs (N equal to particular degree of polynomial that is being evaluated) and one output equal to the value of polynomial for particular input.

The diagram of training the network could be described as follows:


Add additional notes about how to deploy this on a live system

