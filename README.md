# Polynomial Regression

This repo contains an implementation of simple neural network for polynomial regression.


## Depenedencies

The net was implemented using Python 2.7 with Numpy. To get Numpy on Ubuntu run:

```
pip install numpy
```

## Instruction:
### Training the network

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

### Forward pass through the network

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

## The network

The implemented network is a very simple 2 layer neural network.

The diagram of training the network could be described as follows:

![alt text](https://github.com/jakubkarczewski/PolynomialRegression/blob/master/net.png)

### Stochastic Gradient Descent

### Error function

### Forward pass
Firstly the input X which is a float number is being transformed into sequence of N floats where N is a degree of polynomial that is being fitted to the data. For 3rd degree polynomial the X input would be transformed into (X^0, X^1, X^2, X^3) sequence.
Then it is being multiplicated by values of weights (wX) which stand for polynomial coefficients and then summed into y which is interpreted as the value of polynomial for input X proposed by the network.

### Backpropagation
The proposed value of polynomial for input X is being substracted for it's known value Y (both from .csv file) and by means of Stochastic Gradient Descent backpropagated to form dW vector. The vector contains correction values for each of the weights. This is repeated iteratively until the error is within tolerance range or the number of iterations reach it's limit.

This procedure is repeated for each and every degree of polynomial in range (1, POLYNOMIAL_DEGREE).

### Estimation
After running the training job the network will save the degree and coefficients of polynomial that fitted the data best. During forward pass the network computes the polynomial equation for particular input X.

For example if the polynomial saved by the netork would be f(x) = 3 * x^3 + 2 * x ^2 + x - 1 and the input X=1 the net would calculate:
```
3 * 1^3 + 2 * 1^2 + 1 - 1 = 5 
```
and would output ```5```

