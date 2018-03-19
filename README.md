# Polynomial Regression
This repository contains an implementation of a simple neural network for polynomial regression.


## Depenedencies and installation
The network was implemented using Python 2.7.12  with Numpy. To get Numpy activate your virtualenv and run:
```
pip install numpy
```
Afterwards check which python you are using with ```which python``` and copy&paste the path into first line of the script (shebang) so that it looks like this: ```#! /path/to/my/python```. Then run ```mv net.py net``` and ```chmod +x net``` to make it executable. Now it should meet the requirements as far as interface is concerned.


## Instruction:
### Training the network
To train the network you will need .csv training data file with 2 floats per line separated by comma. The floats stand for x and y coordinate accordingly. You will also have to specify the highest order of polynomial that you think will model the data best. Note that high maximal degree will result in longer training.

To train run:
```
./net.py train POLYNOMIAL_DEGREE PATH_TO_CSV
```
where POLYNOMIAL_DEGREE is an integer that stands for highest order of polynomial and PATH_TO_CSV is the path to .csv training data file.

For example, to find the optimal polynomial of order no higher than 6 for data in ```./data_dir/my_csv_data.py``` you want to run:
```
./net.py train 6 ./data_dir/my_csv_data.py
```
The output will be a list with polynomial coefficients, from highest to lowest.

### Forward pass through the network
To run a forward pass through the trained network you need to provide it with an input X for which it will calculate the value of polynomial. The coefficients for this polynomial estimation were computed during training of the network.

To estimate run:
```
./net.py estimate X
```
where X is the input.

For example, to find the value of polynomial at point X=3 you want to run:
```
./net.py estimate 3
```
The output will be a value of polynomial for specified input.

## The network
Implemented network is a very simple 2 layer neural network.

The diagram of training the network could be described as follows:

![alt text](https://github.com/jakubkarczewski/PolynomialRegression/blob/master/pics/net.png)


### Error function 
The error used in this implementation is known as Mean Squared Error (MSE). It is calculated by computing mean of squared errors for each data point. In Python it looks like this: ```mse = (1.0 / len(x)) * np.sum(np.power(y - y_estimate, 2)) ```

### Stochastic Gradient Descent
Values of the weights (which are also polynomial coefficients) were calculated with Stochastic Gradient Descent (SGD). SGD, being stochastic approximation of Gradient Descent, takes in just a part of training data at once. It is used to minimize to cost function. It computes correction values for each weights by computing gradient from batch of training data with equation: ```gradient = -(1.0 / len(x)) * error.dot(x) ``` (error.dot(x) is just matrix multiplication), and then using it to correct the values of weights (w) with ```w -= learning_rate * gradient```.

### Forward pass
Firstly the input X, which is a float number, is being transformed into sequence of N floats where N is a degree of polynomial that is being fitted to the data. For 3rd degree polynomial the X input would be transformed into (X^0, X^1, X^2, X^3) sequence.
Then it is being multiplicated by values of weights (wX) which stand for polynomial coefficients and then summed into y which is interpreted as the value of polynomial for input X proposed by the network.

### Backpropagation
The proposed value of polynomial for input X is being substracted for it's known value Y (from .csv file) and by means of Stochastic Gradient Descent backpropagated to correct weights. This is repeated iteratively until the error is within tolerance range or the number of iterations reach it's limit.

This procedure is repeated for each and every degree of polynomial in range (1, POLYNOMIAL_DEGREE).

### Estimation
After running the training job the network will save the degree and coefficients of polynomial that fitted the data best. During forward pass the network computes the polynomial equation for particular input X.

For example if the polynomial saved by the netork would be f(x) = 3 * x^3 + 2 * x ^2 + x - 1 and the input X=1 the net would calculate:
```
3 * 1^3 + 2 * 1^2 + 1 - 1 = 5 
```
and would output ```5```

## Performance
![alt text](https://github.com/jakubkarczewski/PolynomialRegression/blob/master/pics/performance.png)


