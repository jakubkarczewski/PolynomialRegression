# Polynomial Regression
This repository contains an implementation of a simple neural network for polynomial regression.


## Depenedencies and installation
The network was implemented using Python 2.7.12 with Numpy as only external library. To get Numpy activate your virtualenv (if you have one) and run:
```
pip install numpy
```
Having done that, check which python you are using with:
```
which python
```
and copy&paste the path into first line of the script with ```!# ``` prefix (shebang).

Example: ```#! /path/to/my/python```.
Finally run:
```
mv polynomial.py polynomial
```
and
```
chmod +x polynomial
```
You can also specify the path for saving the network by editing variable ```SAVE_PATH``` line 11th.

You are good to go!


## Instruction:
### Training the network
To train the network you need .csv training data file with 2 floats per line separated by comma. The floats stand for x and y coordinate accordingly. You will also have to specify the highest order of polynomial that you think will model the data best. Note that high maximal degree will result in longer training.

To train run:
```
./polynomial train POLYNOMIAL_DEGREE PATH_TO_CSV
```
where POLYNOMIAL_DEGREE is an integer that stands for highest order of polynomial and PATH_TO_CSV is the path to .csv training data file.

For example, to find the optimal polynomial of order no higher than 6 for data in ```./data_dir/my_csv_data.csv``` you want to run:
```
./polynomial train 6 ./data_dir/my_csv_data.csv
```
The output will be a list with polynomial coefficients, from highest to constant.

### Forward pass through the network
To run a forward pass through the trained network you need to provide it with an input X for which it will calculate the value of polynomial. The coefficients for this polynomial estimation were computed during training of the network.

To estimate run:
```
./polynomial estimate X
```
where X is the input.

For example, to find the value of polynomial at point X=3 you want to run:
```
./polynomial estimate 3
```
The output will be a value of polynomial for specified input.

## The network
Implemented network is a very simple 2 layer neural network.

The diagram of training the network for 3th degree polynomial could be described as follows:

![alt text](https://github.com/jakubkarczewski/PolynomialRegression/blob/master/pics/net.png)

<---> connector - forward pass and backpropagation

----> connector - forward pass only


### Error function 
The error used in this implementation is known as Mean Squared Error (MSE). It is calculated by computing mean of squared errors for each data point. In Python it would look like this: ```mse = (1.0 / len(x)) * np.sum(np.power(y - y_estimate, 2)) ```

### Stochastic Gradient Descent
Values of the weights (which are also scaled polynomial coefficients) were calculated with Stochastic Gradient Descent (SGD). SGD, being stochastic approximation of Gradient Descent, takes in just a part of training data at once. It is used to minimize to cost function. It corrects values for each weight by computing gradient from batch of training data.
### Forward pass
Firstly, the input X, which is a float number, is being transformed into sequence of N floats where N is a degree of polynomial that is being fitted to the data. For 3rd degree polynomial the X input would be transformed into (X^0, X^1, X^2, X^3) sequence.
Then it is being multiplicated (element-wise) by values of weights (wX) which stand for scaled polynomial coefficients. Next the product's elements are summed into ```y``` which is interpreted as the value of polynomial for input X estimated by the network.

### Backpropagation
The estimated value of polynomial for input X is being substracted from it's known value Y (from .csv file) and by means of SGD, backpropagated to correct weights. This is repeated iteratively until the error is within tolerance or the number of iterations expired.

This procedure is repeated for each and every degree of polynomial in range (1, POLYNOMIAL_DEGREE).

### Estimation
After running the training job the network will save the coefficients of polynomial that fitted the data best. During forward pass the network computes the polynomial equation for particular input X.

For example if the polynomial saved by the netork would be ```f(x) = 3 * x^3 + 2 * x ^2 + x - 1``` and the input X=1 the net would calculate:
```
3 * 1^3 + 2 * 1^2 + 1 - 1 = 5 
```
and would output ```5```

## Performance
![alt text](https://github.com/jakubkarczewski/PolynomialRegression/blob/master/pics/performance.png)

To visualize the performance of the network you can use ```evaluation.ipynb``` but note that it requires Pandas and Matplotlib. Both can be downloaded with ```pip``` Python package manager.


