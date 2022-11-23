"""We will make an LSTM that will try to capture the downward and upward trend of the Google stock price.


Well, we're gonna train our LSTM model on five years of the Google stock price, and this is from the beginning of 2012 to the end of 2016 and then,
based on this training, based on the correlations identified or captured by the LSTM of the Google stock price, we will try to predict the first month of 2017.We're gonna try to predict January 2017.
And again, we're not going to try to predict exactly the stock price, we're gonna try to predict the trend, the upward or downward trend of the Google stock price."""

"""**DATA PREPROCESSING**"""
# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# Importing the training dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# selecting the column from the dataset and creating the numpy array of one column
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
# Two ways of feature scaling are:
# -> Standardisation
# -> Normalisation
# We will be using Normalisation
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
# fit means it is going to get minimum stock price and maximium stock price from the dataset
# transform is going to calculate the scaled price for each stock
training_dataset_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output 60 timesteps means that at ecah time t the RNN is going to
# look at 60 stock prices before time t based on this it is going to predict

x_train = []
y_train = []

for i in range(60, 1258):
    x_train.append(training_dataset_scaled[i - 60:i, 0])  # we want to take 60 previous stock 0 is the column number
    # x_train will contain data from 0 to 59 as upper bound is included, based on this training we want our model to
    # predict the price of 60 day stock thats y_train has i
    y_train.append(training_dataset_scaled[i, 0])

# making our list as numpy array
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping Going to add the predictors to our dataset that we made in the previous step which we can use predict the
# price of our stock These predictors are the indicators So we will adding more dimensions to our data structure
# Anytime we want to add a new dimension to our numpy array, we need to use reshape function
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # the first argument is the numpy array that
# we want to reshape, second argument is the shape that we want our numpy array to have
# currently our x_train is 2d matrix, by adding new dimension it will become 3d

"""**Building the RNN**"""
# Importing the Keras libraries and packages
from keras.models import Sequential  # Allow us to create neural network objects representing the squence of layers
from keras.layers import Dense  # To add output layer
from keras.layers import LSTM  # To add LSTM layers
from keras.layers import Dropout  # To prevent any overfitting

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout Regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# first argument is the number of units which is number of LSTM cells
# second argumnet is return squences which will be set to true as we are building a stacked LSTM
# third argument is input_shape that is shape of input(x_train) or input layer

regressor.add(Dropout(0.2))  # 20% neurons of the LSTM will be ignored during the training

# Adding the second LSTM layer and some Dropout Regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the third LSTM layer and some Dropout Regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the fourth LSTM layer and some Dropout Regularisation
regressor.add(LSTM(units=50,
                   return_sequences=False))  # return sequences will be made false as after this layer output layer
# comes so we are not returning anything
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')  # two arguments are the optimizer and loss function

# Fitting the RNN to the training set fit will connect our network to the training dataset and will also execute the
# training over a certain number of epochs that we will choose in the same fit method
regressor.fit(x_train, y_train, epochs=100, batch_size=32)
# argument-1: the input of the training set(features or independent variables which will be the input of the neural
# network)(x-train) argument-2:y_train argument-3:number of epochs argument-4:batch size

"""**Making the predictions and visualizing the results**"""

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

x_test = []

for i in range(60, 80):
    x_test.append(inputs[i - 60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()
