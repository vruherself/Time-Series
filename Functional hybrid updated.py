#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:30:05 2021

@author: vrushali
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import concatenate
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Dense
import matplotlib.pyplot as plt


dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(40, 1258):
    X_train.append(training_set_scaled[i-40:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

##building the model


#Input layer
visible = Input(shape=(40,1))

#building F1
lstm1 = LSTM(units =20)(visible)

#building f2
conv1 = Conv1D(10, 5, 2, activation='relu')(visible)
maxpool1 = MaxPooling1D(pool_size=2)(conv1)
lstm2 = LSTM(10, activation='tanh')(maxpool1)

#building f3
conv2 = Conv1D(20, 7, 2, activation='relu')(visible)
maxpool2 = MaxPooling1D(pool_size=2)(conv2)
conv3 = Conv1D(10, 5, 2, activation='relu')(maxpool2)
maxpool3 = MaxPooling1D(pool_size=2)(conv3)
lstm3 = LSTM(10, activation='tanh')(maxpool3)

merged = concatenate([lstm1, lstm2, lstm3])

dense1 = Dense(10, activation='relu')(merged)
dense2 = Dense(1, activation = 'sigmoid')(dense1)

model = Model(inputs = visible, outputs = dense2)

from keras.utils import plot_model
plot_model(model, to_file='hybrid_functional.png')

optimizer = tf.keras.optimizers.SGD(lr=0.001)
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=tf.keras.metrics.RootMeanSquaredError())
#model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs = 100, batch_size=80)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 40:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(40, 60):
    X_test.append(inputs[i-40:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


