# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

Google stock prices are given in trainset.csv and testset.csv files

![image](https://github.com/laakshit-D/rnn-stock-price-prediction/assets/119559976/003aa5bd-3367-4143-8cec-0a3e89f88428)

## DESIGN STEPS

### Step 1:
Read the csv file and create the Data frame using pandas.

### Step 2:
Select the " Open " column for prediction. Or select any column of your interest and scale the values using MinMaxScaler.

### Step 3:
Create two lists for X_train and y_train. And append the collection of 60 readings in X_train, for which the 61st reading will be the first output in y_train.

### STEP 4:
Make Predictions and plot the graph with the Actual and Predicted values.

## PROGRAM
```py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
```
```py
dataset_train = pd.read_csv('trainset.csv')
dataset_train.head()
```
```py
train_set = dataset_train.iloc[:,1:2].values
```
```py
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
```
```py
X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
```
```py
model = Sequential()
model.add(layers.SimpleRNN(units=50, activation='relu', input_shape=(X_train1.shape[1], 1)))
model.add(layers.Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()
```
```py
model.fit(X_train1,y_train,epochs=100, batch_size=32)
```
```py
dataset_test = pd.read_csv('testset.csv')
```
```py
test_set = dataset_test.iloc[:,1:2].values
```
```py
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
```
```py
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
```
```py
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)
```
```py
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```
## OUTPUT

### True Stock Price, Predicted Stock Price vs time

![image](https://github.com/laakshit-D/rnn-stock-price-prediction/assets/119559976/5f17a73d-455b-410b-8111-1d430c04df68)

### Mean Square Error

![image](https://github.com/laakshit-D/rnn-stock-price-prediction/assets/119559976/8f0b7173-91ba-4472-bf9d-05efd644d39a)

## RESULT
  Hence, we have successfully created a Simple RNN model for Stock Price Prediction.
