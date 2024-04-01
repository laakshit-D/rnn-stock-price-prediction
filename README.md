# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.

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
Include your code here

## OUTPUT

### True Stock Price, Predicted Stock Price vs time

Include your plot here

### Mean Square Error

Include the mean square error

## RESULT
