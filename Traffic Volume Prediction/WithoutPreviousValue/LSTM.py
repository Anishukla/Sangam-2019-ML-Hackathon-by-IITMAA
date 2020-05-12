# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 19:39:42 2019
@author: anishukla
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Train.csv')
res_set = pd.read_csv('Test.csv')
  
#Including previous 15 values
for obs in range(1,15):
    dataset["T_" + str(obs)] = dataset.traffic_volume.shift(obs)
    
dataset.fillna(0.00,inplace=True)

# Clearing dataset
dataset = dataset[['is_holiday', 'air_pollution_index', 'visibility_in_miles', 'temperature', 'T_1', 'T_2', 'T_3', 'T_4', 'T_5', 'T_6', 'T_7', 'T_8', 'T_9', 'T_10', 'T_11', 'T_12', 'T_13', 'T_14', 'traffic_volume']]
        
X = dataset.iloc[:, :].values
df_X = pd.DataFrame(X)

# Changing None to 0 and holidays to 1
n = 0
for i in range(33750):
    if df_X[0][i] == 'None':
        df_X[0][i] = 0
    else:
        df_X[0][i] = 1
        
for i in range(33750):
    if df_X[0][i] == 1:
        n = n+1
        
'''n is number of holidays'''
    
'''# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_1 = LabelEncoder()
X[:, 12] = labelencoder_1.fit_transform(X[:, 12])

labelencoder_2 = LabelEncoder()
X[:, 13] = labelencoder_2.fit_transform(X[:, 13])
df_X = pd.DataFrame(X)


#
onehotencoder = OneHotEncoder(categorical_features = [11])
X = onehotencoder.fit_transform(X[:, 1:]).toarray()
X = X[:, 1:]
df_X = pd.DataFrame(X)

onehotencoder_1 = OneHotEncoder(categorical_features = [21])
X = onehotencoder_1.fit_transform(X[:, :]).toarray()
X = X[:, 1:]
df_X = pd.DataFrame(X)'''

# Splitting the dataset into the Training set and Test set
X_train = X[:26500, 0:18]
df_X_train = pd.DataFrame(X_train)
y_train = X[:26500, 18]
df_y_train = pd.DataFrame(y_train)
X_val = X[26500:28500, 0:18]
df_X_val = pd.DataFrame(X_val)
y_val = X[26500:28500, 18]
df_y_val = pd.DataFrame(y_val)
X_test = X[28500:, 0:18]
df_X_test = pd.DataFrame(X_test)
y_test = X[28500:, 18]
df_y_test = pd.DataFrame(y_test)

#Feature scaling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(-1, 1))

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
X_valid_scaled = scaler.fit_transform(X_val)

df_X_train_scaled = pd.DataFrame(X_train_scaled)
df_X_test_scaled = pd.DataFrame(X_test_scaled)
df_X_valid_scaled = pd.DataFrame(X_valid_scaled)

# Model and Predict
# Importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_valid_scaled = np.reshape(X_valid_scaled, (X_valid_scaled.shape[0], X_valid_scaled.shape[1], 1))
X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Initializing the RNN
regressor = Sequential()

# Adding the first LSTM and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
''' We need to input true in every layer except our last LSTM layer in #return_sequences.
   Also in input shape we are just entering 2Ds for 3D data as 3rd will get recognized by itself
   units are no. of neuron we want we are just taking not so small or very large no.
   as we want a model with good prediction #We can optimize it by grid search'''
''' 20% neuron will be neglected during every round of training i.e 10 for 50'''

# Adding the 2nd LSTM and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

'''# Adding the 3rd LSTM and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the 4th LSTM and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
#We are doing it to increase our dimensionality

# Adding the 4th LSTM and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))'''

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
''' adam is always a safe choice'''

# Fitting the RNN to training set
regressor.fit(X_train_scaled, y_train, epochs = 25, batch_size = 250)

y_pred = regressor.predict(X_test_scaled)
df_y_pred = pd.DataFrame(y_pred)


#RMSE
print("Rmse=",np.sqrt(np.mean((((y_test-y_pred)**2)))))
from numpy import sqrt
sqrt(mean_squared_error(y_test[0],y_pred[0]))




