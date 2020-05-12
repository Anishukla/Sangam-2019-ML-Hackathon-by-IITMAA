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
X_train = X[:27500, 0:18]
df_X_train = pd.DataFrame(X_train)
y_train = X[:27500, 18]
df_y_train = pd.DataFrame(y_train)
X_test = X[27500:, 0:18]
df_X_test = pd.DataFrame(X_test)
y_test = X[27500:, 18]
df_y_test = pd.DataFrame(y_test)

#Random Forest and standard scaling
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error

scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(-1, 1))
rfr  = RandomForestRegressor(n_estimators=100, random_state=5000, verbose=2, n_jobs=10)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


rfr.fit(X_train_scaled,y_train)
rfr.score(X_test_scaled,y_test)

res = rfr.predict(X_test_scaled)
df_res = pd.DataFrame(res)

print("Rmse=",np.sqrt(np.mean((((y_test-res)**2)))))

# Filterring the test set
test_set = pd.read_csv('Test.csv')

#Including previous 15 values
A_req = X[33735:, 18]
df_A_req = pd.DataFrame(A_req)

for obs in range(1,15):
    test_set["T_" + str(obs)] = test_set.weather_description.shift(obs)
    
test_set.fillna(0.00,inplace=True)

# Clearing dataset
test_set = test_set[['is_holiday', 'air_pollution_index', 'visibility_in_miles', 'temperature', 'T_1', 'T_2', 'T_3', 'T_4', 'T_5', 'T_6', 'T_7', 'T_8', 'T_9', 'T_10', 'T_11', 'T_12', 'T_13', 'T_14']]
 
A = test_set.iloc[:, :].values
df_A = pd.DataFrame(A)

# Changing None to 0 and holidays to 1
m = 0
for i in range(14454):
    if df_A[0][i] == 'None':
        df_A[0][i] = 0
    else:
        df_A[0][i] = 1
        
for i in range(14454):
    if df_A[0][i] == 1:
        m = m+1
        
A_scaled = scaler.fit_transform(A)

pred = rfr.predict(A_scaled)