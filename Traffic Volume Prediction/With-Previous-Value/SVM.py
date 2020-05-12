#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:22:48 2019
@author: anishukla
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:58:21 2019
@author: anishukla
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('datasets/Train.csv')
  
#Including previous 15 values
for obs in range(1,49):
    dataset["T_" + str(obs)] = dataset.traffic_volume.shift(obs)
    
dataset.fillna(0.00,inplace=True)

# Clearing dataset
dataset = dataset[['air_pollution_index', 'visibility_in_miles', 'temperature', 'T_1', 'T_2', 'T_3', 'T_4', 
                   'T_5', 'T_6', 'T_7', 'T_8', 'T_9', 'T_10', 'T_11', 'T_12', 'T_13', 'T_14', 'T_15',
                   'T_16', 'T_17', 'T_18', 'T_19', 'T_20', 'T_21', 'T_22', 'T_23', 'T_24','T_25', 'T_26',
                   'T_27', 'T_28', 'T_29', 'T_30', 'T_31', 'T_32', 'T_33', 'T_34', 'T_35', 'T_36', 'T_37', 
                   'T_38', 'T_39', 'T_40', 'T_41', 'T_42', 'T_43', 'T_44', 'T_45', 'T_46', 'T_47',
                   'T_48', 'traffic_volume']]
        
X = dataset.iloc[:, :].values
df_X = pd.DataFrame(X)


        
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
X_train = X[:27500, 0:51]
df_X_train = pd.DataFrame(X_train)
y_train = X[:27500, 51]
df_y_train = pd.DataFrame(y_train)
X_test = X[27500:, 0:51]
df_X_test = pd.DataFrame(X_test)
y_test = X[27500:, 51]
df_y_test = pd.DataFrame(y_test)

#Random Forest and standard scaling
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error

scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(-1, 1))
rfr = SVR(kernel='rbf')

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

rfr.fit(X_train_scaled,y_train)
rfr.score(X_test_scaled,y_test)

res = rfr.predict(X_test_scaled)
df_res = pd.DataFrame(res)

print("Rmse=",np.sqrt(np.mean((((y_test-res)**2)))))
