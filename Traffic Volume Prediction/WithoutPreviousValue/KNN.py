# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 19:09:21 2019
@author: anishukla
"""
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
  
# Clearing dataset
dataset = dataset[['air_pollution_index', 'humidity', 'wind_speed', 'wind_direction', 'visibility_in_miles', 
                   'temperature', 'clouds_all', 'weather_type', 'traffic_volume']]
        
X = dataset.iloc[:, :].values
df_X = pd.DataFrame(X)

'''n is number of holidays'''
    
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_1 = LabelEncoder()
X[:, 7] = labelencoder_1.fit_transform(X[:, 7])

onehotencoder = OneHotEncoder(categorical_features = [7])
X = onehotencoder.fit_transform(X[:, :]).toarray()
X = X[:, 1:]
df_X = pd.DataFrame(X)

'''#
labelencoder_2 = LabelEncoder()
X[:, 13] = labelencoder_2.fit_transform(X[:, 13])
df_X = pd.DataFrame(X)

onehotencoder_1 = OneHotEncoder(categorical_features = [21])
X = onehotencoder_1.fit_transform(X[:, :]).toarray()
X = X[:, 1:]
df_X = pd.DataFrame(X)'''

# Splitting the dataset into the Training set and Test set
X_train = X[:27500, 0:17]
df_X_train = pd.DataFrame(X_train)
y_train = X[:27500, 17]
df_y_train = pd.DataFrame(y_train)
X_test = X[27500:, 0:17]
df_X_test = pd.DataFrame(X_test)
y_test = X[27500:, 17]
df_y_test = pd.DataFrame(y_test)

#Random Forest and standard scaling
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors import KNeighboursRegressor
from sklearn.metrics import r2_score,mean_squared_error

scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(-1, 1))
for i in range(20):
    rfr  = KNeighboursRegressor(n_neighbours=i)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


rfr.fit(X_train_scaled,y_train)
rfr.score(X_test_scaled,y_test)

res = rfr.predict(X_test_scaled)
df_res = pd.DataFrame(res)

print("Rmse=",np.sqrt(np.mean((((y_test-res)**2)))))

# Filterring the test set
test_set = pd.read_csv('Test.csv')

# Clearing dataset
test_set = test_set[['air_pollution_index', 'humidity', 'wind_speed', 'wind_direction', 'visibility_in_miles', 
                   'temperature', 'clouds_all', 'weather_type']]
A = test_set.iloc[:, :].values
df_A = pd.DataFrame(A)

labelencoder_2 = LabelEncoder()
A[:, 7] = labelencoder_2.fit_transform(A[:, 7])

onehotencoder_1 = OneHotEncoder(categorical_features = [7])
A = onehotencoder_1.fit_transform(A[:, :]).toarray()
A = A[:, :]
df_A = pd.DataFrame(A)
        
A_scaled = scaler.fit_transform(A)

pred = rfr.predict(A_scaled)
df_pred =pd.DataFrame(pred)

#Merging
tset = pd.read_csv('Test.csv')
tset = tset.iloc[:, 0].values
df_tset = pd.DataFrame(tset)

df_final = pd.concat([df_tset,df_pred], axis=1)

df_final.to_csv('res2.csv', index=False)