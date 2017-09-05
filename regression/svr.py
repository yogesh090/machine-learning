#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:29:18 2017

@author: yogi
"""


#Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
color = sns.color_palette()

#%matplotlib inline
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')

#Get a feel of the data
dataset.head()
dataset.describe()
dataset.corr()
print("Train variants shape : ",dataset.shape)

#Visualize data -- To be improved
#Frequency distribution
plt.figure(figsize=(12,8))
sns.countplot(x="Salary", data=dataset)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Class Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Classes", fontsize=15)
plt.show()

#Creating independent matrix
X = dataset.iloc[:, 1:2].values

#Creating dependent vector
Y = dataset.iloc[:,2].values

"""
#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encode catagorical data
# Catagory -- yes, No
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])

#Dummy variables as no comparison
oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X = X[:, 1:]

#Splitting the data set into training and test set
from sklearn.cross_validation import train_test_split
x_Train, x_Test, y_Train, y_Test = train_test_split(X, Y, test_size = 0.2, 
                                                    random_state = 0)

"""
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

#Fitting the Regression models to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)

#Predicting with Regression model
y_Pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualize the SVR regression result
plt.scatter(X, Y, color = "red")
plt.plot(X, regressor.predict(X), color = "blue")
plt.title("Truth Vs Hype (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Visualize the SVR regression result in refined curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.title("Truth VS Hype")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()