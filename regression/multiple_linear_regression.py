#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 23:07:32 2017

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
dataset = pd.read_csv('50_Startups.csv')

#Get a feel of the data
dataset.head()
dataset.describe()
dataset.corr()
print("Train variants shape : ",dataset.shape)

#Visualize data -- To be improved
plt.figure(figsize=(12,8))
sns.countplot(x="R&D Spend", data=dataset)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Class Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Classes", fontsize=15)
plt.show()

train_emb = dataset.groupby('R&D Spend')['R&D Spend'].count()
plt.figure(figsize=(12, 8))
plt.hist(train_emb.values, bins=50, log=True)
plt.xlabel('Number of times R&D Spend appeared', fontsize=12)
plt.ylabel('log of Count', fontsize=12)
plt.show()

#Creating independent matrix
X = dataset.iloc[:, :-1].values

#Creating dependent matrix
Y = dataset.iloc[:,4].values

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
x_Train, x_Test, y_Train, y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Fitting multiple linear regression model to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_Train, y_Train)

#Predicting the Test set results
y_Pred = regressor.predict(x_Test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

#Iteratively create X_opt to find the optimal X 
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Iteration 2
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Iteration 3
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Iteration 3
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Iteration 4
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Fitting the model again
#Splitting the data set into training and test set
from sklearn.cross_validation import train_test_split
x_Train, x_Test, y_Train, y_Test = train_test_split(X_opt, Y, test_size = 0.2, random_state = 0)

#Fitting multiple linear regression model to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_Train, y_Train)

#Predicting the Test set results
y_Pred = regressor.predict(x_Test)
