#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:23:12 2017

@author: yogi
"""

#Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Salary_Data.csv')

#Creating independent matrix
X = dataset.iloc[:, :-1].values

#Creating dependent matrix
Y = dataset.iloc[:,1].values

#Splitting the data set into training and test set
from sklearn.cross_validation import train_test_split
x_Train, x_Test, y_Train, y_Test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#fitting the object to training set
regressor.fit(x_Train, y_Train)

#Predicting the Test set results
y_Pred = regressor.predict(x_Test)

#Visualising the Training set result
plt.scatter(x_Train, y_Train, color = "red")
plt.plot(x_Train, regressor.predict(x_Train), color = "blue")
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#Visualising the Test set result
plt.scatter(x_Test, y_Test, color = "red")
plt.plot(x_Train, regressor.predict(x_Train), color = "blue")
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#Visualising the Predicted set result
plt.scatter(x_Test, y_Pred, color = "red")
plt.plot(x_Train, regressor.predict(x_Train), color = "blue")
plt.title("Salary vs Experience (Predicted set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()