#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 23:06:19 2017

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

#No split as data is too less

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)

#Visualize the Linear regression result
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Truth VS Hype")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Visualize the Polynomial regression result
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Truth VS Hype")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Visualize the Polynomial regression result in refined curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Truth VS Hype")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Predicting with Linear regression model
lin_reg.predict(6.5)

#Predicting with Polynomial regression model
lin_reg_2.predict(poly_reg.fit_transform(6.5))