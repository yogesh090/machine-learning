#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 23:27:31 2017

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
dataset = pd.read_csv('Data.csv')

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

#Creating independent matrix
X = dataset.iloc[:, :-1].values

#Creating dependent matrix
Y = dataset.iloc[:,3].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encode catagorical data
# Catagory -- yes, No
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

#Dummy variables as no comparison
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()

#Encoding the dependent variable with label as the algo will know that dependent is
#category and has no order between them
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)


#Splitting the data set into training and test set
from sklearn.cross_validation import train_test_split
x_Train, x_Test, y_Train, y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_Train = sc_X.fit_transform(x_Train)
x_Test = sc_X.transform(x_Test)