#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 23:48:15 2017

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
dataset = pd.read_csv('Social_Network_Ads.csv')

#Get a feel of the data
print("Train variants shape : ",dataset.shape)
dataset.head()
dataset.describe()
dataset.corr()


#Visualize data -- To be improved
#Frequency distribution
plt.figure(figsize=(12,8))
sns.countplot(x="Age", data=dataset)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Class Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Classes", fontsize=15)
plt.show()

#Creating independent matrix
X = dataset.iloc[:, 2:4].values

#Creating dependent vector
Y = dataset.iloc[:,4].values

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

"""

#Splitting the data set into training and test set
from sklearn.cross_validation import train_test_split
x_Train, x_Test, y_Train, y_Test = train_test_split(X, Y, test_size = 0.25, 
                                                    random_state = 0)

#Feature Scaling
#A must do when algo is based on eucledian distance
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_Train = sc_X.fit_transform(x_Train)
x_Test = sc_X.transform(x_Test)

#Fitting Classifier to Training set
from sklearn.tree import DecisionTreeClassifier 
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_Train, y_Train)

#Predicting the Test set results. Vector of prediction
y_Pred = classifier.predict(x_Test)

#Evaluating the predictions of model
#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_Test, y_Pred)

#Visualizing the Training set results
from matplotlib.colors import ListedColormap
x_Set, y_Set =  x_Train, y_Train
X1, X2 = np.meshgrid(np.arange(start = x_Set[:,0].min() - 1, stop = x_Set[:,0].max() + 1, step = 0.01),
                     np.arange(start = x_Set[:,1].min() - 1, stop = x_Set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_Set)):
    plt.scatter(x_Set[y_Set == j,0], x_Set[y_Set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#Visualizing the Test set results
from matplotlib.colors import ListedColormap
x_Set, y_Set =  x_Test, y_Test
X1, X2 = np.meshgrid(np.arange(start = x_Set[:,0].min() - 1, stop = x_Set[:,0].max() + 1, step = 0.01),
                     np.arange(start = x_Set[:,1].min() - 1, stop = x_Set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_Set)):
    plt.scatter(x_Set[y_Set == j,0], x_Set[y_Set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
