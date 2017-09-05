# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

#%matplotlib inline

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

train_df = pd.read_csv("train.csv")
train_text_df = pd.read_csv("train.csv", sep="\,", 
                            engine='python', header=None, skiprows=1, 
                            names=["ID","Name"])

train_df.head()
train_text_df.head()

train_df.corr()

print("Train variants shape : ",train_df.shape)

plt.figure(figsize=(12,8))
sns.countplot(x="Embarked", data=train_df)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Class Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Classes", fontsize=15)
plt.show()

train_emb = train_df.groupby('Embarked')['Embarked'].count()
plt.figure(figsize=(12, 8))
plt.hist(train_emb.values, bins=50, log=True)
plt.xlabel('Number of times Emb appeared', fontsize=12)
plt.ylabel('log of Count', fontsize=12)
plt.show()

