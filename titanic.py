# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:42:10 2020

@author: Sejal Vyas
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset and removing irrelevant columns
dataset = pd.read_csv('train.csv')
dataset = dataset.drop(['Name'], axis=1)
dataset = dataset.drop(['Ticket'], axis=1)
dataset = dataset.drop(['Fare'], axis=1)
dataset = dataset.drop(['Cabin'], axis=1)
dataset = dataset.drop(['SibSp'], axis=1)
dataset = dataset.drop(['Parch'], axis=1)
dataset = dataset.drop(['Embarked'], axis=1)

X = dataset.iloc[:, 2 :8].values
print(X)
y = dataset.iloc[:, 1].values

#Missing data handling
from sklearn.preprocessing import Imputer
imputer = Imputer(axis = 0)
imputer = imputer.fit(X[:,[2]])
X[:,[2]] = imputer.transform(X[:,[2]])


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,1] = labelEncoder_X.fit_transform(X[:,1])
oneHotEncoder = OneHotEncoder(handle_unknown='ignore', categorical_features=[1])
X = oneHotEncoder.fit_transform(X).toarray()
print(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
tp = cm[0][0]
fn = cm[0][1]
fp = cm[1][0]
tn = cm[1][1]

accuracy = (tp + tn)/(tp + fp + tn + fn)
precision = (tp)/(tp+fp)
recall = tp/(tp+fn)
f1 = 2* precision * recall / (precision + recall)
 
