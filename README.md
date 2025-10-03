# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: R.Sairam
RegisterNumber:  25000694
*/
```
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

dataset = pd.read_csv('Placement_Data.csv')

dataset = dataset.drop('sl_no', axis=1)

dataset = dataset.drop('salary', axis=1)

categorical_cols = ["gender", "ssc_b", "hsc_b", "degree_t", "workex", "specialisation", "status","hsc_s"]

for col in categorical_cols:

 dataset[col] = dataset[col].astype('category')
 
for col in categorical_cols:

 dataset[col] = dataset[col].cat.codes
 
X = dataset.iloc[:, :-1].values

Y = dataset.iloc[:, -1].values

theta = np.random.randn(X.shape[1])

def sigmoid(z):

 return 1 / (1 + np.exp(-z))
 
def loss(theta, X, y):

 h = sigmoid(X.dot(theta))
 
 return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
 
def gradient_descent(theta, X, y, alpha, num_iterations):

 m = len(y)
 
 for i in range(num_iterations):
 
 h = sigmoid(X.dot(theta))
 
 gradient = X.T.dot(h - y) / m
 
 theta -= alpha * gradient
 
 return theta
 
theta = gradient_descent(theta, X, Y, alpha=0.01, num_iterations=1000)

def predict(theta, X):

 h = sigmoid(X.dot(theta))
 
 y_pred = np.where(h >= 0.5, 1, 0)
 
 return y_pred
 
y_pred = predict(theta, X)

accuracy = np.mean(y_pred.flatten() == Y)

print("Accuracy:", accuracy)

def plot_decision_boundary(X, y, theta):

 x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
 
 y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
 
 xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
 
 np.arange(y_min, y_max, 0.01))
 
 Z = sigmoid(np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()].dot(theta))
 
 Z = Z.reshape(xx.shape)
 
 plt.contourf(xx, yy, Z, levels=[0, 0.5], colors=['#FFAAAA', '#AAAAFF'], alpha=0.5)
 
 plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
 
 plt.xlabel('Feature 1 (e.g., Gender)')
 
 plt.ylabel('Feature 2 (e.g., SSC Percentage)')
 
 plt.title('Decision Boundary')
 
 plt.show()
 
 plot_decision_boundary(X[:, :2], Y, theta)


## Output:
![logistic regression using gradient descent](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

