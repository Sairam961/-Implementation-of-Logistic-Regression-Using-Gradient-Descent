# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess the dataset (encode categorical data, select features and labels).

2.Initialize model parameters and define the sigmoid function.

3.Apply gradient descent iteratively to minimize the loss function.

4.Predict outcomes, evaluate accuracy, and visualize the decision boundary. 

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

dataset = dataset.drop(['sl_no','salary'], axis=1)

categorical_cols = ["gender","ssc_b","hsc_b","degree_t","workex","specialisation","status","hsc_s"]

for col in categorical_cols:

   dataset[col] = dataset[col].astype('category').cat.codes

X = dataset[['ssc_p','hsc_p']].values

Y = dataset['status'].values

X = np.hstack((np.ones((X.shape[0],1)), X))

theta = np.random.randn(X.shape[1])

def sigmoid(z):

   return 1/(1+np.exp(-z))

def gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000):

   m = len(y)
   
   for _ in range(num_iterations):
   
   h = sigmoid(X.dot(theta))
      
   gradient = X.T.dot(h-y)/m
   
   theta -= alpha*gradient
   
   return theta

theta = gradient_descent(theta, X, Y)

def predict(theta, X):

   return (sigmoid(X.dot(theta))>=0.5).astype(int)

y_pred = predict(theta, X)

print("Accuracy:", np.mean(y_pred==Y))

x_min, x_max = X[:,1].min()-1, X[:,1].max()+1

y_min, y_max = X[:,2].min()-1, X[:,2].max()+1

xx, yy = np.meshgrid(np.linspace(x_min,x_max,200),np.linspace(y_min,y_max,200))

Z = sigmoid(np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()].dot(theta))

Z = Z.reshape(xx.shape)

plt.contourf(xx,yy,Z,levels=[0,0.5,1],colors=['#FFAAAA','#AAAAFF'],alpha=0.5)

plt.scatter(X[:,1],X[:,2],c=Y,edgecolors='k')

plt.xlabel("SSC %")

plt.ylabel("HSC %")

plt.title("Decision Boundary")

plt.show()


## Output:
<img src="ex6 output 1.png" alt="Output" width="500">

<img src="ex6 output 2.png" alt="Output" width="500">


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

