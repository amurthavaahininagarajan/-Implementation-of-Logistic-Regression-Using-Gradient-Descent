# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: AMURTHA VAAHINI.KN
RegisterNumber: 212222240008
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data2.txt",delimiter = ',')
x= data[:,[0,1]]
y= data[:,2]
print('Array Value of x:')
x[:5]

print('Array Value of y:')
y[:5]

print('Exam 1-Score graph')
plt.figure()
plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label=' Not Admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()


def sigmoid(z):
  return 1/(1+np.exp(-z))
  
print('Sigmoid function graph: ')
plt.plot()
x_plot = np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot))
plt.show()


def costFunction(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad = np.dot(x.T,h-y)/x.shape[0]
  return j,grad


print('X_train_grad_value: ')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
j,grad = costFunction(theta,x_train,y)
print(j)
print(grad)


print('y_train_grad_value: ')
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j


def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad

print('res.x:')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)


def plotDecisionBoundary(theta,x,y):
  x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot = np.c_[xx.ravel(),yy.ravel()]
  x_plot = np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot = np.dot(x_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
  plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label='Not Admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel('Exam  1 score')
  plt.ylabel('Exam 2 score')
  plt.legend()
  plt.show()

print('DecisionBoundary-graph for exam score: ')
plotDecisionBoundary(res.x,x,y)

print('Proability value: ')
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)


def predict(theta,x):
  x_train = np.hstack((np.ones((x.shape[0],1)),x))
  prob = sigmoid(np.dot(x_train,theta))
  return (prob >=0.5).astype(int)


print('Prediction value of mean:')
np.mean(predict(res.x,x)==y)
*/
```

## Output:
## Array Value of x:
![image](https://github.com/amurthavaahininagarajan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679102/c56a2367-a570-42f5-bcc9-a90bc774e3bb)
## Array Value of y:
![image](https://github.com/amurthavaahininagarajan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679102/932dd1d6-9a9d-419c-b41c-45b2e15fed1a)
## Score graph:
![image](https://github.com/amurthavaahininagarajan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679102/59403535-b22e-4dac-974e-0f183206b53d)
## Sigmoid function graph:
![image](https://github.com/amurthavaahininagarajan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679102/35dd9310-1732-4f7f-b097-e1951f4174eb)
## X_train_grad value:
![image](https://github.com/amurthavaahininagarajan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679102/9dde8dbd-35f8-4ddd-8571-7d82b76a2393)
## Y_train_grad value:
![image](https://github.com/amurthavaahininagarajan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679102/9dbc888d-12d2-4adf-ad9f-124023d72b30)
## res.x:
![image](https://github.com/amurthavaahininagarajan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679102/c2aac35d-a9b3-4972-9ca7-24f726da15a2)
## Decision boundary:
![image](https://github.com/amurthavaahininagarajan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679102/d6548eaf-9f06-404d-91ad-a539193f7d61)
## Proability value:
![image](https://github.com/amurthavaahininagarajan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679102/81fdb7ed-993f-4f91-ba1c-4400a46c5f97)
## Prediction value of mean:
![image](https://github.com/amurthavaahininagarajan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118679102/06948524-683b-4ff2-97b7-540d03d12e39)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

