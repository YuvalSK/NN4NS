# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:39:15 2023

@author: User
"""

from sklearn.datasets import load_digits
from sklearn.utils import shuffle

mnist = load_digits(n_class=10) # Each datapoint is a vector of 64 pixels, or a 8x8 image of a digit

X, Y = mnist.data[:1800] / 255. , mnist.target[:1800]

x = [ex for ex, ey in zip(X, Y) if ey==1 or ey==8]
y = [1 if ey==1 else -1 for ex, ey in zip(X, Y) if ey==1 or ey==8]
x, y = shuffle(x, y, random_state=1)

import matplotlib.pyplot as plt
plt.figure(1)
for i in range(1,26):
    ax = plt.subplot(5,5,i)
    ax.axis('off')
    if y[i] ==1:
        ax.imshow(x[i].reshape(8, 8),cmap='gray')
    else:
        ax.imshow(255-x[i].reshape(8, 8), cmap='gray')
plt.show()

import numpy as np
import random

m = len(x) #356, since ~180 samples per class
d = len(x[0]) #64, since 8*8

eta = 0.1
w = np.zeros((d,))
T = 8000
for t in range(0, T):
    #print(f"iteration number:{t}")
    i = random.randint(0, m-1)
    y_pred = np.sign(np.dot(w, x[i]))
    
    if y_pred * y[i] <= 0:
        #print("-update weights")
        w+= eta*y[i]*x[i]
        
w_perceptron = w

M_perceptron = 0
for t in range(0,m):
    y_pred = np.sign(np.dot(w_perceptron, x[t]))
    
    #count errors
    if y[t]!= y_pred:
        M_perceptron +=1
        
err = float(M_perceptron/m)
print(f"perceptron error = {err:.5f}")

