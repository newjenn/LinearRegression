# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 03:28:46 2018

@author: Pasumpon
"""
import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression 

#import sklearn.metrics as sm
print("hello")
current_file = os.path.abspath(os.path.dirname(__file__))
#data = pd.read_csv("C:\Users\Pasumpon\Desktop\headbrain.csv")

filename1 = os.path.join(current_file, 'C:/Users/Pasumpon/Desktop/headbrain.csv')

filename2 = 'C:/Users/Pasumpon/Desktop/headbrain.csv'

data = pd.read_csv(filename1)
print(data.shape)
#X = data['Head Size(cm^3)'].values
#Y = data['Brain Weight(grams)'].values
X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values


X = np.array(X).reshape((len(X), 1))

Y = np.array(Y).reshape((len(Y), 1))


#X = Y.reshape(1,-1) 
reg= LinearRegression()
reg=reg.fit(X,Y)
Y_pred =reg.predict(X)
print("hello2")
#mean_square =sm.mean_squared_error(Y,Y_pred)
r2_score = reg.score(X, Y)
print(r2_score)
