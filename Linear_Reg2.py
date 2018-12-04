# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 04:29:16 2018

@author: Pasumpon
"""

import pandas as pd
import numpy as np
import os

os.system('cls')
from sklearn.linear_model import LinearRegression 

current_file = os.path.abspath(os.path.dirname(__file__))

filename1 = os.path.join(current_file, 'C:/Users/Pasumpon/Desktop/student.csv')

#data=pd.read_csv('C:\Users\Pasumpon\Downloads\potential-enigma-master\potential-enigma-master\student.csv')
data =pd.read_csv(filename1)

#Math,Reading,Writing
math =data['Math'].values
read =data['Reading'].values
write =data['Writing'].values


X = np.array([math, read]).T
Y = np.array(write)
reg= LinearRegression()
reg = reg.fit(X, Y)
# Y Prediction
Y_pred = reg.predict(X)

r2 = reg.score(X, Y)

print(r2)
