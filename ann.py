#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:43:55 2019

@author: saikat
"""

import numpy as np
import pandas as pd

#Importing Data Set
dataset = pd.read_csv('Churn_Modelling.csv')

x=dataset.iloc[:,3:13].values#Indipendent Variable
y = dataset.iloc[:,13].values#Dependent Variable

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncodr_x1 = LabelEncoder()
labelEncodr_x2 = LabelEncoder()
oneHotEncoder = OneHotEncoder(categorical_features=[1])
x[:,1] = labelEncodr_x1.fit_transform(x[:,1]) #Converting Country Label To Number
x[:,2] = labelEncodr_x2.fit_transform(x[:,2]) #Converting Gender Label To Number
x = oneHotEncoder.fit_transform(x).toarray() #Createing Dummy Variable
x = x[:,1:] #Removeing Dummy Variable Trap

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)#Spliting Dataset to Train Test

#Normalize The Data 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units=11,kernel_initializer='uniform',activation='relu'))#Input Layer Number of Nuron,Uniformly Distributed Weight,Activation Function is Rectifier
classifier.add(Dense(units= 6,kernel_initializer='uniform',activation='relu'))#First Hidden Layer Activation Function is Rectifier
classifier.add(Dense(units= 1,kernel_initializer='uniform',activation='sigmoid'))#Output Layer Activation Function is Sigmoid
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])#Output Layer
classifier.fit(x_train,y_train,batch_size=10,epochs=100)#Fiting Data to the Network 

#Prediction
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)

#The Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)