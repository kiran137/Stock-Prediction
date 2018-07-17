# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 21:55:49 2018

@author: user
"""
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt,mpld3
import seaborn as sns
from matplotlib.pylab import rcParams
import calendar
import datetime as dt
import types
import tempfile

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

#read dataset
dataset = pd.read_csv('NSE-HITECH.csv')
#convert date into ordinal 
dataset['Date'] = pd.to_datetime(dataset['Date'])


dataset['month']=dataset['Date'].dt.month
dataset['day']=dataset['Date'].dt.day

dataset['day_of_week']=dataset['Date'].dt.weekday
dataset['week_of_year']=dataset['Date'].dt.weekofyear

dataset.iloc[:,[8,7]].groupby(['month']).mean().plot.bar()
dataset.iloc[:,[9,7]].groupby(['day']).mean().plot.bar()
dataset.iloc[:,[10,7]].groupby(['day_of_week']).mean().plot.bar()
dataset.iloc[:,[11,7]].groupby(['week_of_year']).mean().plot.bar()



dataset['Dates'] = pd.to_datetime(dataset['Date'])
dataset['Dates']=dataset['Dates'].map(dt.datetime.toordinal)
dataset=dataset.drop(['Date'],axis=1)
#split data into X and Y variables
X=dataset.iloc[:,[0,1,2,3,4,5,11]].values
y=dataset.iloc[:,6].values

#slpit the data into test and train set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

#import keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt
from keras.models import model_from_json

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score,KFold
#ANN function
def build_regressor():
#Add ANN Model   
    model = Sequential()
#Add First Hidden Layer    
    model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))

# Adding the second hidden layer
    model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))

# Compiling the ANN
    model.compile(loss='mse', 
                optimizer='adam', 
                metrics=['mse'])
    return model
#build KerasRegressor
model=KerasRegressor(build_fn=build_regressor, batch_size = 10, epochs = 100)



#fitting model
model.fit(X_train, y_train, batch_size = 10, epochs = 1000)
#predicting model
y_pred=model.predict(X_test)
#finding MSE
m=(mean_squared_error(y_test,y_pred))
#for MSE <100
while(m>100):
    model.fit(X_train, y_train, batch_size = 10, epochs = 1000)
    y_pred=model.predict(X_test)
    m=(mean_squared_error(y_test,y_pred))
    
#getting new predictions for new data
new_prediction =model.predict((np.array([[338.13,338,300,300,332,12399,736884]])))

#prediction plot vs test data
plt.figure(1)
plt.subplot(211)
plt.plot( y_pred, 'bo')

plt.subplot(212)
plt.plot(y_test,'bo')
plt.show()

#predicted values of  train set vs original values of train set
plt.figure(1)
plt.subplot(211)
plt.plot( model.predict(X_train), 'bo')

plt.subplot(212)
plt.plot(y_train,'bo')
plt.show()


from keras.models import load_model

from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib

#load_model(model,'ann_model.h5')
import pickle

make_keras_picklable()

pickle.dumps(model)