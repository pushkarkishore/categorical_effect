# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 14:35:26 2021

@author: admin
"""

from __future__ import print_function, division

from keras.layers import Input, Dense, Activation
from keras.layers.merge import Maximum, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
df = pd.read_csv("C:\\Users\\admin\\Documents\\Scie_output\\apicmax4gram+final.csv",low_memory=False)
df = pd.read_csv("C:\\Users\\admin\\Documents\\Scie_output\\sccmin3gram+final.csv",low_memory=False)
df= pd.read_csv("C:\\Users\\admin\\apin=2.csv")
for j in range(0,8):
    strt = "c2b" + str(j)
    df[strt] = 0
# apifeature_dims = 3087
# apifeature_dims = 10465
apifeature_dims = 1024
z_dims = 10
# hide_layers = 3100
#hide_layers = 11000
hide_layers = 1000
generator_layers = [apifeature_dims+z_dims,hide_layers, apifeature_dims]
# example = Input(shape=(3087,))
example = Input(shape=(1024,))
noise = Input(shape=(10,))
x = Concatenate(axis=1)([example, noise])
for dim in generator_layers[1:]:
    x = Dense(dim)(x)
    x = Activation(activation='sigmoid')(x)
x = Maximum()([example, x])
generator = Model([example, noise], x, name='generator')
generator.summary()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(oecd_bli,oecd_bli['Label']):
    strat_train_set = oecd_bli.iloc[train_index]
    strat_test_set = oecd_bli.iloc[test_index]    
housing = strat_train_set.copy()
housing = strat_train_set.drop("Label", axis=1) 
housing_labels = strat_train_set["Label"].copy()
X_test = strat_test_set.drop("Label", axis=1)
y_test = strat_test_set["Label"].copy()
housing_labels = pd.Categorical(housing_labels).codes
y_test = pd.Categorical(y_test).codes

oecd_bli = df.drop('label',axis=1)
oecd_bli = oecd_bli.drop('label_updated',axis=1)
xtrain_mal, xtest_mal= train_test_split(oecd_bli,test_size=0.20)
batch_size=48841
col_1=[]
for j in range(0,len(list(housing.columns))):
    col_1.append(j)
housing.columns = col_1
idx = np.random.randint(0,oecd_bli.shape[0],2000)
L= list(idx)
xmal_batch = oecd_bli.iloc[L]
xmal_batch = xmal_batch.to_numpy()
df = oecd_bli.loc[L]
noise = np.random.uniform(0, 1, (2000,10))
noise = pd.DataFrame(noise)
noise = noise.to_numpy()
inputsto = [xmal_batch,noise]
gen_examples = generator.predict([xmal_batch,noise])
gen_examples = pd.DataFrame(gen_examples)
ytr=[]
for j in range(0,len(idx)):
    ytr.append(df.iloc[j]['label'])
    
    
gen_examples['Label']= ytr
gen_examples.to_csv("malgansc_virustotal.csv")



df = df.drop("Label", axis=1) 
ulabel = df["Label"].copy()
for j in range(0,49):
    strt = "c25b" + str(j)
    df[strt] = 0
df['Label']=ulabel
blbdata = pd.read_csv("C:\\Users\\admin\\malgansys.csv")
ulabel = blbdata["Label"].copy()
blbdata = blbdata.drop("Label", axis=1) 
for j in range(0,48):
    strt = "c25b" + str(j)
    blbdata[strt] = 0
blbdata['Label']=ulabel

df.iloc[0]['Label']
keras.backend.clear_session()
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df,df['Label']):
    strat_train_set = df.iloc[train_index]
    strat_test_set = df.iloc[test_index]    
housing = strat_train_set.copy()
housing = strat_train_set.drop("Label", axis=1) 
housing_labels = strat_train_set["Label"].copy()
housing_labels_series = strat_train_set["Label"].copy()
X_test = blbdata.drop("Label", axis=1)
y_test = blbdata["Label"].copy()
y_test_series = blbdata["Label"].copy()
housing_labels = pd.Categorical(housing_labels).codes
    # housing_labels = pd.factorize(housing_labels)[0]
y_test = pd.Categorical(y_test).codes
    # y_test = y_test.astype('category').cat.codes
    # housing_prepared_test = num_pipelines.fit_transform(X_test)
    # housing_prepared = num_pipelines.fit_transform(housing)
housing_prepared_test = X_test.to_numpy()
housing_prepared = housing.to_numpy()

avast_test = housing_prepared_test.reshape((2000,56,56))
avast = housing_prepared.reshape((4412,56,56))
avast = avast[...,np.newaxis]
avast_test=avast_test[...,np.newaxis]
   
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters = 6, 
                     kernel_size = 5, 
                     strides = 1, 
                     activation = 'relu', 
                     input_shape = (56,56,1)))
model.add(keras.layers.MaxPooling2D(pool_size = 2, strides = 2))
model.add(keras.layers.Conv2D(filters = 16, 
                     kernel_size = 5, 
                     strides = 1, 
                     activation = 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size = 2, strides = 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units = 120, activation = 'relu'))
model.add(keras.layers.Dense(units = 84, activation = 'relu'))
model.add(keras.layers.Dense(units = 7, activation = 'softmax'))
model.summary()
model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(avast,housing_labels, epochs=100, validation_split=0.1)


score = model.predict(avast_test)
score_lab=[]
for i in range(0,len(score)):
    score_lab.append(np.argmax(score[i]))             
    indexes = list(y_test_series.index)
    for k in range(0,len(indexes)):
        terms = str(indexes[k])
        d[terms].append(score_lab[k])

