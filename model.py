# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 21:07:03 2020

@author: Keerti
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split


#importing dataset
from sklearn.datasets import load_breast_cancer

cancer= load_breast_cancer()

df=pd.DataFrame(np.c_[cancer['data'],cancer['target']], columns= np.append(cancer['feature_names'],['target']))

# data visualization
sns.countplot(df['target'])
sns.pairplot(df, hue='target',vars=['mean radius','mean smoothness','mean area','mean perimeter'])
sns.heatmap(df.corr())

# Model
X=df.drop(['target'], axis=1)
y=df['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

model=SVC()

# normalization and optimization
min_train=X_train.min()
range_train=(X_train-min_train).max()
X_norm= (X_train-min_train)/range_train

