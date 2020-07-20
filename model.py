# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 21:07:03 2020

@author: Keerti
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing dataset
from sklearn.datasets import load_breast_cancer

cancer= load_breast_cancer()

df=pd.DataFrame(np.c_[cancer['data'],cancer['target']], columns= np.append(cancer['feature_names'],['target']))

# data visualization
sns.countplot(df['target'])
sns.pairplot(df, hue='target',vars=['mean radius','mean smoothness','mean area','mean perimeter'])
sns.heatmap(df.corr())
