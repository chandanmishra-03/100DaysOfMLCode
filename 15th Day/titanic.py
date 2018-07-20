# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 19:13:50 2018

@author: OpenSource
"""
import numpy as np
import pandas as pd
from sklearn import cluster, preprocessing


df = pd.read_excel("titanic.xls")
df.head()
df.drop(['name','body'], 1, inplace=True)
df.head()
df.fillna(0, inplace=True)
df.head()

#now to convert the non numerical data into numerical for convinience
columns = df.columns.values
for i in columns:
    text_value_int = {}
    def text_to_val(val):
        return text_value_int[val]
    if df[i].dtype != np.int64 and df[i].dtype != np.float64:
        all_text = df[i].values.tolist()
        unique_elements = set(all_text)
        
        x = 0
        for unique in unique_elements:
            if unique not in text_value_int:
                text_value_int[unique] = x
                x+=1
        
        df[i] = list(map(text_to_val, df[i]))

df.head()

X = np.array(df.drop('survived', 1))
y = np.array(df['survived'])

clf = cluster.KMeans(n_clusters=2)
clf.fit(X)


found = 0
for i in range(len(X)):
    new_prediction = np.array(X[i].astype(float))
    new_prediction = new_prediction.reshape(-1, len(new_prediction))
    prediction = clf.predict(new_prediction)
    if prediction[0] == y[i]:
        found += 1

accuracy = (found/len(X))*100
accuracy

X = preprocessing.scale(X)
clf.fit(X)
found = 0
for i in range(len(X)):
    new_prediction = np.array(X[i].astype(float))
    new_prediction = new_prediction.reshape(-1, len(new_prediction))
    prediction = clf.predict(new_prediction)
    if prediction[0] == y[i]:
        found += 1

accuracy = (found/len(X))*100
print(accuracy)
