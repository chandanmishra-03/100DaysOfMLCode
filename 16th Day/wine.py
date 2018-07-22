# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:13:45 2018

@author: OpenSource
"""

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


init_data = pd.read_csv("winemag-data_first150k.csv")
print("Length of dataframe before duplicates are removed:", len(init_data))
init_data.head()

parsed_data = init_data[init_data.duplicated('description', keep=False)]
print("Length of dataframe after duplicates are removed:", len(parsed_data))

parsed_data.dropna(subset=['description', 'points'])
print("Length of dataframe after NaNs are removed:", len(parsed_data))

parsed_data.head()

dp = parsed_data[['description','points']]
dp.info()
dp.head()
fig, ax = plt.subplots(figsize=(30,10))
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20) 
ax.set_title('Number of wines per points', fontweight="bold", size=25)
ax.set_ylabel('Number of wines', fontsize = 25)
ax.set_xlabel('Points', fontsize = 25)
dp.groupby(['points']).count()['description'].plot(ax=ax, kind='bar')

dp = dp.assign(description_length = dp['description'].apply(len))
dp.info()
dp.head()


fig, ax = plt.subplots(figsize=(30,10))
sns.boxplot(x='points', y='description_length', data=dp)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20) 
ax.set_title('Description Length per Points', fontweight="bold", size=25)
ax.set_ylabel('Description Length', fontsize = 25)
ax.set_xlabel('Points', fontsize = 25) 
plt.show()

#Transform method taking points as param
def transform_points_simplified(points):
    if points < 84:
        return 1
    elif points >= 84 and points < 88:
        return 2 
    elif points >= 88 and points < 92:
        return 3 
    elif points >= 92 and points < 96:
        return 4 
    else:
        return 5

#Applying transform method and assigning result to new column "points_simplified"
dp = dp.assign(points_simplified = dp['points'].apply(transform_points_simplified))
dp.head()

fig, ax = plt.subplots(figsize=(30,10))
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)
ax.set_title('Number of wines per points', fontweight="bold", size=25)
ax.set_ylabel('Number of wines', fontsize = 25)
ax.set_xlabel('Points', fontsize = 25) 
dp.groupby(['points_simplified']).count()['description'].plot(ax=ax, kind='bar')


fig, ax = plt.subplots(figsize=(30,10))
sns.boxplot(x='points_simplified', y='description_length', data=dp)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_title('Description Length per Points', fontweight="bold", size=25) 
ax.set_ylabel('Description Length', fontsize = 25)
ax.set_xlabel('Points', fontsize = 25) 
plt.show()
X = dp['description']
y = dp['points_simplified']
vectorizer = CountVectorizer()
vectorizer.fit(X)


X = vectorizer.transform(X)
print('Shape of Sparse Matrix: ', X.shape)
print('Amount of Non-Zero occurrences: ', X.nnz)
# Percentage of non-zero values
density = (100.0 * X.nnz / (X.shape[0] * X.shape[1]))
print('Density: {}'.format((density)))

# Training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Testing the model
predictions = rfc.predict(X_test)
print(classification_report(y_test, predictions))

X = dp['description']
y = dp['points_simplified']

# Vectorizing model
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# Training model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Testing model
predictions = rfc.predict(X_test)
print(classification_report(y_test, predictions))

