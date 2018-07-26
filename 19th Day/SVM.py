# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 15:02:37 2018

@author: OpenSource
"""


from yahoo_finance import Share
from pandas import Series,DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

yaho = Share('YHOO') #choose stock
startday='2015-11-1' #choose first day
endday='2016-11-23' #choose end day
train = 15 

#Data processing
StockDate = DataFrame(yaho.get_historical(startday, endday))
StockDate.index = StockDate.Date
StockDate = DataFrame.sort_index(StockDate) #sort

L=len(StockDate)
total_predict_data=L-train

#draw
Data = StockDate.drop(['Date','Symbol','Adj_Close'],axis=1) 
Data=Data.astype(float)
ax=Data.plot(secondary_y=['Volume'])
ax.set_ylabel('Value')
ax.right_ax.set_ylabel('Volume')

#Create more data
value = pd.Series(Data['Open'].shift(-1)-Data['Close'].shift(-1),index=Data.index)
Data['Next_Open'] = Data['Open'].shift(-1) #Next day's Open data.
value[value>=0]=0 #0 means fall
value[value<0]=1 #1 means rise
print(value)


#Data['Value']=value
correct = 0
train_original=train
i=0

#loop training,15 days data for train
while train<L-1:
	Data_train=Data[train-train_original:train]
	value_train = value[train-train_original:train]
	Data_predict=Data[train:train+1]
	value_real = value[train:train+1]
	classifier = svm.SVC(kernel='poly')#kernel='linear')
	classifier.fit(Data_train,value_train)
	value_predict=classifier.predict(Data_predict)
	if(value_real[0]==int(value_predict)):
		correct=correct+1
	train = train+1
correct=correct/total_predict_data*100
print("Correct = ",correct,"%")






