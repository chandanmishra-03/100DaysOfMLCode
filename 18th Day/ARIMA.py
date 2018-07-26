# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:02:37 2018

@author: OpenSource
"""


from yahoo_finance import Share
from pandas import Series,DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import statsmodels.tsa.stattools as ts
from scipy import  stats
import pywt

yaho = Share('YHOO') #choose stock, YAHOO, GOLD
startday='2015-11-1' #choose first day
endday='2016-12-15' #choose end day


#draw
fig=plt.figure()
ax1=fig.add_subplot(711)
ax2=fig.add_subplot(712)
ax3=fig.add_subplot(713)
ax4=fig.add_subplot(714)
ax5=fig.add_subplot(715)
ax6=fig.add_subplot(716)
ax7=fig.add_subplot(717)

#Data processing
StockDate = DataFrame(yaho.get_historical(startday, endday))
StockDate.index = StockDate.Date
StockDate = DataFrame.sort_index(StockDate) #sort

test = DataFrame(yaho.get_historical(startday, '2016-12-1'))
test.index = test.Date
test = DataFrame.sort_index(test)
test = test['Close']
test=test.astype(float)
test.plot(ax=ax5)


#ARIMA
Close_original = StockDate['Close']
Close_original=Close_original.astype(float)
close=pywt.dwt(Close_original, 'db4') #DB4,Wavelet decomposition
Close_db4=pd.Series(close[0])
Close_db4=Close_db4-14
Close_db4.index = pd.Index(sm.tsa.datetools.dates_from_range('2001','2145'))
Close_db4.plot(ax=ax2)
Close=Close_db4.diff(4) #stationary time series
Close=Close[4:]

print("Augmented Dickey-Fuller test:",ts.adfuller(Close,4)) #Augmented Dickey-Fuller test
Close.plot(ax=ax3)
Close_original.plot(ax=ax1)

sm.graphics.tsa.plot_acf(Close,lags=40,ax=ax6) #ARIMA,q
sm.graphics.tsa.plot_pacf(Close,lags=40,ax=ax7) #ARIMA,p

Arma = sm.tsa.ARMA(Close,order=(9,3)).fit(disp=-1, method='mle')
print(Arma.aic,Arma.bic,Arma.hqic)

Arma_stock=Arma.predict()
Arma_stock.plot(ax=ax3)
predict_stock = Arma.predict('2137','2148',dynamic=True)
predict_stock.plot(ax=ax3)

#reduce diff()
L=len(Arma_stock)
i=0
while i<L:
	if(i<4):
		Arma_stock[i]=Arma_stock[i]+Close_db4[i]
	else:
		Arma_stock[i] = Arma_stock[i]+Arma_stock[i-4]
	i=i+1
Arma_stock.plot(ax=ax4)
L=len(predict_stock)
i=0
while i<L:
	if(i<4):
		predict_stock[i] = predict_stock[i]+Arma_stock[-4+i]
	else:
		predict_stock[i] = predict_stock[i]+predict_stock[i-4]
	i=i+1	
predict_stock.plot(ax=ax4)

plt.grid(True)
plt.show()


