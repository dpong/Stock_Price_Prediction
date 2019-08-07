#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspired by sentdex

@author: dpong
"""

import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import style
from datetime import datetime, date, timedelta

style.use('ggplot')

class Stock_price_prediction():
    
    def __init__ (self):
        self.df = pd.DataFrame()
        self.df_origin = pd.DataFrame()
        self.train_accuracy = None
        self.clf = LinearRegression()
        self.predict_date_start = None
        self.predict_length = None
        self.predict_df = pd.DataFrame()
        
    def choose_model(self):
        pass
        
        
    def ML_training(self,ticker,forecast_out=15):
        self.predict_length = forecast_out #提供查詢
        self.ticker = ticker
        self.df_origin = pdr.DataReader(self.ticker,'yahoo')
        
        
        #用talib一定要rename一下key才會對應的到
        self.df_origin.rename(columns = {'Open':'open',
                                  'High':'high',
                                  'Low':'low',
                                  'Close':'close'}, inplace = True)
        self.df_origin.drop(columns=['Adj Close'],inplace=True)
        self.df_origin['HL_PCT'] = (self.df_origin['high']-self.df_origin['low'])/self.df_origin['close'] * 100
        self.df_origin['PCT_change'] = (self.df_origin['close']-self.df_origin['open'])/self.df_origin['open'] * 100
        
        self.df = self.df_origin[:-forecast_out].copy()    #避免pandas一直跳警告
        
        self.predict_date_start = datetime.strptime(str(self.df.iloc[-self.predict_length].name), "%Y-%m-%d %H:%M:%S")
        self.predict_date_start = datetime.date(self.predict_date_start)
        test_df = self.df.dropna(how='any')
        test_df['label'] = self.df['close'].shift(-self.predict_length)
        df_preprocess = test_df.drop(columns=['label'])    
        X = np.array(df_preprocess)
        #X = preprocessing.scale(X)        
        X = X[:-self.predict_length] 
        y = np.array(test_df['label'].dropna(how='any'))  
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
        self.clf.fit(X_train,y_train)  
        self.train_accuracy = self.clf.score(X_test,y_test)            
        
    def prediction_test(self,show_days=20):
        
        self.predict_df = self.df[-self.predict_length * 2:]
        X = np.array(self.predict_df[:-self.predict_length])
        
        #X = preprocessing.scale(X)
        y = self.clf.predict(X)
        #資料整理
        pre_y = np.full(self.predict_length,np.nan)
        y = np.append(pre_y,y)
        self.predict_df['prediction'] = y
        self.df['prediction'] = np.nan
        self.df = self.df[:-self.predict_length-1]
        self.df = pd.concat([self.df,self.predict_df],axis=0)
        #畫圖
        self.df['close'][-show_days:].plot()
        self.df['prediction'][-show_days:].plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('{}'.format(self.ticker))
        
    def predict_future(self):
        self.predict_df = self.df_origin[-self.predict_length:]
        X = np.array(self.predict_df)
        y = self.clf.predict(X)
        print('T: '+str(self.predict_df['close'][-1]))
        for i in range(y.shape[0]):
            print('T+{}: '.format(i+1)+ str(round(y[i],2)))
        

if __name__=='__main__':
    s = Stock_price_prediction()
    s.ML_training('^TWII',forecast_out=10)
    #s.prediction()
    s.predict_future()
    


