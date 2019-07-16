#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspired by sentdex

@author: dpong
"""

import pandas as pd
import matplotlib.pyplot as plt
import quandl, datetime
import math, pickle
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import style
from talib import abstract

style.use('ggplot')


def get_data(ticker):
    df = quandl.get('WIKI/'+ticker)
    df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
    with open('{}_data_from_quandl.pickle'.format(ticker),'wb') as f:
        pickle.dump(df,f)

def ML_training(ticker):
    
    pickle_in = open('{}_data_from_quandl.pickle'.format(ticker),'rb') 
    df = pickle.load(pickle_in)
    
    df['HL_PCT'] = (df['Adj. High']-df['Adj. Low'])/df['Adj. Close'] * 100
    df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100
    #用talib一定要rename一下key才會對應的到
    df.rename(columns = {'Adj. Open':'open',
                         'Adj. High':'high',
                         'Adj. Low':'low',
                         'Adj. Close':'close'}, inplace = True)
    #STOCH = abstract.STOCH(df)  #算出KD
    #df['SlowK'] = STOCH['slowk']
    #df['SlowD'] = STOCH['slowd']
    df['RSI'] = abstract.RSI(df) #算出RSI

    df = df[['close','HL_PCT','PCT_change','Adj. Volume','RSI']]

    df.fillna(-99999,inplace=True) #pandas 把沒資料的都改成-99999

    forecast_out = int(math.ceil(0.005*len(df))) #4捨5入

    df['label'] = df['close'].shift(-forecast_out)
    
    df_preprocess = df[['close','HL_PCT','PCT_change','Adj. Volume','RSI']]
    #把close納入的預測準確度會大幅提升
    
    X = np.array(df_preprocess)
    X = preprocessing.scale(X)
    
    X = X[:-forecast_out]
    X_lately = X[-forecast_out:]
    
    df_origin = df
    df_origin = df_origin['close'] #之後畫圖的對照
    
    df.dropna(inplace=True)
    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    clf = LinearRegression()
                       
    clf.fit(X_train,y_train)
    
    accuracy = clf.score(X_test,y_test)

    print(accuracy)
    
    pickle_save = [clf,df,X_lately,forecast_out,df_origin]
    
    with open('{}_LinearRegression.pickle'.format(ticker),'wb') as f:
        pickle.dump(pickle_save,f)   


def prediction(ticker,days=100):
    pickle_in = open('{}_LinearRegression.pickle'.format(ticker),'rb') 
    clf2,df,X_lately,forecast_out,df_origin = pickle.load(pickle_in) 

    
    forecast_set = clf2.predict(X_lately)
    
    df['Forecast_Close'] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += one_day
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
    
    
    df = df.drop(columns=['HL_PCT','PCT_change','label','RSI'])
  
    df_origin=df_origin[:-6] #調整數據
    df = df[-days:]
    df_origin=df_origin[-days+20:]
    
    df['Combined_Close'] = df['close']
    for i in range(1,forecast_out+1):
        df['Combined_Close'].iloc[-i] = df['Forecast_Close'].iloc[-i]
    
    df_origin.plot()
    df['Forecast_Close'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('{}'.format(ticker))
    







