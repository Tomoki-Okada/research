#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 14:23:50 2017

@author: okadatomoki

Bouchaud et al. (2004)の手法の実装

"""
import numpy as np
import pandas as pd
import os
import datetime
import time

# ティックデータから必要な情報を読み込む
# {'mid_price', 'volume', 'trading_sign'}のリストを作成
file_list = os.listdir('9984/')
D_data = []
for file in file_list:
    tick_data = pd.read_csv('9984/' + file, header=None)
    transacton_flags = tick_data[0]
    best_ask = tick_data[27]
    best_bid = tick_data[16]
    processed_data = []
    for i, flag in enumerate(transacton_flags):
        if flag == 1:
            mid_price = (best_ask[i-1] + best_bid[i-1])/2
            transaction_price = tick_data[11][i]
            volume = tick_data[12][i]
            if transaction_price >= mid_price:
                trading_sign = 1
            else:
                trading_sign = -1
            processed_data.append({'mid_price':mid_price, 'volume':volume, 'trading_sign':trading_sign})
    D_data.append(processed_data)       
     
# response関数の計算
def response(D_data, l):
    _sum = 0
    denominator = 0
    for data in D_data:
        Nd = len(data)
        for n in range(Nd-l):
            _sum += (data[n+l]['mid_price'] - data[n]['mid_price']) * data[n]['trading_sign']
        denominator += Nd - l
    return _sum / denominator

# correlation関数の計算
def correlation(D_data, l):
    _sum = 0
    denominator = 0
    for data in D_data:
        Nd = len(data)
        for n in range(Nd-l):
            _sum += data[n+l]['trading_sign'] * data[n]['trading_sign'] * np.log(data[n]['volume'])
        denominator += Nd - l
    return _sum / denominator    

# 減衰核のノンパラメトリック推定(連立方程式を解く)
