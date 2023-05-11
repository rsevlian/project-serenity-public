import pandas as pd 
import logging 
import requests 
import datetime 
import numpy as np 
import matplotlib.pyplot as plt 
  
import pylab 
  
import scipy.stats as stats 
from matplotlib import pyplot 
  
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.tsa.stattools import acf, pacf 
  
import mining_constants 
  
############################################################################################################ 
#                                                                                                        ## 
############################################################################################################ 

  
def btc_returns_hashrate_statistics(sp, return_series, max_qnt=0.000001): 
  
    max_percentile = return_series.quantile(1 - max_qnt) 
    min_percentile = return_series.quantile(max_qnt) 
  
    fltr_return_series = return_series[ 
        (return_series <= max_percentile) & (return_series >= min_percentile) 
    ].dropna() 
  
    plt.figure(figsize=(7, 5)) 
    plt.title("BTC Returns Frequency") 
    plt.hist(fltr_return_series, bins=20) 
    plt.grid() 
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.title("QQ-Plot price returns")
    _ = stats.probplot(fltr_return_series, dist="norm", plot=pylab) 
    plt.grid() 
    plt.show()

    # plt.figure(figsize=(4, 3))

    plot_acf(fltr_return_series, lags=15) 
    plt.title("Returns ACF")
    plt.grid() 
    plt.show() 
  

    ols_time      = sp._lm_data["x"][:, 1]
    ols_true_hr   = sp._lm_data["y"]
    ols_fitted_hr = sp._lm.fittedvalues 
    offset = ols_fitted_hr[0] - ols_true_hr[0] 

    t0 = 2017
    plt.figure(figsize=(7, 5))
    plt.plot(t0 + ols_time, ols_fitted_hr - offset, label="OLS Fitted Historical Hash Rate") 
    plt.plot(t0 + ols_time, ols_true_hr, label="OLS True Hash Rate") 
    plt.title("Network Hashrate") 
    plt.legend()
    plt.grid() 
    plt.show() 
  


# def plot_backtest_dict_exogenous_data(sp): 
  
#     ols_time      = sp._lm_data["x"]["year"].values 
#     ols_true_hr   = sp._lm_data["y"].values 
#     ols_fitted_hr = sp._lm.fittedvalues.values 
  
#     offset = ols_fitted_hr[0] - ols_true_hr[0] 
  
#     plt.figure(figsize=(7, 5))
#     plt.plot(ols_time, ols_fitted_hr- offset, label="OLS Fitted Historical Hash Rate") 
#     plt.plot(ols_time, ols_true_hr, label="OLS True Hash Rate") 
#     plt.plot(t + ols_time[-1], HRt, label="Projected Hash Rate") 
#     plt.title("Network Hashrate") 
#     plt.grid() 
  
  
    plt.show() 
  

