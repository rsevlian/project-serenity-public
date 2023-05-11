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
def plot_hash_index(weekly_df): 
  
    """  
         subplot 1: {t, hr(t)} & {t, p(t)} 
         subplot 2: {t, ir(t)} & {t, p(t)} 
    """ 
    plt.figure(figsize=(24, 5)) 
    plt.subplot(1, 3, 1) 
    plt.plot(weekly_df.hashrate, label="hash-rate") 
    plt.plot(weekly_df.close/np.max(weekly_df.close), label="price") 
    plt.grid() 
    plt.legend() 
  
    plt.subplot(1, 3, 2) 
    plt.plot(weekly_df.close/np.max(weekly_df.close), label="price") 
    plt.grid() 
    plt.legend() 
    plt.subplot(1, 3, 3) 
    plt.plot(weekly_df.hash_index/np.max(weekly_df.hash_index), label="hash-index") 
    plt.grid() 
    plt.legend() 
  
  
def hash_index_returns_plots(weekly_df): 
  
    plt.figure(figsize=(25, 5)) 
  
    # 
    plt.subplot(1, 4, 1) 
    plt.title("HI Returns") 
    plt.hist(weekly_df.hash_index_return, bins=20) 
    plt.grid() 
  
    plt.subplot(1, 4, 2) 
    plt.title("HI log-returns") 
    plt.hist(weekly_df.hash_index_return_log, bins=20) 
    plt.grid() 

    plt.subplot(1, 4, 3) 
    plt.title("HI returns") 
    _ = stats.probplot(weekly_df.hash_index_return, dist="norm", plot=pylab) 
    plt.grid() 
  
    plt.subplot(1, 4, 4) 
    plt.title("HI log-returns") 
    _ = stats.probplot(weekly_df.hash_index_return_log, dist="norm", plot=pylab) 
    plt.grid() 
  
    plt.show() 
  
  
def hash_index_filtered(return_series, max_qnt=0.01): 
  
    max_percentile = return_series.quantile(1 - max_qnt) 
    min_percentile = return_series.quantile(max_qnt) 
  
    fltr_return_series = return_series[ 
        (return_series <= max_percentile) & (return_series >= min_percentile) 
    ].dropna() 
  
    # fltr_return_series.head() 
  
    plt.figure(figsize=(14, 5)) 
    plt.subplot(1, 2, 1) 
    plt.title("HI log-returns") 
    plt.hist(fltr_return_series, bins=20) 
    plt.grid() 
  
    plt.subplot(1, 2, 2) 
    _ = stats.probplot(fltr_return_series, dist="norm", plot=pylab) 
    plt.grid() 
  
    # plt.subplot(1, 3, 3) 
    plt.figure(figsize=(6, 6)) 
    plot_acf(fltr_return_series, lags=15) 
    plt.grid() 
    plt.show() 
  
  
def plot_hash_rate_and_price(weekly_df): 
  
    fig, axs = plt.subplots(figsize=(17, 7)) 
  
    axs.plot(weekly_df.hashrate, label="hash-rate") 
    axs.axvline(mining_constants.LM_S1, color="red", label="S1") 
    axs.axvline(mining_constants.LM_E1, color="red", label="E1") 
    axs.axvline(mining_constants.LM_S2, color="red", label="S1") 
  
    ax1 = axs.twinx() 
    ax1.plot(weekly_df.close, label="price", color='orange') 
  
    axs.grid() 
    axs.legend() 
  
    plt.legend() 
  
  
def plot_spot_hashrate_hashprice(t, St, HRt, wt,  Ht, mc_size=50): 
  
    plt.figure(figsize=(18, 5)) 
  
    plt.subplot(1, 4, 1) 
    _ = plt.plot(t, St[:, 1:mc_size]) 
    plt.title("GBM-Price") 
    plt.grid() 
  
    plt.subplot(1, 4, 2)
    plt.plot(t, HRt, label="Projected Hash Rate") 
    plt.title("Network-HashRate") 
    plt.legend() 
    plt.grid() 
  
    plt.subplot(1, 4, 3) 
    plt.plot(t, wt) 
    plt.title("Period Reward (BTC)") 
    plt.grid() 
  
    plt.subplot(1, 4, 4) 
    plt.plot(t, Ht[:, 1:mc_size]) 
    plt.title("Hash-Index") 
    plt.grid() 
  
  
def plot_backtest_dict_exogenous_data(backtest_dict, mc_size=50): 
  
    t = backtest_dict["t"] 
    St = backtest_dict["St"] 
    HRt = backtest_dict["HRt"] 
    Ht = backtest_dict["Ht"] 
  
    ols_time = backtest_dict["stochastic_process"]._lm_data["x"]["year"].values 
    ols_true_hr = backtest_dict["stochastic_process"]._lm_data["y"].values 
    ols_fitted_hr = backtest_dict["stochastic_process"]._lm.fittedvalues.values 
  
    offset = ols_fitted_hr[0] - ols_true_hr[0] 
    # offset = 0 
  
    plt.figure(figsize=(18, 5)) 
  
    plt.subplot(1, 3, 1) 
    _ = plt.plot(t, St[:, 1:mc_size]) 
    plt.title("GBM Price") 
    plt.grid() 
  
    plt.subplot(1, 3, 2) 
    plt.plot(ols_time, ols_fitted_hr- offset, label="OLS Fitted Historical Hash Rate") 
    plt.plot(ols_time, ols_true_hr, label="OLS True Hash Rate") 
    plt.plot(t + ols_time[-1], HRt, label="Projected Hash Rate") 
    plt.title("Network Hashrate") 
    plt.legend() 
    plt.grid() 
  
    plt.subplot(1, 3, 3) 
    plt.plot(t, Ht[:, 1:mc_size]) 
    plt.title("Hash Index") 
    plt.grid() 
  
    # plt.subplot(1, 4, 3) 
    # plt.plot(t, wt) 
    # plt.title("Period Reward (BTC)") 
    # plt.grid() 
  
    plt.show() 
  
  
def plot_loan_default(loan_default_df, loan_value, title_str): 
  
    fig = plt.figure(figsize=(15, 6)) 
    ax = fig.add_subplot(111) 
    ax.set_title(title_str) 
    ax.plot(loan_default_df.time, loan_default_df.hash_index, label="hash-index") 
    ax.plot(loan_default_df.time, loan_default_df.hash_index_strike_1, label="h-min: electricity cost") 
    ax.plot(loan_default_df.time, loan_default_df.hash_index_strike_2, label="h-min: electricity + loan-repayment cost") 
    ax.set_ylim((0, loan_default_df.hash_index.max())) 
    ax.set_ylabel("hash-index") 
    ax.legend() 
    ax2 = ax.twinx() 
    ax2.plot(loan_default_df.time, loan_default_df.pr_default, color="red", label="PrDefault") 
    ax2.set_ylim((0, 1.0)) 
    ax2.legend(loc="right") 
    ax2.set_ylabel("PrDefault") 
    ax2.grid() 
  
    fig = plt.figure(figsize=(15, 6)) 
    ax = fig.add_subplot(111) 
    ax.set_title(title_str) 
    ax.plot(loan_default_df.time, loan_default_df.hash_index, label="hash-index") 
    ax.plot(loan_default_df.time, loan_default_df.hash_index_strike_1, label="h-min: electricity cost") 
    ax.plot(loan_default_df.time, loan_default_df.hash_index_strike_2, label="h-min: electricity + loan-repayment cost") 
    ax.set_ylim((0, loan_default_df.hash_index.max())) 
    ax.set_ylabel("hash-index") 
    ax.legend() 
    ax2 = ax.twinx() 
    ax2.plot(loan_default_df.time, abs(loan_default_df.expected_loss / loan_value), color="red", label="Expected Loss") 
    # ax2.set_ylim((0, 1.0)) 
    ax2.legend(loc="right") 
    ax2.set_ylabel("Relative Loss") 
    ax2.grid()


def plot_single_replication(df, K): 
    """
        Analaysis Plots for single strip delta hedging outcome.
        Input:
            df 
                index               - timestamps
                delta               - delta 
                price               - price(t)
                purchase_quantity   - 
                discounted-purchase - 
        Output:
            None.
    """
    n = 6 
    plt.figure(figsize=(20, 10)) 
    plt.subplot(2, 3, 1) 
    plt.plot(df.delta) 
    plt.title("delta(t)") 
    plt.grid() 

    plt.subplot(2, 3, 2) 
    plt.plot(df.price) 
    plt.axhline(K) 
    plt.title("price(t).") 
    plt.grid() 

    plt.subplot(2, 3, 3) 
    plt.plot(df.purchase_quantity) 
    plt.title("purchase_quantity(t).") 
    plt.grid() 

    plt.subplot(2, 3, 4) 
    plt.plot(df.discounted_purchase_value) 
    plt.title("purchase_value(t)") 
    plt.grid() 

    plt.subplot(2, 3, 5) 
    plt.plot(df.purchase_quantity.cumsum()) 
    plt.title("total_purchase_quanity(t)") 
    plt.grid() 

    plt.subplot(2, 3, 6) 
    plt.plot(df.discounted_purchase_value.cumsum()) 
    plt.title("total_purchase_value(t)") 
    plt.grid() 

    plt.show() 
  
  
def plot_aggregate_replication(agg_df, results): 

    logging.info(results) 

    plt.figure(figsize=(20, 8)) 

    plt.subplot(2, 2, 1) 
    plt.plot(agg_df.price, label="price") 
    plt.plot(agg_df.strike[agg_df.strike != 0], linestyle='--', label="strike") 
    plt.title("Price") 
    plt.legend() 
    plt.grid() 

    plt.subplot(2, 2, 2) 
    plt.plot(agg_df.agg_delta) 
    plt.title("Aggregate Delta") 
    plt.grid() 

    plt.subplot(2, 2, 3) 
    plt.plot(-1 * agg_df.purchase_quantity.cumsum(), label="Short BTC Position")
    plt.plot(agg_df.purchase_quantity.cumsum(), label="Short BTC Position") 
    plt.legend()
    plt.grid() 

    plt.subplot(2, 2, 4) 
    plt.plot(agg_df.cash_balance, label="cash_balance") 
    plt.plot(agg_df.net_cash_balance, label="cash_balance - strike") 
    plt.axhline(results["machine_cost"], label="machine_cost") 
    plt.title("cash balance") 
    plt.legend() 
    plt.ylim((0, agg_df.cash_balance.max())) 
    plt.grid() 

    plt.show() 


def plot_replication(df, machine_cost):

    df = result_df
    qmax = df["dh_btc_short_position"].abs().max()

    plt.figure(figsize=(16, 3))
    plt.subplot(1, 2, 1)
    plt.plot(df.price, label="price") 
    plt.plot(df.strike_true.dropna(), linestyle='--', label="strike (TR)") 
    plt.plot(df.strike_init.dropna(), linestyle='--', label="strike") 
    plt.title("Price") 
    plt.legend() 
    plt.grid() 
    plt.subplot(1, 2, 2)
    plt.plot(df.agg_delta) 
    plt.title("Aggregate Delta") 
    plt.grid() 
    plt.show()


    plt.figure( figsize=(16, 3))
    plt.subplot(1, 2, 1)
    plt.plot(df.total_reward_true, color="g", label="Total BTC Reward (I)")
    plt.plot(df.total_reward_init, color="r", label="Total BTC Reward (TR)")
    ax1.legend()
    plt.grid() 
    plt.subplot(1, 2, 2)
    plt.plot(df.dh_btc_short_position, color="g", label="Short BTC Position")
    plt.grid() 
    plt.show()

    plt.figure( figsize=(16, 3))
    plt.subplot(1, 2, 1)
    plt.plot(df.net_btc_position_init, label="Net BTC Position (Init)")
    plt.plot(df.net_btc_position_true, label="Net BTC Position (TR)")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2) 
    plt.plot(df.cash_balance, label="Long Cash Position") 
    plt.plot(df.net_cash_balance, label="Net Cash Position") 
    plt.axhline(machine_cost, label="Machine Cost(t=0)") 
    plt.legend() 
    plt.grid()
    plt.show()