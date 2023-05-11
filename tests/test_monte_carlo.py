
import numpy as np
import random
import pandas as pd
import datetime
import sys
sys.path.append("../lib")

import data_utils
import monte_carlo
import mining_constants

def test_gbm_process(): 
  
	random.seed(0)
	s = 0.3
	M = 100
	N = 1000
	T = 1
	S = 1
	r = 0
	t, St, Rt = monte_carlo.gbm_simulate(S, T, r, sigma = s / np.sqrt(M), steps=M, N=N)

	assert t[0] == 0
	assert t[-1] == T
	assert len(t) == M
	print(St.shape, Rt.shape)
	assert St.shape == (M, N)
	assert Rt.shape == (M, N)

	assert round(np.std(Rt[1:, :]), 3) == s/np.sqrt(M)
	assert round(np.diff(np.log(St.T)).std(), 3) == s/np.sqrt(M)


def test_GBMLinearHashrate():

	## generate features.
	du = data_utils.DataUtility("")
	du._raw_network_price_df = pd.read_csv("../data/raw_network_price.csv").drop("Unnamed: 0", axis="columns")
	du._raw_network_price_df.time = pd.to_datetime(du._raw_network_price_df.time)
	du._df_feature() 
	du._aggregate_weekly_df() 

	## 
	gg = monte_carlo.GBMPriceLinearHashRate()
	gg.fit(
		du._weekly_network_price_df, 
		sample_rate = mining_constants.SAMPLE_RATE_WEEKLY
	)

	assert gg._gbm_params == {'mu': 0.005344673686849799, 'sigma': 0.0873135992840332}
	assert (int(gg._lm.params[0]) == 60684392) & (int(gg._lm.params[1]) == 55942997)

	np.random.seed(0)
	start_time  = datetime.datetime(2022, 4, 14).date()
	hashrate    = 346035351.07423794 
	hash_index  = 0.07690399113505421 
	btc_close   = 29568.332857142854
	T           = 1
	N           = 100
	sample_rate = mining_constants.SAMPLE_RATE_WEEKLY

	ttm_vec, ttm_dt_vec, St, HRt, wt_daily, Ht = gg.simulate(start_time, btc_close, hashrate, T, sample_rate, N)
	assert St.sum().sum() == 167583680.79133713
	
